"""
α 不変量 (関手族の極限 / 余極限) の 20y 拡張版.

背景:
  5y 版 (compute_alpha_invariant.py) では multi_indicators_w30.csv の 12 指標を
  90日 rolling z-score で集約して
    α_naive(t) = #{|z_i| >= 2} / 12        (余極限 colimit)
    α_norm(t)  = sqrt(Σ z_i²) / sqrt(12)   (極限 limit)
  を計算していた。これを 20y データに拡張して長期再現性を確認する。

実装方針 (簡易版):
  multi_indicators_w30.csv は 5y のみ存在し、20y で 12 指標を再計算するのは
  重い (universe 時変も絡む)。20y 既存資産である gamma_timeseries_20y_w30.csv は
  L1_H1 と n_unb の 2 指標を 20y 通しで持つので、まず簡易版として
    α_naive_2(t) = #{|z_L1| >= 2, |z_n_unb| >= 2} / 2
    α_norm_2(t)  = sqrt(z_L1² + z_n_unb²) / sqrt(2)
  を構築する (look-ahead 防止のため expanding window z-score, min=90)。
  これは関手族の指標数を 2 に縮約した「縮退 α」で、5y 版 (12 指標) と直接の
  数値比較はできないが、shock type ごとの Δσ の相対順序 (trade_policy <
  market_structure 等) と長期安定性を確認する目的にはこの方が頑健である。

入力:
  - data/gamma_timeseries_20y_w30.csv
  - data/events_8y.json
出力:
  - data/alpha_invariant_20y_w30.csv
  - data/alpha_invariant_20y_results.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

INPUT_CSV = DATA_DIR / "gamma_timeseries_20y_w30.csv"
EVENTS_JSON = DATA_DIR / "events_8y.json"
OUTPUT_CSV = DATA_DIR / "alpha_invariant_20y_w30.csv"
OUTPUT_JSON = DATA_DIR / "alpha_invariant_20y_results.json"

# 8y OOS event study と同じ
OOS_START = "2017-11-01"
EXPANDING_MIN_PERIODS = 90

# 5y 版と同じ閾値・窓
EVENT_PRE = 30
EVENT_POST = 30
SIGMA_THRESH = 2.0

# event の "category" -> α 不変量との shock type 対応
# 8y events の category は trade_policy / market_structure / geopolitical /
# macro / monetary / tech_shock の 6 種。5y 版の "war" は 8y では "geopolitical" に
# 統合されている (同じ意味)。
SHOCK_TYPES = [
    "trade_policy",
    "market_structure",
    "geopolitical",
    "monetary",
]


def expanding_zscore(s: pd.Series, min_periods: int = EXPANDING_MIN_PERIODS) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=1)
    return (s - mu) / sd.replace(0, np.nan)


def event_delta_sigma(series: pd.Series, event_date: str | pd.Timestamp,
                       pre: int = EVENT_PRE, post: int = EVENT_POST) -> float | None:
    """Δσ = (post window 平均 - pre window 平均) / pre window std."""
    target = pd.Timestamp(event_date)
    s = series.dropna()
    if target < s.index.min() or target > s.index.max():
        return None
    pos = s.index.get_indexer([target], method="nearest")[0]
    if pos < pre:
        return None
    pre_slice = s.iloc[pos - pre:pos]
    end = min(pos + post, len(s))
    post_slice = s.iloc[pos:end]
    if len(pre_slice) < pre // 2 or len(post_slice) < post // 2:
        return None
    pre_std = float(pre_slice.std(ddof=0))
    if pre_std == 0 or math.isnan(pre_std):
        return None
    return float((post_slice.mean() - pre_slice.mean()) / pre_std)


def event_delta_mean(series: pd.Series, event_date: str | pd.Timestamp,
                      pre: int = EVENT_PRE, post: int = EVENT_POST) -> float | None:
    """既に z-score 化された系列に対する単純な mean 差 (= σ 単位差)."""
    target = pd.Timestamp(event_date)
    s = series.dropna()
    if target < s.index.min() or target > s.index.max():
        return None
    pos = s.index.get_indexer([target], method="nearest")[0]
    if pos < pre:
        return None
    pre_slice = s.iloc[pos - pre:pos]
    end = min(pos + post, len(s))
    post_slice = s.iloc[pos:end]
    if len(pre_slice) < pre // 2 or len(post_slice) < post // 2:
        return None
    return float(post_slice.mean() - pre_slice.mean())


def main() -> int:
    print("=== α 不変量 20y 拡張 ===")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date").set_index("date")
    print(f"[load] 20y: {df.shape}, "
          f"range {df.index.min().date()} -> {df.index.max().date()}")

    df = df.dropna(subset=["L1_H1", "n_unb"]).copy()
    df = df[df.index >= pd.Timestamp(OOS_START)].copy()
    print(f"[OOS filter] {OOS_START}+: {df.shape}, "
          f"range {df.index.min().date()} -> {df.index.max().date()}")

    # ----- expanding z-score (look-ahead 排除) -----
    z_L1 = expanding_zscore(df["L1_H1"])
    z_unb = expanding_zscore(df["n_unb"])
    print(f"[z-score] L1 n={z_L1.notna().sum()}, n_unb n={z_unb.notna().sum()}")

    # 既存 e_div (参考比較用)
    e_div = z_unb - z_L1

    # ----- α 不変量 (2 指標版) -----
    z_df = pd.concat([z_L1.rename("z_L1"), z_unb.rename("z_n_unb")], axis=1)
    abs_z = z_df.abs()
    valid_mask = z_df.notna()
    valid_count = valid_mask.sum(axis=1).replace(0, np.nan)

    over_thresh = (abs_z >= SIGMA_THRESH) & valid_mask
    alpha_naive_2 = over_thresh.sum(axis=1) / valid_count

    sq = (z_df.fillna(0.0) ** 2).sum(axis=1)
    alpha_norm_2 = np.sqrt(sq) / np.sqrt(valid_count.fillna(2))

    # ----- event load -----
    events = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    print(f"[events] {len(events)} events")

    # category -> list of events
    by_cat: dict[str, list[dict]] = {}
    for ev in events:
        by_cat.setdefault(ev["category"], []).append(ev)

    # ----- shock event study -----
    print("\n--- shock event study (Δσ, pre 30bd -> post 30bd) ---")
    study: dict = {}
    for shock in SHOCK_TYPES:
        ev_list = by_cat.get(shock, [])
        per_event: dict = {}
        used: list[str] = []
        for ev in ev_list:
            d_alpha_n = event_delta_sigma(alpha_naive_2, ev["date"])
            d_alpha_f = event_delta_sigma(alpha_norm_2, ev["date"])
            d_ediv = event_delta_mean(e_div, ev["date"])
            per_event[ev["date"]] = {
                "label": ev["label"],
                "alpha_naive_2_dsigma": d_alpha_n,
                "alpha_norm_2_dsigma": d_alpha_f,
                "ediv_dsigma": d_ediv,
            }
            if d_alpha_n is not None or d_alpha_f is not None:
                used.append(ev["date"])

        def _mean(key: str) -> float | None:
            vals = [v[key] for v in per_event.values()
                    if v.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        study[shock] = {
            "n_events": len(ev_list),
            "n_used": len(used),
            "events": [{"date": e["date"], "label": e["label"]} for e in ev_list],
            "events_used": used,
            "per_event": per_event,
            "mean_alpha_naive_2_dsigma": _mean("alpha_naive_2_dsigma"),
            "mean_alpha_norm_2_dsigma": _mean("alpha_norm_2_dsigma"),
            "mean_ediv_dsigma": _mean("ediv_dsigma"),
        }
        fmt = lambda x: f"{x:+.3f}" if x is not None else "  n/a"
        print(f"  [{shock:<16}] n={len(used):<2}/{len(ev_list):<2}  "
              f"α_naive={fmt(study[shock]['mean_alpha_naive_2_dsigma'])}  "
              f"α_norm={fmt(study[shock]['mean_alpha_norm_2_dsigma'])}  "
              f"e_div={fmt(study[shock]['mean_ediv_dsigma'])}")

    # ----- CSV 出力 -----
    csv_out = pd.DataFrame({
        "alpha_naive_2": alpha_naive_2,
        "alpha_norm_2": alpha_norm_2,
        "z_L1": z_L1,
        "z_n_unb": z_unb,
        "e_div": e_div,
    })
    csv_out.index.name = "date"
    csv_out.to_csv(OUTPUT_CSV, float_format="%.6f")
    print(f"\n[save] {OUTPUT_CSV}  ({len(csv_out)} rows)")

    # ----- 5y 版との直接比較 (既存 alpha_invariant_eventstudy.json から) -----
    five_y_path = DATA_DIR / "alpha_invariant_eventstudy.json"
    five_y_compare = None
    if five_y_path.exists():
        five_y = json.loads(five_y_path.read_text(encoding="utf-8"))
        five_y_compare = {}
        # 5y では war (= 8y の geopolitical) という名前
        rename_for_5y = {
            "trade_policy": "trade_policy",
            "market_structure": "market_structure",
            "geopolitical": "war",
        }
        for shock_20y, shock_5y in rename_for_5y.items():
            if shock_5y in five_y.get("shock_event_study", {}):
                v5 = five_y["shock_event_study"][shock_5y]
                v20 = study.get(shock_20y, {})
                five_y_compare[shock_20y] = {
                    "5y_alpha_naive_dsigma_12indic": v5.get("mean_alpha_naive_dsigma"),
                    "5y_alpha_norm_dsigma_12indic": v5.get("mean_alpha_norm_dsigma"),
                    "5y_ediv_dsigma": v5.get("mean_ediv_dsigma"),
                    "20y_alpha_naive_2_dsigma": v20.get("mean_alpha_naive_2_dsigma"),
                    "20y_alpha_norm_2_dsigma": v20.get("mean_alpha_norm_2_dsigma"),
                    "20y_ediv_dsigma": v20.get("mean_ediv_dsigma"),
                    "20y_n_events_used": v20.get("n_used"),
                }

    # ----- JSON 出力 -----
    meta = {
        "input_csv": str(INPUT_CSV.relative_to(REPO)),
        "events_json": str(EVENTS_JSON.relative_to(REPO)),
        "oos_start": OOS_START,
        "expanding_min_periods": EXPANDING_MIN_PERIODS,
        "n_indicators": 2,
        "indicators": ["L1_H1", "n_unb"],
        "sigma_threshold_for_alpha_naive": SIGMA_THRESH,
        "event_pre_bdays": EVENT_PRE,
        "event_post_bdays": EVENT_POST,
        "version": "simplified_2indicator",
        "alpha_naive_2_definition": (
            "alpha_naive_2(t) = #{i in {L1, n_unb} : |z_i(t)| >= 2} / 2  "
            "(余極限 colimit / 縮退版)"
        ),
        "alpha_norm_2_definition": (
            "alpha_norm_2(t) = sqrt(z_L1² + z_n_unb²) / sqrt(2)  "
            "(極限 limit / 縮退版)"
        ),
        "note": (
            "20y データ (2006-) には multi_indicators の 12 指標が存在せず, "
            "L1_H1 と n_unb のみが通史で取れるため, α を 2 指標に縮退して構築した. "
            "5y 版 (12 指標) との Δσ の絶対値は直接比較できないが, "
            "shock type ごとの相対順序と長期再現性を検証できる. "
            "z-score は過去のみの expanding window (min_periods=90) で計算し "
            "look-ahead を完全排除している."
        ),
    }
    out_json = {
        "meta": meta,
        "shock_event_study_20y": study,
        "five_y_vs_twenty_y_compare": five_y_compare,
    }
    OUTPUT_JSON.write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {OUTPUT_JSON}")

    # ----- 比較表 -----
    if five_y_compare:
        print("\n=== 5y (12 指標) vs 20y (2 指標縮退) Δσ 比較 ===")
        print(f"{'shock':<18} {'5y α_naive':>12} {'20y α_naive':>14} "
              f"{'5y α_norm':>12} {'20y α_norm':>14}")
        for shock, payload in five_y_compare.items():
            fmt = lambda x: f"{x:+.3f}" if x is not None else "  n/a"
            print(f"{shock:<18} "
                  f"{fmt(payload['5y_alpha_naive_dsigma_12indic']):>12} "
                  f"{fmt(payload['20y_alpha_naive_2_dsigma']):>14} "
                  f"{fmt(payload['5y_alpha_norm_dsigma_12indic']):>12} "
                  f"{fmt(payload['20y_alpha_norm_2_dsigma']):>14}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
