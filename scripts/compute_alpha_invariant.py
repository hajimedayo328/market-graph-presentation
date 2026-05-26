"""
α 不変量 (関手族の極限 / pullback による手法非依存ショックシグナル) の計算.

圏論的解釈:
  各指標 F_i: M -> R を「手法 (method) という小圏 M から実数への関手」と見ると,
  12 指標族 {F_i} の **極限 (limit)** または **余極限 (colimit)** に対応する量は
  「個別の指標選びに依存しない不変量」を与える.

  実装上は z-score 化した 12 指標を以下 2 通りで集約:
    α_naive(t) = #{i : |z_i(t)| >= 2} / 12        (集約 = colimit 的)
    α_norm(t)  = sqrt(Σ_i z_i(t)²) / sqrt(12)     (Frobenius ノルム = limit 的)

入出力:
  入力:
    - data/multi_indicators_w30.csv  (11 指標 + balance_rate 等)
    - data/gamma_timeseries_w30.csv  (L1_H1, n_unb -> e_div 構成用)
  出力:
    - data/alpha_invariant_w30.csv      (date, z_*, alpha_naive, alpha_norm, e_div)
    - data/alpha_invariant_eventstudy.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

MULTI_CSV = DATA_DIR / "multi_indicators_w30.csv"
GAMMA_CSV = DATA_DIR / "gamma_timeseries_w30.csv"
OUTPUT_CSV = DATA_DIR / "alpha_invariant_w30.csv"
OUTPUT_JSON = DATA_DIR / "alpha_invariant_eventstudy.json"

# 12 指標 (multi_indicators の 11 個 + gamma_timeseries 由来の e_div 構成は別)
INDICATORS_12 = [
    "L1", "L2", "Linf", "nH1", "meanP", "entropy",
    "n_unb_total", "n_unb_3", "n_unb_4", "n_unb_5plus",
    "balance_rate", "e_div",
]

# z-score 窓 (既存 robustness_ediv_pair_scan.py と統一)
Z_WINDOW = 90

# event 前後窓
EVENT_PRE = 30
EVENT_POST = 30

# σ 閾値 (α_naive で「動いた」と判定する基準)
SIGMA_THRESH = 2.0

# shock event 群 (既存 robustness_ediv_pair_scan.py と完全一致)
EVENTS = {
    "trade_policy": [
        ("2025-04-02", "Liberation Day (reciprocal tariff)"),
        ("2024-05-14", "Biden 100% EV tariff on China"),
    ],
    "market_structure": [
        ("2024-08-05", "JPY carry unwind / vol spike"),
    ],
    "war": [
        ("2023-10-07", "Hamas attack on Israel"),
        ("2022-02-24", "Russia invades Ukraine"),
    ],
}


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(20, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(20, window // 3)).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def load_indicators() -> pd.DataFrame:
    """12 指標を 1 つの DataFrame に統合する."""
    multi = pd.read_csv(MULTI_CSV, parse_dates=["date"]).set_index("date")
    gamma = pd.read_csv(GAMMA_CSV, parse_dates=["date"]).set_index("date")

    # e_div = z(n_unb) - z(L1_H1) を gamma_timeseries から構成
    z_l1 = rolling_zscore(gamma["L1_H1"], Z_WINDOW)
    z_unb = rolling_zscore(gamma["n_unb"], Z_WINDOW)
    gamma_ediv = (z_unb - z_l1).rename("e_div")

    # multi 側 11 指標 + e_div (12 番目)
    base = multi[[c for c in INDICATORS_12 if c in multi.columns]].copy()
    df = base.join(gamma_ediv, how="outer")

    # 並び順を INDICATORS_12 に揃える
    df = df[INDICATORS_12]
    # 先頭の全 NaN 行は捨てる
    df = df.dropna(how="all")
    return df


def compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """各指標を 90 日 rolling z-score 化."""
    z = pd.DataFrame(index=df.index)
    for col in df.columns:
        z[f"z_{col}"] = rolling_zscore(df[col], Z_WINDOW)
    return z


def compute_alpha(z_df: pd.DataFrame, sigma_thresh: float) -> pd.DataFrame:
    """
    α 不変量 2 種を計算.

    α_naive: 各 t で |z_i| >= sigma_thresh となる指標の割合 (0〜1)
             → 余極限 (colimit) 的: 「複数の手法が同時に発火したか」
    α_norm : 各 t で sqrt(Σ z_i^2) / sqrt(12) (標準化 Frobenius ノルム)
             → 極限 (limit) 的: 「全関手出力の同時 norm」
    """
    z_cols = list(z_df.columns)
    n_indic = len(z_cols)

    # 行ごとに有効 (非 NaN) な指標数で割って欠損に頑健にする
    abs_z = z_df.abs()
    valid_mask = z_df.notna()
    valid_count = valid_mask.sum(axis=1).replace(0, np.nan)

    # α_naive
    over_thresh = (abs_z >= sigma_thresh) & valid_mask
    alpha_naive = over_thresh.sum(axis=1) / valid_count

    # α_norm: NaN を 0 で埋めて二乗和, 標準化は √(有効指標数)
    sq = (z_df.fillna(0.0) ** 2).sum(axis=1)
    alpha_norm = np.sqrt(sq) / np.sqrt(valid_count.fillna(n_indic))

    out = pd.DataFrame({
        "alpha_naive": alpha_naive,
        "alpha_norm": alpha_norm,
    })
    return out


def event_delta_sigma(series: pd.Series, event_date: str,
                       pre: int, post: int) -> float | None:
    """Δσ = (post 平均 - pre 平均) / pre std (= z-score 化された jump)."""
    target = pd.Timestamp(event_date)
    if target < series.index.min() or target > series.index.max():
        return None
    pos = series.index.get_indexer([target], method="nearest")[0]
    pre_slice = series.iloc[max(0, pos - pre):pos].dropna()
    post_slice = series.iloc[pos:min(len(series), pos + post)].dropna()
    if len(pre_slice) < pre // 2 or len(post_slice) < post // 2:
        return None
    pre_std = float(pre_slice.std(ddof=0))
    if pre_std == 0 or math.isnan(pre_std):
        return None
    return float((post_slice.mean() - pre_slice.mean()) / pre_std)


def event_delta_mean(series: pd.Series, event_date: str,
                      pre: int, post: int) -> float | None:
    """既存 robustness_ediv_pair_scan と同じ流儀: 単なる平均差.

    z-score 系列に対しては「σ 単位の差」と等価.
    """
    target = pd.Timestamp(event_date)
    if target < series.index.min() or target > series.index.max():
        return None
    pos = series.index.get_indexer([target], method="nearest")[0]
    pre_slice = series.iloc[max(0, pos - pre):pos].dropna()
    post_slice = series.iloc[pos:min(len(series), pos + post)].dropna()
    if len(pre_slice) < pre // 2 or len(post_slice) < post // 2:
        return None
    return float(post_slice.mean() - pre_slice.mean())


def event_study(alpha_df: pd.DataFrame, ediv_series: pd.Series) -> dict:
    """
    shock type ごとに α_naive / α_norm / e_div の Δσ を計算.

    α は z-score 化していないので, event window 内で更に z-score 標準化した
    値で Δσ を測る (= event_delta_sigma).
    e_div は既に z-score 系列なので, event_delta_mean (= σ 単位差) で測る.
    """
    out: dict = {}
    for shock, events in EVENTS.items():
        per_event: dict = {}
        used: list[str] = []
        for date, label in events:
            d_alpha_n = event_delta_sigma(alpha_df["alpha_naive"], date,
                                           EVENT_PRE, EVENT_POST)
            d_alpha_f = event_delta_sigma(alpha_df["alpha_norm"], date,
                                           EVENT_PRE, EVENT_POST)
            d_ediv = event_delta_mean(ediv_series, date,
                                       EVENT_PRE, EVENT_POST)
            per_event[date] = {
                "label": label,
                "alpha_naive_dsigma": d_alpha_n,
                "alpha_norm_dsigma": d_alpha_f,
                "ediv_dsigma": d_ediv,
            }
            if d_alpha_n is not None or d_alpha_f is not None:
                used.append(date)

        def _mean(key: str) -> float | None:
            vals = [v[key] for v in per_event.values()
                    if v.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        out[shock] = {
            "events": [{"date": d, "label": l} for d, l in events],
            "events_used": used,
            "per_event": per_event,
            "mean_alpha_naive_dsigma": _mean("alpha_naive_dsigma"),
            "mean_alpha_norm_dsigma": _mean("alpha_norm_dsigma"),
            "mean_ediv_dsigma": _mean("ediv_dsigma"),
        }
    return out


def main() -> int:
    df = load_indicators()
    print(f"[load] {len(df)} rows, indicators = {list(df.columns)}")

    z_df = compute_zscores(df)
    alpha_df = compute_alpha(z_df, SIGMA_THRESH)

    # e_div 系列 (z-score 化済) — alpha_df に渡す前に元の指標から取り出す
    ediv_z = z_df["z_e_div"]  # z(e_div) を使うと二重正規化なので, 元の e_div を使う
    # 上述: e_div は既に z(n_unb)-z(L1) として構成済なので, そのまま使う
    ediv_raw = df["e_div"]

    study = event_study(alpha_df, ediv_raw)

    # ----- CSV 出力 -----
    csv_out = pd.concat([alpha_df, ediv_raw.rename("e_div")], axis=1)
    csv_out.index.name = "date"
    csv_out.to_csv(OUTPUT_CSV, float_format="%.6f")
    print(f"[save] {OUTPUT_CSV}  ({len(csv_out)} rows)")

    # ----- JSON 出力 -----
    meta = {
        "input_multi": str(MULTI_CSV.relative_to(REPO)),
        "input_gamma": str(GAMMA_CSV.relative_to(REPO)),
        "n_indicators": len(INDICATORS_12),
        "indicators": INDICATORS_12,
        "z_window": Z_WINDOW,
        "sigma_threshold_for_alpha_naive": SIGMA_THRESH,
        "event_pre_days": EVENT_PRE,
        "event_post_days": EVENT_POST,
        "alpha_naive_definition": (
            "alpha_naive(t) = #{i in 12 indicators : |z_i(t)| >= 2} / 12  "
            "(関手族の余極限 = 複数手法の同時発火)"
        ),
        "alpha_norm_definition": (
            "alpha_norm(t) = sqrt(Σ_i z_i(t)^2) / sqrt(12)  "
            "(関手族の極限 = 全手法の Frobenius ノルム)"
        ),
        "ediv_definition": "e_div(t) = z(n_unb)(t) - z(L1_H1)(t)",
        "events": {k: [{"date": d, "label": l} for d, l in v]
                   for k, v in EVENTS.items()},
    }

    out_json = {
        "meta": meta,
        "shock_event_study": study,
    }
    OUTPUT_JSON.write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[save] {OUTPUT_JSON}")

    # ----- 結果サマリ -----
    print("\n=== Event Study Summary (Δσ, pre 30d -> post 30d) ===")
    print(f"{'shock':<18} {'α_naive':>10} {'α_norm':>10} {'e_div':>10}")
    for shock, payload in study.items():
        an = payload["mean_alpha_naive_dsigma"]
        af = payload["mean_alpha_norm_dsigma"]
        ed = payload["mean_ediv_dsigma"]
        fmt = lambda x: f"{x:+.3f}" if x is not None else "  n/a"
        print(f"{shock:<18} {fmt(an):>10} {fmt(af):>10} {fmt(ed):>10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
