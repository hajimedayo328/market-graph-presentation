"""
8 年完全 OOS event study (2017-11-01 〜 2026-05)
================================================

目的:
  主要 event study (Liberation Day, 円キャリー, 関税系) は 5y データ
  (gamma_timeseries_w30.csv, 2021-06 以降) で実施されてきた。
  本スクリプトでは 20y データ (gamma_timeseries_20y_w30.csv) から
  universe が事実上揃う 2017-11-01 以降の **8 年区間** を抽出し、
  同じ手順を 5y の **外側 (out-of-sample)** に適用して再現性を検証する。

z-score:
  必ず過去のみの **expanding window (min_periods=90)** で計算 (look-ahead 回避)。
  e_div = z_unb_past - z_L1_past.

検定統計量:
  各 event 日について baseline = event 前 30 営業日、event window = 後 30 営業日
  (event 日含む) として
      Δσ = mean(z in event window) - mean(z in baseline window)
  カテゴリ平均は event 数で単純平均する。

permutation:
  各カテゴリで N_PERMUTATIONS=5000 回、event 数と同じ件数の null 日を
  非復元抽出し、同じ pre/post 窓で Δσ を計算 -> 観測値の two-sided p 値。

出力:
  data/events_8y.json
  data/eventstudy_8y_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

INPUT_CSV = DATA_DIR / "gamma_timeseries_20y_w30.csv"
EVENTS_JSON = DATA_DIR / "events_8y.json"
RESULTS_JSON = DATA_DIR / "eventstudy_8y_results.json"

OOS_START = "2017-11-01"
EXPANDING_MIN_PERIODS = 90
BASELINE_BDAYS = 30
EVENT_BDAYS = 30
N_PERMUTATIONS = 5000
RNG_SEED = 20260526


# ============================================================
# 1. 8 年 event DB (Section 6/8 の 5y event + 過去 6 年分の主要 event)
# ============================================================
EVENTS_8Y: list[dict] = [
    # ===== trade_policy =====
    {"date": "2018-03-22", "label": "米中第1弾関税発表 (Section 301)", "category": "trade_policy"},
    {"date": "2018-07-06", "label": "米中第1弾関税発動 (340億ドル)", "category": "trade_policy"},
    {"date": "2019-08-23", "label": "米中再エスカレ (関税報復応酬)", "category": "trade_policy"},
    {"date": "2020-01-15", "label": "米中 Phase One 合意", "category": "trade_policy"},
    {"date": "2021-12-03", "label": "対リトアニア輸入停止", "category": "trade_policy"},
    {"date": "2022-08-09", "label": "米 CHIPS 法成立", "category": "trade_policy"},
    {"date": "2022-10-07", "label": "対中先端半導体輸出規制", "category": "trade_policy"},
    {"date": "2023-07-03", "label": "ガリウム・ゲルマニウム規制", "category": "trade_policy"},
    {"date": "2024-05-14", "label": "バイデン 対中 EV 100% 関税", "category": "trade_policy"},
    {"date": "2025-04-02", "label": "Liberation Day 相互関税", "category": "trade_policy"},
    {"date": "2025-04-04", "label": "中国 34% 報復 + 希土類規制", "category": "trade_policy"},
    {"date": "2025-04-08", "label": "関税 180 日停止・株価反発", "category": "trade_policy"},
    {"date": "2025-04-09", "label": "関税エスカレート 145%", "category": "trade_policy"},
    {"date": "2025-04-15", "label": "Nvidia H20 輸出規制", "category": "trade_policy"},
    {"date": "2026-03-04", "label": "関税再発動", "category": "trade_policy"},

    # ===== market_structure =====
    {"date": "2018-02-05", "label": "Volmageddon (VIX 急騰・XIV 崩壊)", "category": "market_structure"},
    {"date": "2020-03-12", "label": "COVID クラッシュ (株式 -10%)", "category": "market_structure"},
    {"date": "2022-09-23", "label": "UK 年金危機 (LDI ショック)", "category": "market_structure"},
    {"date": "2024-08-05", "label": "円キャリー巻き戻し (日経過去最大下げ)", "category": "market_structure"},
    {"date": "2025-08-01", "label": "サマー・ボラショック", "category": "market_structure"},

    # ===== war / geopolitics =====
    {"date": "2022-02-24", "label": "ウクライナ侵攻", "category": "geopolitical"},
    {"date": "2023-10-07", "label": "ハマス・イスラエル戦争", "category": "geopolitical"},
    {"date": "2024-04-13", "label": "イラン・イスラエル攻撃", "category": "geopolitical"},

    # ===== macro / monetary (Section 6/8 既存) =====
    {"date": "2023-08-02", "label": "Fitch 米国格下げ", "category": "macro"},
    {"date": "2024-01-31", "label": "FOMC タカ派サプライズ", "category": "monetary"},
    {"date": "2024-09-18", "label": "FOMC 50bp 利下げ", "category": "monetary"},
    {"date": "2024-12-18", "label": "FOMC タカ派サプライズ", "category": "monetary"},

    # ===== tech_shock =====
    {"date": "2026-01-27", "label": "DeepSeek AI 投資見直し", "category": "tech_shock"},
]


# ============================================================
# 2. 過去のみ expanding z-score
# ============================================================
def expanding_zscore(s: pd.Series, min_periods: int = EXPANDING_MIN_PERIODS) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=1)
    return (s - mu) / sd


# ============================================================
# 3. event delta sigma (pre baseline vs post event window)
# ============================================================
def event_delta_sigma(series: pd.Series,
                      event_date: str | pd.Timestamp,
                      baseline_bdays: int = BASELINE_BDAYS,
                      event_bdays: int = EVENT_BDAYS,
                      mode: str = "post_minus_pre") -> float | None:
    """
    z 化済 series に対する event Δσ.

    mode="post_minus_pre" (default, 本研究の主指標):
       Δσ = mean(z[event_window]) - mean(z[baseline_window])
       expanding z の場合、baseline 内の平均ドリフトを除去できる.

    mode="post_only" (5y 結果との直接比較用):
       Δσ = mean(z[event_window])
       全期間 z 化を前提とした 5y 既存指標と整合.
    """
    target = pd.Timestamp(event_date)
    s = series.dropna()
    if target < s.index.min() or target > s.index.max():
        return None
    pos = s.index.get_indexer([target], method="nearest")[0]
    if pos < baseline_bdays:
        return None
    base = s.iloc[pos - baseline_bdays:pos]
    end = min(pos + event_bdays, len(s))
    evt = s.iloc[pos:end]
    if len(base) < baseline_bdays // 2 or len(evt) < event_bdays // 2:
        return None
    if mode == "post_only":
        return float(evt.mean())
    return float(evt.mean() - base.mean())


# ============================================================
# 4. permutation test
# ============================================================
def permutation_p(series: pd.Series,
                   obs: float,
                   n_events: int,
                   null_dates: pd.DatetimeIndex,
                   rng: np.random.Generator,
                   mode: str = "post_minus_pre",
                   n_perm: int = N_PERMUTATIONS) -> float:
    null_vals = []
    pool = np.array(null_dates)
    if len(pool) < n_events:
        return float("nan")
    for _ in range(n_perm):
        picks = rng.choice(pool, size=n_events, replace=False)
        vals = []
        for p in picks:
            v = event_delta_sigma(series, pd.Timestamp(p), mode=mode)
            if v is not None:
                vals.append(v)
        if len(vals) >= max(1, n_events // 2):
            null_vals.append(float(np.mean(vals)))
    if not null_vals:
        return float("nan")
    null_arr = np.array(null_vals)
    p = float((np.sum(np.abs(null_arr - null_arr.mean()) >= abs(obs - null_arr.mean())) + 1)
              / (len(null_arr) + 1))
    return p


# ============================================================
# 5. main
# ============================================================
def main() -> None:
    print("=== 8 年 OOS event study ===")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date").set_index("date")
    print(f"loaded 20y: {df.shape}, range {df.index.min().date()} -> {df.index.max().date()}")

    df = df.dropna(subset=["L1_H1", "n_unb"]).copy()
    df = df[df.index >= pd.Timestamp(OOS_START)].copy()
    print(f"after OOS filter ({OOS_START}+): {df.shape}, "
          f"range {df.index.min().date()} -> {df.index.max().date()}")

    # expanding z-score (過去のみ)
    z_L1 = expanding_zscore(df["L1_H1"])
    z_unb = expanding_zscore(df["n_unb"])
    e_div = z_unb - z_L1
    print(f"expanding z: L1 n={z_L1.notna().sum()}, "
          f"unb n={z_unb.notna().sum()}, ediv n={e_div.notna().sum()}")

    # 各指標を index で扱う
    series_dict = {"L1": z_L1.dropna(),
                   "n_unb": z_unb.dropna(),
                   "e_div": e_div.dropna()}

    # event を保存
    EVENTS_JSON.write_text(json.dumps(EVENTS_8Y, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"saved: {EVENTS_JSON} ({len(EVENTS_8Y)} events)")

    # 各カテゴリ
    cats: dict[str, list[dict]] = {}
    for ev in EVENTS_8Y:
        cats.setdefault(ev["category"], []).append(ev)

    # null pool: e_div が利用可能な全日付
    null_pool = series_dict["e_div"].index
    print(f"null pool: {len(null_pool)} dates, "
          f"{null_pool.min().date()} -> {null_pool.max().date()}")

    rng = np.random.default_rng(RNG_SEED)

    MODES = ["post_minus_pre", "post_only"]

    results: dict[str, dict] = {}
    print("\n--- category-level event study ---")
    for cat, ev_list in sorted(cats.items()):
        cat_out: dict = {"n_events": len(ev_list),
                          "events": [e["date"] for e in ev_list],
                          "modes": {}}
        # event 日が universe 外のものはスキップ
        valid_events = [e for e in ev_list
                         if (pd.Timestamp(e["date"]) >= series_dict["e_div"].index.min()
                              and pd.Timestamp(e["date"]) <= series_dict["e_div"].index.max())]
        cat_out["n_valid"] = len(valid_events)

        for mode in MODES:
            mode_out: dict = {"indicators": {}}
            for ind_name, s in series_dict.items():
                vals = []
                per_event = []
                for ev in valid_events:
                    v = event_delta_sigma(s, ev["date"], mode=mode)
                    per_event.append({"date": ev["date"],
                                        "label": ev["label"],
                                        "d_sigma": (round(v, 4) if v is not None else None)})
                    if v is not None:
                        vals.append(v)
                if not vals:
                    mode_out["indicators"][ind_name] = {
                        "d_sigma_mean": None, "p_perm": None,
                        "n_used": 0, "per_event": per_event,
                    }
                    continue
                obs = float(np.mean(vals))
                p = permutation_p(s, obs, len(vals), null_pool, rng, mode=mode)
                mode_out["indicators"][ind_name] = {
                    "d_sigma_mean": round(obs, 4),
                    "p_perm": round(p, 4),
                    "n_used": len(vals),
                    "per_event": per_event,
                }
                print(f"  [{cat:<16}] [{mode:<14}] {ind_name:<6}  n={len(vals):<2} "
                      f"Δσ={obs:+.3f}  p={p:.4f}")
            cat_out["modes"][mode] = mode_out
        results[cat] = cat_out

    # 全 event 合算 (参考)
    print("\n--- all events combined ---")
    all_events = EVENTS_8Y
    all_out: dict = {"n_events": len(all_events), "modes": {}}
    for mode in MODES:
        mode_out: dict = {"indicators": {}}
        for ind_name, s in series_dict.items():
            vals = []
            for ev in all_events:
                v = event_delta_sigma(s, ev["date"], mode=mode)
                if v is not None:
                    vals.append(v)
            if vals:
                obs = float(np.mean(vals))
                p = permutation_p(s, obs, len(vals), null_pool, rng, mode=mode)
                mode_out["indicators"][ind_name] = {
                    "d_sigma_mean": round(obs, 4),
                    "p_perm": round(p, 4),
                    "n_used": len(vals),
                }
                print(f"  [ALL]            [{mode:<14}] {ind_name:<6}  n={len(vals):<2} "
                      f"Δσ={obs:+.3f}  p={p:.4f}")
        all_out["modes"][mode] = mode_out
    results["_ALL"] = all_out

    # meta
    out = {
        "meta": {
            "oos_start": OOS_START,
            "data_range": [str(df.index.min().date()), str(df.index.max().date())],
            "n_days": int(len(df)),
            "expanding_min_periods": EXPANDING_MIN_PERIODS,
            "baseline_bdays": BASELINE_BDAYS,
            "event_bdays": EVENT_BDAYS,
            "n_permutations": N_PERMUTATIONS,
            "rng_seed": RNG_SEED,
            "input_csv": str(INPUT_CSV.name),
            "indicators": ["L1 (z-expanding)", "n_unb (z-expanding)",
                           "e_div = z_unb - z_L1"],
            "note": ("z-score は過去のみ expanding window (min_periods=90) で計算し "
                     "look-ahead を回避。Δσ は 2 mode: "
                     "(1) post_minus_pre: post-30bd 平均 − pre-30bd 平均, "
                     "(2) post_only: post-30bd 平均 (5y 既存指標との直接比較用). "
                     "p_perm は 5000 perm の two-sided。"),
        },
        "results": results,
    }
    RESULTS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nsaved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
