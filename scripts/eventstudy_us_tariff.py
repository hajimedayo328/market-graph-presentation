"""
USA 関税ショック event study (US-issued tariff, n=11)
=====================================================

目的:
  abstract / app.html に記載されてきた「USA 関税ショック (US-issued, n=11)
  で n_unb のみ反応 (Δσ=+0.38, p=0.018)」が別指標の値を混ぜた誤りであった
  ため、US-issued 関税イベント (n=11) を対象に L¹ / n_unb_total / n_unb_4 /
  e_div の Δσ + permutation p を **実データで再計算** して確定する。

イベント定義 (US-issued tariff, n=11):
  events_8y.json の trade_policy 15 件のうち、米国が発令側 (US-issued) で、
  かつ合意ではなく関税/輸出規制ショックであるもの。中国発 (対リトアニア,
  ガリウム規制, 中国34%報復) と Phase One 合意を除外した 11 件。

データ:
  - L¹ / n_unb        : gamma_timeseries_20y_w30.csv (2006-) → 11 件全カバー
                         + gamma_timeseries_w30.csv (5y, 2021-06-) → 8 件
  - n_unb_total / n_unb_4 : multi_indicators_w30.csv (5y, 2021-06-) → 8 件
    (2018-2020 の 3 件は 5y データ範囲外のため n_unb_4 は計算不可)

z-score:
  過去のみ expanding window (min_periods=90) → look-ahead 排除。
  e_div = z(n_unb) - z(L1).

検定:
  baseline = event 前 30 営業日、event window = 当日含む後 30 営業日。
  Δσ = mean(z[event window]) - mean(z[baseline window])。
  permutation test 5000 回 (event 数と同数の null 日を非復元抽出)、two-sided p。

出力:
  data/eventstudy_us_tariff.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

GAMMA_20Y = DATA_DIR / "gamma_timeseries_20y_w30.csv"
GAMMA_5Y = DATA_DIR / "gamma_timeseries_w30.csv"
MULTI_5Y = DATA_DIR / "multi_indicators_w30.csv"
OUT_JSON = DATA_DIR / "eventstudy_us_tariff.json"

EXPANDING_MIN_PERIODS = 90
BASELINE_BDAYS = 30
EVENT_BDAYS = 30
N_PERMUTATIONS = 5000
RNG_SEED = 20260529

# US-issued tariff events (n=11)
US_TARIFF_EVENTS = [
    {"date": "2018-03-22", "label": "米中第1弾関税発表 (Section 301)"},
    {"date": "2018-07-06", "label": "米中第1弾関税発動 (340億ドル)"},
    {"date": "2019-08-23", "label": "米中再エスカレ (関税報復応酬)"},
    {"date": "2022-08-09", "label": "米 CHIPS 法成立"},
    {"date": "2022-10-07", "label": "対中先端半導体輸出規制"},
    {"date": "2024-05-14", "label": "バイデン 対中 EV 100% 関税"},
    {"date": "2025-04-02", "label": "Liberation Day 相互関税"},
    {"date": "2025-04-08", "label": "関税 180 日停止・株価反発"},
    {"date": "2025-04-09", "label": "関税エスカレート 145%"},
    {"date": "2025-04-15", "label": "Nvidia H20 輸出規制"},
    {"date": "2026-03-04", "label": "関税再発動"},
]


def expanding_zscore(s: pd.Series, min_periods: int = EXPANDING_MIN_PERIODS) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=1)
    return (s - mu) / sd


def event_delta_sigma(series: pd.Series,
                      event_date: str | pd.Timestamp,
                      baseline_bdays: int = BASELINE_BDAYS,
                      event_bdays: int = EVENT_BDAYS) -> float | None:
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
    return float(evt.mean() - base.mean())


def permutation_p(series: pd.Series,
                  obs: float,
                  n_events: int,
                  null_dates: pd.DatetimeIndex,
                  rng: np.random.Generator,
                  n_perm: int = N_PERMUTATIONS) -> float:
    pool = np.array(null_dates)
    if len(pool) < n_events:
        return float("nan")
    null_vals = []
    for _ in range(n_perm):
        picks = rng.choice(pool, size=n_events, replace=False)
        vals = [event_delta_sigma(series, pd.Timestamp(p)) for p in picks]
        vals = [v for v in vals if v is not None]
        if len(vals) >= max(1, n_events // 2):
            null_vals.append(float(np.mean(vals)))
    if not null_vals:
        return float("nan")
    null_arr = np.array(null_vals)
    return float((np.sum(np.abs(null_arr - null_arr.mean()) >= abs(obs - null_arr.mean())) + 1)
                 / (len(null_arr) + 1))


def run_indicator(series: pd.Series, label: str, rng: np.random.Generator) -> dict:
    """1 指標について全イベントの Δσ と permutation p を計算."""
    s = series.dropna()
    null_pool = s.index
    per_event = []
    vals = []
    for ev in US_TARIFF_EVENTS:
        v = event_delta_sigma(s, ev["date"])
        per_event.append({"date": ev["date"], "label": ev["label"],
                          "d_sigma": (round(v, 4) if v is not None else None)})
        if v is not None:
            vals.append(v)
    if not vals:
        return {"indicator": label, "d_sigma_mean": None, "p_perm": None,
                "n_used": 0, "per_event": per_event,
                "data_range": [str(s.index.min().date()), str(s.index.max().date())]}
    obs = float(np.mean(vals))
    p = permutation_p(s, obs, len(vals), null_pool, rng)
    return {
        "indicator": label,
        "d_sigma_mean": round(obs, 4),
        "p_perm": round(p, 4),
        "n_used": len(vals),
        "per_event": per_event,
        "data_range": [str(s.index.min().date()), str(s.index.max().date())],
    }


def main() -> None:
    print("=== USA 関税ショック (US-issued, n=11) event study ===")
    print(f"events defined: {len(US_TARIFF_EVENTS)}")

    # --- 20y データ: L1 / n_unb / e_div (11 件全カバー狙い) ---
    g20 = (pd.read_csv(GAMMA_20Y, parse_dates=["date"])
           .sort_values("date").set_index("date"))
    g20 = g20.dropna(subset=["L1_H1", "n_unb"]).copy()
    z_L1_20 = expanding_zscore(g20["L1_H1"])
    z_unb_20 = expanding_zscore(g20["n_unb"])
    e_div_20 = z_unb_20 - z_L1_20

    # --- 5y データ: L1 / n_unb / e_div + n_unb_total / n_unb_4 ---
    g5 = (pd.read_csv(GAMMA_5Y, parse_dates=["date"])
          .sort_values("date").set_index("date"))
    g5 = g5.dropna(subset=["L1_H1", "n_unb"]).copy()
    z_L1_5 = expanding_zscore(g5["L1_H1"])
    z_unb_5 = expanding_zscore(g5["n_unb"])
    e_div_5 = z_unb_5 - z_L1_5

    m5 = (pd.read_csv(MULTI_5Y, parse_dates=["date"])
          .sort_values("date").set_index("date"))
    m5 = m5.dropna(subset=["n_unb_total", "n_unb_4"]).copy()
    z_unb_total = expanding_zscore(m5["n_unb_total"])
    z_unb_4 = expanding_zscore(m5["n_unb_4"])

    rng = np.random.default_rng(RNG_SEED)

    results = {}

    print("\n--- 20y データ (L1 / n_unb / e_div, 2018-2020 含む全 11 件カバー) ---")
    results["20y"] = {
        "L1": run_indicator(z_L1_20, "L1 (z-expanding, 20y)", rng),
        "n_unb": run_indicator(z_unb_20, "n_unb (z-expanding, 20y)", rng),
        "e_div": run_indicator(e_div_20, "e_div = z_unb - z_L1 (20y)", rng),
    }
    for k, r in results["20y"].items():
        print(f"  [20y] {k:<6} n={r['n_used']:<2} "
              f"Δσ={r['d_sigma_mean']}  p={r['p_perm']}")

    print("\n--- 5y データ (L1 / n_unb_total / n_unb_4 / e_div, 8 件のみ) ---")
    results["5y"] = {
        "L1": run_indicator(z_L1_5, "L1 (z-expanding, 5y)", rng),
        "n_unb_total": run_indicator(z_unb_total, "n_unb_total (z-expanding, 5y)", rng),
        "n_unb_4": run_indicator(z_unb_4, "n_unb_4 (z-expanding, 5y)", rng),
        "e_div": run_indicator(e_div_5, "e_div = z_unb - z_L1 (5y)", rng),
    }
    for k, r in results["5y"].items():
        print(f"  [5y]  {k:<12} n={r['n_used']:<2} "
              f"Δσ={r['d_sigma_mean']}  p={r['p_perm']}")

    out = {
        "meta": {
            "event_set": "US-issued tariff",
            "n_events_defined": len(US_TARIFF_EVENTS),
            "events": US_TARIFF_EVENTS,
            "expanding_min_periods": EXPANDING_MIN_PERIODS,
            "baseline_bdays": BASELINE_BDAYS,
            "event_bdays": EVENT_BDAYS,
            "n_permutations": N_PERMUTATIONS,
            "rng_seed": RNG_SEED,
            "note": (
                "z-score は過去のみ expanding window (min_periods=90) で計算し "
                "look-ahead を排除。Δσ = mean(z[post-30bd]) - mean(z[pre-30bd])。"
                "p_perm は 5000 perm の two-sided。"
                "20y ブロックは gamma_timeseries_20y_w30.csv (2006-) を使い "
                "US-issued 関税 11 件全てを対象にできる (L1/n_unb/e_div)。"
                "5y ブロックは multi_indicators_w30.csv / gamma_timeseries_w30.csv "
                "(2021-06-) を使い n_unb_total / n_unb_4 を計算するが、"
                "データ範囲の都合で 2018-2020 の 3 件は対象外 (n_used=8)。"
            ),
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
