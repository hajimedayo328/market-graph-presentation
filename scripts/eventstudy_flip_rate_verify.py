"""
flip_rate (符号反転率) event study 検証
=========================================

目的:
  index.html Section 7 (build_index.py L1708, L1724) の記載を検証:
    - trade_policy ショック前 15 日で flip_rate が Δσ=+0.45 (p=0.0001) で有意上昇
    - 利上げ前にも flip_rate 上昇 (Δσ=+0.54)

データ:
  data/sign_flip_w30_lag30.csv (符号反転ペア時系列, 2021-07 以降)
  指標: flip_rate (直前 30 日相関 vs 当日相関で符号反転したエッジペア割合)
  平常時平均 8.3% (既存検証済み, 本スクリプトでも確認)

手法 (eventstudy_8y_oos.py に準拠):
  z-score: 過去のみ expanding window (min_periods=90), look-ahead 回避
  Δσ mode:
    pre15          : mean(z[-15..0])   ← Section 7 の「前 15 日」アプローチ (記載の主指標)
    pre30          : mean(z[-30..0])
    post_minus_pre : mean(z[+0..+30]) - mean(z[-30..0])  (8y OOS 主指標)
    post_only      : mean(z[+0..+30])
  permutation: N=5000, two-sided, random null date (非復元)

イベント (build_index.py EVENTS_EXTENDED より, flip_rate データ範囲 2021-07 以降):
  trade_policy: Liberation Day 等 USA/CHN 関税
  monetary    : FOMC 利上げ/利下げ (「利上げ前」検証用)

出力: data/eventstudy_flip_rate_verify.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

INPUT_CSV = DATA_DIR / "sign_flip_w30_lag30.csv"
RESULTS_JSON = DATA_DIR / "eventstudy_flip_rate_verify.json"

EXPANDING_MIN_PERIODS = 90
BASELINE_BDAYS = 30
EVENT_BDAYS = 30
N_PERMUTATIONS = 5000
RNG_SEED = 20260526

# build_index.py EVENTS_EXTENDED 準拠 (flip_rate データ範囲 2021-07 以降のもの)
EVENTS = {
    "trade_policy": [
        {"date": "2021-12-03", "label": "対リトアニア輸入停止"},
        {"date": "2022-08-09", "label": "米 CHIPS 法成立"},
        {"date": "2022-10-07", "label": "対中先端半導体輸出規制"},
        {"date": "2023-07-03", "label": "ガリウム・ゲルマニウム規制"},
        {"date": "2024-05-14", "label": "バイデン 対中 EV 100% 関税"},
        {"date": "2025-04-02", "label": "Liberation Day 相互関税"},
        {"date": "2025-04-04", "label": "中国 34% 報復 + 希土類規制"},
        {"date": "2025-04-08", "label": "関税 180 日停止・株価反発"},
        {"date": "2025-04-09", "label": "関税エスカレート 145%"},
        {"date": "2025-04-15", "label": "Nvidia H20 輸出規制"},
        {"date": "2026-03-04", "label": "関税再発動"},
    ],
    # 「利上げ前」= 金融政策イベント (FOMC タカ派/利下げ)
    "monetary": [
        {"date": "2024-01-31", "label": "FOMC タカ派サプライズ"},
        {"date": "2024-09-18", "label": "FOMC 50bp 利下げ"},
        {"date": "2024-12-18", "label": "FOMC タカ派サプライズ"},
    ],
}


def expanding_zscore(s: pd.Series, min_periods: int = EXPANDING_MIN_PERIODS) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=1)
    return (s - mu) / sd


def event_delta_sigma(series: pd.Series, event_date, mode: str = "pre15"):
    target = pd.Timestamp(event_date)
    s = series.dropna()
    if target < s.index.min() or target > s.index.max():
        return None
    pos = s.index.get_indexer([target], method="nearest")[0]

    if mode in ("pre15", "pre30"):
        pre = 15 if mode == "pre15" else 30
        if pos < pre:
            return None
        seg = s.iloc[pos - pre:pos]
        if len(seg) < pre // 2:
            return None
        return float(seg.mean())

    if pos < BASELINE_BDAYS:
        return None
    base = s.iloc[pos - BASELINE_BDAYS:pos]
    end = min(pos + EVENT_BDAYS, len(s))
    evt = s.iloc[pos:end]
    if len(base) < BASELINE_BDAYS // 2 or len(evt) < EVENT_BDAYS // 2:
        return None
    if mode == "post_only":
        return float(evt.mean())
    return float(evt.mean() - base.mean())


def permutation_p(series, obs, n_events, null_dates, rng, mode, n_perm=N_PERMUTATIONS):
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


def run_category(name, events, z_flip, null_pool, rng):
    out = {"n_events": len(events), "events": [e["date"] for e in events], "modes": {}}
    valid = [e for e in events
             if (pd.Timestamp(e["date"]) >= z_flip.index.min()
                 and pd.Timestamp(e["date"]) <= z_flip.index.max())]
    out["n_valid"] = len(valid)
    out["valid_events"] = [e["date"] for e in valid]
    for mode in ["pre15", "pre30", "post_minus_pre", "post_only"]:
        vals, per_event = [], []
        for ev in valid:
            v = event_delta_sigma(z_flip, ev["date"], mode=mode)
            per_event.append({"date": ev["date"], "label": ev["label"],
                              "d_sigma": (round(v, 4) if v is not None else None)})
            if v is not None:
                vals.append(v)
        if not vals:
            out["modes"][mode] = {"d_sigma_mean": None, "p_perm": None,
                                  "n_used": 0, "per_event": per_event}
            continue
        obs = float(np.mean(vals))
        p = permutation_p(z_flip, obs, len(vals), null_pool, rng, mode)
        out["modes"][mode] = {"d_sigma_mean": round(obs, 4), "p_perm": round(p, 4),
                              "n_used": len(vals), "per_event": per_event}
        print(f"  [{name:<13}][{mode:<14}] flip_rate n={len(vals)} "
              f"Δσ={obs:+.3f} p={p:.4f}")
    return out


def main():
    print("=== flip_rate event study 検証 ===")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date").set_index("date")
    df = df.dropna(subset=["flip_rate"]).copy()
    print(f"loaded: {df.shape}, range {df.index.min().date()} -> {df.index.max().date()}")
    print(f"flip_rate 平常時平均 = {df['flip_rate'].mean():.4f} (記載 8.3%)")

    z_flip = expanding_zscore(df["flip_rate"]).dropna()
    print(f"expanding z: n={len(z_flip)}")

    null_pool = z_flip.index
    rng = np.random.default_rng(RNG_SEED)

    results = {}
    for cat, evs in EVENTS.items():
        print(f"\n--- {cat} ---")
        results[cat] = run_category(cat, evs, z_flip, null_pool, rng)

    out = {
        "meta": {
            "input_csv": INPUT_CSV.name,
            "data_range": [str(df.index.min().date()), str(df.index.max().date())],
            "n_days": int(len(df)),
            "flip_rate_baseline_mean": round(float(df["flip_rate"].mean()), 4),
            "expanding_min_periods": EXPANDING_MIN_PERIODS,
            "baseline_bdays": BASELINE_BDAYS,
            "event_bdays": EVENT_BDAYS,
            "n_permutations": N_PERMUTATIONS,
            "rng_seed": RNG_SEED,
            "claim_under_test": ("build_index.py L1708: trade_policy 前 15 日 flip_rate "
                                 "Δσ=+0.45 p=0.0001; L1724: 利上げ前 Δσ=+0.54"),
            "note": ("z は過去のみ expanding (min=90)。mode pre15=「前 15 日」(記載相当), "
                     "pre30, post_minus_pre, post_only。p_perm は 5000 perm two-sided。"),
        },
        "results": results,
    }
    RESULTS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
