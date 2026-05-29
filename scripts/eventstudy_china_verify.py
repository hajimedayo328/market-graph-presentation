"""
中国 A 株 市場構造ショック event study 検証
=============================================

目的:
  app.html L799 の「中国 A 株 5y データで市場構造ショック → L¹ が再現
  (Δσ=+1.47, p=0.007)」を実データで再計算して検証する。

データ:
  data/gamma_cn_timeseries_w30.csv (中国 52 銘柄, 2021-06 以降, w30)
  指標: L1_H1 (L¹), n_unb (不整合サイクル数)

手法 (eventstudy_8y_oos.py に準拠):
  z-score: 過去のみ expanding window (min_periods=90), look-ahead 回避
  Δσ      : post_minus_pre  = mean(z[event window]) - mean(z[baseline window])
            post_only       = mean(z[event window])  (全期間 z 化の 5y 既存指標相当)
            event window = event 後 30 営業日 (event 日含む)
            baseline     = event 前 30 営業日
  permutation: N=5000, two-sided, random null date (非復元)

イベント (market_structure, 2021-06 以降 = データ範囲内のもの):
  2022-09-23 UK 年金危機 (LDI ショック)
  2024-08-05 円キャリー巻き戻し
  2025-08-01 サマー・ボラショック
  (COVID 2020-03, Volmageddon 2018-02 は範囲外)

出力: data/eventstudy_china_verify.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

INPUT_CSV = DATA_DIR / "gamma_cn_timeseries_w30.csv"
RESULTS_JSON = DATA_DIR / "eventstudy_china_verify.json"

EXPANDING_MIN_PERIODS = 90
BASELINE_BDAYS = 30
EVENT_BDAYS = 30
N_PERMUTATIONS = 5000
RNG_SEED = 20260526

# 市場構造ショック (データ範囲 2021-06 以降に存在するもの)
EVENTS_MARKET_STRUCTURE = [
    {"date": "2022-09-23", "label": "UK 年金危機 (LDI ショック)", "category": "market_structure"},
    {"date": "2024-08-05", "label": "円キャリー巻き戻し", "category": "market_structure"},
    {"date": "2025-08-01", "label": "サマー・ボラショック", "category": "market_structure"},
]

# 参考: trade_policy (中国に直接関係するイベント) も計算しておく
EVENTS_TRADE_POLICY = [
    {"date": "2021-12-03", "label": "対リトアニア輸入停止", "category": "trade_policy"},
    {"date": "2022-08-09", "label": "米 CHIPS 法成立", "category": "trade_policy"},
    {"date": "2022-10-07", "label": "対中先端半導体輸出規制", "category": "trade_policy"},
    {"date": "2023-07-03", "label": "ガリウム・ゲルマニウム規制", "category": "trade_policy"},
    {"date": "2024-05-14", "label": "バイデン 対中 EV 100% 関税", "category": "trade_policy"},
    {"date": "2025-04-02", "label": "Liberation Day 相互関税", "category": "trade_policy"},
    {"date": "2025-04-04", "label": "中国 34% 報復 + 希土類規制", "category": "trade_policy"},
    {"date": "2025-04-09", "label": "関税エスカレート 145%", "category": "trade_policy"},
    {"date": "2025-04-15", "label": "Nvidia H20 輸出規制", "category": "trade_policy"},
    {"date": "2026-03-04", "label": "関税再発動", "category": "trade_policy"},
]


def expanding_zscore(s: pd.Series, min_periods: int = EXPANDING_MIN_PERIODS) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=1)
    return (s - mu) / sd


def event_delta_sigma(series: pd.Series, event_date, mode: str = "post_minus_pre"):
    """
    mode:
      post_minus_pre : mean(z[+0..+30]) - mean(z[-30..0])   (eventstudy_8y_oos 主指標)
      post_only      : mean(z[+0..+30])                      (5y 既存指標相当)
      pre15          : mean(z[-15..0])  (Section 6 の「前 15 日」アプローチ)
      pre30          : mean(z[-30..0])
    """
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


def permutation_p(series, obs, n_events, null_dates, rng, mode="post_minus_pre",
                  n_perm=N_PERMUTATIONS):
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


def run_category(name, events, series_dict, null_pool, rng):
    out = {"n_events": len(events), "events": [e["date"] for e in events], "modes": {}}
    valid = [e for e in events
             if (pd.Timestamp(e["date"]) >= series_dict["L1"].index.min()
                 and pd.Timestamp(e["date"]) <= series_dict["L1"].index.max())]
    out["n_valid"] = len(valid)
    out["valid_events"] = [e["date"] for e in valid]
    for mode in ["post_minus_pre", "post_only", "pre15", "pre30"]:
        mode_out = {"indicators": {}}
        for ind_name, s in series_dict.items():
            vals, per_event = [], []
            for ev in valid:
                v = event_delta_sigma(s, ev["date"], mode=mode)
                per_event.append({"date": ev["date"], "label": ev["label"],
                                  "d_sigma": (round(v, 4) if v is not None else None)})
                if v is not None:
                    vals.append(v)
            if not vals:
                mode_out["indicators"][ind_name] = {"d_sigma_mean": None, "p_perm": None,
                                                    "n_used": 0, "per_event": per_event}
                continue
            obs = float(np.mean(vals))
            p = permutation_p(s, obs, len(vals), null_pool, rng, mode=mode)
            mode_out["indicators"][ind_name] = {"d_sigma_mean": round(obs, 4),
                                                "p_perm": round(p, 4), "n_used": len(vals),
                                                "per_event": per_event}
            print(f"  [{name:<16}][{mode:<14}] {ind_name:<6} n={len(vals)} "
                  f"Δσ={obs:+.3f} p={p:.4f}")
        out["modes"][mode] = mode_out
    return out


def main():
    print("=== 中国 A 株 event study 検証 ===")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date").set_index("date")
    df = df.dropna(subset=["L1_H1", "n_unb"]).copy()
    print(f"loaded: {df.shape}, range {df.index.min().date()} -> {df.index.max().date()}")

    z_L1 = expanding_zscore(df["L1_H1"])
    z_unb = expanding_zscore(df["n_unb"])
    print(f"expanding z: L1 n={z_L1.notna().sum()}, unb n={z_unb.notna().sum()}")

    series_dict = {"L1": z_L1.dropna(), "n_unb": z_unb.dropna()}
    null_pool = series_dict["L1"].index
    print(f"null pool: {len(null_pool)} dates")

    rng = np.random.default_rng(RNG_SEED)

    results = {}
    print("\n--- market_structure ---")
    results["market_structure"] = run_category("market_structure",
                                               EVENTS_MARKET_STRUCTURE, series_dict,
                                               null_pool, rng)
    print("\n--- trade_policy (参考) ---")
    results["trade_policy"] = run_category("trade_policy", EVENTS_TRADE_POLICY,
                                          series_dict, null_pool, rng)

    out = {
        "meta": {
            "input_csv": INPUT_CSV.name,
            "data_range": [str(df.index.min().date()), str(df.index.max().date())],
            "n_days": int(len(df)),
            "expanding_min_periods": EXPANDING_MIN_PERIODS,
            "baseline_bdays": BASELINE_BDAYS,
            "event_bdays": EVENT_BDAYS,
            "n_permutations": N_PERMUTATIONS,
            "rng_seed": RNG_SEED,
            "indicators": ["L1 (z-expanding)", "n_unb (z-expanding)"],
            "claim_under_test": "app.html L799: 中国 A 株 市場構造ショック → L¹ Δσ=+1.47, p=0.007",
            "note": "z-score は過去のみ expanding (min=90)。Δσ 2 mode。p_perm は 5000 perm two-sided。",
        },
        "results": results,
    }
    RESULTS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
