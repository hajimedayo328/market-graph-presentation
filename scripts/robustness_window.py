"""
Window sensitivity 検証 (Robustness via Window Scan)
=====================================================

目的:
  「なぜ window=30 なのか」の批判に対する実証検証.
  window = 20, 30, 60, 90 営業日で同じ event study (2025-04 Liberation Day)
  を実施し, Δσ_L1 / Δσ_unb / Δσ_ediv が window 選択にどれだけ依存するか
  を測る.

  結論パターン:
    - 4 種類どれでも Δσ_ediv が大 → 結果は window-robust
    - 30 だけ突出 → cherry-pick の疑い濃厚
    - すべて小さい → そもそも効果が弱い

軽量化:
  PH 計算は重いので, 各 window で 「returns_period = baseline 開始 -
  window*2 営業日前 〜 event_end」 のみ計算する.

出力:
  data/robustness_window_scan.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
LIB = HERE / "lib"
sys.path.insert(0, str(LIB))

from persistent_homology import persistence_diagram, persistence_summary  # noqa: E402
from market_category import MarketCategory  # noqa: E402
from homology import signed_cycle_balance  # noqa: E402

DATA_DIR = HERE.parent / "data"

# ===== パラメータ =====
WINDOWS = [20, 30, 60, 90]
THRESHOLD = 0.3

# event study windows (subsample スクリプトと整合)
BASELINE_START = "2025-02-15"
BASELINE_END = "2025-03-25"
EVENT_START = "2025-04-02"
EVENT_END = "2025-04-15"

# 計算期間: 各 window 分の助走を baseline 開始よりさらに前に取る
# (一番大きい window=90 でも安全な開始日を共通で使う)
RETURNS_START = "2024-08-01"   # 90 営業日助走 + 余裕
RETURNS_END = "2025-04-30"


def compute_indicators_for_window(closes: pd.DataFrame,
                                   symbols: list[str],
                                   window: int,
                                   threshold: float = THRESHOLD) -> pd.DataFrame:
    """指定 window で returns_period 範囲の L1_H1, n_unb を計算."""
    sub = closes[symbols].copy()
    returns = sub.pct_change()
    start_ts = pd.Timestamp(RETURNS_START)
    end_ts = pd.Timestamp(RETURNS_END)
    returns = returns.loc[(returns.index >= start_ts) & (returns.index <= end_ts)]

    rows = []
    n = len(returns)
    for t_idx in range(window, n):
        date = returns.index[t_idx - 1]
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
        if win_clean.shape[1] < 5:
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan})
            continue
        try:
            corr = win_clean.corr()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diag = persistence_diagram(corr, max_dim=1)
            summ = persistence_summary(diag)
            L1 = float(summ["L1_norm_H1"])

            cat = MarketCategory(symbols=list(win_clean.columns),
                                 corr_matrix=corr, threshold=threshold)
            cat._build_graph()
            bal = signed_cycle_balance(cat.G)
            n_unb = int(bal["n_unbalanced"])
            rows.append({
                "date": date,
                "n_symbols": win_clean.shape[1],
                "L1_H1": L1,
                "n_unb": float(n_unb),
            })
        except Exception as e:
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan})
            print(f"  ! {date.date()} failed: {e}")
    return pd.DataFrame(rows)


def event_response(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["L1_H1", "n_unb"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    bmask = (df["date"] >= BASELINE_START) & (df["date"] <= BASELINE_END)
    emask = (df["date"] >= EVENT_START) & (df["date"] <= EVENT_END)
    base = df[bmask]
    evt = df[emask]
    if len(base) < 5 or len(evt) < 3:
        return {"delta_sigma_L1": float("nan"),
                "delta_sigma_unb": float("nan"),
                "delta_sigma_ediv": float("nan"),
                "n_baseline": int(len(base)),
                "n_event": int(len(evt))}

    mu_L1, sd_L1 = base["L1_H1"].mean(), base["L1_H1"].std(ddof=1)
    mu_unb, sd_unb = base["n_unb"].mean(), base["n_unb"].std(ddof=1)
    if sd_L1 == 0 or sd_unb == 0 or not np.isfinite(sd_L1) or not np.isfinite(sd_unb):
        return {"delta_sigma_L1": float("nan"),
                "delta_sigma_unb": float("nan"),
                "delta_sigma_ediv": float("nan"),
                "n_baseline": int(len(base)),
                "n_event": int(len(evt))}

    z_L1_evt = (evt["L1_H1"] - mu_L1) / sd_L1
    z_unb_evt = (evt["n_unb"] - mu_unb) / sd_unb
    z_ediv_evt = z_unb_evt - z_L1_evt
    return {
        "delta_sigma_L1": float(z_L1_evt.mean()),
        "delta_sigma_unb": float(z_unb_evt.mean()),
        "delta_sigma_ediv": float(z_ediv_evt.mean()),
        "n_baseline": int(len(base)),
        "n_event": int(len(evt)),
        "baseline_mean_L1": float(mu_L1),
        "baseline_std_L1": float(sd_L1),
        "baseline_mean_unb": float(mu_unb),
        "baseline_std_unb": float(sd_unb),
    }


def main():
    print("=== Robustness via Window Scan ===")
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    all_symbols = list(closes.columns)
    print(f"Loaded ohlc_40.parquet: {closes.shape}")
    print(f"Windows to test: {WINDOWS}")

    results = {}
    for w in WINDOWS:
        t0 = time.time()
        print(f"\n[window={w}] computing indicators ...")
        df = compute_indicators_for_window(closes, all_symbols, w)
        resp = event_response(df)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s, n_rows={len(df)}, "
              f"n_baseline={resp['n_baseline']}, n_event={resp['n_event']}")
        print(f"  Δσ L1   = {resp['delta_sigma_L1']:+.3f}")
        print(f"  Δσ unb  = {resp['delta_sigma_unb']:+.3f}")
        print(f"  Δσ ediv = {resp['delta_sigma_ediv']:+.3f}")
        results[str(w)] = {
            "window": w,
            "elapsed_sec": elapsed,
            "n_rows_total": len(df),
            **resp,
        }

    out = {
        "meta": {
            "windows": WINDOWS,
            "threshold": THRESHOLD,
            "returns_period": [RETURNS_START, RETURNS_END],
            "baseline_window": [BASELINE_START, BASELINE_END],
            "event_window": [EVENT_START, EVENT_END],
            "event_name": "2025-04 Liberation Day",
            "n_symbols": len(all_symbols),
        },
        "results": results,
        "summary_table": {
            "window": WINDOWS,
            "delta_sigma_L1": [results[str(w)]["delta_sigma_L1"] for w in WINDOWS],
            "delta_sigma_unb": [results[str(w)]["delta_sigma_unb"] for w in WINDOWS],
            "delta_sigma_ediv": [results[str(w)]["delta_sigma_ediv"] for w in WINDOWS],
        },
    }

    out_path = DATA_DIR / "robustness_window_scan.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("\n=== Summary Table ===")
    print(f"{'window':>8} {'Δσ_L1':>10} {'Δσ_unb':>10} {'Δσ_ediv':>10}")
    for w in WINDOWS:
        r = results[str(w)]
        print(f"{w:>8d} {r['delta_sigma_L1']:>+10.3f} {r['delta_sigma_unb']:>+10.3f} {r['delta_sigma_ediv']:>+10.3f}")


if __name__ == "__main__":
    main()
