"""
銘柄数 sensitivity 検証 (Robustness via Subsampling)
======================================================

目的:
  40 銘柄から無作為に 10 銘柄を削除し、30 銘柄で同じ手順
  (window=30 PH, threshold=0.3) を実行。Liberation Day (2025-04-02
  〜 2025-04-15) の event response (Δσ) を 30 trials で測り、
  banner finding (e_div ≫ 0 in LD) が銘柄選択に依存しないことを示す。

出力:
  data/robustness_subsample_40to30.json
    - trials: list[dict]  各 trial の dropped_symbols, Δσ_L1, Δσ_unb, Δσ_ediv
    - summary: 30 trials の mean / std / min / max
    - baseline_40: 40 銘柄での同じ計算 (reference)
    - meta: 設定情報

重い PH 全期間 30 回は避け、event 周辺の短期間 (約 60 営業日)
だけ計算することで軽量化。

設計:
  - returns_period: 2025-01-01 〜 2025-04-30
  - PH 計算 period: window=30 で出力 ≒ 2025-02-12 〜 2025-04-30
  - baseline_window: 2025-02-15 〜 2025-03-25 (event 前、約 28 営業日)
  - event_window:    2025-04-02 〜 2025-04-15 (Liberation Day 期間)
  - z 化: baseline_window の mean/std で z 化
  - Δσ_metric = mean(z_metric in event_window)
                - mean(z_metric in baseline_window)
                  (定義上 baseline は 0 になるので Δσ = mean(z in event))
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
WINDOW = 30
THRESHOLD = 0.3
N_TRIALS = 30
DROP_K = 10  # 40 → 30
RNG_SEED = 20250602

# event study windows
RETURNS_START = "2025-01-01"
RETURNS_END = "2025-04-30"
BASELINE_START = "2025-02-15"
BASELINE_END = "2025-03-25"
EVENT_START = "2025-04-02"
EVENT_END = "2025-04-15"


def compute_indicators_for_period(closes: pd.DataFrame,
                                   symbols: list[str],
                                   period_start: str,
                                   period_end: str,
                                   window: int = WINDOW,
                                   threshold: float = THRESHOLD) -> pd.DataFrame:
    """指定 symbols / period に対して日次の L1_H1, n_unb を計算.

    window 分の助走を取るため、period_start の {window+5} 営業日前から
    closes を取得する想定。返り値の date は period_start 以降。
    """
    sub = closes[symbols].copy()
    returns = sub.pct_change()
    # window 営業日助走を取る (period_start から window 営業日前にずらして開始)
    start_ts = pd.Timestamp(period_start)
    end_ts = pd.Timestamp(period_end)
    # window 分以上の助走確保 (営業日換算で window*1.5 程度を暦日で引く)
    pad_days = int(window * 1.7) + 5
    pad_start = start_ts - pd.Timedelta(days=pad_days)
    returns = returns.loc[(returns.index >= pad_start) & (returns.index <= end_ts)]

    rows = []
    n = len(returns)
    for t_idx in range(window, n):
        date = returns.index[t_idx - 1]
        if date < start_ts or date > end_ts:
            continue
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
    """baseline window で z 化し、event window 平均 (= Δσ) を返す."""
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
    }


def main():
    print("=== Robustness via Subsampling (40 -> 30) ===")
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    all_symbols = list(closes.columns)
    print(f"Loaded ohlc_40.parquet: {closes.shape}, symbols={len(all_symbols)}")

    # ===== baseline (40 銘柄) =====
    t0 = time.time()
    print("\n[baseline] 40 symbols ...")
    base_df = compute_indicators_for_period(closes, all_symbols,
                                            RETURNS_START, RETURNS_END)
    base_resp = event_response(base_df)
    print(f"  done in {time.time()-t0:.1f}s, n_rows={len(base_df)}")
    print(f"  Δσ  L1   = {base_resp['delta_sigma_L1']:+.3f}")
    print(f"  Δσ  unb  = {base_resp['delta_sigma_unb']:+.3f}")
    print(f"  Δσ  ediv = {base_resp['delta_sigma_ediv']:+.3f}")

    # ===== subsample trials =====
    rng = np.random.default_rng(RNG_SEED)
    trials = []
    print(f"\n[subsample] {N_TRIALS} trials, drop {DROP_K} of 40 each ...")
    for t in range(N_TRIALS):
        dropped = sorted(rng.choice(all_symbols, size=DROP_K, replace=False).tolist())
        kept = [s for s in all_symbols if s not in dropped]
        t_start = time.time()
        df = compute_indicators_for_period(closes, kept,
                                           RETURNS_START, RETURNS_END)
        resp = event_response(df)
        trials.append({
            "trial": t,
            "n_kept": len(kept),
            "dropped_symbols": dropped,
            **resp,
        })
        elapsed = time.time() - t_start
        print(f"  trial {t+1:02d}/{N_TRIALS}  ({elapsed:.1f}s)  "
              f"Δσ L1={resp['delta_sigma_L1']:+.3f}  "
              f"unb={resp['delta_sigma_unb']:+.3f}  "
              f"ediv={resp['delta_sigma_ediv']:+.3f}")

    # ===== aggregate =====
    arr_L1 = np.array([t["delta_sigma_L1"] for t in trials], dtype=float)
    arr_unb = np.array([t["delta_sigma_unb"] for t in trials], dtype=float)
    arr_ediv = np.array([t["delta_sigma_ediv"] for t in trials], dtype=float)

    def summarize(arr: np.ndarray) -> dict:
        a = arr[np.isfinite(arr)]
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
            "n_valid": int(len(a)),
        }

    summary = {
        "delta_sigma_L1":   summarize(arr_L1),
        "delta_sigma_unb":  summarize(arr_unb),
        "delta_sigma_ediv": summarize(arr_ediv),
    }

    # baseline と subsample mean の比較
    def diff_block(base, sub):
        return {
            "baseline": float(base),
            "subsample_mean": sub["mean"],
            "diff": sub["mean"] - float(base),
            "within_p05_p95": bool(sub["p05"] <= float(base) <= sub["p95"]),
        }

    comparison = {
        "delta_sigma_L1":   diff_block(base_resp["delta_sigma_L1"],   summary["delta_sigma_L1"]),
        "delta_sigma_unb":  diff_block(base_resp["delta_sigma_unb"],  summary["delta_sigma_unb"]),
        "delta_sigma_ediv": diff_block(base_resp["delta_sigma_ediv"], summary["delta_sigma_ediv"]),
    }

    out = {
        "meta": {
            "window": WINDOW,
            "threshold": THRESHOLD,
            "n_trials": N_TRIALS,
            "drop_k": DROP_K,
            "rng_seed": RNG_SEED,
            "returns_period": [RETURNS_START, RETURNS_END],
            "baseline_window": [BASELINE_START, BASELINE_END],
            "event_window": [EVENT_START, EVENT_END],
            "event_name": "2025-04 Liberation Day",
        },
        "baseline_40": base_resp,
        "trials": trials,
        "summary_30": summary,
        "comparison_baseline_vs_subsample": comparison,
    }

    out_path = DATA_DIR / "robustness_subsample_40to30.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("\n=== Summary (30 trials) ===")
    for k, s in summary.items():
        print(f"  {k:18s}  mean={s['mean']:+.3f}  std={s['std']:.3f}  "
              f"[p05={s['p05']:+.3f}, p95={s['p95']:+.3f}]")
    print("\n=== Baseline 40 vs Subsample 30 ===")
    for k, c in comparison.items():
        flag = "OK (within 90% CI)" if c["within_p05_p95"] else "OUT of 90% CI"
        print(f"  {k:18s}  baseline={c['baseline']:+.3f}  sub_mean={c['subsample_mean']:+.3f}  "
              f"diff={c['diff']:+.3f}  {flag}")


if __name__ == "__main__":
    main()
