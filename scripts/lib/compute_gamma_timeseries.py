"""
γ検証用: L^1 と 不整合サイクル数 を 同じローリング窓で日次計算.

出力: gamma_timeseries_w{w}.csv
  - date
  - n_symbols (window内で全期間NaN無しの銘柄数)
  - L1_H1     (Vietoris-Rips PH の H1 L^1 norm; |corr|ベース、Z係数)
  - n_unb     (符号付きサイクル基底の不整合サイクル数; signベース、Z/2係数)
  - n_edges   (閾値超エッジ数)
  - balance_rate
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from persistent_homology import persistence_diagram, persistence_summary
from market_category import MarketCategory
from homology import signed_cycle_balance

HERE = Path(__file__).parent


def main(window: int = 30, threshold: float = 0.3, save_every: int = 100,
         suffix: str | None = None, input_file: str = "ohlc_40.parquet",
         out_prefix: str = "gamma_timeseries"):
    # 絶対パスが渡された場合はそのまま使い、相対パスの場合は HERE 基準で解決する
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = HERE / input_file
    closes = pd.read_parquet(input_path)
    returns = closes.pct_change()
    print(f"Loaded {input_file}: {closes.shape}, range {closes.index.min().date()} -> {closes.index.max().date()}")

    n = len(returns)
    suffix = suffix if suffix is not None else f"_w{window}"
    # out_prefix が絶対パスの場合はそのまま、相対パスの場合は HERE 基準
    out_prefix_path = Path(out_prefix)
    if out_prefix_path.is_absolute():
        out_path = Path(f"{out_prefix}{suffix}.csv")
    else:
        out_path = HERE / f"{out_prefix}{suffix}.csv"

    print(f"window={window}, threshold={threshold}, output={out_path.name}")
    print(f"Computing for idx {window}..{n} ({n - window} days)")

    rows = []
    t0 = time.time()

    for i, t_idx in enumerate(range(window, n)):
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
        date = returns.index[t_idx - 1]

        if win_clean.shape[1] < 5:
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan,
                         "n_edges": 0, "balance_rate": np.nan})
            continue

        try:
            corr = win_clean.corr()
            # --- L^1 norm of H_1 (Vietoris-Rips PH, |corr|ベース) ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diag = persistence_diagram(corr, max_dim=1)
            summ = persistence_summary(diag)
            L1 = float(summ["L1_norm_H1"])

            # --- 不整合サイクル数 (signed cycle balance, signベース) ---
            cat = MarketCategory(symbols=list(win_clean.columns),
                                 corr_matrix=corr, threshold=threshold)
            cat._build_graph()
            bal = signed_cycle_balance(cat.G)
            n_unb = int(bal["n_unbalanced"])
            n_edges = cat.G.number_of_edges()
            br = float(bal["balance_rate"])

            rows.append({
                "date": date,
                "n_symbols": win_clean.shape[1],
                "L1_H1": round(L1, 6),
                "n_unb": n_unb,
                "n_edges": n_edges,
                "balance_rate": round(br, 4),
            })
        except Exception as e:
            print(f"  [{i}] {date.date()} FAILED: {e}")
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan,
                         "n_edges": 0, "balance_rate": np.nan})

        if (i + 1) % save_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - window - i - 1) / rate
            r = rows[-1]
            print(f"  [{i+1}/{n-window}] {r['date'].date()}  L1={r['L1_H1']}  n_unb={r['n_unb']}  ({elapsed:.0f}s, ETA {eta:.0f}s)")
            pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Saved: {out_path}")
    print(f"Rows: {len(df)}")
    print("\n=== Summary ===")
    print(df[["L1_H1", "n_unb", "n_edges", "balance_rate"]].describe())

    # 簡易相関チェック
    corr_l1_unb = df[["L1_H1", "n_unb"]].corr().iloc[0, 1]
    print(f"\nCorr(L1_H1, n_unb) = {corr_l1_unb:.4f}")


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    inp = sys.argv[2] if len(sys.argv) > 2 else "ohlc_40.parquet"
    pref = sys.argv[3] if len(sys.argv) > 3 else "gamma_timeseries"
    main(window=w, input_file=inp, out_prefix=pref)
