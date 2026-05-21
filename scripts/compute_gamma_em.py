"""新興国市場で γ 時系列計算."""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(HERE / "lib"))

from persistent_homology import persistence_diagram, persistence_summary
from market_category import MarketCategory
from homology import signed_cycle_balance


def main(window: int = 30, threshold: float = 0.3):
    closes = pd.read_parquet(DATA_DIR / "ohlc_em.parquet")
    returns = closes.pct_change()
    print(f"Loaded ohlc_em.parquet: {closes.shape}")
    print(f"Range: {closes.index.min().date()} -> {closes.index.max().date()}")

    n = len(returns)
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diag = persistence_diagram(corr, max_dim=1)
            summ = persistence_summary(diag)
            cat = MarketCategory(symbols=list(win_clean.columns),
                                  corr_matrix=corr, threshold=threshold)
            cat._build_graph()
            bal = signed_cycle_balance(cat.G)
            rows.append({
                "date": date, "n_symbols": win_clean.shape[1],
                "L1_H1": round(float(summ["L1_norm_H1"]), 6),
                "n_unb": int(bal["n_unbalanced"]),
                "n_edges": cat.G.number_of_edges(),
                "balance_rate": round(float(bal["balance_rate"]), 4),
            })
        except Exception as e:
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan,
                         "n_edges": 0, "balance_rate": np.nan})
        if (i + 1) % 200 == 0:
            r = rows[-1]
            print(f"  [{i+1}/{n-window}] {r['date'].date()}  "
                  f"L1={r['L1_H1']}  n_unb={r['n_unb']}  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / f"gamma_em_timeseries_w{window}.csv", index=False)
    print(f"\nSaved: gamma_em_timeseries_w{window}.csv  ({len(df)} rows, {time.time()-t0:.0f}s)")
    print(df[["L1_H1", "n_unb"]].describe())
    valid = df.dropna(subset=["L1_H1", "n_unb"])
    if len(valid) > 0:
        corr = valid[["L1_H1", "n_unb"]].corr().iloc[0, 1]
        print(f"\nEM market corr(L1, n_unb) = {corr:+.4f}")
        print(f"  vs USA 40 = +0.16, CN 52 = +0.41")


if __name__ == "__main__":
    main()
