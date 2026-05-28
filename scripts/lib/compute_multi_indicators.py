"""
指標を 2 → 11 に細分化 (集約スカラーを構成要素に分解).

L¹ (集約スカラー) を分解:
  - L1   = Σ persistence
  - L2   = √(Σ persistence²)
  - Linf = max persistence
  - nH1  = 穴の個数
  - meanP = 平均 persistence
  - entropy = persistence エントロピー (Atienza 2020 風)

不整合サイクル数 (集約スカラー) を分解:
  - n_unb_total = 全不整合サイクル
  - n_unb_3     = 長さ 3 のみ
  - n_unb_4     = 長さ 4 のみ
  - n_unb_5plus = 長さ 5+
  - weighted_unb = エッジ重み |corr| の積で重み付け
  - balance_rate = balanced / total
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from market_category import MarketCategory
from persistent_homology import persistence_diagram

HERE = Path(__file__).parent


def l1_subindicators(diag: dict) -> dict:
    """持続ホモロジーから複数のスカラーを取り出す."""
    H1 = diag["H1"]
    pers = np.array([d - b for b, d in H1])
    if len(pers) == 0:
        return {"L1": 0.0, "L2": 0.0, "Linf": 0.0, "nH1": 0,
                "meanP": 0.0, "entropy": 0.0}
    L1 = float(pers.sum())
    L2 = float(np.sqrt(np.sum(pers ** 2)))
    Linf = float(pers.max())
    nH1 = int(len(pers))
    meanP = float(pers.mean())
    # persistence entropy: -Σ (p_i/L1) log(p_i/L1)
    if L1 > 1e-9:
        probs = pers / L1
        probs = probs[probs > 1e-9]
        entropy = float(-np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0
    return {"L1": L1, "L2": L2, "Linf": Linf, "nH1": nH1,
            "meanP": meanP, "entropy": entropy}


def signed_cycle_decomposition(G: nx.Graph, max_cycles: int = 1500) -> dict:
    """signed cycle balance を サイクル長別 + 重み付きに分解."""
    cycles = nx.cycle_basis(G)
    cycles = cycles[:max_cycles]
    n_total = len(cycles)
    n_bal_3 = n_unb_3 = 0
    n_bal_4 = n_unb_4 = 0
    n_bal_5p = n_unb_5p = 0
    weighted_unb = 0.0
    total_unb = 0
    for cycle in cycles:
        L = len(cycle)
        sign_prod = 1
        weight_prod = 1.0
        for i in range(L):
            u, v = cycle[i], cycle[(i + 1) % L]
            if G.has_edge(u, v):
                ed = G[u][v]
                sign_prod *= int(ed.get("sign", 1))
                weight_prod *= float(ed.get("weight", 1.0))
        is_unb = sign_prod < 0
        if L == 3:
            if is_unb: n_unb_3 += 1
            else: n_bal_3 += 1
        elif L == 4:
            if is_unb: n_unb_4 += 1
            else: n_bal_4 += 1
        else:
            if is_unb: n_unb_5p += 1
            else: n_bal_5p += 1
        if is_unb:
            total_unb += 1
            weighted_unb += weight_prod
    n_unb_total = n_unb_3 + n_unb_4 + n_unb_5p
    return {
        "n_unb_total":  n_unb_total,
        "n_unb_3":      n_unb_3,
        "n_unb_4":      n_unb_4,
        "n_unb_5plus":  n_unb_5p,
        "weighted_unb": float(weighted_unb),
        "balance_rate": float(1 - n_unb_total / n_total) if n_total > 0 else 1.0,
        "n_cycles":     n_total,
    }


def main(
    window: int = 30,
    threshold: float = 0.3,
    save_every: int = 100,
    input_file: str = "ohlc_40.parquet",
    out_prefix: str = "multi_indicators",
) -> None:
    """ローリング窓で 11 指標を日次計算し CSV に保存する."""
    closes = pd.read_parquet(HERE / input_file)
    returns = closes.pct_change()
    print(f"Loaded {input_file}: {closes.shape}, "
          f"{closes.index.min().date()} -> {closes.index.max().date()}")

    n = len(returns)
    out_path = HERE / f"{out_prefix}_w{window}.csv"
    rows = []
    t0 = time.time()
    print(f"Computing 11 indicators for {n - window} days...")

    for i, t_idx in enumerate(range(window, n)):
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
        date = returns.index[t_idx - 1]
        if win_clean.shape[1] < 5:
            rows.append({"date": date, "n_symbols": win_clean.shape[1]})
            continue
        try:
            corr = win_clean.corr()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diag = persistence_diagram(corr, max_dim=1)
            l1_sub = l1_subindicators(diag)
            cat = MarketCategory(symbols=list(win_clean.columns),
                                 corr_matrix=corr, threshold=threshold)
            cat._build_graph()
            cyc_sub = signed_cycle_decomposition(cat.G)
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         **{k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in l1_sub.items()},
                         **{k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in cyc_sub.items()}})
        except Exception as e:
            print(f"  [{i}] {date.date()} FAILED: {e}")
            rows.append({"date": date, "n_symbols": win_clean.shape[1]})
        if (i + 1) % save_every == 0:
            elapsed = time.time() - t0
            r = rows[-1]
            print(f"  [{i+1}/{n-window}] {r['date'].date()}  "
                  f"L1={r.get('L1','-')} Linf={r.get('Linf','-')} "
                  f"unb3={r.get('n_unb_3','-')} unb4={r.get('n_unb_4','-')}  ({elapsed:.0f}s)")
            pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nDone in {time.time()-t0:.0f}s. Saved: {out_path}  ({len(df)} rows)")

    # 各指標の統計
    cols = ["L1", "L2", "Linf", "nH1", "meanP", "entropy",
            "n_unb_total", "n_unb_3", "n_unb_4", "n_unb_5plus",
            "weighted_unb", "balance_rate"]
    valid_cols = [c for c in cols if c in df.columns]
    print("\n=== 11 指標サマリー ===")
    print(df[valid_cols].describe().round(3))

    # 相関行列
    print("\n=== 指標間相関 (Pearson) ===")
    corr_mat = df[valid_cols].corr().round(3)
    print(corr_mat)
    corr_mat.to_csv(HERE / f"{out_prefix}_correlation_w{window}.csv")


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    inp = sys.argv[2] if len(sys.argv) > 2 else "ohlc_40.parquet"
    pref = sys.argv[3] if len(sys.argv) > 3 else "multi_indicators"
    main(window=w, input_file=inp, out_prefix=pref)
