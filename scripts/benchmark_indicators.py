"""
3 指標ベンチマーク: 不整合サイクル数 (我々) vs Walk-based K (Ferreira 2021) vs frustration index (Aref 2018).

すべて符号付きグラフの「バランスの破れ」を測る指標だが、数学的に別物.
日次時系列で計算し、相関 + event study で性質の違いを示す.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import expm

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(HERE / "lib"))

from market_category import MarketCategory
from homology import signed_cycle_balance


def walk_based_K(G: nx.Graph, beta: float = 1.0) -> float:
    """
    Ferreira 2021 / Estrada-Benzi 2014:
      K = tr(exp(β A)) / tr(exp(β |A|))
    K ∈ [0, 1]; K=1 perfectly balanced, K=0 maximally unbalanced.
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 1.0
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n))
    Aabs = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        s = int(d.get("sign", 1))
        w = float(d.get("weight", 1.0))
        A[i, j]    += s * w
        A[j, i]    += s * w
        Aabs[i, j] += w
        Aabs[j, i] += w
    try:
        eA  = expm(beta * A)
        eAa = expm(beta * Aabs)
        tr_A  = float(np.trace(eA))
        tr_Aa = float(np.trace(eAa))
        return tr_A / tr_Aa if tr_Aa > 1e-9 else 1.0
    except Exception:
        return 1.0


def frustration_index_approx(G: nx.Graph, max_iter: int = 200,
                              seed: int = 42) -> int:
    """
    Aref 2018 frustration index F(G):
      最小の「符号反転で balanced 化できるエッジ数」.
      NP-hard, ここでは local search で近似.

    実装: 各ノードに ±1 ラベル σ_v を割り当てて、エッジ符号 σ_uv == σ_u * σ_v
          となるエッジを balanced, ならないエッジを frustrated.
          frustration = min over labelings of #frustrated edges.

    手法: greedy ノードラベル + 単一ノード反転改善ループ.
    """
    if G.number_of_edges() == 0:
        return 0
    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)
    best_F = G.number_of_edges()
    for _ in range(5):  # 5 ランダム初期化
        label = {v: int(rng.choice([-1, 1])) for v in nodes}
        improved = True
        while improved:
            improved = False
            for v in nodes:
                # フリップした時の frustrated edge 数増減を見る
                cur_f = 0; new_f = 0
                for u in G.neighbors(v):
                    s = int(G[v][u].get("sign", 1))
                    cur = s == label[v] * label[u]
                    new = s == (-label[v]) * label[u]
                    if not cur: cur_f += 1
                    if not new: new_f += 1
                if new_f < cur_f:
                    label[v] = -label[v]
                    improved = True
        # 最終的な frustrated edge 数
        F = 0
        for u, v, d in G.edges(data=True):
            s = int(d.get("sign", 1))
            if s != label[u] * label[v]:
                F += 1
        if F < best_F:
            best_F = F
    return int(best_F)


def main(window: int = 30, threshold: float = 0.3, beta: float = 1.0):
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    returns = closes.pct_change()
    n = len(returns)
    print(f"Loaded ohlc_40: {closes.shape}")
    rows = []
    t0 = time.time()
    for i, t_idx in enumerate(range(window, n)):
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
        date = returns.index[t_idx - 1]
        if win_clean.shape[1] < 5:
            rows.append({"date": date, "n_unb": np.nan, "K": np.nan, "F": np.nan})
            continue
        try:
            corr = win_clean.corr()
            cat = MarketCategory(symbols=list(win_clean.columns),
                                  corr_matrix=corr, threshold=threshold)
            cat._build_graph()
            # 1. 不整合サイクル数 (我々)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bal = signed_cycle_balance(cat.G)
            n_unb = int(bal["n_unbalanced"])
            # 2. Walk-based K (Ferreira)
            K = walk_based_K(cat.G, beta=beta)
            # 3. Frustration index (Aref)
            F = frustration_index_approx(cat.G)
            rows.append({"date": date, "n_unb": n_unb,
                         "K": round(K, 6), "F": F,
                         "n_edges": cat.G.number_of_edges()})
        except Exception as e:
            rows.append({"date": date, "n_unb": np.nan, "K": np.nan, "F": np.nan})
        if (i + 1) % 200 == 0:
            r = rows[-1]
            print(f"  [{i+1}/{n-window}] {r['date'].date()}  n_unb={r['n_unb']}  K={r['K']}  F={r['F']}  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "benchmark_indicators_w30.csv", index=False)
    print(f"\nSaved: benchmark_indicators_w30.csv  ({len(df)} rows, {time.time()-t0:.0f}s)")
    valid = df.dropna(subset=["n_unb", "K", "F"])
    if len(valid) > 0:
        cor = valid[["n_unb", "K", "F"]].corr().round(3)
        print(f"\n=== 3 指標相関行列 ===")
        print(cor)
        print(f"\nn_unb 統計: mean={valid['n_unb'].mean():.1f}, std={valid['n_unb'].std():.1f}")
        print(f"K 統計:     mean={valid['K'].mean():.3f}, std={valid['K'].std():.3f}")
        print(f"F 統計:     mean={valid['F'].mean():.1f}, std={valid['F'].std():.1f}")


if __name__ == "__main__":
    main()
