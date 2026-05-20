"""
ホモロジー / 持続ホモロジー / コホモロジー的整合性 の実装.

Adachi 2026 (Martingale Cohomology) + Gidea & Katz 2017 (TDA Crashes) 直結.

実装内容:
  1. ベッチ数 H_0, H_1
     H_0 = 連結成分数（既存 num_communities と等価だが圏論的に明示）
     H_1 = 独立な 1-サイクル数 = m - n + c (m=edges, n=nodes, c=components)

  2. 持続ホモロジー (Persistent Homology, 簡易版)
     閾値 θ を動かすと H_0, H_1 がどう進化するかをbarcode化
     Gidea 2017 のクラッシュ前兆検出と同じ発想

  3. 構造的整合性 (Structural Balance / Cohomological Consistency)
     符号付きエッジの閉路の符号積をチェック
     Adachi 2026 の cohomological arbitrage に近い
     全サイクル積=+1 → balanced（無矛盾）
     -1のサイクル → unbalanced（裁定機会・構造緊張）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx

from market_category import MarketCategory


# ============================================================
#  ベッチ数 (Betti numbers)
# ============================================================

def betti_numbers(G: nx.Graph) -> dict:
    """グラフのベッチ数 (H_0, H_1)."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    c = nx.number_connected_components(G)
    h0 = c
    h1 = m - n + c   # cycle rank (Euler特性数: χ = n - m, dim H_0 - dim H_1 = χ から H_1 = m - n + c)
    return {"H_0": h0, "H_1": max(0, h1), "n": n, "m": m, "components": c}


def euler_characteristic(G: nx.Graph) -> int:
    """オイラー標数 χ = n - m  (1次元複体ではこれで決まる)."""
    return G.number_of_nodes() - G.number_of_edges()


# ============================================================
#  持続ホモロジー (Persistent Homology, 1D simplex版)
# ============================================================

def persistent_homology(
    corr_matrix: pd.DataFrame,
    thresholds: Optional[np.ndarray] = None,
    use_abs: bool = True,
) -> pd.DataFrame:
    """閾値スイープで H_0, H_1 の進化を追う.

    use_abs=True なら相関の絶対値で閾値判定（正負を一旦無視）.
    use_abs=False なら相関値そのままで閾値以上のエッジ.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    symbols = list(corr_matrix.columns)
    n = len(symbols)

    rows = []
    for t in thresholds:
        G = nx.Graph()
        G.add_nodes_from(symbols)
        for i in range(n):
            for j in range(i + 1, n):
                v = corr_matrix.iloc[i, j]
                if not np.isfinite(v):
                    continue
                v_check = abs(v) if use_abs else v
                if v_check >= t:
                    G.add_edge(symbols[i], symbols[j], weight=abs(v), sign=np.sign(v))
        bn = betti_numbers(G)
        rows.append({"threshold": round(float(t), 3), **bn})
    return pd.DataFrame(rows)


def homology_summary(ph: pd.DataFrame) -> dict:
    """持続ホモロジーの集約.
    H_1 が最大化される閾値、H_0=1 になる閾値（連結化点）等."""
    h0_collapse = ph[ph["H_0"] == 1]["threshold"].min() if (ph["H_0"] == 1).any() else None
    h1_max_idx = ph["H_1"].idxmax()
    return {
        "h0_collapse_threshold": float(h0_collapse) if h0_collapse is not None else None,
        "h1_max_value": int(ph["H_1"].max()),
        "h1_max_threshold": float(ph.loc[h1_max_idx, "threshold"]),
        "ph_signature": ph[["threshold", "H_0", "H_1"]].to_dict("records"),
    }


# ============================================================
#  構造的整合性 / コホモロジー的障害 (Adachi 2026 風)
# ============================================================

def signed_cycle_balance(G: nx.Graph, max_cycles: int = 1000) -> dict:
    """符号付きエッジで閉路の符号積をチェック.

    Heider (1946) / Cartwright-Harary (1956) のstructural balance theory:
      全閉路で符号積=+1 → balanced（friend-of-friend is friend）
      -1の閉路 → unbalanced（構造緊張）

    Adachi 2026 "Homological Arbitrage" の発想に近い:
      確率/期待値の閉路で整合性が破れる = コホモロジー的障害 = 裁定可能性
    """
    cycles = nx.cycle_basis(G)
    cycles = cycles[:max_cycles]
    n_total = len(cycles)
    n_balanced = 0
    n_unbalanced = 0
    unbalanced_examples = []
    for cycle in cycles:
        sign_product = 1
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            if G.has_edge(u, v):
                sign_product *= int(G[u][v].get("sign", 1))
        if sign_product > 0:
            n_balanced += 1
        else:
            n_unbalanced += 1
            if len(unbalanced_examples) < 5:
                unbalanced_examples.append(cycle)
    return {
        "n_cycles_in_basis": n_total,
        "n_balanced": n_balanced,
        "n_unbalanced": n_unbalanced,
        "balance_rate": n_balanced / n_total if n_total > 0 else 1.0,
        "unbalanced_examples": unbalanced_examples,
    }


# ============================================================
#  時系列ホモロジー追跡
# ============================================================

def homology_timeseries(
    returns: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.3,
    step: int = 1,
) -> pd.DataFrame:
    """ローリング窓で各時点の H_0, H_1, balance を計算."""
    rows = []
    n = len(returns)
    for i in range(window, n, step):
        win = returns.iloc[i - window: i]
        cat = MarketCategory.from_returns(win, threshold=threshold)
        bn = betti_numbers(cat.G)
        bal = signed_cycle_balance(cat.G)
        rows.append({
            "date": str(returns.index[i].date()),
            "H_0": bn["H_0"],
            "H_1": bn["H_1"],
            "n_edges": bn["m"],
            "balance_rate": round(bal["balance_rate"], 4),
            "n_unbalanced_cycles": bal["n_unbalanced"],
        })
    return pd.DataFrame(rows)


# ============ デモ ============
if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).parent
    closes = pd.read_parquet(here / "ohlc_40.parquet")
    returns = closes.pct_change()

    # ===== 1. 直近の単一時点 =====
    print("=== 1. 直近30日の市場圏のベッチ数 ===")
    win = returns.iloc[-30:]
    cat = MarketCategory.from_returns(win, threshold=0.3)
    bn = betti_numbers(cat.G)
    print(f"  H_0 (連結成分): {bn['H_0']}  ← コミュニティ数 / 「孤立島」の数")
    print(f"  H_1 (独立サイクル): {bn['H_1']}  ← グラフの「穴」 / 独立な閉路")
    print(f"  Euler χ = n - m = {euler_characteristic(cat.G)}")

    # ===== 2. 構造的整合性 =====
    print()
    print("=== 2. Structural Balance / Cohomological Consistency ===")
    bal = signed_cycle_balance(cat.G)
    print(f"  サイクル基底: {bal['n_cycles_in_basis']}")
    print(f"  balanced (符号積+1): {bal['n_balanced']}")
    print(f"  unbalanced (符号積-1): {bal['n_unbalanced']}  ← 構造緊張・裁定可能性")
    print(f"  balance_rate: {bal['balance_rate']:.4f}")
    if bal["unbalanced_examples"]:
        print(f"  unbalanced examples (前3つ):")
        for c in bal["unbalanced_examples"][:3]:
            print(f"    {' → '.join(c)} → ({c[0]})")

    # ===== 3. 持続ホモロジー =====
    print()
    print("=== 3. Persistent Homology (閾値スイープ) ===")
    ph = persistent_homology(cat.corr_matrix)
    print(ph.to_string(index=False))
    summary = homology_summary(ph)
    print()
    print(f"  H_0 が 1 になる閾値（全連結化）: {summary['h0_collapse_threshold']}")
    print(f"  H_1 最大値: {summary['h1_max_value']} at threshold {summary['h1_max_threshold']}")

    # ===== 4. 5年時系列 =====
    print()
    print("=== 4. 5年ホモロジー時系列（30日step） ===")
    ts = homology_timeseries(returns, window=20, threshold=0.3, step=30)
    print(f"  Computed {len(ts)} time points")
    out_path = here / "homology_timeseries.csv"
    ts.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print()

    # 統計
    print("  === 時系列統計 ===")
    print(f"    H_0 平均: {ts['H_0'].mean():.2f},  最大: {ts['H_0'].max()}")
    print(f"    H_1 平均: {ts['H_1'].mean():.2f},  最大: {ts['H_1'].max()}")
    print(f"    balance_rate 平均: {ts['balance_rate'].mean():.3f}")

    # 構造緊張トップ
    print()
    print("  === Top 10 構造緊張日 (balance_rate最低) ===")
    print(ts.nsmallest(10, "balance_rate")[["date", "H_0", "H_1", "balance_rate", "n_unbalanced_cycles"]].to_string(index=False))
