"""
本物の持続ホモロジー（Persistent Homology）.

Gidea & Katz 2017 の手法そのまま:
  - 相関行列から距離行列 d_ij = sqrt(2(1 - r_ij)) を構築
  - Vietoris-Rips フィルトレーション
  - H_0, H_1 の birth-death pairs (persistence diagram) を計算
  - 各特徴の persistence = death - birth が「重要度」

これにより:
  - 「閾値での値」じゃなく「**生死バー**」を可視化（Gidea 2017の論文と同じ形式）
  - persistence の長い feature = 構造的に本質的な「穴」
  - 短い feature = ノイズ
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ripser import ripser  # 軽量・高速の PH ライブラリ


def correlation_to_distance(corr_matrix: pd.DataFrame) -> np.ndarray:
    """Mantegna距離 d_ij = sqrt(2(1 - r_ij)) で相関行列を距離行列に変換."""
    R = corr_matrix.fillna(0).values
    # 対角は0、それ以外は sqrt(2(1-r))
    D = np.sqrt(np.clip(2 * (1 - R), 0, None))
    np.fill_diagonal(D, 0)
    # 対称性確保
    D = (D + D.T) / 2
    return D


def persistence_diagram(corr_matrix: pd.DataFrame, max_dim: int = 1) -> dict:
    """相関行列から持続ホモロジー（H_0, H_1）を計算.

    戻り値:
      {
        "H0": [(birth, death), ...],
        "H1": [(birth, death), ...],
        "max_distance": float
      }
    """
    D = correlation_to_distance(corr_matrix)
    max_edge = float(np.sqrt(2))  # r=0 で d=√2、それ以上は遠すぎ

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ripser(D, maxdim=max_dim, thresh=max_edge, distance_matrix=True)
    # result["dgms"] = [H0_array, H1_array, ...], each array shape (n_features, 2)
    dgms = result["dgms"]

    H0_bars = []
    H1_bars = []
    for dim, arr in enumerate(dgms):
        bars_list = H0_bars if dim == 0 else (H1_bars if dim == 1 else None)
        if bars_list is None:
            continue
        for row in arr:
            birth, death = float(row[0]), float(row[1])
            # 無限大の death は max_edge_length で置き換え
            if not np.isfinite(death):
                death = max_edge
            bars_list.append((birth, death))

    # persistence の降順でソート
    H0_bars.sort(key=lambda b: -(b[1] - b[0]))
    H1_bars.sort(key=lambda b: -(b[1] - b[0]))

    return {
        "H0": H0_bars,
        "H1": H1_bars,
        "max_distance": float(np.sqrt(2)),
    }


def persistence_summary(diagram: dict, top_n: int = 5) -> dict:
    """サマリ: トップNの persistence、平均、最大等."""
    H0 = diagram["H0"]
    H1 = diagram["H1"]
    H0_pers = [d - b for b, d in H0]
    H1_pers = [d - b for b, d in H1]
    return {
        "n_H0": len(H0),
        "n_H1": len(H1),
        "max_H0_persistence": max(H0_pers) if H0_pers else 0,
        "max_H1_persistence": max(H1_pers) if H1_pers else 0,
        "avg_H1_persistence": sum(H1_pers) / len(H1_pers) if H1_pers else 0,
        "top_H1_bars": H1[:top_n],
        "L1_norm_H1": sum(H1_pers),  # Gidea 2017のL^p norm の p=1版
        "L2_norm_H1": sum(p**2 for p in H1_pers) ** 0.5,
    }


# ============ デモ ============
if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).parent
    closes = pd.read_parquet(here / "clean_returns.parquet")
    win = closes.iloc[-30:].dropna()
    corr = win.corr()
    print(f"Computing PH on {win.shape[1]} symbols, 30-day window...")

    diag = persistence_diagram(corr)
    summ = persistence_summary(diag)

    print(f"\n=== Persistence Diagram ===")
    print(f"  H_0 features: {summ['n_H0']}")
    print(f"  H_1 features: {summ['n_H1']}")
    print(f"  Max H_0 persistence: {summ['max_H0_persistence']:.4f}")
    print(f"  Max H_1 persistence: {summ['max_H1_persistence']:.4f}")
    print(f"  Avg H_1 persistence: {summ['avg_H1_persistence']:.4f}")
    print(f"  L^1 norm of H_1: {summ['L1_norm_H1']:.4f}  (Gidea 2017 crash indicator)")
    print(f"  L^2 norm of H_1: {summ['L2_norm_H1']:.4f}")
    print()
    print(f"  Top 5 H_1 bars (longest persistence):")
    for b, d in summ["top_H1_bars"]:
        print(f"    birth={b:.4f}  death={d:.4f}  persistence={d-b:.4f}")
