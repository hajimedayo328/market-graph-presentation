"""
Market Category — 金融市場の圏論的構造体.

圏として:
  対象 (Objects)     = 銘柄
  射  (Morphisms)    = 銘柄間の関係（相関）
  合成 (Composition) = 推移性（A-B + B-C → A-Cの伝播）

このファイルでは:
  - MarketCategory: 単一時点の圏
  - 位相不変量 (density, k-core, spectral, MST, removal impact等) の計算
  - 既存研究 (Mantegna 1999, Onnela 2003, DebtRank 2012) の指標を全部包含

使用例:
    cat = MarketCategory.from_returns(returns_df, threshold=0.3)
    print(cat.invariants())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx


@dataclass
class MarketCategory:
    """単一時点・単一観測者のマーケット圏."""
    symbols: list[str]
    corr_matrix: pd.DataFrame                     # 全ペア相関（n×n）
    threshold: float = 0.3                         # |corr| ≥ threshold のみエッジ
    G: nx.Graph = field(default_factory=nx.Graph)  # threshold適用後の無向グラフ
    timestamp: Optional[pd.Timestamp] = None
    label: str = ""                                # 観測者ラベル（ブローカー名等）

    # ===== ファクトリ =====

    @classmethod
    def from_returns(
        cls,
        returns: pd.DataFrame,
        threshold: float = 0.3,
        timestamp: Optional[pd.Timestamp] = None,
        label: str = "",
        min_periods: int = 10,
    ) -> "MarketCategory":
        """リターン時系列から圏を構築 (pairwise相関で欠損対応)."""
        symbols = list(returns.columns)
        # pandasのcorrはpairwise（NaNをペアごとにskip）。min_periodsで最小サンプル数指定
        corr = returns.corr(min_periods=min_periods)
        cat = cls(symbols=symbols, corr_matrix=corr, threshold=threshold,
                  timestamp=timestamp, label=label)
        cat._build_graph()
        return cat

    def _build_graph(self):
        """corr_matrix と threshold からグラフを構築する.

        weight=|corr|, sign=sign(corr) 属性を各エッジに付与する.
        """
        self.G = nx.Graph()
        self.G.add_nodes_from(self.symbols)
        for i, a in enumerate(self.symbols):
            for j, b in enumerate(self.symbols):
                if i >= j:
                    continue
                w = float(self.corr_matrix.iloc[i, j])
                if not np.isfinite(w):
                    continue
                if abs(w) >= self.threshold:
                    # 重みは |corr|、正負はsign属性で
                    self.G.add_edge(a, b, weight=abs(w), sign=np.sign(w))

    # ===== 基本指標 =====

    def n_objects(self) -> int:
        """銘柄数（圏の対象数）を返す."""
        return len(self.symbols)

    def n_morphisms(self) -> int:
        """エッジ数（圏の射数）を返す."""
        return self.G.number_of_edges()

    def density(self) -> float:
        """グラフ密度 = エッジ数 / 最大エッジ数 を返す."""
        n = self.n_objects()
        max_e = n * (n - 1) / 2
        return self.n_morphisms() / max_e if max_e > 0 else 0.0

    def avg_clustering(self) -> float:
        """重み付き平均クラスタリング係数を返す."""
        return float(nx.average_clustering(self.G, weight="weight"))

    # ===== グラフ理論的位相不変量 =====

    def k_core_distribution(self) -> dict[int, int]:
        """各ノードのk-core番号 → ヒストグラム."""
        cores = nx.core_number(self.G)
        out: dict[int, int] = {}
        for v in cores.values():
            out[v] = out.get(v, 0) + 1
        return out

    def max_k_core(self) -> int:
        """最大 k-core 番号を返す."""
        cores = nx.core_number(self.G)
        return max(cores.values()) if cores else 0

    def isolated_nodes(self) -> list[str]:
        """孤立ノード（次数0）リスト. ゴールド孤立で上昇の発見と同じ."""
        return [n for n in self.G.nodes if self.G.degree(n) == 0]

    def degree_centrality(self) -> dict[str, float]:
        """次数中心性を返す (networkx標準)."""
        return nx.degree_centrality(self.G)

    def eigenvector_centrality(self) -> dict[str, float]:
        """固有ベクトル中心性を返す. 計算失敗時は全ノード 0.0."""
        try:
            return nx.eigenvector_centrality_numpy(self.G, weight="weight")
        except Exception:
            return {n: 0.0 for n in self.G.nodes}

    def spectral_gap(self) -> float:
        """ラプラシアン第2固有値. 連結性の強さの目安."""
        if self.n_morphisms() == 0:
            return 0.0
        try:
            L = nx.normalized_laplacian_matrix(self.G).astype(float).toarray()
            eigs = np.sort(np.linalg.eigvalsh(L))
            # 0(自明)を除く最小 = spectral gap
            return float(eigs[1]) if len(eigs) > 1 else 0.0
        except Exception:
            return 0.0

    def mst_features(self) -> dict[str, float]:
        """MST: total edge weight, normalized tree length (NTL)."""
        if self.n_morphisms() == 0:
            return {"mst_total_weight": 0.0, "ntl": 0.0}
        # Mantegna距離: d_ij = sqrt(2 * (1 - r_ij))
        # weight属性=|corr|なので、距離に変換してからMST
        dist_g = nx.Graph()
        dist_g.add_nodes_from(self.G.nodes)
        for u, v, data in self.G.edges(data=True):
            r = data["weight"] * data.get("sign", 1)  # 元のcorr
            d = np.sqrt(max(0.0, 2 * (1 - r)))
            dist_g.add_edge(u, v, weight=d)
        try:
            mst = nx.minimum_spanning_tree(dist_g)
        except Exception:
            return {"mst_total_weight": 0.0, "ntl": 0.0}
        total = float(sum(d["weight"] for _, _, d in mst.edges(data=True)))
        n = self.n_objects()
        ntl = total / (n - 1) if n > 1 else 0.0  # Onnela 2003 NTL
        return {"mst_total_weight": total, "ntl": ntl, "mst_edges": mst.number_of_edges()}

    def num_max_cliques(self) -> int:
        """最大クリーク数."""
        try:
            return sum(1 for _ in nx.find_cliques(self.G))
        except Exception:
            return 0

    def communities(self) -> list[set[str]]:
        """コミュニティ検出 (Louvain風 / Greedy modularity)."""
        if self.n_morphisms() == 0:
            return [{n} for n in self.G.nodes]
        try:
            return list(nx.community.greedy_modularity_communities(self.G, weight="weight"))
        except Exception:
            return [{n} for n in self.G.nodes]

    def num_communities(self) -> int:
        """コミュニティ数を返す."""
        return len(self.communities())

    # ===== Battiston系: ノード除去耐性 (DebtRank の応用) =====

    def removal_impact(self, sym: str) -> float:
        """対象 sym を圏から除去した時の構造崩壊度.
        密度 + spectral gap + 連結成分数 の変化を合算."""
        if sym not in self.G.nodes:
            return 0.0
        before_density = self.density()
        before_spec = self.spectral_gap()
        before_cc = nx.number_connected_components(self.G)
        H = self.G.copy()
        H.remove_node(sym)
        n2 = H.number_of_nodes()
        max_e2 = n2 * (n2 - 1) / 2
        after_density = H.number_of_edges() / max_e2 if max_e2 > 0 else 0.0
        try:
            L = nx.normalized_laplacian_matrix(H).astype(float).toarray()
            eigs = np.sort(np.linalg.eigvalsh(L))
            after_spec = float(eigs[1]) if len(eigs) > 1 else 0.0
        except Exception:
            after_spec = 0.0
        after_cc = nx.number_connected_components(H)
        # 正規化: 密度差 + spec差(scaled) + CC増加
        score = abs(before_density - after_density) * 5 \
              + abs(before_spec - after_spec) * 2 \
              + (after_cc - before_cc) * 0.3
        return float(score)

    def all_removal_impacts(self) -> dict[str, float]:
        """全ノードの除去耐性."""
        return {n: self.removal_impact(n) for n in self.G.nodes}

    # ===== 統合 =====

    def invariants(self) -> dict:
        """位相不変量を一括計算（バックテスト・レジーム分類用）."""
        return {
            "label": self.label,
            "timestamp": str(self.timestamp) if self.timestamp is not None else None,
            "n_objects": self.n_objects(),
            "n_morphisms": self.n_morphisms(),
            "density": round(self.density(), 4),
            "avg_clustering": round(self.avg_clustering(), 4),
            "max_k_core": self.max_k_core(),
            "k_core_dist": self.k_core_distribution(),
            "spectral_gap": round(self.spectral_gap(), 4),
            "num_communities": self.num_communities(),
            "num_max_cliques": self.num_max_cliques(),
            "isolated_count": len(self.isolated_nodes()),
            **self.mst_features(),
        }

    def __repr__(self) -> str:
        """インスタンスの文字列表現を返す."""
        d = self.density()
        return f"<MarketCategory n={self.n_objects()} edges={self.n_morphisms()} density={d:.3f} label='{self.label}'>"


# ============ デモ実行 ============
if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).parent
    closes = pd.read_parquet(here / "ohlc_40.parquet")
    # pct_change で個別NaN残すが、dropnaしない
    returns = closes.pct_change()

    # 直近30日窓（NaN含むがpairwise corrで対応）
    window = returns.iloc[-30:]
    cat = MarketCategory.from_returns(
        window,
        threshold=0.3,
        timestamp=window.index[-1],
        label="yfinance_30d",
    )
    print(cat)
    print()
    print("=== Invariants ===")
    inv = cat.invariants()
    for k, v in inv.items():
        print(f"  {k}: {v}")
    print()
    print("=== Top 5 by removal impact ===")
    impacts = cat.all_removal_impacts()
    for sym, score in sorted(impacts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {sym}: {score:.4f}")
    print()
    print("=== Communities ===")
    for i, c in enumerate(cat.communities()):
        print(f"  C{i}: {sorted(c)}")
    print()
    print("=== Isolated ===")
    print(f"  {cat.isolated_nodes()}")
