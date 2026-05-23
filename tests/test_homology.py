"""
homology.py の betti_numbers / signed_cycle_balance のスモークテスト.
"""
import networkx as nx
import pytest

from homology import betti_numbers, signed_cycle_balance


# ============================================================
#  betti_numbers
# ============================================================

def test_betti_numbers_triangle():
    """三角形グラフ: H_0=1 (連結), H_1=1 (閉路 1 本)."""
    G = nx.cycle_graph(3)
    bn = betti_numbers(G)
    assert bn["H_0"] == 1
    assert bn["H_1"] == 1


def test_betti_numbers_tree():
    """木 (path 4 ノード): H_0=1, H_1=0 (閉路なし)."""
    G = nx.path_graph(4)
    bn = betti_numbers(G)
    assert bn["H_0"] == 1
    assert bn["H_1"] == 0


def test_betti_numbers_disconnected():
    """2 つの孤立ノード: H_0=2."""
    G = nx.Graph()
    G.add_nodes_from(["A", "B"])
    bn = betti_numbers(G)
    assert bn["H_0"] == 2
    assert bn["H_1"] == 0


def test_betti_numbers_empty_graph():
    """ノードなしグラフ: H_0=0, H_1=0."""
    G = nx.Graph()
    bn = betti_numbers(G)
    assert bn["H_0"] == 0
    assert bn["H_1"] == 0


# ============================================================
#  signed_cycle_balance
# ============================================================

def _triangle_with_signs(s_ab, s_bc, s_ca):
    """符号付き三角形グラフを作るヘルパー."""
    G = nx.Graph()
    G.add_edge("A", "B", sign=s_ab, weight=1)
    G.add_edge("B", "C", sign=s_bc, weight=1)
    G.add_edge("C", "A", sign=s_ca, weight=1)
    return G


def test_signed_cycle_balance_all_positive():
    """全エッジ正符号の三角形: balance_rate=1.0, n_unbalanced=0."""
    G = _triangle_with_signs(1, 1, 1)
    result = signed_cycle_balance(G)
    assert result["balance_rate"] == pytest.approx(1.0)
    assert result["n_unbalanced"] == 0


def test_signed_cycle_balance_one_negative_in_triangle():
    """三角形に負エッジが 1 本: 符号積 = +1*+1*-1 = -1 → unbalanced."""
    G = _triangle_with_signs(1, 1, -1)
    result = signed_cycle_balance(G)
    assert result["n_unbalanced"] == 1
    assert result["n_balanced"] == 0


def test_signed_cycle_balance_two_negatives_in_triangle():
    """三角形に負エッジ 2 本: 符号積 = +1*-1*-1 = +1 → balanced (敵の敵は味方)."""
    G = _triangle_with_signs(1, -1, -1)
    result = signed_cycle_balance(G)
    assert result["n_balanced"] == 1
    assert result["n_unbalanced"] == 0


def test_signed_cycle_balance_no_edges():
    """エッジなし → サイクル 0、balance_rate=1.0 (デフォルト)."""
    G = nx.Graph()
    G.add_nodes_from(["X", "Y"])
    result = signed_cycle_balance(G)
    assert result["n_cycles_in_basis"] == 0
    assert result["balance_rate"] == pytest.approx(1.0)
