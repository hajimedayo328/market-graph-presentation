"""
market_category.py の MarketCategory._build_graph() のスモークテスト.
"""
import numpy as np
import pandas as pd
import pytest

from market_category import MarketCategory


def _make_corr(symbols: list, values: dict) -> pd.DataFrame:
    """
    symbols リストと {(i, j): corr_value} の dict から相関行列を組み立てる.
    対角は 1.0、未指定ペアは 0.0 とする。
    """
    n = len(symbols)
    matrix = np.eye(n)
    for (i, j), v in values.items():
        matrix[i, j] = v
        matrix[j, i] = v
    return pd.DataFrame(matrix, index=symbols, columns=symbols)


# ============================================================
#  _build_graph
# ============================================================

def test_build_graph_basic():
    """3 銘柄、相関 0.5 (threshold 超) と 0.1 (threshold 未満): エッジ数 1."""
    symbols = ["A", "B", "C"]
    # A-B: 0.5 (超), A-C: 0.1 (未満), B-C: 0.1 (未満)
    corr = _make_corr(symbols, {(0, 1): 0.5, (0, 2): 0.1, (1, 2): 0.1})
    cat = MarketCategory(symbols=symbols, corr_matrix=corr, threshold=0.3)
    cat._build_graph()
    assert cat.G.number_of_edges() == 1
    assert cat.G.has_edge("A", "B")


def test_build_graph_empty():
    """全相関が threshold 未満: エッジ 0."""
    symbols = ["X", "Y", "Z"]
    corr = _make_corr(symbols, {(0, 1): 0.1, (0, 2): 0.2, (1, 2): 0.05})
    cat = MarketCategory(symbols=symbols, corr_matrix=corr, threshold=0.3)
    cat._build_graph()
    assert cat.G.number_of_edges() == 0


def test_signed_edges_present():
    """正相関エッジに sign=1、負相関エッジに sign=-1 が付いているか."""
    symbols = ["A", "B", "C"]
    # A-B: +0.6, A-C: -0.5
    corr = _make_corr(symbols, {(0, 1): 0.6, (0, 2): -0.5, (1, 2): 0.0})
    cat = MarketCategory(symbols=symbols, corr_matrix=corr, threshold=0.3)
    cat._build_graph()

    # A-B エッジ: sign = +1
    assert cat.G.has_edge("A", "B")
    assert cat.G["A"]["B"]["sign"] == 1

    # A-C エッジ: sign = -1
    assert cat.G.has_edge("A", "C")
    assert cat.G["A"]["C"]["sign"] == -1


def test_build_graph_all_above_threshold():
    """全相関が threshold 以上の場合、エッジ数が n*(n-1)/2 になる."""
    symbols = ["P", "Q", "R"]
    corr = _make_corr(symbols, {(0, 1): 0.8, (0, 2): 0.7, (1, 2): 0.6})
    cat = MarketCategory(symbols=symbols, corr_matrix=corr, threshold=0.3)
    cat._build_graph()
    n = len(symbols)
    assert cat.G.number_of_edges() == n * (n - 1) // 2
