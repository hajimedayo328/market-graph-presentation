"""
persistent_homology.py の persistence_diagram / persistence_summary のスモークテスト.
"""
import numpy as np
import pandas as pd
import pytest

from persistent_homology import persistence_diagram, persistence_summary


def _random_corr_matrix(n: int, seed: int = 42) -> pd.DataFrame:
    """再現性のある小さい相関行列を生成するヘルパー."""
    rng = np.random.default_rng(seed)
    # ランダム正規分布データから相関行列を作る
    data = rng.standard_normal((50, n))
    df = pd.DataFrame(data, columns=[f"S{i}" for i in range(n)])
    return df.corr()


# ============================================================
#  persistence_diagram
# ============================================================

def test_persistence_diagram_small():
    """5x5 の相関行列で persistence_diagram がエラーなく動く."""
    corr = _random_corr_matrix(5)
    diag = persistence_diagram(corr)
    # 戻り値は dict で H0, H1, max_distance を持つ
    assert isinstance(diag, dict)
    assert "H0" in diag
    assert "H1" in diag
    assert "max_distance" in diag


def test_persistence_diagram_h0_nonempty():
    """H_0 特徴量は少なくとも 1 個以上存在する (全連結成分に対応)."""
    corr = _random_corr_matrix(5)
    diag = persistence_diagram(corr)
    assert len(diag["H0"]) >= 1


def test_persistence_diagram_birth_death_order():
    """全ての (birth, death) ペアで birth <= death が成立する."""
    corr = _random_corr_matrix(6)
    diag = persistence_diagram(corr)
    for b, d in diag["H0"] + diag["H1"]:
        assert b <= d, f"birth={b} > death={d} となっている"


# ============================================================
#  persistence_summary
# ============================================================

def test_persistence_summary_keys():
    """persistence_summary の戻り値に L1_norm_H1 キーが存在する."""
    corr = _random_corr_matrix(5)
    diag = persistence_diagram(corr)
    summary = persistence_summary(diag)
    assert "L1_norm_H1" in summary


def test_persistence_l1_nonneg():
    """L1_norm_H1 >= 0 であること."""
    corr = _random_corr_matrix(5)
    diag = persistence_diagram(corr)
    summary = persistence_summary(diag)
    assert summary["L1_norm_H1"] >= 0.0


def test_persistence_summary_all_keys():
    """必須キーがすべて揃っているか確認."""
    corr = _random_corr_matrix(5)
    diag = persistence_diagram(corr)
    summary = persistence_summary(diag)
    required_keys = [
        "n_H0", "n_H1",
        "max_H0_persistence", "max_H1_persistence",
        "avg_H1_persistence", "L1_norm_H1", "L2_norm_H1",
    ]
    for key in required_keys:
        assert key in summary, f"キー '{key}' が persistence_summary に存在しない"
