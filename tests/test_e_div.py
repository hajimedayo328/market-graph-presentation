"""
e_div = z_unb - z_L1 の計算ロジックのスモークテスト.

e_div は build_index.py でインラインに定義されている:
    gamma["z_L1"]  = (L1_H1 - mean) / std
    gamma["z_unb"] = (n_unb - mean) / std
    gamma["e_div"] = gamma["z_unb"] - gamma["z_L1"]

この計算を単体で検証する。
"""
import numpy as np
import pandas as pd
import pytest


def compute_e_div(L1_series: pd.Series, n_unb_series: pd.Series) -> pd.Series:
    """build_index.py と同じロジックで e_div を計算する."""
    z_L1 = (L1_series - L1_series.mean()) / L1_series.std()
    z_unb = (n_unb_series - n_unb_series.mean()) / n_unb_series.std()
    return z_unb - z_L1


# ============================================================
#  e_div テスト
# ============================================================

def test_e_div_zero_when_same_z():
    """
    L1 と n_unb が完全に相関していて同じ z スコアになる場合、e_div = 0.
    """
    # 同じ値を使うと z_L1 == z_unb → e_div = 0
    vals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    e_div = compute_e_div(vals, vals)
    np.testing.assert_allclose(e_div.values, 0.0, atol=1e-10)


def test_e_div_positive_when_unb_larger():
    """
    n_unb が相対的に高く L1 が低い局面では e_div > 0 になる.
    """
    L1 = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])   # 全部同じ → std=0 になるので調整
    L1 = pd.Series([1.0, 1.5, 1.0, 1.5, 1.0])
    n_unb = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    e_div = compute_e_div(L1, n_unb)
    # L1 が小さく n_unb が右肩上がりなら最後の e_div は正
    assert e_div.iloc[-1] > 0


def test_e_div_shape_matches_input():
    """戻り値の長さが入力と同じ."""
    n = 20
    rng = np.random.default_rng(0)
    L1 = pd.Series(rng.uniform(0.1, 2.0, n))
    n_unb = pd.Series(rng.integers(0, 50, n).astype(float))
    e_div = compute_e_div(L1, n_unb)
    assert len(e_div) == n


def test_e_div_mean_near_zero():
    """z スコアの差の平均は 0 に近い (standardize の性質から)."""
    rng = np.random.default_rng(1)
    L1 = pd.Series(rng.uniform(0.1, 2.0, 100))
    n_unb = pd.Series(rng.uniform(0, 50, 100))
    e_div = compute_e_div(L1, n_unb)
    # z_unb の平均 = 0、z_L1 の平均 = 0 → e_div の平均 ≈ 0
    assert abs(e_div.mean()) < 1e-10
