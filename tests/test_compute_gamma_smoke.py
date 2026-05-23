"""
compute_gamma_timeseries.main() のスモークテスト.

小さい OHLC parquet (10 銘柄 x 60 日) を tmp_path に作り main() を実行。
出力 CSV の存在と必須列を確認する。

時間がかかるため @pytest.mark.slow で隔離。
デフォルト (pytest.ini の addopts = -m "not slow") では実行されない。
明示的に pytest -m slow で実行する。
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.mark.slow
def test_compute_gamma_smoke(tmp_path):
    """
    ランダム OHLC parquet を作り compute_gamma_timeseries.main() を 1 回回す.
    出力 CSV が生成され、L1_H1 と n_unb 列が存在することを確認する。
    """
    from compute_gamma_timeseries import main

    # ---- テスト用 parquet 作成 (10 銘柄 x 60 日) ----
    n_symbols = 10
    n_days = 60
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    # 各銘柄の終値を 100 スタートのランダムウォークで生成
    prices = 100 + np.cumsum(rng.standard_normal((n_days, n_symbols)), axis=0)
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    df = pd.DataFrame(prices, index=dates, columns=symbols)

    input_parquet = tmp_path / "ohlc_test.parquet"
    df.to_parquet(input_parquet)

    out_prefix = str(tmp_path / "gamma_smoke")

    # ---- main() 実行 ----
    # window=10 で小さく回す
    main(
        window=10,
        threshold=0.3,
        save_every=9999,
        suffix="_test",
        input_file=str(input_parquet),
        out_prefix=out_prefix,
    )

    # ---- 出力 CSV の確認 ----
    out_csv = Path(f"{out_prefix}_test.csv")
    assert out_csv.exists(), f"出力 CSV が見つからない: {out_csv}"

    result = pd.read_csv(out_csv)
    assert "L1_H1" in result.columns, "L1_H1 列が CSV に存在しない"
    assert "n_unb" in result.columns, "n_unb 列が CSV に存在しない"
    assert len(result) > 0, "CSV が空"
