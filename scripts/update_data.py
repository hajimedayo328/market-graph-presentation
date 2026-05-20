"""
週次データ更新スクリプト.
GitHub Actions or 手動で実行.

1. yfinance で 40 銘柄の 5 年データ取得 → ohlc_40.parquet 更新
2. gamma_timeseries_w30.csv 再計算
3. multi_indicators_w30.csv 再計算 (オプション、時間かかる)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(HERE / "lib"))


SYMBOL_MAP = [
    ("EURUSD", "EURUSD=X", "FX_MAJOR", "FX"),
    ("USDJPY", "USDJPY=X", "FX_MAJOR", "FX"),
    ("GBPUSD", "GBPUSD=X", "FX_MAJOR", "FX"),
    ("AUDUSD", "AUDUSD=X", "FX_MAJOR", "FX"),
    ("USDCHF", "USDCHF=X", "FX_MAJOR", "FX"),
    ("USDCAD", "USDCAD=X", "FX_MAJOR", "FX"),
    ("NZDUSD", "NZDUSD=X", "FX_MAJOR", "FX"),
    ("EURJPY", "EURJPY=X", "FX_CROSS", "FX"),
    ("GBPJPY", "GBPJPY=X", "FX_CROSS", "FX"),
    ("EURGBP", "EURGBP=X", "FX_CROSS", "FX"),
    ("USDTRY", "USDTRY=X", "FX_EM", "FX"),
    ("USDHUF", "USDHUF=X", "FX_EM", "FX"),
    ("EURTRY", "EURTRY=X", "FX_EM", "FX"),
    ("XAUUSD", "GC=F", "COMMODITY", "METAL"),
    ("XAGUSD", "SI=F", "COMMODITY", "METAL"),
    ("BTCUSD", "BTC-USD", "CRYPTO", "CRYPTO"),
    ("ETHUSD", "ETH-USD", "CRYPTO", "CRYPTO"),
    ("USOUSD", "CL=F", "COMMODITY", "ENERGY"),
    ("UKOUSD", "BZ=F", "COMMODITY", "ENERGY"),
    ("SP500", "^GSPC", "INDEX_US", "INDEX"),
    ("NAS100", "^IXIC", "INDEX_US", "INDEX"),
    ("DJ30", "^DJI", "INDEX_US", "INDEX"),
    ("RUS2000", "^RUT", "INDEX_US", "INDEX"),
    ("GER40", "^GDAXI", "INDEX_EU", "INDEX"),
    ("UK100", "^FTSE", "INDEX_EU", "INDEX"),
    ("FRA40", "^FCHI", "INDEX_EU", "INDEX"),
    ("JP225", "^N225", "INDEX_AS", "INDEX"),
    ("CHINA50", "FXI", "INDEX_AS", "INDEX"),
    ("DXY", "DX=F", "SPECIAL", "SPECIAL"),
    ("VIX", "^VIX", "SPECIAL", "SPECIAL"),
    ("AAPL", "AAPL", "STOCK", "STOCK"),
    ("MSFT", "MSFT", "STOCK", "STOCK"),
    ("GOOG", "GOOG", "STOCK", "STOCK"),
    ("META", "META", "STOCK", "STOCK"),
    ("TSLA", "TSLA", "STOCK", "STOCK"),
    ("US10Y", "^TNX", "BOND", "BOND"),
    ("EUB10Y", "TLT", "BOND", "BOND"),
    ("UKGILT", "IGLT.L", "BOND", "BOND"),
    ("COPPER", "HG=F", "COMMODITY", "COMMODITY"),
    ("NGAS", "NG=F", "COMMODITY", "COMMODITY"),
]


def fetch_data(years: int = 5) -> pd.DataFrame:
    tickers = [m[1] for m in SYMBOL_MAP]
    print(f"Downloading {len(tickers)} tickers, period={years}y...")
    raw = yf.download(tickers, period=f"{years}y", interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    closes = pd.DataFrame()
    for internal, yt, _sec, _ac in SYMBOL_MAP:
        try:
            if (yt, "Close") in raw.columns:
                closes[internal] = raw[(yt, "Close")]
            elif yt in raw.columns and "Close" in raw[yt].columns:
                closes[internal] = raw[yt]["Close"]
        except Exception as e:
            print(f"  WARN {internal} ({yt}): {e}")
    return closes


def compute_gamma_timeseries(closes: pd.DataFrame, window: int = 30,
                              threshold: float = 0.3) -> pd.DataFrame:
    """gamma_timeseries (L¹ + n_unb 日次) 計算."""
    from persistent_homology import persistence_diagram, persistence_summary
    from market_category import MarketCategory
    from homology import signed_cycle_balance

    returns = closes.pct_change()
    n = len(returns)
    rows = []
    t0 = time.time()
    print(f"Computing gamma timeseries for {n - window} days (window={window})...")
    for i, t_idx in enumerate(range(window, n)):
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
        date = returns.index[t_idx - 1]
        if win_clean.shape[1] < 5:
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan,
                         "n_edges": 0, "balance_rate": np.nan})
            continue
        try:
            corr = win_clean.corr()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diag = persistence_diagram(corr, max_dim=1)
            summ = persistence_summary(diag)
            L1 = float(summ["L1_norm_H1"])
            cat = MarketCategory(symbols=list(win_clean.columns),
                                 corr_matrix=corr, threshold=threshold)
            cat._build_graph()
            bal = signed_cycle_balance(cat.G)
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": round(L1, 6),
                         "n_unb": int(bal["n_unbalanced"]),
                         "n_edges": cat.G.number_of_edges(),
                         "balance_rate": round(float(bal["balance_rate"]), 4)})
        except Exception as e:
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan,
                         "n_edges": 0, "balance_rate": np.nan})
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{n-window}] {date.date()}  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    print(f"Done in {time.time()-t0:.0f}s.")
    return df


def main():
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"=== Weekly data update: {years} years ===")

    # 1. yfinance 取得
    closes = fetch_data(years)
    print(f"Shape: {closes.shape}, "
          f"{closes.index.min().date()} -> {closes.index.max().date()}")
    closes.to_parquet(DATA_DIR / "ohlc_40.parquet")
    print(f"Saved: {DATA_DIR / 'ohlc_40.parquet'}")

    # symbol_meta も再保存 (念のため)
    meta = pd.DataFrame(SYMBOL_MAP, columns=["internal", "yticker", "sector", "asset_class"])
    meta.to_csv(DATA_DIR / "symbol_meta.csv", index=False)

    # 2. gamma timeseries
    df = compute_gamma_timeseries(closes)
    df.to_csv(DATA_DIR / "gamma_timeseries_w30.csv", index=False)
    print(f"Saved: {DATA_DIR / 'gamma_timeseries_w30.csv'}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
