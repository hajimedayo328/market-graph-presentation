"""
40銘柄のyfinance過去データ取得（圏論的金融ネットワーク用）.

tick recorder の40銘柄を yfinance で取得できる ticker にマッピング.
スポット系がない指数・先物は ETF/futures で代替.

実行:
  python fetch_40symbols.py [--years 5] [--out ohlc_40.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

# ============ 銘柄マッピング ============
# (内部名, yfinance_ticker, sector, asset_class)
SYMBOL_MAP = [
    # FX Major (yfinance: "<pair>=X")
    ("EURUSD",  "EURUSD=X", "FX_MAJOR",  "FX"),
    ("USDJPY",  "USDJPY=X", "FX_MAJOR",  "FX"),
    ("GBPUSD",  "GBPUSD=X", "FX_MAJOR",  "FX"),
    ("AUDUSD",  "AUDUSD=X", "FX_MAJOR",  "FX"),
    ("USDCHF",  "USDCHF=X", "FX_MAJOR",  "FX"),
    ("USDCAD",  "USDCAD=X", "FX_MAJOR",  "FX"),
    ("NZDUSD",  "NZDUSD=X", "FX_MAJOR",  "FX"),

    # FX Cross
    ("EURJPY",  "EURJPY=X", "FX_CROSS",  "FX"),
    ("GBPJPY",  "GBPJPY=X", "FX_CROSS",  "FX"),
    ("EURGBP",  "EURGBP=X", "FX_CROSS",  "FX"),

    # FX EM
    ("USDTRY",  "USDTRY=X", "FX_EM",     "FX"),
    ("USDHUF",  "USDHUF=X", "FX_EM",     "FX"),
    ("EURTRY",  "EURTRY=X", "FX_EM",     "FX"),

    # Metal (futures or ETF)
    ("XAUUSD",  "GC=F",     "COMMODITY", "METAL"),  # Gold futures
    ("XAGUSD",  "SI=F",     "COMMODITY", "METAL"),  # Silver futures

    # Crypto
    ("BTCUSD",  "BTC-USD",  "CRYPTO",    "CRYPTO"),
    ("ETHUSD",  "ETH-USD",  "CRYPTO",    "CRYPTO"),

    # Energy
    ("USOUSD",  "CL=F",     "COMMODITY", "ENERGY"),  # WTI Crude futures
    ("UKOUSD",  "BZ=F",     "COMMODITY", "ENERGY"),  # Brent futures

    # Index US
    ("SP500",   "^GSPC",    "INDEX_US",  "INDEX"),
    ("NAS100",  "^IXIC",    "INDEX_US",  "INDEX"),
    ("DJ30",    "^DJI",     "INDEX_US",  "INDEX"),
    ("RUS2000", "^RUT",     "INDEX_US",  "INDEX"),

    # Index EU/Asia
    ("GER40",   "^GDAXI",   "INDEX_EU",  "INDEX"),
    ("UK100",   "^FTSE",    "INDEX_EU",  "INDEX"),
    ("FRA40",   "^FCHI",    "INDEX_EU",  "INDEX"),
    ("JP225",   "^N225",    "INDEX_AS",  "INDEX"),
    ("CHINA50", "FXI",      "INDEX_AS",  "INDEX"),  # ETF代替

    # Special
    ("DXY",     "DX=F",     "SPECIAL",   "SPECIAL"),  # Dollar index
    ("VIX",     "^VIX",     "SPECIAL",   "SPECIAL"),

    # Stocks Mag5
    ("AAPL",    "AAPL",     "STOCK",     "STOCK"),
    ("MSFT",    "MSFT",     "STOCK",     "STOCK"),
    ("GOOG",    "GOOG",     "STOCK",     "STOCK"),
    ("META",    "META",     "STOCK",     "STOCK"),
    ("TSLA",    "TSLA",     "STOCK",     "STOCK"),

    # Rates (10y国債利回りindex / ETFで代替)
    ("US10Y",   "^TNX",     "BOND",      "BOND"),  # 米10年利回り
    ("EUB10Y",  "TLT",      "BOND",      "BOND"),  # 欧Bundは取得困難 → TLTで代替（不正確、要差替）
    ("UKGILT",  "IGLT.L",   "BOND",      "BOND"),  # iShares UK Gilts ETF

    # Commodity
    ("COPPER",  "HG=F",     "COMMODITY", "COMMODITY"),
    ("NGAS",    "NG=F",     "COMMODITY", "COMMODITY"),
]


def fetch(years: int = 5) -> pd.DataFrame:
    """全40銘柄の終値時系列をDataFrameで取得."""
    tickers = [m[1] for m in SYMBOL_MAP]
    print(f"Downloading {len(tickers)} tickers, period={years}y...")
    raw = yf.download(
        tickers,
        period=f"{years}y",
        interval="1d",
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )
    # 終値だけ取り出す: 各ticker の "Close" カラム
    closes = pd.DataFrame()
    for internal, yt, _sec, _ac in SYMBOL_MAP:
        try:
            if (yt, "Close") in raw.columns:
                closes[internal] = raw[(yt, "Close")]
            elif yt in raw.columns:
                # group_byがうまく効かない場合のフォールバック
                if "Close" in raw[yt].columns:
                    closes[internal] = raw[yt]["Close"]
        except Exception as e:
            print(f"  WARN {internal} ({yt}): {e}")
    return closes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--out", type=str, default="ohlc_40.parquet")
    args = ap.parse_args()

    out_path = Path(__file__).parent / args.out
    closes = fetch(args.years)
    print(f"\nShape: {closes.shape}")
    print(f"Date range: {closes.index.min()} ~ {closes.index.max()}")
    print(f"Symbols obtained: {len(closes.columns)}/{len(SYMBOL_MAP)}")
    print(f"Missing: {set([m[0] for m in SYMBOL_MAP]) - set(closes.columns)}")

    closes.to_parquet(out_path)
    print(f"Saved: {out_path}")

    # メタデータ別途保存
    meta = pd.DataFrame(SYMBOL_MAP, columns=["internal", "yticker", "sector", "asset_class"])
    meta_path = out_path.parent / "symbol_meta.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
