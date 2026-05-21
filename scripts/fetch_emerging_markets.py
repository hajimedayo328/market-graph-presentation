"""新興国主要 40 銘柄の 5 年データ取得 (yfinance)."""
from pathlib import Path
import sys
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"

# Brazil (Bovespa), India (NIFTY), Turkey, Mexico, Indonesia, S.Africa, Argentina, Vietnam
EM_SYMBOLS = [
    # Brazil B3 (Bovespa large caps)
    ("VALE",   "VALE3.SA",  "BR_MAT",  "EM_STOCK"),
    ("PETR4",  "PETR4.SA",  "BR_ENG",  "EM_STOCK"),
    ("ITUB4",  "ITUB4.SA",  "BR_FIN",  "EM_STOCK"),
    ("BBDC4",  "BBDC4.SA",  "BR_FIN",  "EM_STOCK"),
    ("BBAS3",  "BBAS3.SA",  "BR_FIN",  "EM_STOCK"),
    ("ABEV3",  "ABEV3.SA",  "BR_CON",  "EM_STOCK"),
    # India NSE NIFTY large caps
    ("RELIANCE",  "RELIANCE.NS",  "IN_ENG",  "EM_STOCK"),
    ("TCS",       "TCS.NS",       "IN_TECH", "EM_STOCK"),
    ("HDFCBANK",  "HDFCBANK.NS",  "IN_FIN",  "EM_STOCK"),
    ("INFY",      "INFY.NS",      "IN_TECH", "EM_STOCK"),
    ("ICICIBANK", "ICICIBANK.NS", "IN_FIN",  "EM_STOCK"),
    ("HINDUNILVR","HINDUNILVR.NS","IN_CON",  "EM_STOCK"),
    ("BHARTIARTL","BHARTIARTL.NS","IN_TEL",  "EM_STOCK"),
    # China A (HK proxies for liquidity)
    ("TENCENT",   "0700.HK",      "CN_TECH", "EM_STOCK"),
    ("ALIBABA",   "9988.HK",      "CN_TECH", "EM_STOCK"),
    ("BYD",       "1211.HK",      "CN_EV",   "EM_STOCK"),
    ("MEITUAN",   "3690.HK",      "CN_TECH", "EM_STOCK"),
    # Mexico (BMV)
    ("AMXL",      "AMXL.MX",      "MX_TEL",  "EM_STOCK"),
    ("FEMSA",     "FEMSAUBD.MX",  "MX_CON",  "EM_STOCK"),
    ("GMEXICOB",  "GMEXICOB.MX",  "MX_MAT",  "EM_STOCK"),
    # Indonesia (IDX)
    ("BBCA",      "BBCA.JK",      "ID_FIN",  "EM_STOCK"),
    ("TLKM",      "TLKM.JK",      "ID_TEL",  "EM_STOCK"),
    # South Africa (JSE)
    ("NPN",       "NPN.JO",       "ZA_TECH", "EM_STOCK"),
    ("FSR",       "FSR.JO",       "ZA_FIN",  "EM_STOCK"),
    # EM FX
    ("USDBRL",    "BRL=X",        "EM_FX",   "FX"),
    ("USDINR",    "INR=X",        "EM_FX",   "FX"),
    ("USDTRY",    "TRY=X",        "EM_FX",   "FX"),
    ("USDMXN",    "MXN=X",        "EM_FX",   "FX"),
    ("USDZAR",    "ZAR=X",        "EM_FX",   "FX"),
    ("USDIDR",    "IDR=X",        "EM_FX",   "FX"),
    # EM Country ETF (参照)
    ("EWZ",       "EWZ",          "BR_ETF",  "ETF"),
    ("INDA",      "INDA.NYSE",    "IN_ETF",  "ETF"),
    ("EWW",       "EWW",          "MX_ETF",  "ETF"),
    ("EEM",       "EEM",          "EM_BROAD","ETF"),
    # コモディティ参照 (EM 経済に影響大)
    ("OIL",       "CL=F",         "ENERGY",  "COMM"),
    ("GOLD",      "GC=F",         "METAL",   "COMM"),
    ("COPPER",    "HG=F",         "METAL",   "COMM"),
    # DXY (米ドル指数、EM のドミナント要因)
    ("DXY",       "DX=F",         "SPECIAL", "FX"),
    # 暗号通貨参照
    ("BTC",       "BTC-USD",      "CRYPTO",  "CRYPTO"),
    ("ETH",       "ETH-USD",      "CRYPTO",  "CRYPTO"),
    # VIX 参照
    ("VIX",       "^VIX",         "SPECIAL", "SPECIAL"),
]


def fetch(years=5):
    tickers = [m[1] for m in EM_SYMBOLS]
    print(f"Downloading {len(tickers)} EM tickers, period={years}y...")
    raw = yf.download(tickers, period=f"{years}y", interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    closes = pd.DataFrame()
    for internal, yt, _sec, _ac in EM_SYMBOLS:
        try:
            if (yt, "Close") in raw.columns:
                closes[internal] = raw[(yt, "Close")]
            elif yt in raw.columns and "Close" in raw[yt].columns:
                closes[internal] = raw[yt]["Close"]
        except Exception as e:
            print(f"  WARN {internal} ({yt}): {e}")
    return closes


def main():
    closes = fetch(5)
    print(f"\nShape: {closes.shape}")
    print(f"Date range: {closes.index.min().date()} ~ {closes.index.max().date()}")
    print(f"Symbols obtained: {len(closes.columns)}/{len(EM_SYMBOLS)}")
    nn = (~closes.isna()).sum().sort_values()
    print("\nLow-coverage (bottom 5):")
    print(nn.head())
    closes.to_parquet(DATA_DIR / "ohlc_em.parquet")
    meta = pd.DataFrame(EM_SYMBOLS, columns=["internal", "yticker", "sector", "asset_class"])
    meta.to_csv(DATA_DIR / "symbol_meta_em.csv", index=False)
    print(f"\nSaved: ohlc_em.parquet + symbol_meta_em.csv")


if __name__ == "__main__":
    main()
