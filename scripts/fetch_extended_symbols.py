"""追加 20 銘柄の 5 年 OHLC 取得 (yfinance).

既存 40 銘柄 (ohlc_40.parquet, 2021-05-17 〜 2026-05-18) と同期間で fetch し、
ohlc_60_extended.parquet として全 60 銘柄をマージ保存する。

- 個別株拡張 (10): NVDA, AMZN, JPM, WMT, V, JNJ, KO, XOM, BA, MA
- 暗号拡張 (5): SOL-USD, BNB-USD, XRP-USD, DOGE-USD, ADA-USD
- 地域指数拡張 (5): KS11 (^KS11 KOSPI), BSESN (^BSESN SENSEX), BVSP (^BVSP),
                  STI (^STI), HSI (^HSI)
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"

# (internal_name, yfinance_ticker, sector, asset_class)
EXTENDED_SYMBOLS = [
    # 個別株拡張 (10)
    ("NVDA",   "NVDA",   "STOCK", "STOCK"),
    ("AMZN",   "AMZN",   "STOCK", "STOCK"),
    ("JPM",    "JPM",    "STOCK", "STOCK"),
    ("WMT",    "WMT",    "STOCK", "STOCK"),
    ("V",      "V",      "STOCK", "STOCK"),
    ("JNJ",    "JNJ",    "STOCK", "STOCK"),
    ("KO",     "KO",     "STOCK", "STOCK"),
    ("XOM",    "XOM",    "STOCK", "STOCK"),
    ("BA",     "BA",     "STOCK", "STOCK"),
    ("MA",     "MA",     "STOCK", "STOCK"),
    # 暗号拡張 (5)
    ("SOLUSD", "SOL-USD",  "CRYPTO", "CRYPTO"),
    ("BNBUSD", "BNB-USD",  "CRYPTO", "CRYPTO"),
    ("XRPUSD", "XRP-USD",  "CRYPTO", "CRYPTO"),
    ("DOGEUSD","DOGE-USD", "CRYPTO", "CRYPTO"),
    ("ADAUSD", "ADA-USD",  "CRYPTO", "CRYPTO"),
    # 地域指数拡張 (5)
    ("KS11",   "^KS11",   "INDEX_AS", "INDEX"),  # KOSPI 韓国
    ("BSESN",  "^BSESN",  "INDEX_AS", "INDEX"),  # SENSEX インド
    ("BVSP",   "^BVSP",   "INDEX_AM", "INDEX"),  # BOVESPA ブラジル
    ("STI",    "^STI",    "INDEX_AS", "INDEX"),  # シンガポール STI
    ("HSI",    "^HSI",    "INDEX_AS", "INDEX"),  # 香港 HSI
]


def fetch_one(ticker: str, start, end) -> pd.Series | None:
    """単一銘柄を fetch。失敗時は None を返す。"""
    try:
        raw = yf.download(ticker, start=start, end=end, interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            # ticker レベルを除去
            if "Close" in raw.columns.get_level_values(0):
                ser = raw["Close"].iloc[:, 0]
            else:
                return None
        else:
            if "Close" not in raw.columns:
                return None
            ser = raw["Close"]
        ser = ser.dropna()
        if len(ser) < 50:
            return None
        return ser
    except Exception as e:
        print(f"  WARN {ticker}: {e}")
        return None


def main():
    print("=" * 60)
    print("追加 20 銘柄 fetch (yfinance)")
    print("=" * 60)

    # 既存データの期間に揃える
    base = pd.read_parquet(DATA / "ohlc_40.parquet")
    start = base.index.min()
    end = base.index.max() + pd.Timedelta(days=1)
    print(f"期間: {start.date()} ~ {end.date()} (既存 40 銘柄に同期)")
    print()

    fetched = {}
    skipped = []
    for internal, yt, _sec, _ac in EXTENDED_SYMBOLS:
        print(f"  fetch {internal:<10} ({yt}) ...", end=" ", flush=True)
        ser = fetch_one(yt, start, end)
        if ser is None:
            print("SKIPPED")
            skipped.append(internal)
            continue
        ser.name = internal
        # 既存 index に揃える (営業日不一致対策)
        ser = ser.reindex(base.index)
        n_valid = int(ser.notna().sum())
        print(f"OK ({n_valid} pts)")
        if n_valid < 200:
            print(f"    -> 有効データ不足 ({n_valid} < 200), SKIP")
            skipped.append(internal)
            continue
        fetched[internal] = ser

    print()
    print(f"取得: {len(fetched)}/{len(EXTENDED_SYMBOLS)}  skipped={skipped}")

    if not fetched:
        print("有効な追加銘柄なし、終了")
        sys.exit(1)

    extra_df = pd.DataFrame(fetched)
    merged = base.join(extra_df, how="left")
    print(f"\n合計 shape: {merged.shape}")
    print(f"列: {merged.columns.tolist()}")

    out = DATA / "ohlc_60_extended.parquet"
    merged.to_parquet(out)
    print(f"\nSaved: {out}")

    # meta 拡張版
    meta = pd.read_csv(DATA / "symbol_meta.csv")
    extra_rows = [
        {"internal": s[0], "yticker": s[1], "sector": s[2], "asset_class": s[3]}
        for s in EXTENDED_SYMBOLS if s[0] in fetched
    ]
    meta_ext = pd.concat([meta, pd.DataFrame(extra_rows)], ignore_index=True)
    meta_ext.to_csv(DATA / "symbol_meta_extended.csv", index=False)
    print(f"Saved: symbol_meta_extended.csv ({len(meta_ext)} rows)")

    # skip 情報を JSON
    import json
    info = {
        "fetched": list(fetched.keys()),
        "skipped": skipped,
        "n_total": int(merged.shape[1]),
        "period_start": str(start.date()),
        "period_end": str(end.date()),
    }
    (DATA / "fetch_extended_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: fetch_extended_info.json")


if __name__ == "__main__":
    main()
