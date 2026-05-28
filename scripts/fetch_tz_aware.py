"""
tz-aware daily fetch (close-time-aligned to 20:00 UTC).

Section 11.2 (4) の future work「tz-aware resampling」を実装する。

問題:
  yfinance.download(group_by="ticker", interval="1d") は、各銘柄の bar timestamp を
  「native timezone の midnight (bar start)」で返し、最終的に tz-naive UTC midnight
  に揃えてしまう。実際の close 時刻はマーケットごとに異なる:

    - US 株/指数 (^GSPC, AAPL, ...): NY 16:00 = 20:00 UTC (EDT) / 21:00 UTC (EST)
    - US 先物 (GC=F, CL=F, HG=F):    ほぼ 24h (CME) だが日足は NY タイムで切られる
    - EU 指数 (^GDAXI, ^FTSE, ...):  Frankfurt/London 17:30 = ~16:30 UTC
    - Asia 指数 (^N225):              Tokyo 15:00 = 06:00 UTC
    - FX (=X):                         London midnight 切替 = 23:00 UTC (BST) / 0:00 UTC
    - Crypto (BTC-USD, ETH-USD):       UTC midnight (24h)

  つまり同じ "2025-04-02 daily bar" でも、各 close が実時刻軸上では 14h 以上ばらつく。
  Liberation Day のような同日午後イベントでは、US 株は当日 close に反映するが、
  Asia 指数は「翌日 open まで反映されない」 → tz-naive 並べでは最大 30h ずれが残る。

解法:
  各銘柄を yfinance で 個別 ticker.history(period='5y', interval='1d') で取得。
  これだと index は tz-aware に保たれる。
  各 bar を「実際の close 時刻 UTC」に shift し、共通の 20:00 UTC 日次グリッドに
  asof-resample (各日 20:00 UTC 時点で最後に観測した close を使う)。

  これにより:
    - 同じ日付の値は「20:00 UTC 時点のスナップショット」になり tz 不一致が解消
    - Asia の 4/2 close は (06:00 UTC < 20:00 UTC) なので 4/2 に乗る
    - US の 4/2 close は (20:00 UTC ≈ NY 16:00) なので 4/2 にギリギリ間に合う
    - FX の 4/2 23:00 UTC close は 4/3 のスロットに反映 (前日扱い)

出力:
  data/ohlc_40_tz_aware.parquet  : tz-naive UTC 日次グリッドの close 行列 (40 cols)
  data/fetch_tz_aware_log.json   : 各銘柄の close 時刻情報、欠損、skip 理由

実行:
  python scripts/fetch_tz_aware.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
sys.path.insert(0, str(HERE / "lib"))

from fetch_40symbols import SYMBOL_MAP  # noqa: E402


# ===== 各銘柄の実 close 時刻 (UTC) を返すための shift マップ =====
# bar_start_in_native_tz から close までの 「時間差 (hours)」 を定義する。
# yfinance daily bar は native tz の 00:00 を timestamp として持つ。
# その timestamp を UTC に変換した後、ここで定義した shift を足すと
# 「だいたいの実 close 時刻 UTC」になる。
#
# DST は無視 (年に数時間しか影響しない、日次粒度で十分)。
CLOSE_SHIFT_HOURS = {
    # US 株 / 指数 / 先物: bar 00:00 NY → close 16:00 NY (= 20:00-21:00 UTC)
    "^GSPC": 16, "^IXIC": 16, "^DJI": 16, "^RUT": 16,
    "AAPL": 16, "MSFT": 16, "GOOG": 16, "META": 16, "TSLA": 16,
    "^TNX": 16, "^VIX": 16,
    # 先物 (CME): やや長い取引時間だが daily bar は NY タイム
    "GC=F": 17, "SI=F": 17, "CL=F": 17, "BZ=F": 17,
    "HG=F": 17, "NG=F": 17, "DX=F": 17,
    # EU 指数: Frankfurt 17:30 / London 16:30
    "^GDAXI": 17, "^FTSE": 16, "^FCHI": 17,
    # Asia 指数: Tokyo 15:00 JST / Hong Kong 16:00 HKT
    "^N225": 15, "FXI": 16,  # FXI は US listed なので NY 16:00
    # FX (London midnight 切替): bar start 00:00 London → close 23:59 London
    "EURUSD=X": 23, "USDJPY=X": 23, "GBPUSD=X": 23, "AUDUSD=X": 23,
    "USDCHF=X": 23, "USDCAD=X": 23, "NZDUSD=X": 23,
    "EURJPY=X": 23, "GBPJPY=X": 23, "EURGBP=X": 23,
    "USDTRY=X": 23, "USDHUF=X": 23, "EURTRY=X": 23,
    # Crypto: 24h, bar 00:00 UTC → 「その日 24:00 UTC = 翌日 00:00 UTC」を close と考える
    "BTC-USD": 24, "ETH-USD": 24,
    # ETF
    "TLT": 16, "IGLT.L": 16,  # IGLT は London listed
}


# ===== 共通 grid: 20:00 UTC ごと =====
DAILY_ANCHOR_HOUR_UTC = 20


def fetch_one(yticker: str, years: int = 5) -> Optional[pd.Series]:
    """1 銘柄を yfinance.Ticker(yt).history で取得し close Series (tz-aware UTC) を返す.

    Returns:
        close 時刻 (UTC) を index とする ``pd.Series``。失敗時は ``None``。
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist = yf.Ticker(yticker).history(
                period=f"{years}y", interval="1d", auto_adjust=True
            )
    except Exception as e:
        print(f"  ! {yticker}: history() raised {type(e).__name__}: {e}")
        return None
    if hist is None or len(hist) == 0 or "Close" not in hist.columns:
        return None
    close = hist["Close"].dropna()
    if len(close) == 0:
        return None
    idx = close.index
    # tz が無い場合は UTC とみなす
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    # bar timestamp を 「概ねの実 close 時刻 UTC」 に shift
    shift_h = CLOSE_SHIFT_HOURS.get(yticker, 16)  # default: NY 16:00
    idx_utc = idx.tz_convert("UTC") + pd.Timedelta(hours=shift_h)
    close.index = idx_utc
    close = close[~close.index.duplicated(keep="last")]
    close = close.sort_index()
    return close


def resample_asof_20utc(series: pd.Series,
                        grid: pd.DatetimeIndex) -> pd.Series:
    """20:00 UTC daily grid に asof (前方探索) で並べ直す.

    各 grid 時刻 ``t`` に対し、``series.index <= t`` のうち最も新しい値を取る。
    pandas.merge_asof で直接実装する (ベクトル化)。
    """
    df = pd.DataFrame({"ts": series.index, "close": series.values})
    df = df.sort_values("ts").reset_index(drop=True)
    target = pd.DataFrame({"ts": grid}).sort_values("ts").reset_index(drop=True)
    merged = pd.merge_asof(target, df, on="ts", direction="backward")
    out = pd.Series(merged["close"].values, index=grid)
    return out


def main() -> None:
    """40 銘柄を個別 yfinance fetch して tz-aware に揃え、parquet に出力する."""
    years = 5
    print(f"=== tz-aware daily fetch (period={years}y, anchor=20:00 UTC) ===")
    t0 = time.time()

    closes_by_symbol: dict[str, pd.Series] = {}
    log: dict[str, dict] = {}

    for internal, yt, sector, asset_class in SYMBOL_MAP:
        print(f"  [{internal:8s} <- {yt}] ...", end=" ")
        s = fetch_one(yt, years=years)
        if s is None or len(s) < 60:  # 60 営業日くらい無いと使い物にならない
            print(f"SKIP (n={0 if s is None else len(s)})")
            log[internal] = {
                "yticker": yt,
                "fetched": False,
                "skip_reason": "empty or <60 rows",
                "n_rows": 0 if s is None else int(len(s)),
            }
            continue
        closes_by_symbol[internal] = s
        log[internal] = {
            "yticker": yt,
            "fetched": True,
            "n_rows": int(len(s)),
            "close_time_first_utc": str(s.index[0]),
            "close_time_last_utc": str(s.index[-1]),
            "shift_hours_from_bar_start": CLOSE_SHIFT_HOURS.get(yt, 16),
        }
        print(f"ok (n={len(s)})")

    if not closes_by_symbol:
        raise RuntimeError("no symbols fetched, abort.")

    # 共通の 20:00 UTC grid を作る (全銘柄の min/max カバー)
    g_start = min(s.index.min() for s in closes_by_symbol.values())
    g_end = max(s.index.max() for s in closes_by_symbol.values())
    g_start_d = pd.Timestamp(g_start.date(), tz="UTC") + pd.Timedelta(hours=DAILY_ANCHOR_HOUR_UTC)
    g_end_d = pd.Timestamp(g_end.date(), tz="UTC") + pd.Timedelta(hours=DAILY_ANCHOR_HOUR_UTC)
    grid = pd.date_range(start=g_start_d, end=g_end_d, freq="1D", tz="UTC")
    print(f"\nGrid: {grid[0]} → {grid[-1]} ({len(grid)} days)")

    # 各銘柄を asof で grid に並べる
    closes_aligned = pd.DataFrame(index=grid)
    for internal, s in closes_by_symbol.items():
        closes_aligned[internal] = resample_asof_20utc(s, grid)

    # tz を落として naive UTC index にする (既存 parquet と互換)
    closes_aligned.index = closes_aligned.index.tz_localize(None)
    # date のみ保持 (20:00 UTC を midnight に再 anchor)
    closes_aligned.index = pd.to_datetime(closes_aligned.index.normalize())
    # 重複日 (asof で同 date が複数になるケース) を最後勝ち
    closes_aligned = closes_aligned[~closes_aligned.index.duplicated(keep="last")]

    # weekend (土日) は flat になるので残しておく (既存 ohlc_40 と同じ振る舞い)
    # …ただし FX/crypto 以外は土日 close が更新されない。
    # 既存 ohlc_40.parquet も全カレンダー日を持っているわけではないので、
    # 営業日 (Mon-Fri) のみに絞る。
    closes_aligned = closes_aligned[closes_aligned.index.dayofweek < 5]

    print(f"\nFinal shape: {closes_aligned.shape}")
    print(f"Range: {closes_aligned.index.min()} → {closes_aligned.index.max()}")
    print(f"Fetched symbols: {len(closes_aligned.columns)}/{len(SYMBOL_MAP)}")
    miss = [m[0] for m in SYMBOL_MAP if m[0] not in closes_aligned.columns]
    if miss:
        print(f"Missing: {miss}")

    out_path = DATA_DIR / "ohlc_40_tz_aware.parquet"
    closes_aligned.to_parquet(out_path)
    print(f"Saved: {out_path}")

    log_path = DATA_DIR / "fetch_tz_aware_log.json"
    log_path.write_text(
        json.dumps(
            {
                "anchor_hour_utc": DAILY_ANCHOR_HOUR_UTC,
                "years": years,
                "n_symbols_requested": len(SYMBOL_MAP),
                "n_symbols_fetched": len(closes_by_symbol),
                "n_symbols_in_parquet": int(closes_aligned.shape[1]),
                "n_rows": int(closes_aligned.shape[0]),
                "date_range": [str(closes_aligned.index.min()),
                               str(closes_aligned.index.max())],
                "missing": miss,
                "by_symbol": log,
                "elapsed_sec": round(time.time() - t0, 1),
            },
            indent=2, ensure_ascii=False, default=str
        ),
        encoding="utf-8",
    )
    print(f"Saved log: {log_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
