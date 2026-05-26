"""
VPS daily job:
1. yfinance で 40 銘柄取得 (last 5 years)
2. γ 計算 (L¹, n_unb, e_div) - 当日のみ (last 30-day window)
3. SQLite DB に追記
4. 判定変化を log
5. (Phase 2) Vantage MT5 で自動売買
6. (Phase 3) GitHub Contents API で結果を push

実行: python C:\\tools\\market-graph\\scripts\\vps_daily.py
ログ: C:\\tools\\market-graph\\logs\\vps_daily.log
DB:   C:\\tools\\market-graph\\data\\market_graph.db
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE / "lib"))

from persistent_homology import persistence_diagram, persistence_summary
from market_category import MarketCategory
from homology import signed_cycle_balance

# 40 銘柄 (presentation の SYMBOL_MAP と同じ)
SYMBOL_MAP = [
    ("EURUSD", "EURUSD=X"), ("USDJPY", "USDJPY=X"), ("GBPUSD", "GBPUSD=X"),
    ("AUDUSD", "AUDUSD=X"), ("USDCHF", "USDCHF=X"), ("USDCAD", "USDCAD=X"),
    ("NZDUSD", "NZDUSD=X"), ("EURJPY", "EURJPY=X"), ("GBPJPY", "GBPJPY=X"),
    ("EURGBP", "EURGBP=X"), ("USDTRY", "USDTRY=X"), ("USDHUF", "USDHUF=X"),
    ("EURTRY", "EURTRY=X"), ("XAUUSD", "GC=F"), ("XAGUSD", "SI=F"),
    ("BTCUSD", "BTC-USD"), ("ETHUSD", "ETH-USD"), ("USOUSD", "CL=F"),
    ("UKOUSD", "BZ=F"), ("SP500", "^GSPC"), ("NAS100", "^IXIC"),
    ("DJ30", "^DJI"), ("RUS2000", "^RUT"), ("GER40", "^GDAXI"),
    ("UK100", "^FTSE"), ("FRA40", "^FCHI"), ("JP225", "^N225"),
    ("CHINA50", "FXI"), ("VIX", "^VIX"),  # DX=F retired
    ("AAPL", "AAPL"), ("MSFT", "MSFT"), ("GOOG", "GOOG"),
    ("META", "META"), ("TSLA", "TSLA"), ("US10Y", "^TNX"),
    ("EUB10Y", "TLT"), ("UKGILT", "IGLT.L"), ("COPPER", "HG=F"), ("NGAS", "NG=F"),
]


def log(msg: str) -> None:
    """タイムスタンプ付きメッセージを stdout とログファイルに書く."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_DIR / "vps_daily.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def init_db(conn: object) -> None:
    """SQLite テーブル初期化."""
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS gamma_daily (
      date         TEXT PRIMARY KEY,
      n_symbols    INTEGER,
      L1_H1        REAL,
      n_unb        INTEGER,
      n_edges      INTEGER,
      balance_rate REAL,
      z_L1         REAL,
      z_unb        REAL,
      e_div        REAL,
      classification TEXT,
      computed_at  TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS classification_changes (
      changed_at   TEXT PRIMARY KEY,
      date         TEXT,
      prev_class   TEXT,
      new_class    TEXT,
      e_div        REAL,
      L1_H1        REAL,
      n_unb        INTEGER,
      note         TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS trades (
      trade_id     INTEGER PRIMARY KEY AUTOINCREMENT,
      executed_at  TEXT,
      symbol       TEXT,
      side         TEXT,        -- buy / sell
      volume       REAL,
      open_price   REAL,
      close_price  REAL,
      pnl          REAL,
      classification_trigger TEXT,
      mt5_ticket   INTEGER,
      note         TEXT
    )
    """)
    conn.commit()


def fetch_recent(years: int = 1) -> pd.DataFrame:
    """直近 1 年分のデータ取得 (window=30 計算には 60 日くらいあれば十分だが余裕で 1 年)."""
    tickers = [m[1] for m in SYMBOL_MAP]
    log(f"yfinance: fetching {len(tickers)} tickers, period={years}y...")
    raw = yf.download(tickers, period=f"{years}y", interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    closes = pd.DataFrame()
    for internal, yt in SYMBOL_MAP:
        try:
            if (yt, "Close") in raw.columns:
                closes[internal] = raw[(yt, "Close")]
            elif yt in raw.columns and "Close" in raw[yt].columns:
                closes[internal] = raw[yt]["Close"]
        except Exception:
            pass
    log(f"yfinance: got {closes.shape[0]} days x {closes.shape[1]} symbols")
    return closes


def compute_gamma_for_day(closes: pd.DataFrame, window: int = 30, threshold: float = 0.3) -> dict | None:
    """最新日の γ 計算."""
    returns = closes.pct_change()
    if len(returns) < window + 1:
        log(f"Not enough data: {len(returns)} < {window+1}")
        return None
    # 直近 90 日に絞り、銘柄ごとの全期間 NaN を ffill/bfill で埋める
    returns_recent = returns.tail(window + 60).ffill().bfill()
    win = returns_recent.iloc[-window:]
    # 80% 以上値がある銘柄のみ採用
    n_valid = (~win.isna()).sum()
    keep_cols = n_valid[n_valid >= int(0.8 * window)].index.tolist()
    win_clean = win[keep_cols].dropna(axis=1, how="all")
    # 残った NaN は 0 で埋める (corr 計算は欠損許容しないので)
    win_clean = win_clean.fillna(0)
    date = returns.index[-1]
    if win_clean.shape[1] < 5:
        log(f"Too few clean symbols: {win_clean.shape[1]}")
        return None
    corr = win_clean.corr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag = persistence_diagram(corr, max_dim=1)
    summ = persistence_summary(diag)
    cat = MarketCategory(symbols=list(win_clean.columns),
                          corr_matrix=corr, threshold=threshold)
    cat._build_graph()
    bal = signed_cycle_balance(cat.G)
    return {
        "date": str(date.date()),
        "n_symbols": win_clean.shape[1],
        "L1_H1": float(round(summ["L1_norm_H1"], 6)),
        "n_unb": int(bal["n_unbalanced"]),
        "n_edges": cat.G.number_of_edges(),
        "balance_rate": float(round(bal["balance_rate"], 4)),
    }


def classify(e_div: float, l1: float) -> str:
    """e_div と L1 から市場レジームを分類して文字列ラベルを返す."""
    if e_div >= 0.8: return "政策ショック型"
    if e_div <= -0.5: return "強さ変化型"
    if l1 >= 0.7: return "一般ボラ上昇"
    return "平常"


def get_recent_stats(conn: object, n_days: int = 90) -> tuple[float, float, float, float]:
    """過去 N 日の L¹/n_unb の mean/std (z-score 用)."""
    c = conn.cursor()
    c.execute("SELECT L1_H1, n_unb FROM gamma_daily WHERE date >= date('now', ?)",
              (f"-{n_days} days",))
    rows = c.fetchall()
    if len(rows) < 30:
        # 統計不足、ヒストリカル定数を使う (5y mean/std)
        return (1.045, 0.312, 29.0, 14.1)
    L1s = [r[0] for r in rows if r[0] is not None]
    unbs = [r[1] for r in rows if r[1] is not None]
    if len(L1s) < 30:
        return (1.045, 0.312, 29.0, 14.1)
    return (np.mean(L1s), np.std(L1s), np.mean(unbs), np.std(unbs))


def get_last_classification(conn: object) -> str | None:
    """DB から最新の classification ラベルを取得する."""
    c = conn.cursor()
    c.execute("SELECT classification FROM gamma_daily ORDER BY date DESC LIMIT 1")
    r = c.fetchone()
    return r[0] if r else None


def main():
    """VPS 日次ジョブ: データ取得 → γ計算 → DB保存 → publish を実行する."""
    log("=" * 60)
    log("VPS daily job started")
    t0 = time.time()
    db_path = DATA_DIR / "market_graph.db"
    conn = sqlite3.connect(str(db_path))
    init_db(conn)

    closes = fetch_recent(years=2)
    result = compute_gamma_for_day(closes)
    if not result:
        log("ERROR: gamma computation failed")
        return

    # z-score & e_div
    l1_mean, l1_std, unb_mean, unb_std = get_recent_stats(conn, n_days=90)
    z_L1 = (result["L1_H1"] - l1_mean) / l1_std if l1_std > 1e-9 else 0
    z_unb = (result["n_unb"] - unb_mean) / unb_std if unb_std > 1e-9 else 0
    e_div = z_unb - z_L1
    classification = classify(e_div, result["L1_H1"])
    result["z_L1"] = float(round(z_L1, 4))
    result["z_unb"] = float(round(z_unb, 4))
    result["e_div"] = float(round(e_div, 4))
    result["classification"] = classification
    result["computed_at"] = datetime.now().isoformat()

    # 判定変化を検知
    prev_class = get_last_classification(conn)
    classification_changed = False
    if prev_class and prev_class != classification:
        log(f"[!]  CLASSIFICATION CHANGED: {prev_class} -> {classification}")
        classification_changed = True
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO classification_changes
        (changed_at, date, prev_class, new_class, e_div, L1_H1, n_unb, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), result["date"], prev_class, classification,
              e_div, result["L1_H1"], result["n_unb"], ""))
    elif prev_class is None:
        log(f"Initial classification: {classification}")
    else:
        log(f"Classification unchanged: {classification}")

    # gamma_daily に upsert
    c = conn.cursor()
    c.execute("""
    INSERT OR REPLACE INTO gamma_daily
    (date, n_symbols, L1_H1, n_unb, n_edges, balance_rate, z_L1, z_unb, e_div, classification, computed_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (result["date"], result["n_symbols"], result["L1_H1"], result["n_unb"],
          result["n_edges"], result["balance_rate"], z_L1, z_unb, e_div,
          classification, result["computed_at"]))
    conn.commit()
    log(f"DB updated: date={result['date']} L1={result['L1_H1']:.3f} "
        f"n_unb={result['n_unb']} e_div={e_div:+.3f} class={classification}")

    # 状況サマリー
    c.execute("SELECT COUNT(*) FROM gamma_daily")
    n_rows = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM classification_changes")
    n_changes = c.fetchone()[0]
    log(f"DB total: {n_rows} daily rows, {n_changes} classification changes recorded")

    conn.close()

    # 判定変化があれば trade_executor を呼ぶ
    if classification_changed:
        log(f"Triggering trade_executor...")
        try:
            from trade_executor import execute_trade
            tr_res = execute_trade(prev_class, classification,
                                    trigger_e_div=e_div, dry_run=False)
            log(f"Trade result: {tr_res}")
        except Exception as e:
            log(f"trade_executor failed: {e}")
            import traceback
            log(traceback.format_exc())

    # GitHub Pages に最新データを publish
    log(f"Publishing to GitHub Pages...")
    try:
        from vps_publish import main as publish_main
        publish_main()
    except Exception as e:
        log(f"vps_publish failed: {e}")
        import traceback
        log(traceback.format_exc())

    log(f"VPS daily job done in {time.time()-t0:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
