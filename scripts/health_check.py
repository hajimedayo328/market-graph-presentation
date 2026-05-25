"""VPS 運用ヘルスチェック (ローカルから ssh 越しに実行する想定)."""
import sys, json, sqlite3
from pathlib import Path
from datetime import datetime, timedelta

print("=" * 50)
print("Market Graph VPS Health Check")
print("=" * 50)

# 1. Python deps
print("\n[1] Python deps:")
mods = ["yfinance", "MetaTrader5", "pandas", "numpy", "requests", "networkx", "scipy"]
for m in mods:
    try:
        __import__(m)
        print(f"  OK: {m}")
    except Exception as e:
        print(f"  NG: {m} -> {e}")

# 2. MT5 demo connection
print("\n[2] MT5-Demo connection:")
try:
    import MetaTrader5 as mt5
    if mt5.initialize(path=r"C:\MT5-Demo\terminal64.exe"):
        info = mt5.account_info()
        print(f"  login: {info.login}")
        print(f"  server: {info.server}")
        print(f"  balance: {info.balance}")
        print(f"  trade_mode: {info.trade_mode} (0=demo, 1=contest, 2=LIVE)")
        if info.trade_mode == 2:
            print("  !!! WARNING: LIVE ACCOUNT !!!")
        sym = mt5.symbol_info("SP500.r")
        if sym:
            print(f"  SP500.r: filling_mode={sym.filling_mode}, trade_mode={sym.trade_mode}")
        mt5.shutdown()
    else:
        print(f"  NG: init failed err={mt5.last_error()}")
except Exception as e:
    print(f"  NG: {e}")

# 3. DB latest row
print("\n[3] DB freshness:")
db = Path(r"C:\tools\market-graph\data\market_graph.db")
if db.exists():
    conn = sqlite3.connect(str(db))
    c = conn.cursor()
    c.execute("SELECT date, e_div, classification, computed_at FROM gamma_daily ORDER BY date DESC LIMIT 3")
    for r in c.fetchall():
        print(f"  date={r[0]}  e_div={r[1]:+.3f}  class={r[2]}  computed={r[3]}")
    c.execute("SELECT COUNT(*) FROM trades")
    n_trades = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE mt5_ticket IS NOT NULL AND mt5_ticket != '0' AND mt5_ticket != 0")
    n_real_trades = c.fetchone()[0]
    print(f"  trades total: {n_trades}, real-ticket: {n_real_trades}")
    conn.close()
else:
    print(f"  NG: DB not found at {db}")

# 4. .env / PAT
print("\n[4] .env presence:")
env = Path(r"C:\tools\market-graph\.env")
if env.exists():
    content = env.read_text(encoding="utf-8")
    has_pat = "GITHUB_PAT=" in content and len([l for l in content.splitlines() if l.startswith("GITHUB_PAT=") and len(l) > 20]) > 0
    print(f"  .env exists, GITHUB_PAT set: {has_pat}")
else:
    print(f"  NG: .env not found")

# 5. yfinance smoke
print("\n[5] yfinance smoke (SPY 5d):")
try:
    import yfinance as yf
    df = yf.download("SPY", period="5d", progress=False, auto_adjust=True)
    print(f"  rows: {len(df)}, last: {df.index[-1].date() if len(df) else 'NONE'}")
except Exception as e:
    print(f"  NG: {e}")

# 6. Log freshness
print("\n[6] Log freshness:")
for log in ["vps_daily.log", "vps_publish.log"]:
    p = Path(r"C:\tools\market-graph\logs") / log
    if p.exists():
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        age = datetime.now() - mtime
        print(f"  {log}: {mtime}  age={age}")

print("\n" + "=" * 50)
print("done")
