"""
MT5 デモ口座の取引履歴 (deals) から資産残高の推移を再構築する.
読み取り専用 — 注文は一切出さない。デモ口座以外なら即中断。

出力: C:\\tools\\market-graph\\equity_history.json
  - current_balance / current_equity (現在値)
  - history: [{date, balance}] の日次推移 (初期残高からの累積)
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(r"C:\tools\market-graph")


def load_env() -> dict:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def main():
    import MetaTrader5 as mt5
    env = load_env()
    path = env.get("VANTAGE_DEMO_PATH")
    login = env.get("VANTAGE_DEMO_LOGIN")
    pw = env.get("VANTAGE_DEMO_PASSWORD")
    server = env.get("VANTAGE_DEMO_SERVER")

    if all([login, pw, server]):
        ok = mt5.initialize(path=path, login=int(login), password=pw, server=server)
    else:
        ok = mt5.initialize(path=path)
    if not ok:
        print("INIT FAIL:", mt5.last_error())
        return

    info = mt5.account_info()
    if info is None:
        print("account_info None"); mt5.shutdown(); return
    # 安全: デモ以外なら中断 (trade_mode 0 = demo)
    if info.trade_mode != 0:
        print(f"NOT DEMO (trade_mode={info.trade_mode}) — abort")
        mt5.shutdown(); return

    print(f"login={info.login} balance={info.balance} equity={info.equity} {info.currency}")

    # 取引履歴 (口座開設〜現在)
    deals = mt5.history_deals_get(datetime(2020, 1, 1), datetime.now() + timedelta(days=1))
    n = len(deals) if deals else 0
    print(f"deals count: {n}")

    rows = []
    if deals:
        # 入金 deal (DEAL_TYPE_BALANCE) を初期残高の起点にし、売買 deal の損益を累積
        BAL = mt5.DEAL_TYPE_BALANCE
        bal_deals = [d for d in deals if d.type == BAL]
        trade_deals = [d for d in deals if d.type != BAL]
        init_balance = sum(d.profit for d in bal_deals) or info.balance
        cum = init_balance
        daily = {}
        # 入金日を推移の起点として記録
        if bal_deals:
            d0 = min(bal_deals, key=lambda x: x.time)
            daily[datetime.fromtimestamp(d0.time).strftime("%Y-%m-%d")] = round(init_balance, 2)
        for d in sorted(trade_deals, key=lambda x: x.time):
            cum += d.profit + d.swap + d.commission
            day = datetime.fromtimestamp(d.time).strftime("%Y-%m-%d")
            daily[day] = round(cum, 2)
        rows = [{"date": k, "balance": v} for k, v in sorted(daily.items())]
        print(f"init_balance={round(init_balance,2)}  days={len(rows)}")
        for r in rows[:3]:
            print("  ", r)
        if len(rows) > 3:
            print("   ...")
            for r in rows[-3:]:
                print("  ", r)

    out = {
        "as_of": datetime.now().isoformat(),
        "currency": info.currency,
        "current_balance": round(info.balance, 2),
        "current_equity": round(info.equity, 2),
        "n_deals": n,
        "history": rows,
    }
    (ROOT / "equity_history.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", ROOT / "equity_history.json")
    mt5.shutdown()


if __name__ == "__main__":
    main()
