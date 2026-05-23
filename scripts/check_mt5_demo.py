"""デモ MT5 (C:\\MT5-Demo) への接続テスト. 安全装置あり."""
import MetaTrader5 as mt5

DEMO_PATH = r"C:\MT5-Demo\terminal64.exe"

print(f"Connecting to: {DEMO_PATH}")
if not mt5.initialize(path=DEMO_PATH):
    print(f"❌ initialize() failed: {mt5.last_error()}")
    print("Hint: MT5 may not be running, or path is wrong")
else:
    info = mt5.account_info()
    if info:
        print(f"\n=== Account ===")
        print(f"  Login:     {info.login}")
        print(f"  Server:    {info.server}")
        print(f"  Balance:   {info.balance:,.2f} {info.currency}")
        print(f"  Leverage:  {info.leverage}")
        mode_name = {0: "DEMO ✓", 1: "CONTEST", 2: "REAL ⚠️ LIVE!!!"}.get(info.trade_mode, f"UNKNOWN({info.trade_mode})")
        print(f"  Mode:      {mode_name}")

        # 安全装置: ライブだったら警告
        if info.trade_mode == 2:
            print(f"\n⛔⛔⛔ SAFETY: This is a LIVE account! Do NOT use for auto-trading. ⛔⛔⛔")
        else:
            print(f"\n✅ Demo account confirmed. Safe to use.")
    term = mt5.terminal_info()
    if term:
        print(f"\n=== Terminal ===")
        print(f"  Name:      {term.name}")
        print(f"  Path:      {term.path}")
        print(f"  Build:     {term.build}")
        print(f"  Connected: {term.connected}")

    # US500 系シンボル探索
    print(f"\n=== US500 symbol search ===")
    candidates = ["US500", "SPX500", "SP500", "USTEC", "US500.cash", "SPX500.cash",
                   "USTECH", "US100", "NAS100"]
    found = []
    for sym in candidates:
        info_s = mt5.symbol_info(sym)
        if info_s is not None:
            found.append(sym)
            print(f"  ✓ {sym}: {info_s.description if hasattr(info_s, 'description') else '?'}")
    if not found:
        # 全シンボル走査
        all_syms = mt5.symbols_get() or []
        candidates_found = [s.name for s in all_syms
                              if "500" in s.name.upper() or "SPX" in s.name.upper()
                              or "US100" in s.name.upper()]
        print(f"  All symbols with 500/SPX/US100: {candidates_found[:10]}")

    mt5.shutdown()
