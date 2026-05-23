"""現在の MT5 接続情報確認."""
import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"initialize() failed: {mt5.last_error()}")
else:
    info = mt5.account_info()
    if info:
        print(f"Account login: {info.login}")
        print(f"Server: {info.server}")
        print(f"Balance: {info.balance}")
        print(f"Currency: {info.currency}")
        print(f"trade_mode: {info.trade_mode}  (0=demo, 1=contest, 2=real)")
        print(f"Leverage: {info.leverage}")
    else:
        print("No account info available")
    term = mt5.terminal_info()
    if term:
        print(f"\nTerminal: {term.name}")
        print(f"Path: {term.path}")
        print(f"Build: {term.build}")
        print(f"Connected: {term.connected}")
    mt5.shutdown()
