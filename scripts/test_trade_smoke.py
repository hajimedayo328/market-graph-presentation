"""
実約定スモークテスト: 最小ロット 0.01 で SP500.r buy → 即 close.

目的:
- trade_executor の発注経路が本当に約定できるかを 1 回だけ確認
- ticket が返ってくれば OK、retcode != 10009 なら NG
- 約定後すぐ close するので持ち越し損益は spread 分のみ

実行:
  ssh fx-vps 'C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python312\\python.exe C:\\tools\\market-graph\\scripts\\test_trade_smoke.py'

絶対ルール:
- demo 口座 (trade_mode=0) でしか動かない (LIVE なら即 abort)
- 0.01 lot (最小)
- 1 回のみ買って即 close
"""
from __future__ import annotations
import sys
import time
import MetaTrader5 as mt5

DEMO_PATH = r"C:\MT5-Demo\terminal64.exe"
SYMBOL = "SP500.r"
LOT = 0.1  # SP500.r の volume_min は 0.1

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def best_filling(symbol_info) -> int:
    fm = getattr(symbol_info, "filling_mode", 0)
    if fm & 1: return mt5.ORDER_FILLING_FOK
    if fm & 2: return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

def main() -> int:
    log("[1] init MT5-Demo...")
    if not mt5.initialize(path=DEMO_PATH):
        log(f"NG: init failed err={mt5.last_error()}")
        return 1

    info = mt5.account_info()
    if info is None:
        log("NG: account_info() None")
        mt5.shutdown(); return 1

    log(f"  login={info.login}  server={info.server}  balance={info.balance}  trade_mode={info.trade_mode}")
    if info.trade_mode != 0:
        log(f"!! ABORT: trade_mode={info.trade_mode} (not demo). Will NOT trade.")
        mt5.shutdown(); return 2

    log(f"[2] symbol_info({SYMBOL})...")
    sym = mt5.symbol_info(SYMBOL)
    if sym is None:
        log(f"NG: symbol_info None. trying symbol_select...")
        if not mt5.symbol_select(SYMBOL, True):
            log(f"NG: symbol_select failed")
            mt5.shutdown(); return 1
        sym = mt5.symbol_info(SYMBOL)
    log(f"  visible={sym.visible}  trade_mode={sym.trade_mode}  filling_mode={sym.filling_mode}")

    tick = mt5.symbol_info_tick(SYMBOL)
    log(f"  bid={tick.bid}  ask={tick.ask}  time={tick.time}")

    log(f"[3] sending BUY {LOT} {SYMBOL}...")
    req_buy = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 20,
        "magic": 99999,
        "comment": "smoke_test",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": best_filling(sym),
    }
    res = mt5.order_send(req_buy)
    log(f"  retcode={res.retcode}  comment={res.comment}  ticket={res.order}  deal={res.deal}")

    if res.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"NG: BUY failed retcode={res.retcode} ({res.comment})")
        log("  ヒント: 10018=MARKET_CLOSED (週末/時間外)、10030=INVALID_FILL、10027=AUTOTRADING_OFF")
        mt5.shutdown(); return 1

    log(f"OK: BUY filled, ticket={res.order}")

    # 即 close (反対売買)
    log(f"[4] closing position ticket={res.order}...")
    positions = mt5.positions_get(ticket=res.order)
    if not positions:
        log(f"WARN: position not found, may already closed")
        mt5.shutdown(); return 0

    pos = positions[0]
    tick2 = mt5.symbol_info_tick(SYMBOL)
    req_close = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL,
        "position": pos.ticket,
        "price": tick2.bid,
        "deviation": 20,
        "magic": 99999,
        "comment": "smoke_test_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": best_filling(sym),
    }
    res_close = mt5.order_send(req_close)
    log(f"  retcode={res_close.retcode}  comment={res_close.comment}")

    if res_close.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"OK: CLOSED. round-trip spread loss = {(tick.ask - tick2.bid):.2f} pts on {LOT} lot")
    else:
        log(f"WARN: close failed retcode={res_close.retcode} ({res_close.comment}). manual close 必要")

    mt5.shutdown()
    log("[5] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
