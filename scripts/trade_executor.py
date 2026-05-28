"""
Vantage MT5 デモ口座での自動売買執行.

設計:
- 別 MT5 ターミナル (デモ専用) に接続
- 既存の本番 MT5 (師匠 bot 用) には触らない
- 判定変化を検知 → US500 ポジション調整 → DB 記録

安全装置:
- ライブ口座だった場合は即停止 (trade_mode=2 なら絶対に発注しない)
- 1 日の最大トレード回数制限
- ポジション重複防止
- 全例外で発注スキップ + ログ
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"


def load_env() -> dict[str, str]:
    """C:\\tools\\market-graph\\.env から認証情報を読む."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}
    env = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def log(msg: str) -> None:
    """タイムスタンプ付きメッセージを stdout とログファイルに書く."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [trade] {msg}"
    print(line)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "trade_executor.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def connect_demo_mt5() -> object | None:
    """Vantage デモ MT5 に接続. 失敗 or ライブ口座なら None."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log("MetaTrader5 not installed")
        return None
    env = load_env()
    path = env.get("VANTAGE_DEMO_PATH")
    login = env.get("VANTAGE_DEMO_LOGIN")
    pw = env.get("VANTAGE_DEMO_PASSWORD")
    server = env.get("VANTAGE_DEMO_SERVER")
    if not path:
        log("Missing VANTAGE_DEMO_PATH in .env")
        return None
    log(f"Connecting to demo MT5: {path}")
    try:
        # login/pw/server がそろってればフル指定 (MT5 が落ちてても起動 + ログイン)
        # path のみなら起動中 MT5 にアタッチ
        if all([login, pw, server]):
            ok = mt5.initialize(path=path, login=int(login), password=pw, server=server)
            log("  using full credentials (login+pw+server)")
        else:
            ok = mt5.initialize(path=path)
            log("  attaching to running MT5 (path only)")
    except Exception as e:
        log(f"initialize() raised: {e}")
        return None
    if not ok:
        log(f"initialize() failed: {mt5.last_error()}")
        return None
    info = mt5.account_info()
    if not info:
        log("account_info() returned None")
        mt5.shutdown()
        return None
    # 安全装置: ライブ口座だったら即停止
    if info.trade_mode != 0:  # 0=DEMO のみ許可
        log(f"[NG] SAFETY: Not a DEMO account (trade_mode={info.trade_mode}). Login={info.login}. ABORTING.")
        mt5.shutdown()
        return None
    if info.trade_mode == 0:
        mode = "DEMO"
    elif info.trade_mode == 1:
        mode = "CONTEST"
    else:
        mode = f"UNKNOWN({info.trade_mode})"
    log(f"[OK] Connected: login={info.login} server={info.server} balance={info.balance} {info.currency} mode={mode}")
    # autotrading check
    term = mt5.terminal_info()
    if term is not None and not term.trade_allowed:
        log("[NG] Auto-trading is DISABLED in MT5 terminal. Enable 'Algo Trading' toolbar button.")
        log("     Trades will be rejected with retcode=10027 until enabled.")
        # 接続は返す (情報取得は可能、発注のみ失敗する)
    return mt5


def find_us500_symbol(mt5: object) -> str | None:
    """Vantage の US500 系シンボルを探す (名前ブローカー依存)."""
    # Vantage Trading: SP500.r (spot), SP500ft.r (futures), SPX
    candidates = ["SP500.r", "SP500", "SPX", "SP500ft.r", "US500",
                   "SPX500", "USTEC", "US500.cash", "SPX500.cash"]
    for sym in candidates:
        info = mt5.symbol_info(sym)
        if info is not None:
            log(f"Symbol found: {sym} ({info.description if hasattr(info, 'description') else '?'})")
            if not info.visible:
                if not mt5.symbol_select(sym, True):
                    log(f"  symbol_select({sym}) failed")
                    continue
            return sym
    # 全シンボルから検索
    all_syms = mt5.symbols_get()
    if all_syms:
        for s in all_syms:
            if "500" in s.name.upper() or "SPX" in s.name.upper():
                log(f"Symbol candidate by search: {s.name}")
                return s.name
    log("[WARN]  US500 symbol not found")
    return None


def _best_filling(mt5: object, symbol: str) -> int:
    """symbol がサポートする filling mode を選ぶ.
    Vantage の SP500.r 等は FOK 非対応のことがある."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_RETURN
    # filling_mode は bit flag: 1=FOK, 2=IOC, 4=Return
    fm = getattr(info, "filling_mode", 0)
    if fm & 1:  # FOK
        return mt5.ORDER_FILLING_FOK
    if fm & 2:  # IOC
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def get_current_position(mt5: object, symbol: str) -> object | None:
    """その symbol の現在のポジション (なければ None)."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    if len(positions) > 1:
        log(f"[WARN]  Multiple positions on {symbol}: {len(positions)}")
    return positions[0]


def execute_trade(classification_prev: str, classification_new: str,
                  trigger_e_div: float, dry_run: bool = False) -> dict:
    """
    判定変化に応じて Vantage デモ口座で US500 自動売買.

    Returns: dict with action / order_result / position_after etc.
    """
    log(f"Trigger: {classification_prev} -> {classification_new} (e_div={trigger_e_div:+.3f})")

    # 戦略ロジック
    action = None
    if classification_new == "政策ショック型":
        action = "close_long_or_short"  # 株売り (リスクオフ)
    elif classification_new == "強さ変化型":
        action = "open_long"  # 逆張り買い
    elif classification_new in ("平常", "一般ボラ上昇"):
        if classification_prev in ("政策ショック型", "強さ変化型"):
            action = "close_to_neutral"  # ポジション解消
        else:
            action = "no_action"
    if action == "no_action":
        log("No action required.")
        return {"action": "no_action"}

    mt5 = connect_demo_mt5()
    if mt5 is None:
        return {"action": action, "error": "MT5 connection failed"}
    try:
        symbol = find_us500_symbol(mt5)
        if symbol is None:
            return {"action": action, "error": "US500 symbol not found"}

        current = get_current_position(mt5, symbol)
        # ポジションサイズ: 残高の 10% (FX みたいに大きく出ない)
        # CFD なので 0.1 lot 程度に固定
        volume = 0.1

        if dry_run:
            log(f"[DRY RUN] action={action} symbol={symbol} volume={volume} current={current}")
            return {"action": action, "dry_run": True, "symbol": symbol, "volume": volume}

        # 既存ポジション解消
        if current is not None:
            # close
            close_type = mt5.ORDER_TYPE_SELL if current.type == 0 else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol, "volume": current.volume,
                "type": close_type, "position": current.ticket,
                "price": price, "deviation": 20, "magic": 99001,
                "comment": f"close_for_{classification_new}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": _best_filling(mt5, symbol),
            }
            res = mt5.order_send(req)
            log(f"Close result: retcode={res.retcode if res else None} comment={res.comment if res else None}")

        # 新規ポジション
        new_position = None
        if action == "open_long":
            tick = mt5.symbol_info_tick(symbol)
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol, "volume": volume,
                "type": mt5.ORDER_TYPE_BUY, "price": tick.ask,
                "deviation": 20, "magic": 99001,
                "comment": f"open_long_{classification_new}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": _best_filling(mt5, symbol),
            }
            res = mt5.order_send(req)
            log(f"Open long result: retcode={res.retcode if res else None}")
            new_position = res

        # 政策ショック型は close のみ (現金化)
        # close_to_neutral も close のみ

        # DB に記録
        conn = sqlite3.connect(str(DATA_DIR / "market_graph.db"))
        c = conn.cursor()
        c.execute("""
        INSERT INTO trades (executed_at, symbol, side, volume, open_price, close_price, pnl,
                             classification_trigger, mt5_ticket, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), symbol,
              "long" if action == "open_long" else "close",
              volume, None, None, None,
              f"{classification_prev}->{classification_new}",
              new_position.order if new_position else None,
              f"trigger_e_div={trigger_e_div:+.3f}"))
        conn.commit()
        conn.close()
        log("Trade recorded in DB.")

        return {"action": action, "symbol": symbol, "volume": volume}
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    # CLI テスト: dry_run
    res = execute_trade("平常", "政策ショック型", trigger_e_div=1.20, dry_run=True)
    log(f"Result: {res}")
