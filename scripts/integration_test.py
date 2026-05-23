"""
trade_executor の統合テスト. デモ口座で実発注 → 即 close.

シナリオ:
1. 政策ショック型 (平常 → 政策ショック型) 通知 → close 系 (ポジションなければ何もせず終わる)
2. 強さ変化型 (平常 → 強さ変化型) → open_long 0.1 lot
3. 平常戻り (強さ変化型 → 平常) → close_to_neutral
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from trade_executor import execute_trade, log

log("=" * 60)
log("INTEGRATION TEST START")

# シナリオ 1: 政策ショック型 (ポジションなければ close は no-op)
log("--- Scenario 1: 平常 -> 政策ショック型 (close 系) ---")
r1 = execute_trade("平常", "政策ショック型", trigger_e_div=+1.20, dry_run=False)
log(f"Result: {r1}")
time.sleep(2)

# シナリオ 2: 強さ変化型 (open_long 0.1 lot)
log("--- Scenario 2: 平常 -> 強さ変化型 (open_long) ---")
r2 = execute_trade("平常", "強さ変化型", trigger_e_div=-0.80, dry_run=False)
log(f"Result: {r2}")
time.sleep(2)

# シナリオ 3: 平常戻り (open したポジを close)
log("--- Scenario 3: 強さ変化型 -> 平常 (close to neutral) ---")
r3 = execute_trade("強さ変化型", "平常", trigger_e_div=0.05, dry_run=False)
log(f"Result: {r3}")

log("INTEGRATION TEST END")
log("=" * 60)
