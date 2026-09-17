"""学会質問の続き: e_div は「直近20日の実現ボラで売る」単純ルールに前進検証で勝てるか.

backtest_oos_random_vix.walkforward_oos の VIX 比較枠に、VIX の代わりに
「直近20日の実現ボラ（過去のみ）」を流す。fold 毎に e_div と同じ現金化率になる
しきい値を train から決めるので、公平な比較になる。
価格は手元の parquet の SP500 終値を使う（元スクリプトの SPY 取得はネットワーク依存）。
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import backtest_oos_random_vix as bo

ROOT = Path(__file__).parent.parent
period = sys.argv[1] if len(sys.argv) > 1 else "20y"
n_random = int(sys.argv[2]) if len(sys.argv) > 2 else 100

df = bo.load_indicators_with_vix(bo.CSV_MAP[period], bo.PARQUET_MAP[period])
px = pd.read_parquet(ROOT / "data" / bo.PARQUET_MAP[period])["SP500"].dropna()
rets = px.pct_change()
rv20 = rets.rolling(20).std() * np.sqrt(252)          # 過去20日のみ → look-ahead なし
df["VIX"] = rv20.reindex(df.index).ffill()               # VIX 枠に実現ボラを入れる
ohlc = pd.DataFrame({"Close": px})

res = bo.walkforward_oos(df, ohlc, bo.TRAIN_YEARS, bo.TEST_YEARS, bo.PERCENTILE, n_random, bo.SEED)
e, b, vc = res["ediv"], res["buy_hold"], res["vix_comparison"]
print(f"period={period}  folds={res['n_folds']}  OOS {res['oos_span']}  n_random={n_random}")
print(f"{'戦略':<22}{'Return':>9}{'Sharpe':>8}{'MaxDD':>9}")
print(f"{'e_div (OOS)':<22}{e['total_return']*100:>+8.1f}%{e['sharpe']:>+8.2f}{e['max_drawdown']*100:>+8.1f}%")
print(f"{'持ち続け':<22}{b['total_return']*100:>+8.1f}%{b['sharpe']:>+8.2f}{b['max_drawdown']*100:>+8.1f}%")
if vc:
    v, es = vc["vix"], vc["ediv_same_span"]
    print(f"{'実現ボラ20日 (同現金化率)':<22}{v['total_return']*100:>+8.1f}%{v['sharpe']:>+8.2f}{v['max_drawdown']*100:>+8.1f}%")
    print(f"e_div が実現ボラに勝つ: Sharpe={vc['ediv_beats_vix_sharpe']} MaxDD={vc['ediv_beats_vix_maxdd']} Return={vc['ediv_beats_vix_return']}")
print(f"ランダム{n_random}本が e_div に勝つ確率: MaxDD={res['p_random_beats_ediv_maxdd']*100:.1f}% Sharpe={res['p_random_beats_ediv_sharpe']*100:.1f}%")

out = {"description": "e_div vs 直近20日実現ボラ(過去のみ) の前進検証比較。VIX枠に実現ボラを流し、fold毎に同じ現金化率でしきい値を学習",
       "period": period, "n_random": n_random, "price_source": "data/ohlc parquet SP500 (offline)",
       "ediv": e, "buy_hold": b, "realized_vol_20d": (vc["vix"] if vc else None),
       "ediv_beats_rv": {k: vc[k] for k in ("ediv_beats_vix_sharpe", "ediv_beats_vix_maxdd", "ediv_beats_vix_return")} if vc else None}
(ROOT / "data" / f"realized_vol_oos_{period}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"saved: data/realized_vol_oos_{period}.json")
