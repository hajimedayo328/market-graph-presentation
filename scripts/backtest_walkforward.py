"""
Walk-forward Out-of-Sample 評価.

学会査読対策の核心: 「閾値を過去最適化してない」を実証.

設計:
  1. 全期間を 1 年スライドで分割
  2. 各 split で:
     - 前 3 年 (train): e_div の閾値を 80 percentile に決める
     - 次 1 年 (test): その閾値で S1 戦略を実行 → OOS リターン記録
  3. 全 split の OOS リターンを連結 → walk-forward equity curve
  4. In-sample (固定閾値 0.8) と比較

評価:
  - OOS Sharpe vs IS Sharpe
  - OOS 年率リターン vs IS
  - 各 fold での Sharpe ばらつき
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_v2 import (load_indicators, fetch_spy_ohlc, simulate_v2,
                          apply_hysteresis, TRANSACTION_COST, HYSTERESIS_DAYS)

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"


def walkforward_s1(df: pd.DataFrame, ohlc: pd.DataFrame,
                    train_years: int = 3, test_years: int = 1,
                    percentile: float = 80) -> dict:
    """S1 (e_div_high short) を walk-forward で評価."""
    common = ohlc.index.intersection(df.index)
    df_c = df.loc[common]; ohlc_c = ohlc.loc[common]
    rets = ohlc_c["Close"].pct_change().fillna(0)

    train_days = train_years * 252
    test_days = test_years * 252
    n = len(df_c)
    folds = []
    fold_id = 0
    # 連続する OOS 期間を集めるための equity curve
    oos_strategy_rets = pd.Series(0.0, index=ohlc_c.index)
    oos_mask = pd.Series(False, index=ohlc_c.index)

    start = 0
    while start + train_days + test_days <= n:
        train = df_c.iloc[start: start + train_days]
        test = df_c.iloc[start + train_days: start + train_days + test_days]

        # 学習: train 期間の e_div 80 パーセンタイルを閾値とする
        thr = float(np.percentile(train["e_div"], percentile))
        # テスト: その閾値で test 期間の S1 short 戦略実行
        sig_test = test["e_div"] >= thr
        sig_test_h = apply_hysteresis(sig_test, HYSTERESIS_DAYS)
        # 翌日寄付き約定
        sig_pos = sig_test_h.shift(1).fillna(False).astype(bool)
        rets_test = rets.loc[test.index]
        strat_rets = rets_test * (~sig_pos).astype(float)
        pos_change = (sig_pos.astype(int).diff().abs() > 0)
        strat_rets = strat_rets - pos_change.astype(float) * TRANSACTION_COST

        # 累積
        eq = (1 + strat_rets).cumprod()
        bh_eq = (1 + rets_test).cumprod()
        n_years_fold = len(rets_test) / 252
        fold_ret = float(eq.iloc[-1] - 1)
        bh_ret = float(bh_eq.iloc[-1] - 1)
        vol_fold = float(strat_rets.std() * np.sqrt(252))
        cagr_fold = float(eq.iloc[-1] ** (1 / n_years_fold) - 1) if n_years_fold > 0 else 0
        sharpe_fold = cagr_fold / vol_fold if vol_fold > 1e-9 else 0
        dd_fold = float((eq / eq.cummax() - 1).min())

        folds.append({
            "fold_id": fold_id,
            "train_start": str(train.index.min().date()),
            "train_end": str(train.index.max().date()),
            "test_start": str(test.index.min().date()),
            "test_end": str(test.index.max().date()),
            "threshold": thr,
            "fold_return": fold_ret,
            "bh_return": bh_ret,
            "sharpe": sharpe_fold,
            "max_dd": dd_fold,
            "n_signal_days": int(sig_pos.sum()),
            "n_test_days": int(len(test)),
        })
        # OOS 連結
        oos_strategy_rets.loc[test.index] = strat_rets
        oos_mask.loc[test.index] = True

        fold_id += 1
        start += test_days

    # OOS 全体の連結評価
    oos_rets = oos_strategy_rets[oos_mask]
    oos_bh_rets = rets[oos_mask]
    oos_eq = (1 + oos_rets).cumprod()
    oos_bh_eq = (1 + oos_bh_rets).cumprod()
    n_years_oos = len(oos_rets) / 252
    oos_total_ret = float(oos_eq.iloc[-1] - 1)
    oos_bh_ret = float(oos_bh_eq.iloc[-1] - 1)
    oos_cagr = float(oos_eq.iloc[-1] ** (1 / n_years_oos) - 1) if n_years_oos > 0 else 0
    oos_vol = float(oos_rets.std() * np.sqrt(252))
    oos_sharpe = oos_cagr / oos_vol if oos_vol > 1e-9 else 0
    oos_dd = float((oos_eq / oos_eq.cummax() - 1).min())

    return {
        "folds": folds,
        "n_folds": fold_id,
        "oos_total_return": oos_total_ret,
        "oos_bh_total_return": oos_bh_ret,
        "oos_cagr": oos_cagr,
        "oos_sharpe": oos_sharpe,
        "oos_max_drawdown": oos_dd,
        "oos_dates": oos_eq.index.strftime("%Y-%m-%d").tolist(),
        "oos_equity": oos_eq.round(4).tolist(),
        "oos_bh_equity": oos_bh_eq.round(4).tolist(),
    }


def main():
    df = load_indicators()
    start = df.index.min().strftime("%Y-%m-%d")
    end = (df.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    ohlc = fetch_spy_ohlc(start, end)
    print(f"Loaded: indicators {df.shape}, ohlc {ohlc.shape}")

    res = walkforward_s1(df, ohlc, train_years=3, test_years=1, percentile=80)
    print(f"\n=== Walk-Forward S1 (train=3y, test=1y, percentile=80%) ===")
    print(f"{'fold':<5} {'train':<25} {'test':<25} {'thr':>6} "
          f"{'OOS_ret':>8} {'BH_ret':>8} {'sharpe':>7} {'MaxDD':>7}")
    print("-" * 110)
    for f in res["folds"]:
        print(f"{f['fold_id']:<5} {f['train_start']}〜{f['train_end']:<14}  "
              f"{f['test_start']}〜{f['test_end']:<14}  {f['threshold']:>+6.3f} "
              f"{f['fold_return']*100:>+7.2f}% {f['bh_return']*100:>+7.2f}% "
              f"{f['sharpe']:>+7.2f} {f['max_dd']*100:>+6.2f}%")

    print(f"\n=== OOS 全体 (連結) ===")
    print(f"  期間: {len(res['oos_dates'])} 営業日 ({res['n_folds']} folds)")
    print(f"  OOS Total Return:    {res['oos_total_return']*100:+.2f}%")
    print(f"  OOS B&H Return:      {res['oos_bh_total_return']*100:+.2f}%")
    print(f"  OOS CAGR:            {res['oos_cagr']*100:+.2f}%")
    print(f"  OOS Sharpe:          {res['oos_sharpe']:+.2f}")
    print(f"  OOS Max Drawdown:    {res['oos_max_drawdown']*100:+.2f}%")

    # 各 fold の Sharpe ばらつき
    fold_sharpes = [f["sharpe"] for f in res["folds"]]
    print(f"\n  Fold Sharpe stats: mean={np.mean(fold_sharpes):+.2f}  "
          f"std={np.std(fold_sharpes):.2f}  min={min(fold_sharpes):+.2f}  max={max(fold_sharpes):+.2f}")

    out_path = DATA_DIR / "backtest_walkforward.json"
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
