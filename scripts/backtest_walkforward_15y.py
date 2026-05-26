"""Walk-forward 15 年データ (2011-2026): 2015 China shock, 2018, COVID 2020 含む."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
sys.path.insert(0, str(Path(__file__).parent))
from backtest_v2 import apply_hysteresis, TRANSACTION_COST, HYSTERESIS_DAYS
from backtest_walkforward_10y import walkforward_eval

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"


def load_15y(zscore_min_periods: int = 30):
    """15y 指標を expanding z-score で計算 (look-ahead 完全排除)."""
    df = pd.read_csv(DATA_DIR / "gamma_timeseries_15y_w30.csv", parse_dates=["date"])
    df = df.dropna(subset=["L1_H1", "n_unb"]).set_index("date")
    mp = zscore_min_periods
    df["z_L1"]  = (
        (df["L1_H1"] - df["L1_H1"].expanding(min_periods=mp).mean())
        / df["L1_H1"].expanding(min_periods=mp).std()
    )
    df["z_unb"] = (
        (df["n_unb"] - df["n_unb"].expanding(min_periods=mp).mean())
        / df["n_unb"].expanding(min_periods=mp).std()
    )
    df["e_div"] = df["z_unb"] - df["z_L1"]
    return df


def main():
    df = load_15y()
    print(f"15y indicators: {df.shape}, {df.index.min().date()} -> {df.index.max().date()}")
    start = df.index.min().strftime("%Y-%m-%d")
    end = (df.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    print(f"Fetching ^GSPC ({start} to {end})...")
    ohlc = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(ohlc.columns, pd.MultiIndex):
        ohlc.columns = ohlc.columns.get_level_values(0)
    ohlc.index = pd.to_datetime(ohlc.index)
    ohlc = ohlc.dropna(subset=["Close"])
    print(f"SPY OHLC: {ohlc.shape}")

    configs = [
        (3, 1, 80, "short"),
        (5, 1, 80, "short"),
        (3, 1, 90, "short"),  # より厳しい閾値
        (3, 1, 80, "long"),
    ]
    all_res = {}
    for tr, te, pct, dir_ in configs:
        key = f"15y_train{tr}y_test{te}y_pct{pct}_{dir_}"
        res = walkforward_eval(df, ohlc, tr, te, pct, dir_)
        all_res[key] = res
        print(f"\n--- {key} ---")
        print(f"  n_folds={res['n_folds']}, OOS Ret={res['oos_total_return']*100:+.2f}%, "
              f"BH={res['oos_bh_return']*100:+.2f}%, Alpha={res['oos_alpha']*100:+.2f}%")
        print(f"  OOS Sharpe={res['oos_sharpe']:+.2f}, MaxDD={res['oos_max_dd']*100:+.2f}%, "
              f"Win={res['win_rate_vs_bh']*100:.0f}% ({sum(1 for f in res['folds'] if f['alpha']>0)}/{res['n_folds']})")

    out_path = DATA_DIR / "backtest_walkforward_15y.json"
    out_path.write_text(json.dumps(all_res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # メイン設定の fold 詳細
    main_key = "15y_train3y_test1y_pct80_short"
    if main_key in all_res:
        r = all_res[main_key]
        print(f"\n=== {main_key} fold 詳細 ===")
        print(f"{'fold':<5} {'train':<28} {'test':<28} {'OOS':>8} {'BH':>8} {'α':>8} {'Sharpe':>7}")
        print("-" * 100)
        for f in r["folds"]:
            print(f"{f['fold_id']:<5} {f['train_period']:<28} {f['test_period']:<28} "
                  f"{f['OOS_return']*100:>+7.2f}% {f['BH_return']*100:>+7.2f}% "
                  f"{f['alpha']*100:>+7.2f}% {f['sharpe']:>+7.2f}")


if __name__ == "__main__":
    main()
