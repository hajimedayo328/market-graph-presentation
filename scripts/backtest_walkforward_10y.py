"""
Walk-forward 10 年データ版. fold 数増やして OOS 信頼性向上.
複数の (train, test) 組み合わせでロバスト性確認.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_v2 import apply_hysteresis, TRANSACTION_COST, HYSTERESIS_DAYS

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"


def load_10y(zscore_min_periods: int = 30) -> pd.DataFrame:
    """10y 指標を expanding z-score で計算 (look-ahead 完全排除)."""
    df = pd.read_csv(DATA_DIR / "gamma_timeseries_10y_w30.csv", parse_dates=["date"])
    df = df.dropna(subset=["L1_H1", "n_unb"]).set_index("date")
    mp = zscore_min_periods
    df["z_L1"] = (
        (df["L1_H1"] - df["L1_H1"].expanding(min_periods=mp).mean())
        / df["L1_H1"].expanding(min_periods=mp).std()
    )
    df["z_unb"] = (
        (df["n_unb"] - df["n_unb"].expanding(min_periods=mp).mean())
        / df["n_unb"].expanding(min_periods=mp).std()
    )
    df["e_div"] = df["z_unb"] - df["z_L1"]
    return df


def fetch_spy_10y(start: str, end: str) -> pd.DataFrame:
    print(f"Fetching ^GSPC OHLC ({start} to {end})...")
    df = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.dropna(subset=["Open", "Close"])


def walkforward_eval(df: pd.DataFrame, ohlc: pd.DataFrame,
                      train_years: float, test_years: float,
                      percentile: float = 80,
                      direction: str = "short") -> dict:
    common = ohlc.index.intersection(df.index)
    df_c = df.loc[common]; ohlc_c = ohlc.loc[common]
    rets = ohlc_c["Close"].pct_change().fillna(0)
    train_days = int(train_years * 252)
    test_days = int(test_years * 252)
    n = len(df_c)
    folds = []
    fold_id = 0
    oos_strategy_rets = pd.Series(0.0, index=ohlc_c.index)
    oos_mask = pd.Series(False, index=ohlc_c.index)
    start = 0
    while start + train_days + test_days <= n:
        train = df_c.iloc[start: start + train_days]
        test = df_c.iloc[start + train_days: start + train_days + test_days]
        # expanding z-score 化で最初の min_periods 日は NaN なので除外
        train_ediv = train["e_div"].dropna()
        if len(train_ediv) < 30:
            # train 期間がほぼ全部 NaN なら fold スキップ
            start += test_days
            continue
        thr = float(np.percentile(train_ediv, percentile))
        # test 側は NaN を False (シグナル発火しない) 扱い
        sig_test = (test["e_div"] >= thr).fillna(False)
        sig_test_h = apply_hysteresis(sig_test, HYSTERESIS_DAYS)
        sig_pos = sig_test_h.shift(1).fillna(False).astype(bool)
        rets_test = rets.loc[test.index]
        if direction == "short":
            strat_rets = rets_test * (~sig_pos).astype(float)
        else:
            strat_rets = rets_test * sig_pos.astype(float)
        pos_change = (sig_pos.astype(int).diff().abs() > 0)
        strat_rets = strat_rets - pos_change.astype(float) * TRANSACTION_COST
        eq = (1 + strat_rets).cumprod()
        bh_eq = (1 + rets_test).cumprod()
        n_y = len(rets_test) / 252
        fold_ret = float(eq.iloc[-1] - 1)
        bh_ret = float(bh_eq.iloc[-1] - 1)
        vol = float(strat_rets.std() * np.sqrt(252))
        cagr = float(eq.iloc[-1] ** (1 / n_y) - 1) if n_y > 0 else 0
        sharpe = cagr / vol if vol > 1e-9 else 0
        dd = float((eq / eq.cummax() - 1).min())
        folds.append({
            "fold_id": fold_id,
            "train_period": f"{train.index.min().date()}〜{train.index.max().date()}",
            "test_period":  f"{test.index.min().date()}〜{test.index.max().date()}",
            "threshold": thr,
            "OOS_return": fold_ret, "BH_return": bh_ret,
            "alpha": fold_ret - bh_ret,
            "sharpe": sharpe, "max_dd": dd,
            "n_signal_days": int(sig_pos.sum()),
            "n_test_days": int(len(test)),
        })
        oos_strategy_rets.loc[test.index] = strat_rets
        oos_mask.loc[test.index] = True
        fold_id += 1
        start += test_days

    oos_rets = oos_strategy_rets[oos_mask]
    oos_bh_rets = rets[oos_mask]
    oos_eq = (1 + oos_rets).cumprod()
    oos_bh_eq = (1 + oos_bh_rets).cumprod()
    n_y_oos = len(oos_rets) / 252
    return {
        "params": {"train_years": train_years, "test_years": test_years,
                    "percentile": percentile, "direction": direction},
        "n_folds": fold_id,
        "folds": folds,
        "oos_total_return": float(oos_eq.iloc[-1] - 1),
        "oos_bh_return":    float(oos_bh_eq.iloc[-1] - 1),
        "oos_alpha":        float(oos_eq.iloc[-1] - oos_bh_eq.iloc[-1]),
        "oos_cagr":         float(oos_eq.iloc[-1] ** (1 / n_y_oos) - 1) if n_y_oos > 0 else 0,
        "oos_sharpe":       float(oos_rets.mean() / oos_rets.std() * np.sqrt(252)) if oos_rets.std() > 1e-9 else 0,
        "oos_max_dd":       float((oos_eq / oos_eq.cummax() - 1).min()),
        "fold_sharpe_mean": float(np.mean([f["sharpe"] for f in folds])),
        "fold_sharpe_std":  float(np.std([f["sharpe"] for f in folds])),
        "win_rate_vs_bh":   float(sum(1 for f in folds if f["alpha"] > 0) / max(1, fold_id)),
        "oos_dates": oos_eq.index.strftime("%Y-%m-%d").tolist(),
        "oos_equity": oos_eq.round(4).tolist(),
        "oos_bh_equity": oos_bh_eq.round(4).tolist(),
    }


def main():
    df = load_10y()
    print(f"10y indicators: {df.shape}, {df.index.min().date()} -> {df.index.max().date()}")
    start = df.index.min().strftime("%Y-%m-%d")
    end = (df.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    ohlc = fetch_spy_10y(start, end)
    print(f"SPY OHLC: {ohlc.shape}")

    print("\n=== Walk-Forward (multiple train/test setups) ===")
    configs = [
        (3, 1, 80, "short"),  # 3 年学習 / 1 年テスト
        (2, 1, 80, "short"),  # 2 年学習 / 1 年テスト (fold 多め)
        (2, 0.5, 80, "short"), # 2 年学習 / 半年テスト
        (3, 1, 80, "long"),   # long 版
    ]
    all_results = {}
    for tr, te, pct, dir_ in configs:
        key = f"train{tr}y_test{te}y_pct{pct}_{dir_}"
        res = walkforward_eval(df, ohlc, tr, te, pct, dir_)
        all_results[key] = res
        print(f"\n--- {key} ---")
        print(f"  n_folds: {res['n_folds']}")
        print(f"  OOS Total: {res['oos_total_return']*100:+.2f}%  "
              f"BH: {res['oos_bh_return']*100:+.2f}%  "
              f"Alpha: {res['oos_alpha']*100:+.2f}%")
        print(f"  OOS Sharpe: {res['oos_sharpe']:+.2f}  "
              f"Fold Sharpe mean±std: {res['fold_sharpe_mean']:+.2f}±{res['fold_sharpe_std']:.2f}")
        print(f"  Win rate vs BH: {res['win_rate_vs_bh']*100:.0f}% ({sum(1 for f in res['folds'] if f['alpha']>0)}/{res['n_folds']})")
        print(f"  OOS MaxDD: {res['oos_max_dd']*100:+.2f}%")

    out_path = DATA_DIR / "backtest_walkforward_10y.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # メインの推奨設定 (3y train, 1y test, short) の詳細
    main_key = "train3y_test1y_pct80_short"
    main_res = all_results[main_key]
    print(f"\n=== {main_key} の fold 詳細 ===")
    print(f"{'fold':<5} {'train':<28} {'test':<28} {'thr':>7} {'OOS':>8} {'BH':>8} {'α':>8} {'Sharpe':>7}")
    print("-" * 110)
    for f in main_res["folds"]:
        print(f"{f['fold_id']:<5} {f['train_period']:<28} {f['test_period']:<28} "
              f"{f['threshold']:>+7.3f} {f['OOS_return']*100:>+7.2f}% {f['BH_return']*100:>+7.2f}% "
              f"{f['alpha']*100:>+7.2f}% {f['sharpe']:>+7.2f}")


if __name__ == "__main__":
    main()
