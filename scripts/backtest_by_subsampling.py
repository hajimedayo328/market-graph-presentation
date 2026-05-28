"""
銘柄数頑健性: 段階縮小バックテスト (35/30/25/20/15 各 N_TRIALS trial)
================================================================

目的:
  Section 8.6 の event study で「20-30 銘柄なら頑健、10 以下ブレ大」
  という結果が出ている。本スクリプトは **バックテスト** (S1 戦略: e_div ≥ +0.8 short)
  でも同じ性質が成立するかを確認する。

手順:
  1. ohlc_40.parquet から N 個 (N = 35, 30, 25, 20, 15) を無作為サンプル
  2. seed = 42, 43, 44 の 3 trial を各サイズで実施
  3. 各 trial で γ 時系列を再計算 (window=30, threshold=0.3)
  4. 既存 backtest.py のロジックを再利用して S1_ediv_high_short を実行
  5. Sharpe / MaxDD / α (vs BH) を集計

出力:
  data/backtest_by_subsampling.json
    - results: list[dict]  各 (N, trial) の Sharpe, MaxDD, α など
    - summary: 各サイズの mean / std
    - baseline_40: 既存 backtest_results.json から流用 (reference)
    - meta: 設定

CLI:
  python backtest_by_subsampling.py            # 3 trial (デフォルト)
  python backtest_by_subsampling.py --trials 2 # 軽量化 (重い場合)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
LIB = HERE / "lib"
sys.path.insert(0, str(LIB))

from persistent_homology import persistence_diagram, persistence_summary  # noqa: E402
from market_category import MarketCategory  # noqa: E402
from homology import signed_cycle_balance  # noqa: E402

DATA_DIR = HERE.parent / "data"

# ===== パラメータ =====
WINDOW = 30
THRESHOLD = 0.3
ZSCORE_MIN_PERIODS = 30
EDIV_THRESHOLD = 0.8  # S1 戦略: e_div >= 0.8 で short (現金退避)
SIZES = [35, 30, 25, 20, 15]
DEFAULT_SEEDS = [42, 43, 44]
MIN_SYMBOLS_IN_WINDOW = 5
DAYS_PER_YEAR = 252


def compute_gamma_timeseries_for_symbols(
    closes: pd.DataFrame,
    symbols: list[str],
    window: int = WINDOW,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """指定 symbols について、closes 全期間で日次の L1_H1 / n_unb を計算.

    Returns:
        DataFrame with columns: date, n_symbols, L1_H1, n_unb
    """
    sub = closes[symbols].copy()
    returns = sub.pct_change()
    n = len(returns)

    rows: list[dict] = []
    for t_idx in range(window, n):
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
        date = returns.index[t_idx - 1]

        if win_clean.shape[1] < MIN_SYMBOLS_IN_WINDOW:
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan})
            continue

        try:
            corr = win_clean.corr()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diag = persistence_diagram(corr, max_dim=1)
            summ = persistence_summary(diag)
            L1 = float(summ["L1_norm_H1"])

            cat = MarketCategory(symbols=list(win_clean.columns),
                                 corr_matrix=corr, threshold=threshold)
            cat._build_graph()
            bal = signed_cycle_balance(cat.G)
            n_unb = int(bal["n_unbalanced"])

            rows.append({
                "date": date,
                "n_symbols": win_clean.shape[1],
                "L1_H1": L1,
                "n_unb": float(n_unb),
            })
        except Exception as e:
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan})
            print(f"  ! {date.date()} failed: {e}")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_zscores(gamma_df: pd.DataFrame,
                    min_periods: int = ZSCORE_MIN_PERIODS) -> pd.DataFrame:
    """L1_H1 / n_unb から expanding window で z_L1, z_unb, e_div を計算."""
    df = gamma_df.dropna(subset=["L1_H1", "n_unb"]).copy()
    df = df.set_index("date").sort_index()
    mp = min_periods
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


def s1_backtest(prices: pd.Series, indicators: pd.DataFrame) -> dict:
    """S1 戦略 (e_div >= +0.8, short = 現金退避) と Buy & Hold を比較."""
    # 共通日付
    common = prices.index.intersection(indicators.index)
    if len(common) < 30:
        raise ValueError(f"insufficient common dates: {len(common)}")
    px = prices.loc[common]
    ind = indicators.loc[common]

    # シグナル: e_div >= 0.8 → ON (現金), OFF → 株式保有
    sig = (ind["e_div"] >= EDIV_THRESHOLD).fillna(False)

    rets = px.pct_change().fillna(0)
    # short 方向: ON=現金 (リターン0), OFF=株式保有
    strat_rets = rets * (~sig).astype(float)

    eq = (1 + strat_rets).cumprod()
    bench = (1 + rets).cumprod()

    n_years = len(rets) / DAYS_PER_YEAR
    total_ret = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    vol_ann = float(strat_rets.std() * np.sqrt(DAYS_PER_YEAR))
    sharpe = float(cagr / vol_ann) if vol_ann > 1e-9 else 0
    peak = eq.cummax()
    max_dd = float((eq / peak - 1).min())

    # Buy & Hold
    bh_total_ret = float(bench.iloc[-1] - 1)
    bh_cagr = float(bench.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    bh_vol = float(rets.std() * np.sqrt(DAYS_PER_YEAR))
    bh_sharpe = float(bh_cagr / bh_vol) if bh_vol > 1e-9 else 0
    bh_peak = bench.cummax()
    bh_max_dd = float((bench / bh_peak - 1).min())

    alpha = total_ret - bh_total_ret  # 単純なリターン差

    return {
        "s1_total_return": total_ret,
        "s1_CAGR": cagr,
        "s1_sharpe": sharpe,
        "s1_max_drawdown": max_dd,
        "bh_total_return": bh_total_ret,
        "bh_sharpe": bh_sharpe,
        "bh_max_drawdown": bh_max_dd,
        "alpha_total_return": alpha,
        "n_signal_days": int(sig.sum()),
        "n_total_days": int(len(sig)),
        "signal_pct": float(sig.sum() / len(sig)) if len(sig) > 0 else 0.0,
    }


def fetch_spy_prices(start: str, end: str) -> pd.Series:
    print(f"Fetching ^GSPC from {start} to {end} ...")
    df = yf.download("^GSPC", start=start, end=end, progress=False,
                     auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].squeeze()
    else:
        close = df["Close"]
    close.index = pd.to_datetime(close.index)
    return close.dropna()


def summarize_metric(values: list[float]) -> dict:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3,
                        help="trial 数 (デフォルト 3, 軽量化なら 2)")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES,
                        help="銘柄数リスト")
    args = parser.parse_args()

    n_trials = max(1, args.trials)
    seeds = DEFAULT_SEEDS[:n_trials]
    sizes = args.sizes

    print("=== Backtest by Subsampling (S1: e_div >= +0.8 short) ===")
    print(f"sizes={sizes}, seeds={seeds}, trials_per_size={n_trials}")

    # 入力
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    all_symbols = list(closes.columns)
    print(f"Loaded ohlc_40.parquet: {closes.shape}, "
          f"{closes.index.min().date()} -> {closes.index.max().date()}")

    # SPY 価格 (一度だけ取得)
    start = closes.index.min().strftime("%Y-%m-%d")
    end = (closes.index.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    prices = fetch_spy_prices(start, end)
    print(f"SPY prices: {len(prices)} days")

    # baseline (40 銘柄): 既存 backtest_results.json から流用
    try:
        bl = json.loads((DATA_DIR / "backtest_results.json").read_text(encoding="utf-8"))
        s1_bl = bl["summary"]["S1_ediv_high_short"]
        bh_bl = bl["summary"]["Z_buy_and_hold"]
        baseline_40 = {
            "s1_sharpe": s1_bl["sharpe"],
            "s1_max_drawdown": s1_bl["max_drawdown"],
            "s1_total_return": s1_bl["total_return"],
            "bh_sharpe": bh_bl["sharpe"],
            "bh_total_return": bh_bl["total_return"],
            "alpha_total_return": s1_bl["total_return"] - bh_bl["total_return"],
        }
        print(f"\n[baseline 40] sharpe={baseline_40['s1_sharpe']:+.3f}  "
              f"MaxDD={baseline_40['s1_max_drawdown']*100:+.2f}%  "
              f"alpha={baseline_40['alpha_total_return']*100:+.2f}%")
    except Exception as e:
        print(f"[baseline 40] load failed: {e}")
        baseline_40 = None

    # ===== 各サイズ × trial =====
    results: list[dict] = []
    summary_by_size: dict[int, dict] = {}

    t_global = time.time()
    for N in sizes:
        print(f"\n----- N = {N} -----")
        sharpes = []
        maxdds = []
        alphas = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            kept = sorted(rng.choice(all_symbols, size=N, replace=False).tolist())
            t0 = time.time()
            print(f"  [seed={seed}] computing gamma for {N} symbols ...", flush=True)
            gamma_df = compute_gamma_timeseries_for_symbols(closes, kept)
            ind = compute_zscores(gamma_df)
            res = s1_backtest(prices, ind)
            elapsed = time.time() - t0
            print(f"    done in {elapsed:.0f}s  Sharpe={res['s1_sharpe']:+.3f}  "
                  f"MaxDD={res['s1_max_drawdown']*100:+.2f}%  "
                  f"alpha={res['alpha_total_return']*100:+.2f}%  "
                  f"sig_pct={res['signal_pct']*100:.1f}%")
            row = {
                "N": N,
                "seed": int(seed),
                "kept_symbols": kept,
                "elapsed_sec": elapsed,
                **res,
            }
            results.append(row)
            sharpes.append(res["s1_sharpe"])
            maxdds.append(res["s1_max_drawdown"])
            alphas.append(res["alpha_total_return"])

        summary_by_size[N] = {
            "sharpe": summarize_metric(sharpes),
            "max_drawdown": summarize_metric(maxdds),
            "alpha_total_return": summarize_metric(alphas),
        }
        s = summary_by_size[N]
        print(f"  -> N={N} summary  Sharpe {s['sharpe']['mean']:+.3f} "
              f"± {s['sharpe']['std']:.3f}  |  "
              f"MaxDD {s['max_drawdown']['mean']*100:+.2f}% "
              f"± {s['max_drawdown']['std']*100:.2f}%  |  "
              f"α {s['alpha_total_return']['mean']*100:+.2f}% "
              f"± {s['alpha_total_return']['std']*100:.2f}%")

    total_elapsed = time.time() - t_global
    print(f"\nTotal elapsed: {total_elapsed:.0f}s "
          f"({len(results)} backtests)")

    # ===== 出力 =====
    out = {
        "meta": {
            "window": WINDOW,
            "threshold": THRESHOLD,
            "zscore_min_periods": ZSCORE_MIN_PERIODS,
            "ediv_threshold": EDIV_THRESHOLD,
            "sizes": sizes,
            "seeds": seeds,
            "n_trials_per_size": n_trials,
            "data_range": [
                closes.index.min().strftime("%Y-%m-%d"),
                closes.index.max().strftime("%Y-%m-%d"),
            ],
            "benchmark": "^GSPC (S&P500) Buy & Hold",
            "total_elapsed_sec": total_elapsed,
        },
        "baseline_40": baseline_40,
        "summary_by_size": {str(N): v for N, v in summary_by_size.items()},
        "results": [
            {k: v for k, v in r.items() if k != "kept_symbols"}
            | {"kept_symbols": r["kept_symbols"]}
            for r in results
        ],
    }
    out_path = DATA_DIR / "backtest_by_subsampling.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                   default=str),
                        encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # ===== 最終テーブル =====
    print("\n=== Summary table (mean ± std across trials) ===")
    print(f"{'N':>4}  {'Sharpe':>16}  {'MaxDD':>16}  {'α (vs BH)':>16}")
    if baseline_40 is not None:
        print(f"{'40':>4}  "
              f"{baseline_40['s1_sharpe']:+.3f}{'':>10}  "
              f"{baseline_40['s1_max_drawdown']*100:+.2f}%{'':>9}  "
              f"{baseline_40['alpha_total_return']*100:+.2f}%{'':>9}  (baseline)")
    for N in sizes:
        s = summary_by_size[N]
        print(f"{N:>4}  "
              f"{s['sharpe']['mean']:+.3f} ± {s['sharpe']['std']:.3f}    "
              f"{s['max_drawdown']['mean']*100:+.2f}% ± {s['max_drawdown']['std']*100:.2f}%  "
              f"{s['alpha_total_return']['mean']*100:+.2f}% ± {s['alpha_total_return']['std']*100:.2f}%")


if __name__ == "__main__":
    main()
