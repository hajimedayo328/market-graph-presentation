"""
バックテスト: γ指標シグナルベース戦略 vs Buy & Hold.

ベンチマーク: S&P500 (^GSPC) の単純 Buy & Hold

戦略一覧 (各 short / long の 2 方向):
  S1: e_div ≥ +0.8 (政策ショック検知でリスクオフ)
  S2: e_div ≤ -0.5 (強さ変化検知でリスクオフ)
  S3: |z_L1|  ≥ 1.0 (L¹ シグナル)
  S4: |z_unb| ≥ 1.0 (n_unb シグナル)
  S5: z_L1 ≥ 1.0 OR z_unb ≥ 1.0
  S6: z_L1 ≥ 1.0 AND z_unb ≥ 1.0

short = シグナル発火日に株式売却 (現金 100%)、解除で買い戻し
long  = シグナル発火日に株式買い増し、解除で平常化

Limitation:
  - 取引コスト 0% (実際は 0.05-0.1% かかる)
  - スリッページなし
  - シグナル当日終値で売買 (実際は翌営業日始値が現実的)
  - look-ahead bias: 指標は当日 close まで使うが取引も同日
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"


def load_indicators(zscore_min_periods: int = 30) -> pd.DataFrame:
    """指標時系列をロードし、z-score と e_div を計算.

    z-score は過去のみの expanding window で計算し look-ahead bias を完全排除.
    """
    df = pd.read_csv(DATA_DIR / "gamma_timeseries_w30.csv", parse_dates=["date"])
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


def fetch_spy_prices(start: str, end: str) -> pd.Series:
    """S&P500 の Close 取得."""
    print(f"Fetching ^GSPC from {start} to {end}...")
    df = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].squeeze()
    else:
        close = df["Close"]
    close.index = pd.to_datetime(close.index)
    return close.dropna()


# シグナル生成関数
SIGNALS = {
    "S1_ediv_high":     lambda d: d["e_div"] >= +0.8,
    "S2_ediv_low":      lambda d: d["e_div"] <= -0.5,
    "S3_abs_zL1":       lambda d: d["z_L1"].abs() >= 1.0,
    "S4_abs_zunb":      lambda d: d["z_unb"].abs() >= 1.0,
    "S5_zL1_OR_zunb":   lambda d: (d["z_L1"] >= 1.0) | (d["z_unb"] >= 1.0),
    "S6_zL1_AND_zunb":  lambda d: (d["z_L1"] >= 1.0) & (d["z_unb"] >= 1.0),
}


def simulate_strategy(prices: pd.Series, signal: pd.Series,
                       direction: str = "short") -> dict:
    """
    direction='short': シグナル ON で現金 (株式売却), OFF で買い戻し
    direction='long':  シグナル ON で 1.5x 株式 (借入なしなら 1x), OFF で 1x
                       簡略化: ON=1.0 (no change), OFF=1.0, ベンチと同じ
                       → ロング版は「シグナル ON で買い、OFF で売り」逆張り想定:
                       ON=1.0 株式, OFF=現金 (順張り長期持ち戦略の逆)
                       わかりやすさ重視で「OFF で現金に逃げる、ON で買い」のパターンも
                       実装したいが、今回は短期版とロング版を同じ「ON=保有」
                       で逆ロジックとして実装
    """
    # signal を prices に揃える
    sig = signal.reindex(prices.index, method="ffill").fillna(False)
    rets = prices.pct_change().fillna(0)
    if direction == "short":
        # ON=現金 (リターン0), OFF=株式保有 (リターン=日次)
        strat_rets = rets * (~sig).astype(float)
    elif direction == "long":
        # ON=株式保有 (rets), OFF=現金
        strat_rets = rets * sig.astype(float)
    else:
        raise ValueError(direction)

    eq = (1 + strat_rets).cumprod()
    bench = (1 + rets).cumprod()

    # 指標
    days_per_year = 252
    n_years = len(rets) / days_per_year
    total_ret = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    vol_ann = float(strat_rets.std() * np.sqrt(days_per_year))
    sharpe = float(cagr / vol_ann) if vol_ann > 1e-9 else 0
    # max drawdown
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    max_dd = float(dd)
    # 勝率
    n_signal_days = int(sig.sum())
    days_in_market = int((~sig).sum() if direction == "short" else sig.sum())
    # 戦略リターン vs Buy&Hold で勝った日数 (cumulative ベース)
    # 「ベンチに勝った日」の割合 (簡単版)
    win_days = int((strat_rets > rets).sum())
    win_rate = float(win_days / len(rets)) if len(rets) > 0 else 0

    return {
        "direction": direction,
        "total_return": total_ret,
        "CAGR": cagr,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_signal_days": n_signal_days,
        "n_total_days": int(len(sig)),
        "signal_pct": float(n_signal_days / len(sig)) if len(sig) > 0 else 0,
        "win_rate_vs_bh": win_rate,
        "equity_curve": eq.round(4).tolist(),
        "dates": eq.index.strftime("%Y-%m-%d").tolist(),
    }


def main():
    df = load_indicators()
    print(f"Loaded indicators: {df.shape}, {df.index.min().date()} -> {df.index.max().date()}")

    start = df.index.min().strftime("%Y-%m-%d")
    end = (df.index.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    prices = fetch_spy_prices(start, end)
    print(f"SPY prices: {len(prices)} days")

    # 共通の期間で揃える
    common = prices.index.intersection(df.index)
    df = df.loc[common]
    prices = prices.loc[common]

    # ベンチマーク
    bh_rets = prices.pct_change().fillna(0)
    bh_eq = (1 + bh_rets).cumprod()
    print(f"\nBuy & Hold: 最終リターン = {(bh_eq.iloc[-1] - 1) * 100:.2f}%")

    # 全戦略
    results = {}
    for name, gen in SIGNALS.items():
        sig = gen(df)
        for direction in ["short", "long"]:
            key = f"{name}_{direction}"
            r = simulate_strategy(prices, sig, direction)
            results[key] = r
            print(f"  {key:<24} ret={r['total_return']*100:>+7.2f}%  "
                  f"sharpe={r['sharpe']:>+5.2f}  maxDD={r['max_drawdown']*100:>+6.2f}%  "
                  f"sig_days={r['n_signal_days']:>4} ({r['signal_pct']*100:.1f}%)")

    # Buy & Hold もエントリ追加
    bh_strat_rets = bh_rets.copy()
    n_years = len(bh_rets) / 252
    total_ret_bh = float(bh_eq.iloc[-1] - 1)
    cagr_bh = float(bh_eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    vol_bh = float(bh_strat_rets.std() * np.sqrt(252))
    sharpe_bh = cagr_bh / vol_bh if vol_bh > 1e-9 else 0
    peak_bh = bh_eq.cummax()
    dd_bh = float((bh_eq / peak_bh - 1).min())
    results["Z_buy_and_hold"] = {
        "direction": "bh", "total_return": total_ret_bh, "CAGR": cagr_bh,
        "vol_ann": vol_bh, "sharpe": sharpe_bh, "max_drawdown": dd_bh,
        "n_signal_days": 0, "n_total_days": int(len(bh_eq)),
        "signal_pct": 0, "win_rate_vs_bh": 0.5,
        "equity_curve": bh_eq.round(4).tolist(),
        "dates": bh_eq.index.strftime("%Y-%m-%d").tolist(),
    }
    print(f"\n  Z_buy_and_hold           ret={total_ret_bh*100:>+7.2f}%  "
          f"sharpe={sharpe_bh:>+5.2f}  maxDD={dd_bh*100:>+6.2f}%")

    # JSON 出力 (equity_curve は別ファイル)
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("equity_curve", "dates")}
                for k, v in results.items()}
    # 共通日付 (Buy & Hold と同じ)
    common_dates = results["Z_buy_and_hold"]["dates"]
    equity_curves = {k: v["equity_curve"] for k, v in results.items()}

    out = {
        "as_of": str(df.index.max().date()),
        "common_dates": common_dates,
        "summary": summary,
        "equity_curves": equity_curves,
        "benchmark": "^GSPC (S&P500) Buy & Hold",
    }
    out_path = DATA_DIR / "backtest_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # ランキング
    print(f"\n=== Sharpe ランキング (上位 5) ===")
    ranked = sorted(summary.items(), key=lambda x: -x[1]["sharpe"])
    for k, v in ranked[:5]:
        print(f"  {k:<24} Sharpe={v['sharpe']:>+5.2f}  "
              f"Ret={v['total_return']*100:>+7.2f}%  MaxDD={v['max_drawdown']*100:>+6.2f}%")

    print(f"\n=== トータルリターン ランキング (上位 5) ===")
    ranked = sorted(summary.items(), key=lambda x: -x[1]["total_return"])
    for k, v in ranked[:5]:
        print(f"  {k:<24} Ret={v['total_return']*100:>+7.2f}%  "
              f"Sharpe={v['sharpe']:>+5.2f}  MaxDD={v['max_drawdown']*100:>+6.2f}%")


if __name__ == "__main__":
    main()
