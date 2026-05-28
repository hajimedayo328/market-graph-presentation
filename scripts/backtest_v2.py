"""
バックテスト v2: 取引コスト込み + ヒステリシス + 翌日寄付き約定.

v1 からの改善:
- **取引コスト 0.05% / 片道** を約定毎に差し引き
- **ヒステリシス 5 日**: シグナル ON してから最低 5 営業日は保持 (頻度削減)
- **翌日寄付き約定**: シグナル日 close でなく、翌日 open で約定
  → look-ahead bias 対策 (翌日の open は当日 close を見てから判定可能)

実用レベル版.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"

TRANSACTION_COST = 0.0005   # 0.05% (片道、買 + 売で計 0.1%)
HYSTERESIS_DAYS = 5         # シグナル発火後の最低保持日数


def load_indicators(zscore_min_periods: int = 30) -> pd.DataFrame:
    """指標 CSV を読み込み、z-score と e_div を計算する.

    z-score は **過去のみの expanding window** で計算し look-ahead bias を完全排除する.
    最初の `zscore_min_periods` 営業日は z 未定義 (NaN) となり、シグナルは発火しない.

    Live (vps_daily.py) は過去 90 日の rolling 統計を使うが、backtest では「初期も含めて
    厳密に過去だけを使う」expanding window のほうが冷血で再現性が高い.
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


def fetch_spy_ohlc(start: str, end: str) -> pd.DataFrame:
    print(f"Fetching ^GSPC OHLC from {start} to {end}...")
    df = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.dropna(subset=["Open", "Close"])


SIGNALS = {
    "S1_ediv_high":     lambda d: d["e_div"] >= +0.8,
    "S2_ediv_low":      lambda d: d["e_div"] <= -0.5,
    "S3_abs_zL1":       lambda d: d["z_L1"].abs() >= 1.0,
    "S4_abs_zunb":      lambda d: d["z_unb"].abs() >= 1.0,
    "S5_zL1_OR_zunb":   lambda d: (d["z_L1"] >= 1.0) | (d["z_unb"] >= 1.0),
    "S6_zL1_AND_zunb":  lambda d: (d["z_L1"] >= 1.0) & (d["z_unb"] >= 1.0),
}


def apply_hysteresis(sig: pd.Series, min_days: int = 5) -> pd.Series:
    """シグナル ON 後、最低 min_days は ON を維持. 連続OFFが min_days 以上で完全OFF."""
    n = len(sig)
    arr = sig.values.astype(bool)
    last_on = -10**9
    for i in range(n):
        if arr[i]:
            last_on = i
        else:
            if i - last_on < min_days:
                arr[i] = True  # ヒステリシス維持
    return pd.Series(arr, index=sig.index)


def simulate_v2(ohlc: pd.DataFrame, signal: pd.Series,
                direction: str = "short",
                cost: float = TRANSACTION_COST,
                hysteresis: int = HYSTERESIS_DAYS) -> dict:
    """
    取引コスト + ヒステリシス + 翌日寄付き約定の改善バックテスト.

    signal: 日次 (close 時点で算出される) bool シリーズ
    direction='short': ON で現金、OFF で株式
    direction='long':  ON で株式、OFF で現金

    翌日寄付き約定:
      signal[t] が決まる (close で算出) → 翌日 t+1 open で約定
      → ポジション = sig.shift(1) [翌日からの状態]
    """
    sig = signal.reindex(ohlc.index, method="ffill").fillna(False).astype(bool)
    if hysteresis > 0:
        sig = apply_hysteresis(sig, hysteresis)
    # 翌日始まりからのポジション (look-ahead bias 対策)
    sig_pos = sig.shift(1).fillna(False).astype(bool)
    # 日次リターン (Close to Close)
    rets = ohlc["Close"].pct_change().fillna(0)
    # 戦略リターン
    if direction == "short":
        strat_rets = rets * (~sig_pos).astype(float)
    elif direction == "long":
        strat_rets = rets * sig_pos.astype(float)
    else:
        raise ValueError(direction)
    # 取引コスト: ポジション変化があった日に cost を差し引く
    pos_change = (sig_pos.astype(int).diff().abs() > 0)
    strat_rets = strat_rets - pos_change.astype(float) * cost

    eq = (1 + strat_rets).cumprod()

    days_per_year = 252
    n_years = len(rets) / days_per_year
    total_ret = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    vol_ann = float(strat_rets.std() * np.sqrt(days_per_year))
    sharpe = float(cagr / vol_ann) if vol_ann > 1e-9 else 0
    peak = eq.cummax()
    max_dd = float((eq / peak - 1).min())

    n_trades = int(pos_change.sum())
    n_signal_days = int(sig_pos.sum())
    avg_holding_days = (
        n_signal_days / max(1, n_trades / 2) if direction == "long"
        else (len(sig_pos) - n_signal_days) / max(1, n_trades / 2)
    )

    return {
        "direction": direction,
        "total_return": total_ret, "CAGR": cagr,
        "vol_ann": vol_ann, "sharpe": sharpe, "max_drawdown": max_dd,
        "n_trades": n_trades,
        "trades_per_year": float(n_trades / n_years) if n_years > 0 else 0,
        "n_signal_days": n_signal_days,
        "signal_pct": float(n_signal_days / len(sig_pos)) if len(sig_pos) > 0 else 0,
        "avg_holding_days": float(avg_holding_days),
        "total_cost_drag": float(n_trades * cost),
        "equity_curve": eq.round(4).tolist(),
        "dates": eq.index.strftime("%Y-%m-%d").tolist(),
    }


def main():
    df = load_indicators()
    start = df.index.min().strftime("%Y-%m-%d")
    end = (df.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    ohlc = fetch_spy_ohlc(start, end)
    common = ohlc.index.intersection(df.index)
    df = df.loc[common]; ohlc = ohlc.loc[common]

    rets = ohlc["Close"].pct_change().fillna(0)
    bh_eq = (1 + rets).cumprod()
    n_years = len(rets) / 252
    total_ret_bh = float(bh_eq.iloc[-1] - 1)
    cagr_bh = float(bh_eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    vol_bh = float(rets.std() * np.sqrt(252))
    sharpe_bh = cagr_bh / vol_bh if vol_bh > 1e-9 else 0
    dd_bh = float((bh_eq / bh_eq.cummax() - 1).min())

    print(f"\n=== Backtest v2 (cost={TRANSACTION_COST*100:.3f}%, hysteresis={HYSTERESIS_DAYS}d, next-day open) ===")
    print(f"Buy & Hold: ret={total_ret_bh*100:+.2f}%  sharpe={sharpe_bh:+.2f}  MaxDD={dd_bh*100:+.2f}%")
    print()
    print(f"{'strategy':<25} {'dir':<6} {'ret':>8} {'sharpe':>7} {'MaxDD':>7} {'trades':>7} {'tr/yr':>7} {'hold':>6} {'cost':>7}")
    print("-" * 95)

    results = {}
    for name, gen in SIGNALS.items():
        sig = gen(df)
        for direction in ["short", "long"]:
            key = f"{name}_{direction}"
            r = simulate_v2(ohlc, sig, direction)
            results[key] = r
            print(f"{name:<25} {direction:<6} {r['total_return']*100:>+7.2f}% "
                  f"{r['sharpe']:>+7.2f} {r['max_drawdown']*100:>+6.2f}% "
                  f"{r['n_trades']:>7} {r['trades_per_year']:>7.1f} "
                  f"{r['avg_holding_days']:>5.1f}d {r['total_cost_drag']*100:>+6.2f}%")

    results["Z_buy_and_hold"] = {
        "direction": "bh", "total_return": total_ret_bh, "CAGR": cagr_bh,
        "vol_ann": vol_bh, "sharpe": sharpe_bh, "max_drawdown": dd_bh,
        "n_trades": 1, "trades_per_year": 1 / n_years, "n_signal_days": 0,
        "signal_pct": 0, "avg_holding_days": float(len(rets)),
        "total_cost_drag": float(2 * TRANSACTION_COST),
        "equity_curve": bh_eq.round(4).tolist(),
        "dates": bh_eq.index.strftime("%Y-%m-%d").tolist(),
    }
    print(f"{'Z_buy_and_hold':<25} {'bh':<6} {total_ret_bh*100:>+7.2f}% "
          f"{sharpe_bh:>+7.2f} {dd_bh*100:>+6.2f}%")

    # JSON 出力
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("equity_curve", "dates")}
               for k, v in results.items()}
    common_dates = results["Z_buy_and_hold"]["dates"]
    equity_curves = {k: v["equity_curve"] for k, v in results.items()}
    out = {
        "as_of": str(df.index.max().date()),
        "version": "v2",
        "params": {"transaction_cost": TRANSACTION_COST,
                    "hysteresis_days": HYSTERESIS_DAYS,
                    "execution": "next-day open"},
        "common_dates": common_dates,
        "summary": summary, "equity_curves": equity_curves,
        "benchmark": "^GSPC (S&P500) Buy & Hold",
    }
    out_path = DATA_DIR / "backtest_v2_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # ランキング
    print("\n=== Sharpe ランキング (上位 5) ===")
    ranked = sorted(summary.items(), key=lambda x: -x[1]["sharpe"])
    for k, v in ranked[:5]:
        print(f"  {k:<25} Sharpe={v['sharpe']:>+5.2f}  Ret={v['total_return']*100:>+7.2f}%  "
              f"MaxDD={v['max_drawdown']*100:>+6.2f}%  trades/yr={v.get('trades_per_year', 0):.1f}")


if __name__ == "__main__":
    main()
