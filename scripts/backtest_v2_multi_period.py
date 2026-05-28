"""
バックテスト v2 を 4 期間 (5y / 10y / 15y / 20y) で in-sample 実行する.

- ロジックは scripts/backtest_v2.py と同一:
  - 取引コスト 0.05% / 片道 (TRANSACTION_COST)
  - ヒステリシス 5 営業日 (HYSTERESIS_DAYS)
  - 翌日寄付き約定 (look-ahead bias 対策で sig.shift(1))
  - z-score は **expanding window (min_periods=30)** で過去のみを使用
- 各期間ごとに ^GSPC OHLC を yfinance から取得 (期間に合わせた start-end)
- 全戦略 (S1 / S2 / S6) を short/long 両方向で計算 (= 6 戦略 + B&H)
- 結果は data/backtest_v2_multi_period.json に出力

注意:
- backtest_v2.py は壊さない (本ファイルは別実装)
- 個人名 / 学会名は出力に含めない
- 出力 JSON は equity_curve / dates を保持 (要約と再現の両用)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"

TRANSACTION_COST = 0.0005   # 0.05% / 片道
HYSTERESIS_DAYS = 5         # 最低保持営業日

# 期間ごとの入力 CSV
PERIOD_FILES: dict[str, str] = {
    "5y":  "gamma_timeseries_w30.csv",
    "10y": "gamma_timeseries_10y_w30.csv",
    "15y": "gamma_timeseries_15y_w30.csv",
    "20y": "gamma_timeseries_20y_w30.csv",
}

# 評価対象戦略 (S1 / S2 / S6)
SIGNALS = {
    "S1_ediv_high":    lambda d: d["e_div"] >= +0.8,
    "S2_ediv_low":     lambda d: d["e_div"] <= -0.5,
    "S6_zL1_AND_zunb": lambda d: (d["z_L1"] >= 1.0) & (d["z_unb"] >= 1.0),
}


def load_indicators(csv_name: str, zscore_min_periods: int = 30) -> pd.DataFrame:
    """指標 CSV を読み込み、expanding window で z-score / e_div を計算."""
    df = pd.read_csv(DATA_DIR / csv_name, parse_dates=["date"])
    df = df.dropna(subset=["L1_H1", "n_unb"]).set_index("date").sort_index()
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
    print(f"  Fetching ^GSPC OHLC: {start} -> {end} ...")
    df = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.dropna(subset=["Open", "Close"])


def apply_hysteresis(sig: pd.Series, min_days: int = 5) -> pd.Series:
    out = sig.copy()
    arr = sig.values.astype(bool)
    last_on = -10**9
    for i in range(len(arr)):
        if arr[i]:
            last_on = i
        elif i - last_on < min_days:
            arr[i] = True
    return pd.Series(arr, index=sig.index)


def simulate_v2(
    ohlc: pd.DataFrame,
    signal: pd.Series,
    direction: str = "short",
    cost: float = TRANSACTION_COST,
    hysteresis: int = HYSTERESIS_DAYS,
) -> dict:
    """backtest_v2.simulate_v2 と完全同一ロジック (移植)."""
    sig = signal.reindex(ohlc.index, method="ffill").fillna(False).astype(bool)
    if hysteresis > 0:
        sig = apply_hysteresis(sig, hysteresis)
    sig_pos = sig.shift(1).fillna(False).astype(bool)
    rets = ohlc["Close"].pct_change().fillna(0)
    if direction == "short":
        strat_rets = rets * (~sig_pos).astype(float)
    elif direction == "long":
        strat_rets = rets * sig_pos.astype(float)
    else:
        raise ValueError(direction)
    pos_change = (sig_pos.astype(int).diff().abs() > 0)
    strat_rets = strat_rets - pos_change.astype(float) * cost

    eq = (1 + strat_rets).cumprod()

    days_per_year = 252
    n_years = len(rets) / days_per_year
    total_ret = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol_ann = float(strat_rets.std() * np.sqrt(days_per_year))
    sharpe = float(cagr / vol_ann) if vol_ann > 1e-9 else 0.0
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
        "total_return": total_ret,
        "CAGR": cagr,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "trades_per_year": float(n_trades / n_years) if n_years > 0 else 0.0,
        "n_signal_days": n_signal_days,
        "signal_pct": float(n_signal_days / len(sig_pos)) if len(sig_pos) > 0 else 0.0,
        "avg_holding_days": float(avg_holding_days),
        "total_cost_drag": float(n_trades * cost),
        "equity_curve": eq.round(4).tolist(),
        "dates": eq.index.strftime("%Y-%m-%d").tolist(),
    }


def run_period(period_label: str, csv_name: str) -> dict:
    """1 期間分のバックテストを実行し、要約 dict を返す."""
    print(f"\n[{period_label}] indicator: {csv_name}")
    df = load_indicators(csv_name)
    start = df.index.min().strftime("%Y-%m-%d")
    end = (df.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    ohlc = fetch_spy_ohlc(start, end)
    common = ohlc.index.intersection(df.index)
    df = df.loc[common]
    ohlc = ohlc.loc[common]
    print(f"  effective days: {len(common)}  range: {df.index.min().date()} -> {df.index.max().date()}")

    # Buy & Hold benchmark
    rets = ohlc["Close"].pct_change().fillna(0)
    bh_eq = (1 + rets).cumprod()
    n_years = len(rets) / 252
    total_ret_bh = float(bh_eq.iloc[-1] - 1)
    cagr_bh = float(bh_eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol_bh = float(rets.std() * np.sqrt(252))
    sharpe_bh = cagr_bh / vol_bh if vol_bh > 1e-9 else 0.0
    dd_bh = float((bh_eq / bh_eq.cummax() - 1).min())

    strategies: dict[str, dict] = {}
    for name, gen in SIGNALS.items():
        sig = gen(df)
        for direction in ("short", "long"):
            key = f"{name}_{direction}"
            r = simulate_v2(ohlc, sig, direction)
            # 容量削減: 期間別 JSON では equity_curve を捨てて要約のみ
            r_summary = {k: v for k, v in r.items() if k not in ("equity_curve", "dates")}
            strategies[key] = r_summary
            print(
                f"  {key:<28} ret={r['total_return']*100:+7.2f}%  "
                f"sharpe={r['sharpe']:+5.2f}  MaxDD={r['max_drawdown']*100:+6.2f}%  "
                f"trades/yr={r['trades_per_year']:5.1f}"
            )

    bh_summary = {
        "direction": "bh",
        "total_return": total_ret_bh,
        "CAGR": cagr_bh,
        "vol_ann": vol_bh,
        "sharpe": sharpe_bh,
        "max_drawdown": dd_bh,
        "n_trades": 1,
        "trades_per_year": float(1 / n_years) if n_years > 0 else 0.0,
        "n_signal_days": 0,
        "signal_pct": 0.0,
        "avg_holding_days": float(len(rets)),
        "total_cost_drag": float(2 * TRANSACTION_COST),
    }
    strategies["Z_buy_and_hold"] = bh_summary
    print(
        f"  Z_buy_and_hold              ret={total_ret_bh*100:+7.2f}%  "
        f"sharpe={sharpe_bh:+5.2f}  MaxDD={dd_bh*100:+6.2f}%"
    )

    return {
        "period": period_label,
        "csv": csv_name,
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "n_days": int(len(common)),
        "n_years": float(n_years),
        "strategies": strategies,
    }


def main() -> None:
    print("=" * 72)
    print("Backtest v2 (multi-period): 5y / 10y / 15y / 20y in-sample")
    print(f"cost={TRANSACTION_COST*100:.3f}%/leg, hysteresis={HYSTERESIS_DAYS}d, "
          f"execution=next-day open, z-score=expanding(min_periods=30)")
    print("=" * 72)

    results: dict[str, dict] = {}
    for period, csv_name in PERIOD_FILES.items():
        if not (DATA_DIR / csv_name).exists():
            print(f"  SKIP {period}: missing {csv_name}", file=sys.stderr)
            continue
        results[period] = run_period(period, csv_name)

    out = {
        "version": "v2_multi_period",
        "params": {
            "transaction_cost": TRANSACTION_COST,
            "hysteresis_days": HYSTERESIS_DAYS,
            "execution": "next-day open",
            "zscore": "expanding(min_periods=30)",
        },
        "benchmark": "^GSPC (S&P500) Buy & Hold",
        "periods": results,
    }
    out_path = DATA_DIR / "backtest_v2_multi_period.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # 比較テーブル
    print("\n" + "=" * 88)
    print("Period comparison (S1_ediv_high_short vs Buy & Hold)")
    print("=" * 88)
    print(f"{'period':<6} {'n_years':>8} {'BH_Sharpe':>10} {'BH_MaxDD':>10} "
          f"{'S1_Sharpe':>10} {'S1_MaxDD':>10} {'S1_alpha':>10}")
    for p, r in results.items():
        s1 = r["strategies"].get("S1_ediv_high_short", {})
        bh = r["strategies"]["Z_buy_and_hold"]
        s1_sh = s1.get("sharpe", float("nan"))
        bh_sh = bh.get("sharpe", float("nan"))
        s1_dd = s1.get("max_drawdown", float("nan"))
        bh_dd = bh.get("max_drawdown", float("nan"))
        alpha = s1.get("total_return", 0) - bh.get("total_return", 0)
        print(f"{p:<6} {r['n_years']:>8.2f} {bh_sh:>+10.2f} {bh_dd*100:>+9.2f}% "
              f"{s1_sh:>+10.2f} {s1_dd*100:>+9.2f}% {alpha*100:>+9.2f}%")


if __name__ == "__main__":
    main()
