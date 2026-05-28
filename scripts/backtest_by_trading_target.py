"""
売買対象別 S1 バックテスト
==========================

背景:
  発表で「他の銘柄でも試した?」「なぜ S&P 500 を売買対象に選んだか?」と
  問われたときに、実証で先回り反論するためのスクリプト。

  既存 backtest_v2.py は売買対象を **^GSPC (S&P 500) だけ** に固定している。
  本スクリプトは **観測 40 銘柄 (= e_div シグナル)** をそのまま使い、
  **売買対象だけを 11 種類** に振り替えて S1_ediv_high_short 戦略
  (e_div >= 0.8 で売り、それ以外保有) を比較する。

  e_div シグナルは look-ahead bias を回避済の data/gamma_timeseries_w30.csv
  をそのまま使用 (expanding window z-score)。

対象 (11):
  S&P 500 (^GSPC)             米国大型 (baseline)
  NASDAQ Composite (^IXIC)    米国テック
  Dow Jones (^DJI)            米国大型 (別指数)
  Russell 2000 (^RUT)         米国小型
  Nikkei 225 (^N225)          日本
  FTSE 100 (^FTSE)            英国
  DAX (^GDAXI)                ドイツ
  CHINA50 (FXI)               中国 (ETF)
  AAPL (AAPL)                 個別株
  Gold (GC=F)                 コモディティ
  BTC (BTC-USD)               暗号

  fetch 失敗 (delisted 等) は skip 記録。

出力:
  data/backtest_by_trading_target.json
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

# ===== パラメータ (backtest_v2 と統一) =====
ZSCORE_MIN_PERIODS = 30
EDIV_THRESHOLD = 0.8
TRANSACTION_COST = 0.0005
HYSTERESIS_DAYS = 5

# (target_id, label, yticker, category)
TARGETS: list[tuple[str, str, str, str]] = [
    ("SP500",   "S&P 500",         "^GSPC",   "米国大型 (baseline)"),
    ("NAS100",  "NASDAQ Composite","^IXIC",   "米国テック"),
    ("DJ30",    "Dow Jones",       "^DJI",    "米国大型 (別指数)"),
    ("RUS2000", "Russell 2000",    "^RUT",    "米国小型"),
    ("JP225",   "Nikkei 225",      "^N225",   "日本"),
    ("UK100",   "FTSE 100",        "^FTSE",   "英国"),
    ("GER40",   "DAX",             "^GDAXI",  "ドイツ"),
    ("CHINA50", "China Large-Cap", "FXI",     "中国 (ETF)"),
    ("AAPL",    "Apple",           "AAPL",    "個別株 (US)"),
    ("GOLD",    "Gold Futures",    "GC=F",    "コモディティ (金)"),
    ("BTC",     "Bitcoin",         "BTC-USD", "暗号"),
]


def load_indicators(min_periods: int = ZSCORE_MIN_PERIODS) -> pd.DataFrame:
    """既存 gamma_timeseries_w30.csv を読み、expanding window z-score を計算."""
    df = pd.read_csv(DATA_DIR / "gamma_timeseries_w30.csv", parse_dates=["date"])
    df = df.dropna(subset=["L1_H1", "n_unb"]).set_index("date")
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


def fetch_close(ticker: str, start: str, end: str) -> pd.Series | None:
    """yfinance Close 時系列取得 (失敗時 None)."""
    try:
        df = yf.download(ticker, start=start, end=end,
                          progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"].squeeze()
        else:
            close = df["Close"]
        close.index = pd.to_datetime(close.index)
        close = close.dropna()
        if len(close) == 0:
            return None
        return close
    except Exception as e:
        print(f"  ! fetch {ticker} failed: {e}")
        return None


def apply_hysteresis(sig: pd.Series, min_days: int = HYSTERESIS_DAYS) -> pd.Series:
    arr = sig.values.astype(bool).copy()
    last_on = -10**9
    for i in range(len(arr)):
        if arr[i]:
            last_on = i
        else:
            if i - last_on < min_days:
                arr[i] = True
    return pd.Series(arr, index=sig.index)


def simulate_s1(close: pd.Series, e_div: pd.Series,
                threshold: float = EDIV_THRESHOLD,
                cost: float = TRANSACTION_COST,
                hysteresis: int = HYSTERESIS_DAYS) -> dict:
    """S1_ediv_high_short: e_div >= threshold で売却 (= 現金保有)."""
    sig_raw = (e_div >= threshold).reindex(close.index, method="ffill").fillna(False).astype(bool)
    sig = apply_hysteresis(sig_raw, hysteresis) if hysteresis > 0 else sig_raw
    # 翌日約定 (look-ahead bias 防止). fillna 由来の FutureWarning は infer_objects で抑止
    sig_pos = sig.shift(1).fillna(False).infer_objects(copy=False).astype(bool)

    rets = close.pct_change().fillna(0)
    # direction='short': ON=現金 / OFF=保有
    strat_rets = rets * (~sig_pos).astype(float)
    pos_change = (sig_pos.astype(int).diff().abs() > 0)
    strat_rets = strat_rets - pos_change.astype(float) * cost

    eq = (1 + strat_rets).cumprod()
    bench_eq = (1 + rets).cumprod()

    n_years = len(rets) / 252
    total_ret = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol_ann = float(strat_rets.std() * np.sqrt(252))
    sharpe = float(cagr / vol_ann) if vol_ann > 1e-9 else 0.0
    peak = eq.cummax()
    max_dd = float((eq / peak - 1).min())

    n_trades = int(pos_change.sum())
    n_signal_days = int(sig_pos.sum())

    total_ret_bh = float(bench_eq.iloc[-1] - 1)
    cagr_bh = float(bench_eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol_bh = float(rets.std() * np.sqrt(252))
    sharpe_bh = cagr_bh / vol_bh if vol_bh > 1e-9 else 0.0
    dd_bh = float((bench_eq / bench_eq.cummax() - 1).min())

    return {
        "total_return": total_ret,
        "CAGR": cagr,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "trades_per_year": float(n_trades / n_years) if n_years > 0 else 0.0,
        "n_signal_days": n_signal_days,
        "signal_pct": float(n_signal_days / len(sig_pos)) if len(sig_pos) > 0 else 0.0,
        "n_days": int(len(rets)),
        "n_years": float(n_years),
        "bh_total_return": total_ret_bh,
        "bh_CAGR": cagr_bh,
        "bh_sharpe": sharpe_bh,
        "bh_max_drawdown": dd_bh,
        "alpha_vs_bh_return": total_ret - total_ret_bh,
        "alpha_vs_bh_sharpe": sharpe - sharpe_bh,
    }


def compute_target_correlation(indicators_df: pd.DataFrame,
                                target_closes: dict[str, pd.Series]) -> dict[str, float]:
    """各 target の日次リターンと L1/n_unb との相関 (sanity 用、参考値)."""
    out: dict[str, float] = {}
    for tid, close in target_closes.items():
        rets = close.pct_change().dropna()
        common = rets.index.intersection(indicators_df.index)
        if len(common) < 252:
            out[tid] = float("nan")
            continue
        s = rets.loc[common]
        e = indicators_df.loc[common, "e_div"]
        # |return| と e_div の相関 (リスクオフ時に株式 down するなら正相関期待)
        try:
            out[tid] = float(s.abs().corr(e))
        except Exception:
            out[tid] = float("nan")
    return out


def main() -> None:
    print("=== 売買対象別 S1 バックテスト ===")
    indicators = load_indicators()
    print(f"Loaded indicators: {indicators.shape}  "
          f"{indicators.index.min().date()} -> {indicators.index.max().date()}")
    start = indicators.index.min().strftime("%Y-%m-%d")
    end = (indicators.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    results: list[dict] = []
    target_closes: dict[str, pd.Series] = {}

    for tid, label, ticker, category in TARGETS:
        print(f"\n[{tid}] {label} ({ticker}) - {category}")
        close = fetch_close(ticker, start, end)
        if close is None:
            print("  SKIP: fetch 失敗 (delisted / network 等)")
            results.append({
                "target_id": tid, "label": label, "ticker": ticker,
                "category": category, "skipped": True,
                "reason": "yfinance fetch failed (delisted or no data)",
            })
            continue
        # 指標と日付を揃える
        common = close.index.intersection(indicators.index)
        if len(common) < 252:
            print(f"  SKIP: common days={len(common)} < 252")
            results.append({
                "target_id": tid, "label": label, "ticker": ticker,
                "category": category, "skipped": True,
                "reason": f"common days={len(common)} < 252",
            })
            continue
        close_sub = close.loc[common]
        ediv_sub = indicators.loc[common, "e_div"]
        target_closes[tid] = close_sub

        t0 = time.time()
        res = simulate_s1(close_sub, ediv_sub)
        elapsed = time.time() - t0
        print(f"  S1   sharpe={res['sharpe']:+.3f}  MaxDD={res['max_drawdown']*100:+.2f}%  "
              f"trades/yr={res['trades_per_year']:.1f}  total_ret={res['total_return']*100:+.2f}%")
        print(f"  B&H  sharpe={res['bh_sharpe']:+.3f}  MaxDD={res['bh_max_drawdown']*100:+.2f}%  "
              f"total_ret={res['bh_total_return']*100:+.2f}%")
        print(f"  alpha ret={res['alpha_vs_bh_return']*100:+.2f}%  "
              f"sharpe={res['alpha_vs_bh_sharpe']:+.3f}  (in {elapsed:.1f}s)")
        results.append({
            "target_id": tid, "label": label, "ticker": ticker,
            "category": category, "skipped": False,
            **res,
        })

    # ===== 比較表 (ターミナル) =====
    print("\n=== 比較表 ===")
    header = (f"{'target':<9} {'category':<20} {'Sharpe':>8} {'MaxDD':>8} "
              f"{'B&H Sh':>8} {'B&H DD':>8} {'ΔSharpe':>9} {'tr/yr':>7}")
    print(header)
    print("-" * len(header))
    for r in results:
        if r.get("skipped"):
            print(f"{r['target_id']:<9} {r['category']:<20} SKIPPED ({r['reason']})")
            continue
        print(f"{r['target_id']:<9} {r['category']:<20} "
              f"{r['sharpe']:>+8.3f} {r['max_drawdown']*100:>+7.2f}% "
              f"{r['bh_sharpe']:>+8.3f} {r['bh_max_drawdown']*100:>+7.2f}% "
              f"{r['alpha_vs_bh_sharpe']:>+9.3f} {r['trades_per_year']:>7.1f}")

    # ===== 解釈 =====
    valid = [r for r in results if not r.get("skipped")]
    if valid:
        # ベスト Sharpe
        best_sharpe = max(valid, key=lambda x: x["sharpe"])
        # SP500 baseline
        sp = next((r for r in valid if r["target_id"] == "SP500"), None)
        # 米国系
        us_idx_ids = {"SP500", "NAS100", "DJ30", "RUS2000"}
        us_results = [r for r in valid if r["target_id"] in us_idx_ids]
        non_us_ids = {"JP225", "UK100", "GER40", "CHINA50"}
        non_us_results = [r for r in valid if r["target_id"] in non_us_ids]
        other_ids = {"AAPL", "GOLD", "BTC"}
        other_results = [r for r in valid if r["target_id"] in other_ids]

        def avg_sh(rs):
            if not rs:
                return float("nan")
            return sum(x["sharpe"] for x in rs) / len(rs)

        def avg_alpha(rs):
            if not rs:
                return float("nan")
            return sum(x["alpha_vs_bh_sharpe"] for x in rs) / len(rs)

        # 勝ち負け集計 (αSharpe > 0)
        n_beat = sum(1 for r in valid if r["alpha_vs_bh_sharpe"] > 0)

        finding_parts: list[str] = []
        if sp is not None:
            finding_parts.append(
                f"S&P 500 baseline: Sharpe={sp['sharpe']:+.3f} / "
                f"MaxDD={sp['max_drawdown']*100:+.2f}% / "
                f"αSharpe vs B&H={sp['alpha_vs_bh_sharpe']:+.3f}."
            )
        finding_parts.append(
            f"全 {len(valid)} 通り中、{n_beat} 通りで Buy&Hold を Sharpe で上回った."
        )
        finding_parts.append(
            f"ベスト Sharpe = {best_sharpe['label']} ({best_sharpe['ticker']}) "
            f"Sharpe={best_sharpe['sharpe']:+.3f}."
        )
        finding_parts.append(
            f"平均 Sharpe: 米国指数 {avg_sh(us_results):+.3f} / "
            f"非米国指数 {avg_sh(non_us_results):+.3f} / "
            f"その他 (個別株/金/BTC) {avg_sh(other_results):+.3f}."
        )
        finding_parts.append(
            f"平均 αSharpe (vs 各 B&H): 米国指数 {avg_alpha(us_results):+.3f} / "
            f"非米国指数 {avg_alpha(non_us_results):+.3f} / "
            f"その他 {avg_alpha(other_results):+.3f}."
        )
        if sp is not None:
            cmp_us = [r for r in us_results if r["target_id"] != "SP500"]
            n_us_beat_sp = sum(1 for r in cmp_us if r["sharpe"] > sp["sharpe"])
            n_us_total = len(cmp_us)
            finding_parts.append(
                f"米国系他指数 ({n_us_total} 銘柄) のうち S&P 500 baseline Sharpe を超えたのは {n_us_beat_sp} 銘柄."
            )
        finding = " ".join(finding_parts)
    else:
        finding = "有効な結果なし"

    # 参考: 観測指標との相関 (|return| と e_div の相関)
    corrs = compute_target_correlation(indicators, target_closes)

    out = {
        "meta": {
            "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "indicators_source": "data/gamma_timeseries_w30.csv (40 銘柄観測)",
            "indicators_range": [
                str(indicators.index.min().date()),
                str(indicators.index.max().date()),
            ],
            "strategy": "S1_ediv_high_short (e_div>=0.8 で売却=現金保有)",
            "params": {
                "ediv_threshold": EDIV_THRESHOLD,
                "transaction_cost": TRANSACTION_COST,
                "hysteresis_days": HYSTERESIS_DAYS,
                "zscore_min_periods": ZSCORE_MIN_PERIODS,
                "execution": "next-day close (close-to-close)",
            },
            "note": (
                "観測 40 銘柄 = e_div シグナル源は変えず、売買対象だけを差し替えた."
                " yfinance fetch 失敗銘柄 (delisted 等) は skipped=true で記録."
            ),
        },
        "results": results,
        "abs_ret_vs_ediv_corr": corrs,
        "finding": finding,
    }
    out_path = DATA_DIR / "backtest_by_trading_target.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")
    print(f"\nFinding: {finding}")


if __name__ == "__main__":
    main()
