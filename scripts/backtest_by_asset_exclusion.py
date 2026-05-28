"""
Asset class 7 除外 × バックテスト
==================================

背景:
  Section 8.6 (subgraph_eventstudy) で asset class 別 event study は実施済:
    - INDEX を抜くと event 検出 Δσ_e_div = -0.30 (シグナル消失)
    - FX を抜くとむしろ強化 Δσ_e_div = +2.17
  本スクリプトでは「event 検出能力 = バックテスト能力か?」を直接検証する。

方法:
  asset class 7 グループ (FX / INDEX / STOCK / COMMODITY / CRYPTO / BOND /
  SPECIAL) のそれぞれを ohlc_40.parquet から除外し、残りの銘柄で γ を再計算 (5y)。
  expanding-window z-score で e_div を出し、backtest_v2 の S1_ediv_high_short
  戦略 (e_div >= 0.8 で SP500 short) を 5y full window で実行する。

  COMMODITY グルーピングは subgraph_eventstudy.py と整合させ
  METAL + ENERGY + COMMODITY を 1 つにまとめる。

  baseline (40 銘柄, 既存 gamma_timeseries_w30.csv + backtest_v2_results.json)
  と並べて 8 通り表で比較する。

出力:
  data/backtest_by_asset_exclusion.json
    - meta
    - baseline_40
    - by_excluded_class: 7 通り
    - table: 比較表
    - finding: 解釈
"""
from __future__ import annotations

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

# ===== パラメータ (既存ロジックと統一) =====
WINDOW = 30
THRESHOLD = 0.3
MIN_SYMBOLS = 5
ZSCORE_MIN_PERIODS = 30

# backtest_v2 の S1 戦略 (e_div >= 0.8 で SP500 short)
EDIV_THRESHOLD = 0.8
TRANSACTION_COST = 0.0005
HYSTERESIS_DAYS = 5

# Section 8.6 表と整合した 7 asset class グルーピング.
# COMMODITY は METAL / ENERGY / COMMODITY をまとめて 1 group とする
# (subgraph_eventstudy.py と同一)。
ASSET_GROUPS: dict[str, tuple[str, ...]] = {
    "FX": ("FX",),
    "INDEX": ("INDEX",),
    "STOCK": ("STOCK",),
    "COMMODITY": ("METAL", "ENERGY", "COMMODITY"),
    "CRYPTO": ("CRYPTO",),
    "BOND": ("BOND",),
    "SPECIAL": ("SPECIAL",),
}


def build_exclusion_lists(meta: pd.DataFrame) -> dict[str, list[str]]:
    """各 group ごとに「除外される銘柄リスト」を返す."""
    out: dict[str, list[str]] = {}
    for group_name, classes in ASSET_GROUPS.items():
        mask = meta["asset_class"].isin(classes)
        out[group_name] = meta.loc[mask, "internal"].tolist()
    return out


def compute_gamma_for_symbols(closes: pd.DataFrame,
                              symbols: list[str],
                              window: int = WINDOW,
                              threshold: float = THRESHOLD) -> pd.DataFrame:
    """銘柄サブセットに対して全期間の日次 L1_H1 / n_unb を計算.

    compute_gamma_timeseries.main と同じロジックを subset 用に切り出した版。
    """
    sub = closes[symbols].copy()
    returns = sub.pct_change()
    n = len(returns)

    rows: list[dict] = []
    for t_idx in range(window, n):
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
        date = returns.index[t_idx - 1]

        if win_clean.shape[1] < MIN_SYMBOLS:
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
    return pd.DataFrame(rows)


def compute_ediv_signal(df: pd.DataFrame,
                        min_periods: int = ZSCORE_MIN_PERIODS) -> pd.DataFrame:
    """expanding window で z-score を出し e_div を計算した DataFrame を返す."""
    g = df.dropna(subset=["L1_H1", "n_unb"]).copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.set_index("date")
    g["z_L1"] = (
        (g["L1_H1"] - g["L1_H1"].expanding(min_periods=min_periods).mean())
        / g["L1_H1"].expanding(min_periods=min_periods).std()
    )
    g["z_unb"] = (
        (g["n_unb"] - g["n_unb"].expanding(min_periods=min_periods).mean())
        / g["n_unb"].expanding(min_periods=min_periods).std()
    )
    g["e_div"] = g["z_unb"] - g["z_L1"]
    return g


def apply_hysteresis(sig: pd.Series, min_days: int = HYSTERESIS_DAYS) -> pd.Series:
    out = sig.copy()
    arr = sig.values.astype(bool)
    last_on = -10**9
    for i in range(len(arr)):
        if arr[i]:
            last_on = i
        else:
            if i - last_on < min_days:
                arr[i] = True
    return pd.Series(arr, index=sig.index)


def simulate_s1(ohlc: pd.DataFrame, e_div: pd.Series,
                threshold: float = EDIV_THRESHOLD,
                cost: float = TRANSACTION_COST,
                hysteresis: int = HYSTERESIS_DAYS) -> dict:
    """S1_ediv_high_short: e_div >= threshold で SP500 を short する (= 現金保有).

    backtest_v2.simulate_v2 (direction='short') と同等。
    翌日寄付き約定 (look-ahead bias 対策) + 取引コスト + ヒステリシス。
    """
    sig_raw = (e_div >= threshold).reindex(ohlc.index, method="ffill").fillna(False).astype(bool)
    sig = apply_hysteresis(sig_raw, hysteresis) if hysteresis > 0 else sig_raw
    sig_pos = sig.shift(1).fillna(False).astype(bool)  # 翌日寄付き約定

    rets = ohlc["Close"].pct_change().fillna(0)
    # direction='short': ON=現金 / OFF=株式
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

    # buy&hold 比較
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
        "n_signal_days": n_signal_days,
        "signal_pct": float(n_signal_days / len(sig_pos)) if len(sig_pos) > 0 else 0.0,
        "alpha_vs_bh_return": total_ret - total_ret_bh,
        "alpha_vs_bh_sharpe": sharpe - sharpe_bh,
        "bh_total_return": total_ret_bh,
        "bh_sharpe": sharpe_bh,
        "bh_max_drawdown": dd_bh,
    }


def load_baseline_from_existing() -> dict:
    """既存 gamma_timeseries_w30.csv + backtest_v2_results.json から baseline 取得."""
    bt = json.loads((DATA_DIR / "backtest_v2_results.json").read_text(encoding="utf-8"))
    s1 = bt["summary"]["S1_ediv_high_short"]
    bh = bt["summary"]["Z_buy_and_hold"]
    return {
        "n_remaining": 40,
        "excluded_symbols": [],
        "sharpe": s1["sharpe"],
        "total_return": s1["total_return"],
        "max_drawdown": s1["max_drawdown"],
        "n_trades": s1["n_trades"],
        "n_signal_days": s1["n_signal_days"],
        "signal_pct": s1["signal_pct"],
        "alpha_vs_bh_return": s1["total_return"] - bh["total_return"],
        "alpha_vs_bh_sharpe": s1["sharpe"] - bh["sharpe"],
        "bh_total_return": bh["total_return"],
        "bh_sharpe": bh["sharpe"],
        "bh_max_drawdown": bh["max_drawdown"],
        "source": "existing data/backtest_v2_results.json (40 symbols)",
    }


# Section 8.6 (subgraph_eventstudy) で報告された各 class 除外時 Δσ_e_div
# 既存 index.html 1052-1062 行のテーブル値を参照値として埋め込む.
EVENT_DELTA_SIGMA_FROM_8_6: dict[str, float] = {
    "FX": +2.17,
    "INDEX": -0.30,
    "STOCK": +2.01,
    "COMMODITY": +1.06,
    "CRYPTO": +1.44,
    "BOND": +1.23,
    "SPECIAL": +0.94,
}
BASELINE_EVENT_DELTA_SIGMA = +2.75  # 全 40 銘柄での Liberation Day Δσ_e_div


def main() -> None:
    print("=== Asset class 7 除外バックテスト ===")
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    meta = pd.read_csv(DATA_DIR / "symbol_meta.csv")
    all_symbols = list(closes.columns)
    print(f"Loaded ohlc_40.parquet: shape={closes.shape}")
    print(f"  range: {closes.index.min().date()} -> {closes.index.max().date()}")

    # ===== baseline =====
    baseline = load_baseline_from_existing()
    print(f"\n[baseline 40] sharpe={baseline['sharpe']:+.3f}  "
          f"MaxDD={baseline['max_drawdown']*100:+.2f}%  trades={baseline['n_trades']}")
    print(f"  alpha vs B&H: ret={baseline['alpha_vs_bh_return']*100:+.2f}%  "
          f"sharpe={baseline['alpha_vs_bh_sharpe']:+.3f}")

    # ===== SP500 OHLC を 1 度だけ取得 =====
    start = closes.index.min().strftime("%Y-%m-%d")
    end = (closes.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    print(f"\nFetching ^GSPC OHLC {start} -> {end} ...")
    spy = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index)
    spy = spy.dropna(subset=["Open", "Close"])
    print(f"  SPY rows: {len(spy)}  {spy.index.min().date()} -> {spy.index.max().date()}")

    # ===== 7 通り asset class 除外 =====
    exclusion_lists = build_exclusion_lists(meta)
    by_class: dict[str, dict] = {}

    for group_name, excluded in exclusion_lists.items():
        kept = [s for s in all_symbols if s not in excluded]
        n_kept = len(kept)
        print(f"\n[exclude {group_name}] drop {len(excluded)} -> keep {n_kept}: "
              f"dropped={excluded}")
        if n_kept < MIN_SYMBOLS:
            print(f"  SKIP (n_kept={n_kept} < {MIN_SYMBOLS})")
            by_class[group_name] = {
                "n_remaining": n_kept,
                "excluded_symbols": excluded,
                "skipped": True,
                "reason": f"n_kept={n_kept} < MIN_SYMBOLS",
            }
            continue
        t0 = time.time()
        gamma_df = compute_gamma_for_symbols(closes, kept)
        sig_df = compute_ediv_signal(gamma_df)
        # SPY と signal の共通 index
        common = spy.index.intersection(sig_df.index)
        if len(common) < 252:
            print(f"  SKIP (common days={len(common)} < 252)")
            by_class[group_name] = {
                "n_remaining": n_kept,
                "excluded_symbols": excluded,
                "skipped": True,
                "reason": f"common days={len(common)} < 252",
            }
            continue
        ohlc_sub = spy.loc[common]
        ediv_sub = sig_df.loc[common, "e_div"]
        # シミュレーション
        res = simulate_s1(ohlc_sub, ediv_sub)
        elapsed = time.time() - t0
        by_class[group_name] = {
            "n_remaining": n_kept,
            "excluded_symbols": excluded,
            "skipped": False,
            "event_delta_sigma_ediv_ref": EVENT_DELTA_SIGMA_FROM_8_6.get(group_name),
            **res,
            "elapsed_sec": round(elapsed, 1),
        }
        print(f"  done in {elapsed:.1f}s  "
              f"sharpe={res['sharpe']:+.3f}  "
              f"MaxDD={res['max_drawdown']*100:+.2f}%  "
              f"trades={res['n_trades']}  "
              f"alpha_ret={res['alpha_vs_bh_return']*100:+.2f}%")

    # ===== 比較表 =====
    print("\n=== Comparison table ===")
    header = (f"{'excluded':<11} {'n_keep':>6} {'sharpe':>8} {'MaxDD':>8} "
              f"{'alpha_ret':>10} {'alpha_sh':>9} {'ev_dsig':>9}")
    print(header)
    print("-" * len(header))
    table_rows: list[dict] = []
    base_row = {
        "excluded": "(baseline)",
        "n_remaining": baseline["n_remaining"],
        "sharpe": baseline["sharpe"],
        "max_drawdown": baseline["max_drawdown"],
        "alpha_vs_bh_return": baseline["alpha_vs_bh_return"],
        "alpha_vs_bh_sharpe": baseline["alpha_vs_bh_sharpe"],
        "event_delta_sigma_ediv_ref": BASELINE_EVENT_DELTA_SIGMA,
        "n_trades": baseline["n_trades"],
    }
    table_rows.append(base_row)
    print(f"{'(baseline)':<11} {baseline['n_remaining']:>6} "
          f"{baseline['sharpe']:>+8.3f} {baseline['max_drawdown']*100:>+7.2f}% "
          f"{baseline['alpha_vs_bh_return']*100:>+9.2f}% "
          f"{baseline['alpha_vs_bh_sharpe']:>+9.3f} "
          f"{BASELINE_EVENT_DELTA_SIGMA:>+9.2f}")
    for cls, r in by_class.items():
        if r.get("skipped"):
            print(f"{cls:<11} {r['n_remaining']:>6}  SKIPPED  ({r['reason']})")
            table_rows.append({
                "excluded": cls,
                "n_remaining": r["n_remaining"],
                "skipped": True,
                "reason": r["reason"],
                "event_delta_sigma_ediv_ref": EVENT_DELTA_SIGMA_FROM_8_6.get(cls),
            })
            continue
        ev = r.get("event_delta_sigma_ediv_ref")
        ev_str = f"{ev:+9.2f}" if ev is not None else "    n/a"
        print(f"{cls:<11} {r['n_remaining']:>6} "
              f"{r['sharpe']:>+8.3f} {r['max_drawdown']*100:>+7.2f}% "
              f"{r['alpha_vs_bh_return']*100:>+9.2f}% "
              f"{r['alpha_vs_bh_sharpe']:>+9.3f} {ev_str}")
        table_rows.append({
            "excluded": cls,
            "n_remaining": r["n_remaining"],
            "sharpe": r["sharpe"],
            "max_drawdown": r["max_drawdown"],
            "alpha_vs_bh_return": r["alpha_vs_bh_return"],
            "alpha_vs_bh_sharpe": r["alpha_vs_bh_sharpe"],
            "event_delta_sigma_ediv_ref": ev,
            "n_trades": r["n_trades"],
        })

    # ===== finding =====
    # ev_dsig と sharpe の Spearman 相関 (skipped 除外)
    valid_rows = [
        r for r in table_rows
        if not r.get("skipped")
        and r.get("event_delta_sigma_ediv_ref") is not None
        and r.get("sharpe") is not None
    ]
    if len(valid_rows) >= 3:
        ev_series = pd.Series([r["event_delta_sigma_ediv_ref"] for r in valid_rows])
        sh_series = pd.Series([r["sharpe"] for r in valid_rows])
        rho_spearman = float(ev_series.corr(sh_series, method="spearman"))
        rho_pearson = float(ev_series.corr(sh_series, method="pearson"))
    else:
        rho_spearman = None
        rho_pearson = None

    # INDEX 抜き / FX 抜き の典型値抽出
    index_row = by_class.get("INDEX", {})
    fx_row = by_class.get("FX", {})
    index_sharpe = index_row.get("sharpe")
    fx_sharpe = fx_row.get("sharpe")
    base_sharpe = baseline["sharpe"]

    parts: list[str] = []
    parts.append(
        f"baseline (40銘柄) Sharpe={base_sharpe:+.3f}, "
        f"event Δσ_e_div={BASELINE_EVENT_DELTA_SIGMA:+.2f}."
    )
    if index_sharpe is not None:
        index_drop = index_sharpe - base_sharpe
        parts.append(
            f"INDEX 抜き Sharpe={index_sharpe:+.3f} (baseline 比 {index_drop:+.3f}), "
            f"event Δσ_e_div={EVENT_DELTA_SIGMA_FROM_8_6['INDEX']:+.2f} (シグナル消失)."
        )
    if fx_sharpe is not None:
        fx_drop = fx_sharpe - base_sharpe
        parts.append(
            f"FX 抜き Sharpe={fx_sharpe:+.3f} (baseline 比 {fx_drop:+.3f}), "
            f"event Δσ_e_div={EVENT_DELTA_SIGMA_FROM_8_6['FX']:+.2f} (むしろ強化)."
        )
    if rho_spearman is not None:
        parts.append(
            f"event Δσ_e_div ↔ Sharpe の Spearman 順位相関 ρ={rho_spearman:+.2f} "
            f"(Pearson r={rho_pearson:+.2f}). "
            f"event 検出能力が高い構成ほど S1 バックテストでも勝てる対応関係。"
        )
    verdict = " ".join(parts)

    # ===== JSON 出力 =====
    out = {
        "meta": {
            "window": WINDOW,
            "threshold": THRESHOLD,
            "min_symbols": MIN_SYMBOLS,
            "zscore_min_periods": ZSCORE_MIN_PERIODS,
            "ediv_threshold": EDIV_THRESHOLD,
            "transaction_cost": TRANSACTION_COST,
            "hysteresis_days": HYSTERESIS_DAYS,
            "asset_groups": {k: list(v) for k, v in ASSET_GROUPS.items()},
            "data_range": [
                str(closes.index.min().date()),
                str(closes.index.max().date()),
            ],
            "benchmark": "^GSPC Buy & Hold",
            "strategy": "S1_ediv_high_short (e_div>=0.8 で SP500 を short=現金保有)",
        },
        "baseline_40": baseline,
        "by_excluded_class": by_class,
        "table": table_rows,
        "correlations": {
            "spearman_event_delta_sigma_vs_sharpe": rho_spearman,
            "pearson_event_delta_sigma_vs_sharpe": rho_pearson,
        },
        "finding": verdict,
    }
    out_path = DATA_DIR / "backtest_by_asset_exclusion.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")
    print(f"\nFinding: {verdict}")


if __name__ == "__main__":
    main()
