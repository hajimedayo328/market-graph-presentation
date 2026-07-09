"""ポスター記載数値の再現・永続化スクリプト.

2026-07-09 の使い捨てセッションで計算されたままリポジトリに残っていなかった
2 つの数値群を、data/ の永続データから再現して記録する:

  [A] 30 通り指標変種の前半/後半分割 → MaxDD ランキングの順位相関
      (ポスター「今後の取り組み」の 順位相関 −0.12)
  [B] 20 年単独指標バックテスト
      (ポスター 3.5 表: e_div −20%/0.76, n_unb だけ −45%/0.47,
       L1 だけ −56%/0.22, B&H −57%/0.48)

方法:
- バックテストエンジンは scripts/backtest_v2.py の simulate_v2 と同一ロジック
  (コスト 0.05%/片道, ヒステリシス 5 営業日, sig.shift(1), expanding z mp=30)
- 価格はオフライン再現性のため data/ohlc_40*.parquet の SP500 列を使用
- 校正: 全期間 30 通りの MaxDD が data/indicator_variation_backtest.csv と
  一致することを確認してから分割検証に進む (一致しなければ規則が違う = 失敗)

出力: data/reproduce_poster_numbers.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"

TRANSACTION_COST = 0.0005   # 0.05% / 片道 (backtest_v2 と同一)
HYSTERESIS_DAYS = 5
Z_MIN_PERIODS = 30
SIGNAL_THRESHOLD = 0.8      # e_div >= 0.8 で現金化 (S1 と同一)

HOLE_COLS = ["L1", "L2", "Linf", "nH1", "meanP", "entropy"]
UNB_COLS = ["n_unb_total", "n_unb_3", "n_unb_4", "n_unb_5plus", "weighted_unb"]


def apply_hysteresis(sig: pd.Series, min_days: int = HYSTERESIS_DAYS) -> pd.Series:
    """backtest_v2.apply_hysteresis と同一."""
    arr = sig.values.astype(bool).copy()
    last_on = -10**9
    for i in range(len(arr)):
        if arr[i]:
            last_on = i
        elif i - last_on < min_days:
            arr[i] = True
    return pd.Series(arr, index=sig.index)


def simulate(close: pd.Series, signal: pd.Series,
             cost: float = TRANSACTION_COST,
             hysteresis: int = HYSTERESIS_DAYS) -> dict:
    """backtest_v2.simulate_v2 の short 方向と同一ロジック (現金化戦略)."""
    sig = signal.reindex(close.index, method="ffill").fillna(False).astype(bool)
    if hysteresis > 0:
        sig = apply_hysteresis(sig, hysteresis)
    sig_pos = sig.shift(1).fillna(False).astype(bool)
    rets = close.pct_change().fillna(0)
    strat_rets = rets * (~sig_pos).astype(float)
    pos_change = (sig_pos.astype(int).diff().abs() > 0)
    strat_rets = strat_rets - pos_change.astype(float) * cost

    eq = (1 + strat_rets).cumprod()
    n_years = len(rets) / 252
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol_ann = float(strat_rets.std() * np.sqrt(252))
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "sharpe": float(cagr / vol_ann) if vol_ann > 1e-9 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "cash_pct": float(sig_pos.mean()),
    }


def buy_and_hold(close: pd.Series) -> dict:
    rets = close.pct_change().fillna(0)
    eq = (1 + rets).cumprod()
    n_years = len(rets) / 252
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol_ann = float(rets.std() * np.sqrt(252))
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "sharpe": float(cagr / vol_ann) if vol_ann > 1e-9 else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "cash_pct": 0.0,
    }


def expanding_z(s: pd.Series, mp: int = Z_MIN_PERIODS) -> pd.Series:
    return (s - s.expanding(min_periods=mp).mean()) / s.expanding(min_periods=mp).std()


def load_sp500(parquet_name: str) -> pd.Series:
    px = pd.read_parquet(DATA_DIR / parquet_name)["SP500"].dropna()
    px.index = pd.to_datetime(px.index)
    return px.sort_index()


def spearman(xs: list[float], ys: list[float]) -> float:
    """scipy 非依存の Spearman 順位相関."""
    rx = pd.Series(xs).rank().values
    ry = pd.Series(ys).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def variant_backtest(ind: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """30 通り (unb × hole) の e_div 変種で現金化バックテスト."""
    rows = []
    for unb in UNB_COLS:
        for hole in HOLE_COLS:
            e = expanding_z(ind[unb]) - expanding_z(ind[hole])
            sig = (e >= SIGNAL_THRESHOLD).fillna(False)
            r = simulate(close, sig)
            rows.append({"unb": unb, "hole": hole, **r})
    return pd.DataFrame(rows)


def part_a_variants() -> dict:
    """[A] 全期間校正 + 前半/後半分割の順位相関."""
    ind = (pd.read_csv(DATA_DIR / "multi_indicators_w30.csv", parse_dates=["date"])
           .dropna(subset=HOLE_COLS + UNB_COLS)
           .set_index("date").sort_index())
    px = load_sp500("ohlc_40.parquet")
    common = px.index.intersection(ind.index)
    ind, px = ind.loc[common], px.loc[common]
    print(f"[A] 共通営業日: {len(common)}  {common.min().date()} -> {common.max().date()}")

    # --- 校正: 全期間 30 通りを既存 CSV と突合 ---
    full = variant_backtest(ind, px)
    ref = pd.read_csv(DATA_DIR / "indicator_variation_backtest.csv")
    ref = ref.rename(columns={"矛盾": "unb", "穴": "hole"})
    merged = full.merge(ref[["unb", "hole", "MaxDD"]], on=["unb", "hole"],
                        suffixes=("", "_ref"))
    merged["dd_diff"] = (merged["max_drawdown"] * 100 - merged["MaxDD"]).abs()
    max_diff = float(merged["dd_diff"].max())
    rank_corr_vs_ref = spearman(list(merged["max_drawdown"]), list(merged["MaxDD"] / 100))
    print(f"[A] 校正: MaxDD 最大乖離 = {max_diff:.3f}pp / 既存CSVとの順位相関 = {rank_corr_vs_ref:.3f}")

    # --- 前半/後半分割 (営業日ベースで半分に割り、各半で独立に z を再計算) ---
    mid = len(common) // 2
    idx_h1, idx_h2 = common[:mid], common[mid:]
    h1 = variant_backtest(ind.loc[idx_h1], px.loc[idx_h1])
    h2 = variant_backtest(ind.loc[idx_h2], px.loc[idx_h2])
    both = h1.merge(h2, on=["unb", "hole"], suffixes=("_h1", "_h2"))
    rank_corr_halves = spearman(list(both["max_drawdown_h1"]),
                                list(both["max_drawdown_h2"]))
    print(f"[A] 前半 {idx_h1.min().date()}~{idx_h1.max().date()} / "
          f"後半 {idx_h2.min().date()}~{idx_h2.max().date()}")
    print(f"[A] 前半/後半 MaxDD 順位相関 (Spearman) = {rank_corr_halves:.4f}")

    # 上位の顔ぶれ
    top_h1 = both.nlargest(3, "max_drawdown_h1")[["unb", "hole", "max_drawdown_h1", "max_drawdown_h2"]]
    print("[A] 前半トップ3 (MaxDD 浅い順):")
    for _, r in top_h1.iterrows():
        print(f"     {r['unb']} × {r['hole']}: H1 {r['max_drawdown_h1']*100:+.1f}% -> "
              f"H2 {r['max_drawdown_h2']*100:+.1f}%")

    return {
        "n_days": int(len(common)),
        "calibration": {
            "maxdd_max_abs_diff_pp": round(max_diff, 4),
            "rank_corr_vs_existing_csv": round(rank_corr_vs_ref, 4),
        },
        "split": {
            "h1_range": [str(idx_h1.min().date()), str(idx_h1.max().date())],
            "h2_range": [str(idx_h2.min().date()), str(idx_h2.max().date())],
            "rank_corr_maxdd_h1_vs_h2_spearman": round(rank_corr_halves, 4),
        },
        "halves_table": [
            {"unb": r["unb"], "hole": r["hole"],
             "maxdd_h1": round(r["max_drawdown_h1"], 4),
             "maxdd_h2": round(r["max_drawdown_h2"], 4)}
            for _, r in both.iterrows()
        ],
    }


def part_b_20y_table() -> dict:
    """[B] 20 年単独指標バックテスト (ポスター 3.5 表)."""
    ind = (pd.read_csv(DATA_DIR / "gamma_timeseries_20y_w30.csv", parse_dates=["date"])
           .dropna(subset=["L1_H1", "n_unb"])
           .set_index("date").sort_index())
    px = load_sp500("ohlc_40_20y.parquet")
    common = px.index.intersection(ind.index)
    ind, px = ind.loc[common], px.loc[common]
    print(f"\n[B] 共通営業日: {len(common)}  {common.min().date()} -> {common.max().date()}")

    z_l1 = expanding_z(ind["L1_H1"])
    z_unb = expanding_z(ind["n_unb"])
    strategies = {
        "e_div (差分)": ((z_unb - z_l1) >= SIGNAL_THRESHOLD),
        "n_unb だけ": (z_unb >= SIGNAL_THRESHOLD),
        "L1 だけ": (z_l1 >= SIGNAL_THRESHOLD),
    }
    poster = {
        "e_div (差分)": {"maxdd": -20, "sharpe": 0.76},
        "n_unb だけ": {"maxdd": -45, "sharpe": 0.47},
        "L1 だけ": {"maxdd": -56, "sharpe": 0.22},
        "B&H": {"maxdd": -57, "sharpe": 0.48},
    }
    out = {}
    for name, sig in strategies.items():
        r = simulate(px, sig.fillna(False))
        out[name] = r
        p = poster[name]
        print(f"[B] {name:<12} MaxDD {r['max_drawdown']*100:+6.1f}% (ポスター {p['maxdd']}%)  "
              f"Sharpe {r['sharpe']:+.2f} (ポスター {p['sharpe']})  現金化 {r['cash_pct']*100:.0f}%")
    bh = buy_and_hold(px)
    out["B&H"] = bh
    p = poster["B&H"]
    print(f"[B] {'B&H':<12} MaxDD {bh['max_drawdown']*100:+6.1f}% (ポスター {p['maxdd']}%)  "
          f"Sharpe {bh['sharpe']:+.2f} (ポスター {p['sharpe']})")

    return {
        "n_days": int(len(common)),
        "range": [str(common.min().date()), str(common.max().date())],
        "strategies": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                       for k, v in out.items()},
        "poster_reference": poster,
    }


def main() -> None:
    print("=" * 72)
    print("ポスター数値の再現 (規則: e_div>=0.8 現金化, cost 0.05%, hyst 5d, "
          "expanding z mp=30, 価格=parquet SP500)")
    print("=" * 72)
    result = {
        "params": {
            "engine": "backtest_v2.simulate_v2 互換 (short=現金化)",
            "transaction_cost": TRANSACTION_COST,
            "hysteresis_days": HYSTERESIS_DAYS,
            "zscore": f"expanding(min_periods={Z_MIN_PERIODS})",
            "signal_threshold": SIGNAL_THRESHOLD,
            "price_source": "data/ohlc_40*.parquet SP500 列 (オフライン再現)",
        },
        "A_variant_split": part_a_variants(),
        "B_20y_single_indicators": part_b_20y_table(),
    }
    out_path = DATA_DIR / "reproduce_poster_numbers.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
