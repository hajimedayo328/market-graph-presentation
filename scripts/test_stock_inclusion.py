"""個別株 5 (AAPL/MSFT/GOOG/META/TSLA) を除外して結果が変わるかざっとテスト."""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from persistent_homology import persistence_diagram, persistence_summary
from market_category import MarketCategory
from homology import signed_cycle_balance

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

INDIVIDUAL_STOCKS = ["AAPL", "MSFT", "GOOG", "META", "TSLA"]

EVENT_DATE = pd.Timestamp("2025-04-02")
WINDOW = 30
THRESHOLD = 0.3
PRE_DAYS = 30
POST_DAYS = 30


def compute_one_day(returns: pd.DataFrame, end_idx: int) -> dict:
    win = returns.iloc[end_idx - WINDOW : end_idx].dropna(axis=1, how="any")
    if win.shape[1] < 5:
        return {"L1": np.nan, "n_unb": np.nan}
    corr = win.corr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag = persistence_diagram(corr, max_dim=1)
    summ = persistence_summary(diag)
    cat = MarketCategory(symbols=list(win.columns), corr_matrix=corr, threshold=THRESHOLD)
    cat._build_graph()
    bal = signed_cycle_balance(cat.G)
    return {"L1": float(summ["L1_norm_H1"]), "n_unb": int(bal["n_unbalanced"])}


def event_study(closes: pd.DataFrame, label: str) -> dict:
    returns = closes.pct_change()
    event_pos = returns.index.get_indexer([EVENT_DATE], method="nearest")[0]
    # 計算範囲: event ± POST_DAYS、window 分余裕を見る
    rows = []
    for i in range(event_pos - PRE_DAYS - 5, event_pos + POST_DAYS + 5):
        if i < WINDOW or i >= len(returns):
            continue
        r = compute_one_day(returns, i)
        rows.append({"date": returns.index[i - 1], **r})
    df = pd.DataFrame(rows).dropna()
    # z-score (期間内 expanding でなく単純全期間、ざっとテスト用)
    df["z_L1"] = (df["L1"] - df["L1"].mean()) / df["L1"].std()
    df["z_unb"] = (df["n_unb"] - df["n_unb"].mean()) / df["n_unb"].std()
    df["e_div"] = df["z_unb"] - df["z_L1"]
    pre = df[df["date"] < EVENT_DATE].tail(PRE_DAYS)
    post = df[df["date"] >= EVENT_DATE].head(POST_DAYS)
    delta_L1 = post["z_L1"].mean() - pre["z_L1"].mean()
    delta_unb = post["z_unb"].mean() - pre["z_unb"].mean()
    delta_ediv = post["e_div"].mean() - pre["e_div"].mean()
    return {
        "label": label,
        "n_symbols": closes.shape[1],
        "delta_L1": round(delta_L1, 3),
        "delta_n_unb": round(delta_unb, 3),
        "delta_e_div": round(delta_ediv, 3),
    }


def main():
    closes = pd.read_parquet(DATA / "ohlc_40.parquet")
    print(f"Loaded ohlc_40.parquet: {closes.shape}, period {closes.index.min().date()} -> {closes.index.max().date()}")
    print(f"Symbols: {list(closes.columns)}")
    print()

    # 40 銘柄 baseline
    base = event_study(closes, label="baseline 40 全部")

    # 個別株 5 除外
    no_stock_cols = [c for c in closes.columns if c not in INDIVIDUAL_STOCKS]
    closes_no_stock = closes[no_stock_cols]
    no_stock = event_study(closes_no_stock, label="35 (個別株 5 除外)")

    # 個別株だけ除外 → 株指数 (SP500/NAS100/DJ30) も残ってるからセクター情報は維持
    print("=" * 70)
    print(f"Event: Liberation Day {EVENT_DATE.date()}  window={WINDOW}d")
    print(f"Pre {PRE_DAYS}d vs Post {POST_DAYS}d (z-score 全期間 mean/std)")
    print("=" * 70)
    print(f"{'パターン':<25} {'n':>4} {'ΔL¹':>8} {'Δn_unb':>8} {'Δe_div':>8}")
    print("-" * 70)
    for r in [base, no_stock]:
        print(f"{r['label']:<25} {r['n_symbols']:>4} {r['delta_L1']:>+8.3f} {r['delta_n_unb']:>+8.3f} {r['delta_e_div']:>+8.3f}")
    print()
    diff_ediv = no_stock["delta_e_div"] - base["delta_e_div"]
    print(f"差 (Δe_div): {diff_ediv:+.3f}σ")
    if abs(diff_ediv) < 0.5:
        print("=> 個別株の有無で主要発見 (Δe_div の符号と大きさ) はほぼ不変。頑健。")
    else:
        print(f"=> 影響あり: {diff_ediv:+.3f}σ の差。要 limitation 記載。")


if __name__ == "__main__":
    main()
