"""銘柄構成バリエーション総当たり: LOO / asset_class 除外 / 段階的縮小."""
from __future__ import annotations
import sys
import json
import time
import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from persistent_homology import persistence_diagram, persistence_summary
from market_category import MarketCategory
from homology import signed_cycle_balance

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

EVENT_DATE = pd.Timestamp("2025-04-02")
WINDOW = 30
THRESHOLD = 0.3
PRE_DAYS = 30
POST_DAYS = 30

# data/symbol_meta.csv を読んで asset_class マッピング
META = pd.read_csv(DATA / "symbol_meta.csv").set_index("internal")
ASSET_GROUPS = {
    "FX": META[META["asset_class"] == "FX"].index.tolist(),
    "INDEX": META[META["asset_class"] == "INDEX"].index.tolist(),
    "STOCK": META[META["asset_class"] == "STOCK"].index.tolist(),
    "COMMODITY": META[META["sector"].isin(["COMMODITY", "METAL", "ENERGY"])].index.tolist(),
    "CRYPTO": META[META["asset_class"] == "CRYPTO"].index.tolist(),
    "BOND": META[META["asset_class"] == "BOND"].index.tolist(),
    "SPECIAL": META[META["asset_class"] == "SPECIAL"].index.tolist(),
}


def compute_day(returns: pd.DataFrame, end_idx: int) -> tuple[float, int]:
    win = returns.iloc[end_idx - WINDOW : end_idx].dropna(axis=1, how="any")
    if win.shape[1] < 5:
        return np.nan, np.nan
    corr = win.corr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag = persistence_diagram(corr, max_dim=1)
    summ = persistence_summary(diag)
    cat = MarketCategory(symbols=list(win.columns), corr_matrix=corr, threshold=THRESHOLD)
    cat._build_graph()
    bal = signed_cycle_balance(cat.G)
    return float(summ["L1_norm_H1"]), int(bal["n_unbalanced"])


def event_study(closes: pd.DataFrame) -> dict:
    """指定銘柄群で Liberation Day Δe_div を計算."""
    returns = closes.pct_change()
    if EVENT_DATE not in returns.index:
        event_pos = returns.index.get_indexer([EVENT_DATE], method="nearest")[0]
    else:
        event_pos = returns.index.get_loc(EVENT_DATE)
    rows = []
    for i in range(event_pos - PRE_DAYS - 5, event_pos + POST_DAYS + 5):
        if i < WINDOW or i >= len(returns):
            continue
        l1, n_unb = compute_day(returns, i)
        rows.append({"date": returns.index[i - 1], "L1": l1, "n_unb": n_unb})
    df = pd.DataFrame(rows).dropna()
    if len(df) < 20:
        return {"n_symbols": closes.shape[1], "delta_L1": np.nan, "delta_n_unb": np.nan, "delta_e_div": np.nan}
    df["z_L1"] = (df["L1"] - df["L1"].mean()) / max(df["L1"].std(), 1e-9)
    df["z_unb"] = (df["n_unb"] - df["n_unb"].mean()) / max(df["n_unb"].std(), 1e-9)
    df["e_div"] = df["z_unb"] - df["z_L1"]
    pre = df[df["date"] < EVENT_DATE].tail(PRE_DAYS)
    post = df[df["date"] >= EVENT_DATE].head(POST_DAYS)
    return {
        "n_symbols": closes.shape[1],
        "delta_L1": round(post["z_L1"].mean() - pre["z_L1"].mean(), 3),
        "delta_n_unb": round(post["z_unb"].mean() - pre["z_unb"].mean(), 3),
        "delta_e_div": round(post["e_div"].mean() - pre["e_div"].mean(), 3),
    }


def main():
    closes_all = pd.read_parquet(DATA / "ohlc_40.parquet")
    all_symbols = list(closes_all.columns)
    print(f"Loaded: {closes_all.shape}, symbols={len(all_symbols)}")
    print()

    t0 = time.time()
    results = {}

    # === Baseline ===
    print("[Baseline] 40 銘柄...")
    base = event_study(closes_all)
    results["baseline_40"] = base
    print(f"  Δe_div = {base['delta_e_div']:+.3f}")

    # === Test 1: Leave-One-Out ===
    print("\n[Test 1] Leave-One-Out (40 通り)...")
    loo_results = []
    for i, sym in enumerate(all_symbols):
        cols = [c for c in all_symbols if c != sym]
        r = event_study(closes_all[cols])
        loo_results.append({"removed": sym, **r})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(all_symbols)}  last: {sym}  Δe_div={r['delta_e_div']:+.3f}")
    results["leave_one_out"] = loo_results

    # === Test 2: Asset class 別除外 ===
    print("\n[Test 2] Asset class 別除外...")
    asset_results = {}
    for grp, syms in ASSET_GROUPS.items():
        present = [s for s in syms if s in all_symbols]
        if not present:
            continue
        cols = [c for c in all_symbols if c not in present]
        if len(cols) < 5:
            asset_results[grp] = {"n_removed": len(present), "skipped": "残り < 5 銘柄"}
            continue
        r = event_study(closes_all[cols])
        asset_results[grp] = {"n_removed": len(present), "removed": present, **r}
        print(f"  抜き: {grp:<10} ({len(present)} 銘柄, 残 {len(cols)}) → Δe_div={r['delta_e_div']:+.3f}")
    results["asset_class_excluded"] = asset_results

    # === Test 3: 段階的縮小 (ランダム削減) ===
    print("\n[Test 3] 段階的縮小 (各サイズ 3 trial 平均)...")
    rng = random.Random(42)
    size_results = {}
    for n_keep in [35, 30, 25, 20, 15, 10]:
        deltas = []
        for trial in range(3):
            keep = rng.sample(all_symbols, n_keep)
            r = event_study(closes_all[keep])
            if not np.isnan(r["delta_e_div"]):
                deltas.append(r["delta_e_div"])
        if deltas:
            size_results[n_keep] = {
                "n_trials": len(deltas),
                "mean": round(np.mean(deltas), 3),
                "std": round(np.std(deltas), 3),
                "min": round(min(deltas), 3),
                "max": round(max(deltas), 3),
            }
            print(f"  n={n_keep}: mean={size_results[n_keep]['mean']:+.3f} ± {size_results[n_keep]['std']:.3f}  range [{size_results[n_keep]['min']:+.3f}, {size_results[n_keep]['max']:+.3f}]")
    results["progressive_reduction"] = size_results

    # === 出力 ===
    elapsed = time.time() - t0
    out = DATA / "symbol_variation_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nDone in {elapsed:.0f}s. Saved: {out}")

    # === LOO ランキング表示 ===
    print("\n=== LOO ランキング (Δe_div への寄与) ===")
    loo_sorted = sorted(loo_results, key=lambda r: r["delta_e_div"], reverse=True)
    print(f"{'rank':>4} {'removed':<10} {'Δe_div':>8}  解釈")
    print("-" * 65)
    base_ed = base["delta_e_div"]
    for i, r in enumerate(loo_sorted[:5], 1):
        diff = r["delta_e_div"] - base_ed
        print(f"{i:>4} {r['removed']:<10} {r['delta_e_div']:>+8.3f}  baseline+{diff:+.3f} ← この銘柄を抜くと強くなる (ノイズ気味)")
    print("...")
    for i, r in enumerate(loo_sorted[-5:], len(loo_sorted) - 4):
        diff = r["delta_e_div"] - base_ed
        print(f"{i:>4} {r['removed']:<10} {r['delta_e_div']:>+8.3f}  baseline+{diff:+.3f} ← この銘柄を抜くと弱くなる (シグナルに貢献)")


if __name__ == "__main__":
    main()
