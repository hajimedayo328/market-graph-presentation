"""複数の独立ショックで Leave-One-Out (LOO) を再現し、cross-asset 震源の普遍性を検証する.

背景:
  既存 (test_symbol_variations.py, Section 8.6) は 2025-04 Liberation Day の 1 イベントのみで
  LOO を実施し、源泉 = CHINA50/EUB10Y/COPPER/USDJPY/VIX、米株はノイズ という結論を得た。
  本スクリプトは複数の独立ショックで同じ LOO を回し、「源泉銘柄が共通か / イベント依存か」を判定する。

既存との差分:
  - 複数ショック (Liberation Day / 円キャリー巻き戻し / ハマス・イスラエル / ウクライナ侵攻)
  - z-score は expanding window (各日まで過去のみで標準化 = look-ahead bias 排除)
"""
from __future__ import annotations

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from persistent_homology import persistence_diagram, persistence_summary  # noqa: E402
from market_category import MarketCategory  # noqa: E402
from homology import signed_cycle_balance  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

WINDOW = 30
THRESHOLD = 0.3
PRE_DAYS = 30
POST_DAYS = 30
# expanding z-score の最低サンプル数 (これ未満の日は NaN)
MIN_EXPANDING = 20

# 検証する独立ショック。各イベントを 5y / 20y のどちらでカバーするか source で指定。
# 4 イベントすべて 5y (2021-05 以降) でカバーされるため source は ohlc_40 を基本とする。
SHOCKS = [
    {
        "key": "liberation_day",
        "label": "Liberation Day (米相互関税)",
        "date": "2025-04-02",
        "category": "trade_war",
        "source": "ohlc_40.parquet",
    },
    {
        "key": "yen_carry_unwind",
        "label": "円キャリー巻き戻し",
        "date": "2024-08-05",
        "category": "market_structure",
        "source": "ohlc_40.parquet",
    },
    {
        "key": "hamas_israel",
        "label": "ハマス・イスラエル衝突",
        "date": "2023-10-07",
        "category": "war",
        "source": "ohlc_40.parquet",
    },
    {
        "key": "ukraine_invasion",
        "label": "ウクライナ侵攻",
        "date": "2022-02-24",
        "category": "war",
        "source": "ohlc_40.parquet",
    },
]


def _load_meta() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    meta = pd.read_csv(DATA / "symbol_meta.csv").set_index("internal")
    groups = {
        "FX": meta[meta["asset_class"] == "FX"].index.tolist(),
        "INDEX": meta[meta["asset_class"] == "INDEX"].index.tolist(),
        "STOCK": meta[meta["asset_class"] == "STOCK"].index.tolist(),
        "COMMODITY": meta[meta["sector"].isin(["COMMODITY", "METAL", "ENERGY"])].index.tolist(),
        "CRYPTO": meta[meta["asset_class"] == "CRYPTO"].index.tolist(),
        "BOND": meta[meta["asset_class"] == "BOND"].index.tolist(),
        "SPECIAL": meta[meta["asset_class"] == "SPECIAL"].index.tolist(),
    }
    return meta, groups


def compute_day(returns: pd.DataFrame, end_idx: int) -> tuple[float, int]:
    """指定日の (L1_norm_H1, n_unbalanced) を計算する (既存 test_symbol_variations と同一ロジック)."""
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


def event_study(closes: pd.DataFrame, event_date: pd.Timestamp) -> dict:
    """指定銘柄群 × 指定イベントで Δe_div を計算する (expanding window z-score).

    e_div = z_unb - z_L1。z は各日まで過去のみで標準化 (expanding) するため look-ahead なし。
    Δe_div = post 平均 - pre 平均。
    """
    # 既存 test_symbol_variations.py と同一挙動を保つため fill_method='pad' (既定) を踏襲する。
    returns = closes.pct_change(fill_method="pad")
    if event_date in returns.index:
        event_pos = returns.index.get_loc(event_date)
    else:
        event_pos = returns.index.get_indexer([event_date], method="nearest")[0]

    rows = []
    # expanding 標準化に十分な過去を含めるため、イベントの大きく手前から計算する。
    start = max(WINDOW, event_pos - PRE_DAYS - 200)
    end = min(len(returns), event_pos + POST_DAYS + 5)
    for i in range(start, end):
        l1, n_unb = compute_day(returns, i)
        rows.append({"date": returns.index[i - 1], "L1": l1, "n_unb": n_unb})
    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    if len(df) < MIN_EXPANDING + PRE_DAYS:
        return {
            "n_symbols": closes.shape[1],
            "delta_L1": np.nan,
            "delta_n_unb": np.nan,
            "delta_e_div": np.nan,
        }

    # === expanding window z-score (look-ahead bias なし) ===
    exp_mean_l1 = df["L1"].expanding(min_periods=MIN_EXPANDING).mean()
    exp_std_l1 = df["L1"].expanding(min_periods=MIN_EXPANDING).std()
    exp_mean_unb = df["n_unb"].expanding(min_periods=MIN_EXPANDING).mean()
    exp_std_unb = df["n_unb"].expanding(min_periods=MIN_EXPANDING).std()
    df["z_L1"] = (df["L1"] - exp_mean_l1) / exp_std_l1.replace(0, np.nan).clip(lower=1e-9)
    df["z_unb"] = (df["n_unb"] - exp_mean_unb) / exp_std_unb.replace(0, np.nan).clip(lower=1e-9)
    df = df.dropna(subset=["z_L1", "z_unb"]).reset_index(drop=True)
    df["e_div"] = df["z_unb"] - df["z_L1"]

    pre = df[df["date"] < event_date].tail(PRE_DAYS)
    post = df[df["date"] >= event_date].head(POST_DAYS)
    if len(pre) < 10 or len(post) < 10:
        return {
            "n_symbols": closes.shape[1],
            "delta_L1": np.nan,
            "delta_n_unb": np.nan,
            "delta_e_div": np.nan,
        }
    return {
        "n_symbols": closes.shape[1],
        "n_pre": len(pre),
        "n_post": len(post),
        "delta_L1": round(post["z_L1"].mean() - pre["z_L1"].mean(), 3),
        "delta_n_unb": round(post["z_unb"].mean() - pre["z_unb"].mean(), 3),
        "delta_e_div": round(post["e_div"].mean() - pre["e_div"].mean(), 3),
    }


def run_shock(shock: dict, groups: dict[str, list[str]]) -> dict:
    """1 ショックで baseline / LOO / asset-class 除外を計算する."""
    closes_all = pd.read_parquet(DATA / shock["source"])
    all_symbols = list(closes_all.columns)
    event_date = pd.Timestamp(shock["date"])
    print(f"\n{'='*70}\n[{shock['key']}] {shock['label']}  {shock['date']}  src={shock['source']}")

    base = event_study(closes_all, event_date)
    print(f"  baseline Δe_div = {base['delta_e_div']:+.3f}  (pre={base.get('n_pre')}, post={base.get('n_post')})")

    # === LOO ===
    loo = []
    for i, sym in enumerate(all_symbols):
        cols = [c for c in all_symbols if c != sym]
        r = event_study(closes_all[cols], event_date)
        loo.append({"removed": sym, "delta_e_div": r["delta_e_div"]})
        if (i + 1) % 10 == 0:
            print(f"    LOO {i+1}/{len(all_symbols)}")

    base_ed = base["delta_e_div"]
    for r in loo:
        if not np.isnan(r["delta_e_div"]) and not np.isnan(base_ed):
            # contribution = baseline - LOO値。正なら「抜くと弱くなる」= シグナル源泉
            r["contribution"] = round(base_ed - r["delta_e_div"], 3)
        else:
            r["contribution"] = np.nan

    valid_loo = [r for r in loo if not np.isnan(r.get("contribution", np.nan))]
    by_contrib = sorted(valid_loo, key=lambda r: r["contribution"], reverse=True)
    weakens = by_contrib[:5]   # 抜くと弱くなる (源泉) TOP5
    strengthens = by_contrib[-5:][::-1]  # 抜くと強くなる (ノイズ) TOP5

    # === asset class 除外 ===
    asset = {}
    for grp, syms in groups.items():
        present = [s for s in syms if s in all_symbols]
        if not present:
            continue
        cols = [c for c in all_symbols if c not in present]
        if len(cols) < 5:
            asset[grp] = {"n_removed": len(present), "skipped": "残り<5"}
            continue
        r = event_study(closes_all[cols], event_date)
        asset[grp] = {
            "n_removed": len(present),
            "delta_e_div": r["delta_e_div"],
            "drop_vs_base": round(base_ed - r["delta_e_div"], 3) if not np.isnan(r["delta_e_div"]) and not np.isnan(base_ed) else np.nan,
        }

    print(f"  源泉 TOP5 (抜くと弱くなる): {[(r['removed'], r['contribution']) for r in weakens]}")
    print(f"  ノイズ TOP5 (抜くと強くなる): {[(r['removed'], r['contribution']) for r in strengthens]}")

    return {
        "label": shock["label"],
        "date": shock["date"],
        "category": shock["category"],
        "source": shock["source"],
        "baseline_delta_e_div": base_ed,
        "baseline_detail": base,
        "leave_one_out": loo,
        "source_top5": [{"removed": r["removed"], "contribution": r["contribution"], "loo_delta_e_div": r["delta_e_div"]} for r in weakens],
        "noise_top5": [{"removed": r["removed"], "contribution": r["contribution"], "loo_delta_e_div": r["delta_e_div"]} for r in strengthens],
        "asset_class_excluded": asset,
    }


def cross_analysis(shocks_out: list[dict]) -> dict:
    """横断分析: 源泉銘柄の共通性 / asset class レベルの崩壊."""
    valid = [s for s in shocks_out if not np.isnan(s["baseline_delta_e_div"])]
    keys = [s["label"] for s in valid]

    # 源泉 TOP5 の出現回数
    src_count: dict[str, int] = {}
    src_events: dict[str, list[str]] = {}
    for s in valid:
        for r in s["source_top5"]:
            sym = r["removed"]
            src_count[sym] = src_count.get(sym, 0) + 1
            src_events.setdefault(sym, []).append(s["label"])
    src_ranked = sorted(src_count.items(), key=lambda kv: kv[1], reverse=True)

    # ノイズ TOP5 の出現回数
    noise_count: dict[str, int] = {}
    for s in valid:
        for r in s["noise_top5"]:
            noise_count[r["removed"]] = noise_count.get(r["removed"], 0) + 1
    noise_ranked = sorted(noise_count.items(), key=lambda kv: kv[1], reverse=True)

    # 各銘柄の contribution 行列 (shock × symbol)
    all_syms = sorted({r["removed"] for s in valid for r in s["leave_one_out"]})
    contrib_matrix = {}
    for sym in all_syms:
        row = {}
        for s in valid:
            m = next((r for r in s["leave_one_out"] if r["removed"] == sym), None)
            row[s["label"]] = m["contribution"] if m and not np.isnan(m.get("contribution", np.nan)) else None
        contrib_matrix[sym] = row

    # 全ショックで源泉 (TOP5 入り) = 普遍震源候補
    universal = [sym for sym, c in src_ranked if c == len(valid)]
    majority = [sym for sym, c in src_ranked if c >= max(2, len(valid) - 1) and c < len(valid)]

    # asset class: INDEX 抜きで全ショック崩壊するか (drop_vs_base が大きい / Δe_div が baseline から大きく低下)
    asset_breakdown = {}
    for grp in ["INDEX", "FX", "COMMODITY", "BOND", "SPECIAL", "STOCK", "CRYPTO"]:
        drops = []
        for s in valid:
            a = s["asset_class_excluded"].get(grp, {})
            if "drop_vs_base" in a and a["drop_vs_base"] is not None and not (isinstance(a["drop_vs_base"], float) and np.isnan(a["drop_vs_base"])):
                drops.append({"shock": s["label"], "drop_vs_base": a["drop_vs_base"], "loo_delta_e_div": a["delta_e_div"]})
        if drops:
            mean_drop = round(np.mean([d["drop_vs_base"] for d in drops]), 3)
            # 「崩壊」= 抜いた後 Δe_div が baseline 比で大きく低下した shock の数
            n_collapse = sum(1 for d in drops if d["drop_vs_base"] > 0.5)
            asset_breakdown[grp] = {
                "mean_drop_vs_base": mean_drop,
                "n_shocks_collapsed": n_collapse,
                "n_shocks": len(drops),
                "per_shock": drops,
            }

    return {
        "n_valid_shocks": len(valid),
        "shocks": keys,
        "source_appearance_count": dict(src_ranked),
        "source_events": src_events,
        "noise_appearance_count": dict(noise_ranked),
        "universal_sources": universal,       # 全ショックで TOP5
        "near_universal_sources": majority,   # ほぼ全ショックで TOP5
        "contribution_matrix": contrib_matrix,
        "asset_class_breakdown": asset_breakdown,
    }


def core_block_test(cross: dict) -> dict:
    """普遍+準普遍震源を 1 ブロックで抜くと、全ショックで信号が崩壊するか検証する.

    LOO は 1 銘柄ずつだが、震源が「ブロックとして連鎖」しているなら
    まとめて抜いた方が崩壊が顕著なはず。これが cross-asset 震源仮説の核心テスト。
    """
    core = list(dict.fromkeys(cross["universal_sources"] + cross["near_universal_sources"]))
    if not core:
        return {"core_symbols": [], "note": "震源核なし"}
    closes = pd.read_parquet(DATA / "ohlc_40.parquet")
    cols = [c for c in closes.columns if c not in core]
    per_shock = []
    n_flip = 0
    for shock in SHOCKS:
        ev = pd.Timestamp(shock["date"])
        base = event_study(closes, ev)["delta_e_div"]
        rem = event_study(closes[cols], ev)["delta_e_div"]
        if np.isnan(base) or np.isnan(rem):
            continue
        flipped = (base > 0) and (rem <= 0)
        n_flip += int(flipped)
        per_shock.append({
            "shock": shock["label"],
            "baseline": base,
            "core_removed": round(rem, 3),
            "drop": round(base - rem, 3),
            "sign_flip_to_neg": flipped,
        })
    drops = [p["drop"] for p in per_shock]
    return {
        "core_symbols": core,
        "n_shocks": len(per_shock),
        "mean_drop": round(float(np.mean(drops)), 3) if drops else None,
        "min_drop": round(float(np.min(drops)), 3) if drops else None,
        "n_sign_flip_to_neg": n_flip,
        "all_weakened": all(d > 0 for d in drops) if drops else False,
        "per_shock": per_shock,
    }


def main():
    t0 = time.time()
    _, groups = _load_meta()

    shocks_out = []
    for shock in SHOCKS:
        shocks_out.append(run_shock(shock, groups))

    cross = cross_analysis(shocks_out)
    cross["core_block_test"] = core_block_test(cross)

    out = {
        "meta": {
            "window": WINDOW,
            "threshold": THRESHOLD,
            "pre_days": PRE_DAYS,
            "post_days": POST_DAYS,
            "z_score_method": "expanding_window (look-ahead bias なし)",
            "min_expanding": MIN_EXPANDING,
            "n_shocks": len(SHOCKS),
        },
        "shocks": {s["key"]: out_s for s, out_s in zip(SHOCKS, shocks_out)},
        "cross_analysis": cross,
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    out_path = DATA / "loo_multi_shock.json"
    out_path.write_text(json.dumps(_clean(out), ensure_ascii=False, indent=1), encoding="utf-8")
    elapsed = time.time() - t0

    print(f"\n{'='*70}\nDone in {elapsed:.0f}s. Saved: {out_path}")
    print("\n=== 横断分析サマリ ===")
    print(f"有効ショック: {cross['n_valid_shocks']} / {len(SHOCKS)}")
    print(f"普遍震源 (全ショックで TOP5): {cross['universal_sources']}")
    print(f"準普遍震源 (ほぼ全ショック): {cross['near_universal_sources']}")
    print(f"源泉出現回数: {cross['source_appearance_count']}")
    print(f"ノイズ出現回数: {cross['noise_appearance_count']}")
    print("\nasset class 崩壊度 (mean_drop_vs_base, 崩壊ショック数):")
    for grp, info in cross["asset_class_breakdown"].items():
        print(f"  {grp:<10} drop={info['mean_drop_vs_base']:+.3f}  崩壊 {info['n_shocks_collapsed']}/{info['n_shocks']}")


if __name__ == "__main__":
    main()
