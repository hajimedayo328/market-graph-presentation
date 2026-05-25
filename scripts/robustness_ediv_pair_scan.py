"""
e_div ロバストネス検証: 全 11 指標ペアスイープ event study スキャン.

目的:
  e_div = z(n_unb_total) - z(L1) が「同じパイプラインで作れる多数の指標
  ペア差分の中で、たまたま良いものを選んだだけ (data dredging / p-hacking)」
  という批判への実証反論データを生成する.

手順:
  1. data/multi_indicators_w30.csv から 11 指標を抽出
  2. 各指標の 90 日 rolling z-score を計算
  3. C(11, 2) = 55 ペアの差分 z_a - z_b を作る
  4. shock type 別 event 群で event study (Δσ = post mean − pre mean)
  5. e_div のランクを 55 ペア分布の中で算出

出力: data/robustness_ediv_pair_scan.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ----- パス -----
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
INPUT_CSV = DATA_DIR / "multi_indicators_w30.csv"
OUTPUT_JSON = DATA_DIR / "robustness_ediv_pair_scan.json"

# ----- パラメータ -----
INDICATORS = [
    "L1", "L2", "Linf", "nH1", "meanP", "entropy",
    "n_unb_total", "n_unb_3", "n_unb_4", "n_unb_5plus",
    "balance_rate",
]
Z_WINDOW = 90       # rolling z-score 窓
EVENT_PRE = 30      # event 前 N 日
EVENT_POST = 30     # event 後 N 日

# ----- shock type 別イベント (データ範囲 2021-06 以降のみ採用) -----
EVENTS = {
    "trade_policy": [
        ("2025-04-02", "Liberation Day (reciprocal tariff)"),
        ("2024-05-14", "Biden 100% EV tariff on China"),
        # 2018-07-06 (USA China 25%) はデータ範囲外 (2021-06 開始)
    ],
    "market_structure": [
        ("2024-08-05", "JPY carry unwind / vol spike"),
        # 2020-03-12 (COVID) はデータ範囲外
    ],
    "war": [
        ("2023-10-07", "Hamas attack on Israel"),
        ("2022-02-24", "Russia invades Ukraine"),
    ],
}


def load_indicators() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).set_index("date")
    df = df[INDICATORS].copy()
    # 先頭の NaN 1 行 (n_symbols=0 の日) は落とす
    df = df.dropna(how="all")
    return df


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(20, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(20, window // 3)).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def event_delta_sigma(z_series: pd.Series, event_date: str,
                      pre: int, post: int) -> float | None:
    """Δσ = event 後 N 日の z 平均 − event 前 N 日の z 平均."""
    target = pd.Timestamp(event_date)
    if target < z_series.index.min() or target > z_series.index.max():
        return None
    pos = z_series.index.get_indexer([target], method="nearest")[0]
    pre_slice = z_series.iloc[max(0, pos - pre):pos]
    post_slice = z_series.iloc[pos:min(len(z_series), pos + post)]
    pre_slice = pre_slice.dropna()
    post_slice = post_slice.dropna()
    if len(pre_slice) < pre // 2 or len(post_slice) < post // 2:
        return None
    return float(post_slice.mean() - pre_slice.mean())


def build_pair_diffs(z_df: pd.DataFrame) -> dict[str, pd.Series]:
    """全 C(11,2) = 55 ペアの差分 z_a - z_b を作る (a < b 辞書順)."""
    pairs: dict[str, pd.Series] = {}
    for a, b in combinations(INDICATORS, 2):
        key = f"{a}__minus__{b}"
        pairs[key] = z_df[a] - z_df[b]
    return pairs


def shock_type_event_study(pair_series: dict[str, pd.Series]) -> dict:
    """
    shock type ごとに各ペア差分の event 平均 Δσ を計算.

    Returns
    -------
    {
      "trade_policy": {
        "events": [...],
        "n_events_with_data": int,
        "per_pair": { pair_key: {"mean_dsigma": float, "per_event": {...}} }
      },
      ...
    }
    """
    out: dict = {}
    for shock, events in EVENTS.items():
        per_pair: dict = {}
        used_events: list[str] = []
        for pair_key, s in pair_series.items():
            event_vals: dict[str, float | None] = {}
            for date, label in events:
                v = event_delta_sigma(s, date, EVENT_PRE, EVENT_POST)
                event_vals[date] = v
            valid = [v for v in event_vals.values() if v is not None]
            mean_dsigma = float(np.mean(valid)) if valid else None
            per_pair[pair_key] = {
                "mean_dsigma": mean_dsigma,
                "per_event": event_vals,
            }
        # used_events: 少なくとも 1 ペアで data あった event
        for date, _ in events:
            if any(per_pair[p]["per_event"].get(date) is not None
                   for p in per_pair):
                used_events.append(date)
        out[shock] = {
            "events": [{"date": d, "label": l} for d, l in events],
            "n_events_with_data": len(used_events),
            "events_used": used_events,
            "per_pair": per_pair,
        }
    return out


def rank_ediv(study: dict) -> dict:
    """
    e_div = z(n_unb_total) - z(L1) のランクを各 shock type の 55 ペア分布
    の中で算出 (絶対値での順位).

    Returns
    -------
    {
      shock: {
        "ediv_key": str,
        "ediv_dsigma": float,
        "n_pairs": int,
        "rank_by_abs": int,      # 1 が絶対値最大
        "percentile_by_abs": float,  # 0-100, 高いほど絶対値で上位 (上位 5% = 95+)
        "rank_signed_positive": int,
        "rank_signed_negative": int,
        "pattern_classification": "A" | "B" | "C" | "indeterminate",
        "top10_by_abs": [(pair, dsigma), ...],
        "distribution": {"min", "p10", "p25", "median", "p75", "p90", "max"}
      }
    }
    """
    EDIV_KEY = "L1__minus__n_unb_total"  # 辞書順で L1 < n_unb_total
    # 注: 仕様の e_div = z(n_unb_total) - z(L1) は符号が逆.
    # ペアキーは a < b で固定, 値は z_a - z_b なので, 真の e_div は -1 倍.
    out: dict = {}
    for shock, payload in study.items():
        items = [(k, v["mean_dsigma"]) for k, v in payload["per_pair"].items()
                 if v["mean_dsigma"] is not None]
        if not items:
            out[shock] = {"error": "no valid pairs"}
            continue
        # e_div は z(n_unb_total) - z(L1) = -(z(L1) - z(n_unb_total))
        ediv_dsigma_raw = next((d for k, d in items if k == EDIV_KEY), None)
        if ediv_dsigma_raw is None:
            out[shock] = {"error": "ediv pair missing"}
            continue
        ediv_dsigma = -ediv_dsigma_raw  # 仕様に合わせて符号反転

        # 絶対値ランク (1 が大きい)
        sorted_by_abs = sorted(items, key=lambda kv: -abs(kv[1]))
        ediv_abs = abs(ediv_dsigma)
        n_pairs = len(items)
        rank_abs = sum(1 for _, d in items if abs(d) > ediv_abs) + 1
        percentile_abs = 100.0 * (1.0 - (rank_abs - 1) / n_pairs)

        # signed ranks
        rank_pos = sum(1 for _, d in items if d > ediv_dsigma) + 1
        rank_neg = sum(1 for _, d in items if d < ediv_dsigma) + 1

        # 分布の percentile (絶対値ベース)
        abs_arr = np.array([abs(d) for _, d in items])
        distribution = {
            "min": float(np.min(abs_arr)),
            "p10": float(np.percentile(abs_arr, 10)),
            "p25": float(np.percentile(abs_arr, 25)),
            "median": float(np.percentile(abs_arr, 50)),
            "p75": float(np.percentile(abs_arr, 75)),
            "p90": float(np.percentile(abs_arr, 90)),
            "max": float(np.max(abs_arr)),
        }

        # パターン分類
        if percentile_abs >= 90:
            cluster_top = sorted_by_abs[:max(5, n_pairs // 10)]
            cluster_vals = [abs(d) for _, d in cluster_top]
            # 上位 5-10 個が「密集」しているか (max と min の比 < 1.5)
            if min(cluster_vals) > 0 and max(cluster_vals) / min(cluster_vals) < 1.5:
                pattern = "C"  # 複数指標が同程度に良い
            else:
                pattern = "A"  # e_div が突出
        elif percentile_abs >= 50:
            pattern = "B"  # e_div は中央値〜上位だが特別ではない
        else:
            pattern = "B"  # e_div は並 (中央値以下)

        # 符号も考慮した e_div の正当性: e_div は「ストレス時に正方向に動く」
        # 設計だが (n_unb 増 + L1 減 = strong sign-flip), 期待方向との一致も見る
        out[shock] = {
            "ediv_key": "z(n_unb_total) - z(L1)",
            "ediv_dsigma": ediv_dsigma,
            "n_pairs": n_pairs,
            "rank_by_abs": rank_abs,
            "percentile_by_abs": percentile_abs,
            "rank_signed_above": rank_pos,
            "rank_signed_below": rank_neg,
            "pattern_classification": pattern,
            "top10_by_abs": [
                {"pair": k.replace("__minus__", " - "),
                 "dsigma": float(d),
                 "abs_dsigma": float(abs(d))}
                for k, d in sorted_by_abs[:10]
            ],
            "distribution_abs": distribution,
        }
    return out


def overall_histogram(study: dict, n_bins: int = 30) -> dict:
    """全 shock type をまとめた効果量分布 (絶対値) のヒストグラム."""
    all_vals: list[float] = []
    for shock, payload in study.items():
        for k, v in payload["per_pair"].items():
            if v["mean_dsigma"] is not None:
                all_vals.append(abs(v["mean_dsigma"]))
    if not all_vals:
        return {"counts": [], "edges": []}
    counts, edges = np.histogram(all_vals, bins=n_bins)
    return {
        "n_samples": len(all_vals),
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "mean": float(np.mean(all_vals)),
        "median": float(np.median(all_vals)),
        "p90": float(np.percentile(all_vals, 90)),
        "p95": float(np.percentile(all_vals, 95)),
    }


def main() -> None:
    print(f"[load] {INPUT_CSV}")
    raw = load_indicators()
    print(f"  date range: {raw.index.min().date()} -> {raw.index.max().date()}")
    print(f"  n_rows: {len(raw)}, indicators: {len(INDICATORS)}")

    print(f"[zscore] rolling window = {Z_WINDOW}")
    z_df = pd.DataFrame({c: rolling_zscore(raw[c], Z_WINDOW)
                          for c in INDICATORS}, index=raw.index)

    print(f"[pairs] C({len(INDICATORS)}, 2) = "
          f"{len(list(combinations(INDICATORS, 2)))} pairs")
    pair_series = build_pair_diffs(z_df)

    print(f"[events] {sum(len(v) for v in EVENTS.values())} events "
          f"across {len(EVENTS)} shock types")
    study = shock_type_event_study(pair_series)
    for shock, payload in study.items():
        print(f"  {shock}: n_events_with_data="
              f"{payload['n_events_with_data']}")

    print(f"[rank] e_div = z(n_unb_total) - z(L1) ranking within "
          f"{len(pair_series)} pairs")
    ranks = rank_ediv(study)
    for shock, info in ranks.items():
        if "error" in info:
            print(f"  {shock}: {info['error']}")
            continue
        print(f"  {shock}: ediv Δσ = {info['ediv_dsigma']:+.4f}, "
              f"abs rank = {info['rank_by_abs']}/{info['n_pairs']} "
              f"(top {100 - info['percentile_by_abs']:.1f}%), "
              f"pattern = {info['pattern_classification']}")

    print("[hist] overall effect size distribution")
    hist = overall_histogram(study)

    result = {
        "meta": {
            "input": str(INPUT_CSV.relative_to(REPO)),
            "date_range": [str(raw.index.min().date()),
                           str(raw.index.max().date())],
            "n_indicators": len(INDICATORS),
            "indicators": INDICATORS,
            "n_pairs": len(pair_series),
            "z_window": Z_WINDOW,
            "event_pre": EVENT_PRE,
            "event_post": EVENT_POST,
            "events": EVENTS,
            "ediv_definition": "z(n_unb_total) - z(L1)",
            "ediv_pair_key_internal": "L1__minus__n_unb_total (= -ediv)",
        },
        "shock_event_study": study,
        "ediv_ranks": ranks,
        "overall_distribution": hist,
    }

    OUTPUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
