"""
Cluster permutation test (Autocorrelation 批判への検証)
========================================================

目的:
  Liberation Day 関連 5 イベント (2025-04-02, 04, 08, 09, 15) は
  「2 週間に集中した因果的に連鎖した cluster」であり, 互いに独立とは言えない.
  これらを n=5 として permutation すると有意性を過大評価する恐れがある.

  本検証では:
    1. 5 件を independent (n=5) として permutation test
    2. 1 つの cluster event (=4/2 〜 4/15 期間) として permutation test (n=1)
  両者の p 値を比較し, 結論の頑健性を判定する.

  - cluster (n=1) でも有意 → 結論は autocorrelation バイアスに頑健
  - cluster (n=1) で有意性消失 → 正直に Limitation として記載

検定統計量 (subsample スクリプトと整合):
  e_div = z(n_unb_total) - z(L1) , baseline window (2025-02-15〜2025-03-25)
                                   の mean/std を使った固定 z 化.
  本スクリプトでは「baseline」を「event 日の直前 28 営業日」として一般化
  し, 任意の null date に対しても同じ手順で Δσ を計算できるようにする.

  Δσ = mean(z in event window)
  - n=5 の場合: 各 event 日について pre 28 営業日 baseline で z 化し,
                  当該 event 日の z 値を取り, 5 日分を平均
  - n=1 cluster の場合: event window = [4/2, 4/15] (14 日), baseline =
                          直前 28 営業日, Δσ = 期間内 z 平均

permutation:
  各 trial で random null date(s) を選び, 同じ pre/event window 幅で計算.

出力:
  data/robustness_cluster_perm.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
INPUT_CSV = DATA_DIR / "multi_indicators_w30.csv"
OUTPUT_JSON = DATA_DIR / "robustness_cluster_perm.json"

# ===== パラメータ =====
BASELINE_BDAYS = 28           # event 直前の baseline 営業日数
CLUSTER_EVENT_BDAYS = 10      # 4/2〜4/15 は ≒ 10 営業日
N_PERMUTATIONS = 5000
RNG_SEED = 20250602

# Liberation Day cluster (n=5)
LD_EVENTS = [
    "2025-04-02",  # reciprocal tariff 発表
    "2025-04-04",  # 中国報復関税
    "2025-04-08",  # 90 日 pause 表明
    "2025-04-09",  # 関税一時停止 (一部除外)
    "2025-04-15",  # 中国関税 245% 引き上げ
]
LD_CLUSTER_REPRESENTATIVE = "2025-04-02"  # cluster の起点 (n=1 の代表日)

# random null date 抽出範囲: baseline + cluster event window 分のマージン
NULL_START = "2021-09-01"
NULL_END = "2025-02-15"


def event_delta_sigma_pointwise(series: pd.Series,
                                  event_date: str | pd.Timestamp,
                                  baseline_bdays: int = BASELINE_BDAYS
                                  ) -> float | None:
    """
    指定 event_date 1 点の Δσ.
    event 直前の baseline_bdays 営業日 mean/std で z 化し, event 日の値.
    """
    target = pd.Timestamp(event_date)
    series = series.dropna()
    if target < series.index.min() or target > series.index.max():
        return None
    pos = series.index.get_indexer([target], method="nearest")[0]
    if pos < baseline_bdays:
        return None
    base = series.iloc[pos - baseline_bdays:pos]
    if len(base) < baseline_bdays // 2:
        return None
    mu = base.mean()
    sd = base.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return None
    return float((series.iloc[pos] - mu) / sd)


def event_delta_sigma_window(series: pd.Series,
                              start_date: str | pd.Timestamp,
                              window_bdays: int = CLUSTER_EVENT_BDAYS,
                              baseline_bdays: int = BASELINE_BDAYS
                              ) -> float | None:
    """
    指定 start_date から window_bdays 営業日の Δσ 平均.
    baseline = start_date 直前 baseline_bdays 営業日.
    """
    start = pd.Timestamp(start_date)
    series = series.dropna()
    if start < series.index.min() or start > series.index.max():
        return None
    pos = series.index.get_indexer([start], method="nearest")[0]
    if pos < baseline_bdays or pos + window_bdays > len(series):
        return None
    base = series.iloc[pos - baseline_bdays:pos]
    evt = series.iloc[pos:pos + window_bdays]
    if len(base) < baseline_bdays // 2 or len(evt) < window_bdays // 2:
        return None
    mu = base.mean()
    sd = base.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return None
    z_evt = (evt - mu) / sd
    return float(z_evt.mean())


def main():
    print("=== Cluster Permutation Test ===")
    raw = pd.read_csv(INPUT_CSV, parse_dates=["date"]).set_index("date")
    print(f"Loaded: {raw.shape}, range "
          f"{raw.index.min().date()} -> {raw.index.max().date()}")

    # e_div の素材として L1, n_unb_total の生時系列を使う
    L1 = raw["L1"].dropna()
    n_unb = raw["n_unb_total"].dropna()
    common_idx = L1.index.intersection(n_unb.index)
    L1 = L1.loc[common_idx].sort_index()
    n_unb = n_unb.loc[common_idx].sort_index()
    print(f"raw L1 / n_unb_total: {len(common_idx)} common dates, "
          f"range {common_idx.min().date()} -> {common_idx.max().date()}")

    def ediv_dsigma_pointwise(date):
        z_L1 = event_delta_sigma_pointwise(L1, date)
        z_unb = event_delta_sigma_pointwise(n_unb, date)
        if z_L1 is None or z_unb is None:
            return None
        return z_unb - z_L1

    def ediv_dsigma_window(start_date, window_bdays=CLUSTER_EVENT_BDAYS):
        z_L1 = event_delta_sigma_window(L1, start_date, window_bdays)
        z_unb = event_delta_sigma_window(n_unb, start_date, window_bdays)
        if z_L1 is None or z_unb is None:
            return None
        return z_unb - z_L1

    # ===== 1. observed effects =====
    # n=5 (independent): 各 event 日 1 点の e_div Δσ
    n5_per_event = {}
    n5_vals = []
    for d in LD_EVENTS:
        v = ediv_dsigma_pointwise(d)
        n5_per_event[d] = v
        if v is not None:
            n5_vals.append(v)
    obs_n5 = float(np.mean(n5_vals)) if n5_vals else None

    # n=1 cluster: 代表日 4/2 から 10 営業日 (4/2〜4/15) の窓平均
    obs_n1 = ediv_dsigma_window(LD_CLUSTER_REPRESENTATIVE,
                                 CLUSTER_EVENT_BDAYS)

    print(f"\n[observed]")
    print(f"  n=5 (independent) Δσ mean = {obs_n5:+.4f}")
    for d, v in n5_per_event.items():
        v_str = f"{v:+.4f}" if v is not None else "N/A"
        print(f"     {d}: {v_str}")
    print(f"  n=1 (cluster: {LD_CLUSTER_REPRESENTATIVE} + "
          f"{CLUSTER_EVENT_BDAYS}bdays window) Δσ = {obs_n1:+.4f}")

    # ===== 2. permutation null distributions =====
    null_pool_mask = ((common_idx >= pd.Timestamp(NULL_START))
                      & (common_idx <= pd.Timestamp(NULL_END)))
    null_pool = common_idx[null_pool_mask]
    n_pool = len(null_pool)
    print(f"\n[null pool] {n_pool} candidate dates "
          f"({NULL_START} -> {NULL_END})")

    rng = np.random.default_rng(RNG_SEED)

    # n=5 null
    null_n5 = []
    for _ in range(N_PERMUTATIONS):
        # 5 個の null date を非復元抽出 (cluster と同じ「5 点」サンプル)
        picks = rng.choice(null_pool, size=5, replace=False)
        vals = []
        for p in picks:
            v = ediv_dsigma_pointwise(p)
            if v is not None:
                vals.append(v)
        if len(vals) >= 3:
            null_n5.append(float(np.mean(vals)))
    null_n5 = np.array(null_n5)

    # n=1 null (cluster と同じ window 幅で評価)
    null_n1 = []
    for _ in range(N_PERMUTATIONS):
        p = rng.choice(null_pool, size=1, replace=False)[0]
        v = ediv_dsigma_window(p, CLUSTER_EVENT_BDAYS)
        if v is not None:
            null_n1.append(v)
    null_n1 = np.array(null_n1)

    # ===== 3. p values (two-sided & one-sided positive) =====
    def p_two_sided(obs, null):
        return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))

    def p_one_sided(obs, null):
        return float((np.sum(null >= obs) + 1) / (len(null) + 1))

    p_n5_two = p_two_sided(obs_n5, null_n5)
    p_n5_one = p_one_sided(obs_n5, null_n5)
    p_n1_two = p_two_sided(obs_n1, null_n1)
    p_n1_one = p_one_sided(obs_n1, null_n1)

    print(f"\n[permutation results]")
    print(f"  n=5 (independent):")
    print(f"    null distribution: n={len(null_n5)}, "
          f"mean={null_n5.mean():+.4f}, std={null_n5.std(ddof=1):.4f}, "
          f"p05={np.percentile(null_n5, 5):+.4f}, "
          f"p95={np.percentile(null_n5, 95):+.4f}")
    print(f"    p (two-sided)   = {p_n5_two:.4f}")
    print(f"    p (one-sided +) = {p_n5_one:.4f}")
    print(f"  n=1 (cluster):")
    print(f"    null distribution: n={len(null_n1)}, "
          f"mean={null_n1.mean():+.4f}, std={null_n1.std(ddof=1):.4f}, "
          f"p05={np.percentile(null_n1, 5):+.4f}, "
          f"p95={np.percentile(null_n1, 95):+.4f}")
    print(f"    p (two-sided)   = {p_n1_two:.4f}")
    print(f"    p (one-sided +) = {p_n1_one:.4f}")

    # ===== 4. 解釈 =====
    alpha = 0.05
    sig_n5 = p_n5_two < alpha
    sig_n1 = p_n1_two < alpha
    if sig_n5 and sig_n1:
        verdict = ("cluster (n=1) でも有意 -> autocorrelation バイアスに頑健. "
                   "結論は cluster として扱っても変わらない.")
    elif sig_n5 and not sig_n1:
        verdict = ("n=5 で有意だが cluster (n=1) では有意性消失. "
                   "Liberation Day を 5 独立イベントとして数えると過大評価. "
                   "Limitation として明記すべき.")
    elif not sig_n5 and sig_n1:
        verdict = ("n=5 では有意でないが cluster では有意. "
                   "(通常起きにくい. 個別イベントの分散が大きい可能性)")
    else:
        verdict = "両者とも有意でない. event response は noise レベル."

    out = {
        "meta": {
            "baseline_bdays": BASELINE_BDAYS,
            "cluster_event_bdays": CLUSTER_EVENT_BDAYS,
            "n_permutations": N_PERMUTATIONS,
            "rng_seed": RNG_SEED,
            "ld_events": LD_EVENTS,
            "ld_cluster_representative": LD_CLUSTER_REPRESENTATIVE,
            "null_pool_range": [NULL_START, NULL_END],
            "n_null_pool_dates": int(n_pool),
            "indicator": ("e_div = z(n_unb_total) - z(L1), "
                          "baseline = pre 28 business days mean/std"),
        },
        "observed": {
            "n5_independent_mean_dsigma": obs_n5,
            "n5_per_event": n5_per_event,
            "n1_cluster_dsigma": obs_n1,
        },
        "permutation": {
            "n5_independent": {
                "n_valid_trials": int(len(null_n5)),
                "null_mean": float(null_n5.mean()),
                "null_std": float(null_n5.std(ddof=1)),
                "null_p05": float(np.percentile(null_n5, 5)),
                "null_p95": float(np.percentile(null_n5, 95)),
                "p_value_two_sided": p_n5_two,
                "p_value_one_sided_pos": p_n5_one,
                "significant_at_0.05_two_sided": bool(sig_n5),
            },
            "n1_cluster": {
                "n_valid_trials": int(len(null_n1)),
                "null_mean": float(null_n1.mean()),
                "null_std": float(null_n1.std(ddof=1)),
                "null_p05": float(np.percentile(null_n1, 5)),
                "null_p95": float(np.percentile(null_n1, 95)),
                "p_value_two_sided": p_n1_two,
                "p_value_one_sided_pos": p_n1_one,
                "significant_at_0.05_two_sided": bool(sig_n1),
            },
        },
        "verdict": verdict,
    }

    OUTPUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"\n[verdict] {verdict}")


if __name__ == "__main__":
    main()
