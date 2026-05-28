"""
tz-aware ohlc で Liberation Day event study を再計算し、tz-naive 版と比較する.

Section 11.2 (4) の future work「tz-aware resampling で高頻度 event detection 改善」
の検証スクリプト。

入力:
  data/ohlc_40.parquet          : 既存 (tz-naive UTC midnight)
  data/ohlc_40_tz_aware.parquet : 新規 (20:00 UTC anchor / asof close-time-shifted)

処理:
  ``scripts/subgraph_eventstudy.py`` の compute_indicators_for_period /
  event_response と同一ロジックで、両 parquet について全 40 銘柄での
  Δσ_L¹ / Δσ_n_unb / Δσ_e_div を計算する。
  baseline / event window は同じ:
    baseline = 2025-02-15 → 2025-03-25
    event    = 2025-04-02 → 2025-04-15

出力:
  data/eventstudy_tz_aware_results.json
    {
      meta: {...},
      tz_naive: { delta_sigma_L1, delta_sigma_unb, delta_sigma_ediv, ... },
      tz_aware: { delta_sigma_L1, delta_sigma_unb, delta_sigma_ediv, ... },
      diff:     { L1: tz_aware - tz_naive, ... },
      verdict:  "日次粒度では tz-naive で十分" or "tz-aware で要修正",
    }

判定:
  |Δ| < 0.5σ なら "日次粒度では tz-naive で十分"、それ以上なら "要修正"。
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))

from persistent_homology import persistence_diagram, persistence_summary  # noqa: E402
from market_category import MarketCategory  # noqa: E402
from homology import signed_cycle_balance  # noqa: E402

# subgraph_eventstudy から定数だけ拝借 (実装は再掲する)
from subgraph_eventstudy import (  # noqa: E402
    WINDOW, THRESHOLD, MIN_SYMBOLS,
    RETURNS_START, RETURNS_END,
    BASELINE_START, BASELINE_END,
    EVENT_START, EVENT_END,
    compute_indicators_for_period,
    event_response,
)

DATA_DIR = HERE.parent / "data"

# tz-aware 版で「Δσ が tz-naive とどれくらい違ったら修正必須か」の閾値
DELTA_TOLERANCE_SIGMA = 0.5


def run(label: str, parquet_path: Path,
        restrict_to: list[str] | None = None) -> dict:
    """1 つの parquet で全銘柄 (or restrict_to) の Δσ を計算する.

    Args:
        label: ログ表示用ラベル。
        parquet_path: 入力 parquet パス。
        restrict_to: ``None`` なら parquet 内全銘柄、リスト指定なら部分集合で計算
            する。fair な比較のため tz-naive vs tz-aware を同じ銘柄集合で揃える
            ときに使う。
    """
    print(f"\n[{label}] {parquet_path.name} ...")
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    closes = pd.read_parquet(parquet_path)
    if restrict_to is not None:
        syms = [s for s in restrict_to if s in closes.columns]
    else:
        syms = list(closes.columns)
    print(f"  shape={closes.shape}, symbols={len(syms)}")
    t0 = time.time()
    df = compute_indicators_for_period(closes, syms, RETURNS_START, RETURNS_END)
    resp = event_response(df)
    print(f"  done in {time.time()-t0:.1f}s")
    if resp.get("delta_sigma_ediv") is None:
        print(f"  DEGENERATE: {resp.get('degenerate_reason')}")
    else:
        print(f"  Δσ L¹   = {resp['delta_sigma_L1']:+.3f}")
        print(f"  Δσ n_unb = {resp['delta_sigma_unb']:+.3f}")
        print(f"  Δσ e_div = {resp['delta_sigma_ediv']:+.3f}")
    return {
        "parquet": parquet_path.name,
        "n_symbols": len(syms),
        "symbols": syms,
        **resp,
    }


def main() -> None:
    """tz-naive と tz-aware の Liberation Day event study 結果を比較・保存する."""
    print("=== Liberation Day event study: tz-naive vs tz-aware ===")

    # tz-aware の銘柄集合 (= 取得失敗を除いた共通集合) を fair な比較対象とする
    tz_aware_cols = list(
        pd.read_parquet(DATA_DIR / "ohlc_40_tz_aware.parquet").columns
    )

    # 既存実装と完全一致する 40 銘柄版 (公開済みの数値と整合)
    tz_naive_full = run("tz-naive (full 40)", DATA_DIR / "ohlc_40.parquet")
    # tz-aware と同じ 39 銘柄に揃えた版 (公平比較用)
    tz_naive = run(
        "tz-naive (matched 39)", DATA_DIR / "ohlc_40.parquet",
        restrict_to=tz_aware_cols,
    )
    tz_aware = run(
        "tz-aware (39)", DATA_DIR / "ohlc_40_tz_aware.parquet",
        restrict_to=tz_aware_cols,
    )

    # diff (tz_aware - tz_naive)
    diff: dict[str, float | None] = {}
    for k in ("delta_sigma_L1", "delta_sigma_unb", "delta_sigma_ediv"):
        a, b = tz_aware.get(k), tz_naive.get(k)
        if a is None or b is None:
            diff[k] = None
        else:
            diff[k] = float(a - b)

    # verdict
    e_div_diff = diff.get("delta_sigma_ediv")
    if e_div_diff is None:
        verdict = "Δσ_e_div が片方で算出不能 → 比較できない (要再調査)"
    elif abs(e_div_diff) < DELTA_TOLERANCE_SIGMA:
        verdict = (
            f"Δσ_e_div の差 {e_div_diff:+.3f}σ は {DELTA_TOLERANCE_SIGMA}σ 以内 → "
            "日次粒度では tz-naive で十分、tz 揃えで主要結論は変わらない"
        )
    else:
        verdict = (
            f"Δσ_e_div の差 {e_div_diff:+.3f}σ は {DELTA_TOLERANCE_SIGMA}σ を超える → "
            "tz-aware 版を main results に採用するか、bias を limitation に明記すべき"
        )
    print("\n=== diff (tz_aware - tz_naive) ===")
    for k, v in diff.items():
        print(f"  {k}: {v:+.3f}" if v is not None else f"  {k}: None")
    print(f"\nVerdict: {verdict}")

    out = {
        "meta": {
            "window": WINDOW,
            "threshold": THRESHOLD,
            "min_symbols": MIN_SYMBOLS,
            "returns_period": [RETURNS_START, RETURNS_END],
            "baseline_window": [BASELINE_START, BASELINE_END],
            "event_window": [EVENT_START, EVENT_END],
            "event_name": "2025-04 Liberation Day (reciprocal tariff cluster)",
            "delta_tolerance_sigma": DELTA_TOLERANCE_SIGMA,
            "note": (
                "tz-naive = UTC midnight close-to-close (既存 ohlc_40.parquet)。"
                "tz-aware = 各銘柄を yfinance individual fetch し、native tz の close 時刻に "
                "shift してから 20:00 UTC daily grid に asof-resample (新規 ohlc_40_tz_aware.parquet)。"
                "DXY (DX=F) は yfinance で delisted のため tz-aware では skip → 39 銘柄。"
                "公平比較のため tz-naive 側も同じ 39 銘柄で再計算した版を main diff に使用。"
            ),
        },
        "tz_naive_full_40_reference": tz_naive_full,
        "tz_naive": tz_naive,
        "tz_aware": tz_aware,
        "diff_tz_aware_minus_naive": diff,
        "verdict": verdict,
    }

    out_path = DATA_DIR / "eventstudy_tz_aware_results.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
