"""
Granger 因果性検定 (政策イベント → e_div の方向性実証)
================================================================

目的:
  permutation test では「政策ショック前後で e_div がランダム日と統計的に違う動き」
  までは示せた (Section 11 の robustness 群参照). 残る批判は
    「相関は示せても causation じゃないよね」
  である. 本スクリプトは Granger 因果性検定で
    「政策イベント → e_div の動き」
  の方向性に踏み込み, 因果方向性まで実証する.

設定:
  - 入力時系列:
      e_div = z(n_unb) - z(L1_H1)   (Section 10.5 と同じ expanding z-score)
      L1    = L1_H1 の生値
      n_unb = unbalanced cycle 数
    いずれも data/gamma_timeseries_w30.csv
  - event dummy:
      trade_policy_dummy(t)     = 1 if t in trade_policy event date or ±2 営業日
      market_structure_dummy(t) = 1 if t in market_structure event date or ±2 営業日
  - statsmodels.tsa.stattools.grangercausalitytests を lag=1,3,5,10 で実施
  - 2 方向: (event → indicator) と (indicator → event)

p 値解釈:
  - event → e_div で p < 0.05 → event が e_div を Granger-cause する
                                = 「政策ショックが e_div の動きを統計的に先行する」
                                = 主張強化 (causation 方向性まで実証)
  - 両方向 p ≥ 0.05            → Granger では因果は検出できず, permutation の枠を超えない
  - 両方向 p < 0.05            → event 自体が予測可能 (市場が織り込み済み) で双方向に動く

出力:
  data/causal_granger_results.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
INPUT_CSV = DATA_DIR / "gamma_timeseries_w30.csv"
EVENTS_JSON = DATA_DIR / "events_8y.json"
OUTPUT_JSON = DATA_DIR / "causal_granger_results.json"

# ===== パラメータ =====
EVENT_HALF_WIDTH_BDAYS = 2          # event ±2 営業日を 1 として扱う
LAGS = [1, 3, 5, 10]                # Granger 検定の lag
ALPHA = 0.05                        # 有意水準
RNG_SEED = 20260526


def expanding_zscore(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """過去のみ expanding window mean/std で z 化 (look-ahead free)."""
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=1)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def build_event_dummy(idx: pd.DatetimeIndex,
                      event_dates: list[pd.Timestamp],
                      half_width_bdays: int = EVENT_HALF_WIDTH_BDAYS
                      ) -> pd.Series:
    """
    event 日 ±half_width 営業日に 1, それ以外 0 のバイナリ時系列.
    """
    dummy = pd.Series(0, index=idx, dtype=float)
    for d in event_dates:
        pos = idx.get_indexer([d], method="nearest")[0]
        if pos < 0:
            continue
        lo = max(0, pos - half_width_bdays)
        hi = min(len(idx), pos + half_width_bdays + 1)
        dummy.iloc[lo:hi] = 1.0
    return dummy


def run_granger(y: pd.Series, x: pd.Series,
                lags: list[int]) -> dict:
    """
    H0: x は y を Granger-cause しない (= y を予測する役に立たない)
    statsmodels の grangercausalitytests は内部で
       y_t を y_{t-1..t-L} で説明する restricted model と
       y_t を y_{t-1..t-L}, x_{t-1..t-L} で説明する unrestricted model
    を比較する. 入力 DataFrame は [y, x] の順.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    # 定数列は捨てる (gcatests がエラーになる)
    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return {"n_obs": int(len(df)), "lags": {},
                "error": "constant series"}
    if len(df) < max(lags) * 5:
        return {"n_obs": int(len(df)), "lags": {},
                "error": f"insufficient data (n={len(df)} for max_lag={max(lags)})"}
    out: dict = {"n_obs": int(len(df)), "lags": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = grangercausalitytests(df[["y", "x"]].values,
                                        maxlag=max(lags), verbose=False)
        except Exception as exc:
            return {"n_obs": int(len(df)), "lags": {}, "error": str(exc)}
    for lag in lags:
        if lag not in res:
            continue
        stats = res[lag][0]
        # 4 つの検定統計量がある (ssr_ftest, ssr_chi2test, lrtest, params_ftest)
        # 最も保守的に F 検定 (ssr_ftest) を主指標とする
        f_stat, f_p, f_df1, f_df2 = stats["ssr_ftest"]
        chi2_stat, chi2_p, chi2_df = stats["ssr_chi2test"]
        out["lags"][lag] = {
            "f_stat": float(f_stat),
            "f_p": float(f_p),
            "chi2_stat": float(chi2_stat),
            "chi2_p": float(chi2_p),
            "significant_at_0.05": bool(f_p < ALPHA),
        }
    return out


def summarize_direction(name: str, result: dict) -> str:
    """short 1-行サマリーを print 用に."""
    if "error" in result and not result.get("lags"):
        return f"  [{name:<40}] ERROR: {result['error']}"
    p_min_lag = None
    p_min_val = None
    for lag, stats in result["lags"].items():
        if p_min_val is None or stats["f_p"] < p_min_val:
            p_min_val = stats["f_p"]
            p_min_lag = lag
    if p_min_val is None:
        return f"  [{name:<40}] no result"
    mark = "***" if p_min_val < 0.01 else ("** " if p_min_val < 0.05 else "   ")
    return (f"  [{name:<40}] n={result['n_obs']:<4} "
            f"min p (F) = {p_min_val:.4f} at lag={p_min_lag} {mark}")


def main() -> None:
    print("=== Granger 因果性検定 (政策ショック → e_div) ===")

    # ----- 1. 時系列ロード -----
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date").set_index("date")
    df = df.dropna(subset=["L1_H1", "n_unb"]).copy()
    print(f"loaded: {df.shape}, range "
          f"{df.index.min().date()} -> {df.index.max().date()}")

    z_L1 = expanding_zscore(df["L1_H1"])
    z_unb = expanding_zscore(df["n_unb"])
    e_div = (z_unb - z_L1).dropna()
    L1_raw = df["L1_H1"].dropna()
    n_unb_raw = df["n_unb"].dropna()
    print(f"e_div n = {len(e_div)}, "
          f"range {e_div.index.min().date()} -> {e_div.index.max().date()}")

    # ----- 2. event dummy 構築 -----
    events = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    trade_dates = [pd.Timestamp(e["date"]) for e in events
                   if e["category"] == "trade_policy"]
    market_dates = [pd.Timestamp(e["date"]) for e in events
                    if e["category"] == "market_structure"]
    print(f"events loaded: trade_policy={len(trade_dates)}, "
          f"market_structure={len(market_dates)}")

    common_idx = e_div.index
    trade_dummy = build_event_dummy(common_idx, trade_dates)
    market_dummy = build_event_dummy(common_idx, market_dates)
    print(f"trade dummy 1's: {int(trade_dummy.sum())}/{len(trade_dummy)} "
          f"({100 * trade_dummy.mean():.2f}%)")
    print(f"market dummy 1's: {int(market_dummy.sum())}/{len(market_dummy)} "
          f"({100 * market_dummy.mean():.2f}%)")

    # ----- 3. Granger 検定 -----
    print("\n--- forward direction (event → indicator) ---")
    fwd_pairs = [
        ("trade_policy → e_div",       e_div,                       trade_dummy),
        ("trade_policy → L1",          L1_raw.reindex(common_idx),  trade_dummy),
        ("trade_policy → n_unb",       n_unb_raw.reindex(common_idx), trade_dummy),
        ("market_structure → e_div",   e_div,                       market_dummy),
        ("market_structure → L1",      L1_raw.reindex(common_idx),  market_dummy),
        ("market_structure → n_unb",   n_unb_raw.reindex(common_idx), market_dummy),
    ]
    forward_results: dict[str, dict] = {}
    for name, y, x in fwd_pairs:
        # x → y を検定: grangercausalitytests は [y, x] の順で x → y の H0 を検定
        res = run_granger(y, x, LAGS)
        forward_results[name] = res
        print(summarize_direction(name, res))

    print("\n--- reverse direction (indicator → event) ---")
    rev_pairs = [
        ("e_div → trade_policy",       trade_dummy,  e_div),
        ("L1 → trade_policy",          trade_dummy,  L1_raw.reindex(common_idx)),
        ("n_unb → trade_policy",       trade_dummy,  n_unb_raw.reindex(common_idx)),
        ("e_div → market_structure",   market_dummy, e_div),
        ("L1 → market_structure",      market_dummy, L1_raw.reindex(common_idx)),
        ("n_unb → market_structure",   market_dummy, n_unb_raw.reindex(common_idx)),
    ]
    reverse_results: dict[str, dict] = {}
    for name, y, x in rev_pairs:
        res = run_granger(y, x, LAGS)
        reverse_results[name] = res
        print(summarize_direction(name, res))

    # ----- 4. 解釈 (主要ペア) -----
    def min_p(res: dict) -> float | None:
        if "lags" not in res or not res["lags"]:
            return None
        return min(s["f_p"] for s in res["lags"].values())

    p_fwd_main = min_p(forward_results.get("trade_policy → e_div", {}))
    p_rev_main = min_p(reverse_results.get("e_div → trade_policy", {}))

    if p_fwd_main is None or p_rev_main is None:
        interp = "main pair で結果が得られず, 解釈不能 (データ不足)."
    elif p_fwd_main < ALPHA and p_rev_main >= ALPHA:
        interp = (f"trade_policy → e_div の p={p_fwd_main:.4f} < {ALPHA} かつ "
                  f"逆方向 p={p_rev_main:.4f} ≥ {ALPHA}: "
                  f"政策ショックが e_div を Granger-cause する一方向因果. "
                  f"permutation の相関を超え, 因果方向性まで実証.")
    elif p_fwd_main < ALPHA and p_rev_main < ALPHA:
        interp = (f"trade_policy → e_div: p={p_fwd_main:.4f}, "
                  f"e_div → trade_policy: p={p_rev_main:.4f}, 両方向有意. "
                  f"event 自体が予測可能 (市場が織り込み済み) で双方向に動く解釈.")
    elif p_fwd_main >= ALPHA and p_rev_main < ALPHA:
        interp = (f"e_div → trade_policy が有意 (p={p_rev_main:.4f}) で逆方向の方が強い. "
                  f"政策の事前リーク可能性 / 市場が政策を先読みする方向の弱い因果.")
    else:
        interp = (f"両方向とも有意でない (fwd p={p_fwd_main:.4f}, "
                  f"rev p={p_rev_main:.4f}). Granger では因果は示せず, "
                  f"permutation の枠を超えない. より厳密な手法 (DAG, IV) は future work.")
    print(f"\n[interpretation] {interp}")

    # ----- 5. 出力 -----
    output = {
        "meta": {
            "input_csv": str(INPUT_CSV.name),
            "events_json": str(EVENTS_JSON.name),
            "n_obs": int(len(e_div)),
            "date_range": [str(e_div.index.min().date()),
                           str(e_div.index.max().date())],
            "n_trade_events": len(trade_dates),
            "n_market_events": len(market_dates),
            "event_half_width_bdays": EVENT_HALF_WIDTH_BDAYS,
            "lags": LAGS,
            "alpha": ALPHA,
            "test_statistic": "ssr_ftest (F-test on SSR ratio)",
        },
        "forward_direction": {
            "description": "event dummy が indicator を Granger-cause するか",
            "results": forward_results,
        },
        "reverse_direction": {
            "description": "indicator が event dummy を Granger-cause するか",
            "results": reverse_results,
        },
        "summary": {
            "main_pair_forward": {
                "name": "trade_policy → e_div",
                "min_f_p": p_fwd_main,
            },
            "main_pair_reverse": {
                "name": "e_div → trade_policy",
                "min_f_p": p_rev_main,
            },
            "interpretation": interp,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nsaved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
