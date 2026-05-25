"""
VIX 内生性検証 (Robustness via VIX Exclusion)
==============================================

疑い:
  VIX は S&P 500 オプションから計算される指標であり、
  SP500 / NAS100 / 個別株 (AAPL/MSFT/...) などと数学的に独立とは言えない。
  → 相関ネットワークの入力に VIX を含めると、L¹/n_unb の "ショック検知" が
    実際は VIX の自己相関でブーストされている可能性 (内生性リーク疑い)。

検証:
  ohlc_40.parquet から VIX を 1 銘柄だけ除外して 39 銘柄で再計算。
  2025-04 Liberation Day を event window として Δσ (L1 / n_unb / e_div) を測り、
  VIX 込み (baseline 40 銘柄) と VIX 除外 (39 銘柄) で結果を比較する。
  「ほぼ同じ」なら VIX の内生性は実質的に banner finding に影響しない、
  と言える。

出力:
  data/robustness_vix_exclusion.json
    - baseline_40:    40 銘柄 (VIX 込み) の Δσ
    - exclude_vix_39: 39 銘柄 (VIX 抜き) の Δσ
    - comparison:     差分と判定
    - meta:           設定情報

実装方針:
  robustness_subsample.py の compute_indicators_for_period / event_response を流用。
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
LIB = HERE / "lib"
sys.path.insert(0, str(LIB))

from persistent_homology import persistence_diagram, persistence_summary  # noqa: E402
from market_category import MarketCategory  # noqa: E402
from homology import signed_cycle_balance  # noqa: E402

DATA_DIR = HERE.parent / "data"

# ===== パラメータ (robustness_subsample.py と同一にして fair に比較) =====
WINDOW = 30
THRESHOLD = 0.3

RETURNS_START = "2025-01-01"
RETURNS_END = "2025-04-30"
BASELINE_START = "2025-02-15"
BASELINE_END = "2025-03-25"
EVENT_START = "2025-04-02"
EVENT_END = "2025-04-15"


def compute_indicators_for_period(closes: pd.DataFrame,
                                   symbols: list[str],
                                   period_start: str,
                                   period_end: str,
                                   window: int = WINDOW,
                                   threshold: float = THRESHOLD) -> pd.DataFrame:
    """指定 symbols / period に対して日次の L1_H1, n_unb を計算."""
    sub = closes[symbols].copy()
    returns = sub.pct_change()
    start_ts = pd.Timestamp(period_start)
    end_ts = pd.Timestamp(period_end)
    pad_days = int(window * 1.7) + 5
    pad_start = start_ts - pd.Timedelta(days=pad_days)
    returns = returns.loc[(returns.index >= pad_start) & (returns.index <= end_ts)]

    rows = []
    n = len(returns)
    for t_idx in range(window, n):
        date = returns.index[t_idx - 1]
        if date < start_ts or date > end_ts:
            continue
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
        if win_clean.shape[1] < 5:
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


def event_response(df: pd.DataFrame) -> dict:
    """baseline window で z 化し、event window 平均 (= Δσ) を返す."""
    df = df.dropna(subset=["L1_H1", "n_unb"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    bmask = (df["date"] >= BASELINE_START) & (df["date"] <= BASELINE_END)
    emask = (df["date"] >= EVENT_START) & (df["date"] <= EVENT_END)
    base = df[bmask]
    evt = df[emask]
    if len(base) < 5 or len(evt) < 3:
        return {"delta_sigma_L1": float("nan"),
                "delta_sigma_unb": float("nan"),
                "delta_sigma_ediv": float("nan"),
                "n_baseline": int(len(base)),
                "n_event": int(len(evt))}

    mu_L1, sd_L1 = base["L1_H1"].mean(), base["L1_H1"].std(ddof=1)
    mu_unb, sd_unb = base["n_unb"].mean(), base["n_unb"].std(ddof=1)
    if sd_L1 == 0 or sd_unb == 0 or not np.isfinite(sd_L1) or not np.isfinite(sd_unb):
        return {"delta_sigma_L1": float("nan"),
                "delta_sigma_unb": float("nan"),
                "delta_sigma_ediv": float("nan"),
                "n_baseline": int(len(base)),
                "n_event": int(len(evt))}

    z_L1_evt = (evt["L1_H1"] - mu_L1) / sd_L1
    z_unb_evt = (evt["n_unb"] - mu_unb) / sd_unb
    z_ediv_evt = z_unb_evt - z_L1_evt
    return {
        "delta_sigma_L1": float(z_L1_evt.mean()),
        "delta_sigma_unb": float(z_unb_evt.mean()),
        "delta_sigma_ediv": float(z_ediv_evt.mean()),
        "n_baseline": int(len(base)),
        "n_event": int(len(evt)),
        "mu_L1_baseline": float(mu_L1),
        "sd_L1_baseline": float(sd_L1),
        "mu_unb_baseline": float(mu_unb),
        "sd_unb_baseline": float(sd_unb),
    }


def main():
    print("=== Robustness via VIX Exclusion (40 -> 39, drop VIX) ===")
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    all_symbols = list(closes.columns)
    print(f"Loaded ohlc_40.parquet: shape={closes.shape}, n_symbols={len(all_symbols)}")
    if "VIX" not in all_symbols:
        raise RuntimeError("VIX が ohlc_40.parquet に含まれていません")

    symbols_no_vix = [s for s in all_symbols if s != "VIX"]
    print(f"  dropped: ['VIX']  ->  kept {len(symbols_no_vix)} symbols")

    # ===== baseline (40 銘柄 / VIX 込み) =====
    t0 = time.time()
    print("\n[baseline] 40 symbols (incl. VIX) ...")
    base_df = compute_indicators_for_period(closes, all_symbols,
                                            RETURNS_START, RETURNS_END)
    base_resp = event_response(base_df)
    print(f"  done in {time.time()-t0:.1f}s, n_rows={len(base_df)}")
    print(f"  Δσ  L1   = {base_resp['delta_sigma_L1']:+.3f}")
    print(f"  Δσ  unb  = {base_resp['delta_sigma_unb']:+.3f}")
    print(f"  Δσ  ediv = {base_resp['delta_sigma_ediv']:+.3f}")

    # ===== exclude VIX (39 銘柄) =====
    t0 = time.time()
    print("\n[exclude VIX] 39 symbols ...")
    novix_df = compute_indicators_for_period(closes, symbols_no_vix,
                                             RETURNS_START, RETURNS_END)
    novix_resp = event_response(novix_df)
    print(f"  done in {time.time()-t0:.1f}s, n_rows={len(novix_df)}")
    print(f"  Δσ  L1   = {novix_resp['delta_sigma_L1']:+.3f}")
    print(f"  Δσ  unb  = {novix_resp['delta_sigma_unb']:+.3f}")
    print(f"  Δσ  ediv = {novix_resp['delta_sigma_ediv']:+.3f}")

    # ===== 比較 =====
    def diff_block(base_v: float, novix_v: float) -> dict:
        diff = novix_v - base_v
        rel = (abs(diff) / abs(base_v)) if (base_v not in (0.0,) and np.isfinite(base_v)) else float("nan")
        return {
            "baseline_40": float(base_v),
            "exclude_vix_39": float(novix_v),
            "diff_abs": float(diff),
            "diff_rel": float(rel),
            "sign_preserved": bool(np.sign(base_v) == np.sign(novix_v)),
        }

    comparison = {
        "delta_sigma_L1":   diff_block(base_resp["delta_sigma_L1"],   novix_resp["delta_sigma_L1"]),
        "delta_sigma_unb":  diff_block(base_resp["delta_sigma_unb"],  novix_resp["delta_sigma_unb"]),
        "delta_sigma_ediv": diff_block(base_resp["delta_sigma_ediv"], novix_resp["delta_sigma_ediv"]),
    }

    # 判定: 主要 finding (Δσ_ediv) について「方向が保たれ、絶対差が 0.5σ 以下」を OK とする
    ediv_diff_abs = abs(comparison["delta_sigma_ediv"]["diff_abs"])
    sign_ok = comparison["delta_sigma_ediv"]["sign_preserved"]
    verdict_ediv = bool(sign_ok and ediv_diff_abs <= 0.5)

    summary_txt = (
        f"VIX 込み Δσ_ediv = {base_resp['delta_sigma_ediv']:+.3f}, "
        f"VIX 抜き Δσ_ediv = {novix_resp['delta_sigma_ediv']:+.3f}, "
        f"差 = {comparison['delta_sigma_ediv']['diff_abs']:+.3f}σ. "
        f"{'方向一致 + 0.5σ 以内 → VIX の内生性は banner finding に実質影響なし。' if verdict_ediv else '差が大きい → VIX 寄与を要再評価。'}"
    )

    out = {
        "meta": {
            "window": WINDOW,
            "threshold": THRESHOLD,
            "returns_period": [RETURNS_START, RETURNS_END],
            "baseline_window": [BASELINE_START, BASELINE_END],
            "event_window": [EVENT_START, EVENT_END],
            "event_name": "2025-04 Liberation Day",
            "n_symbols_baseline": len(all_symbols),
            "n_symbols_exclude_vix": len(symbols_no_vix),
            "dropped_symbols": ["VIX"],
        },
        "baseline_40": base_resp,
        "exclude_vix_39": novix_resp,
        "comparison_baseline_vs_no_vix": comparison,
        "verdict": {
            "ediv_sign_preserved": sign_ok,
            "ediv_abs_diff_sigma": float(ediv_diff_abs),
            "ediv_passes_0p5sigma": verdict_ediv,
            "summary": summary_txt,
        },
    }

    out_path = DATA_DIR / "robustness_vix_exclusion.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("\n=== Comparison: VIX in (40) vs VIX out (39) ===")
    for k, c in comparison.items():
        flag = "OK (sign preserved)" if c["sign_preserved"] else "FLIP"
        print(f"  {k:18s}  base={c['baseline_40']:+.3f}  no_vix={c['exclude_vix_39']:+.3f}  "
              f"diff={c['diff_abs']:+.3f}  rel={c['diff_rel']:.2%}  {flag}")
    print(f"\nVerdict (Δσ_ediv): {summary_txt}")


if __name__ == "__main__":
    main()
