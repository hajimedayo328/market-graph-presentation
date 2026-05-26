"""
Asset-class 別 sub-graph での event study (2025-04 Liberation Day)
==================================================================

背景:
  Section 11.2 (4) で残課題として挙げた「24h 銘柄と日中銘柄を tz-naive UTC
  midnight に揃えると最大 ~30h ずれ」が、main banner finding (Δσ_e_div) に
  どの asset class から効いているのかを調べる。
  もし「FX サブグラフだけで反応している」なら震源は FX、
  「全 asset class で同方向に反応」なら市場横断的ショック、と分けられる。

実装:
  data/symbol_meta.csv の asset_class でグルーピングし、
  各サブセット銘柄だけで 30 日 rolling correlation graph → L¹_H1 / n_unb /
  e_div を計算し、Liberation Day で event study (Δσ) する。
  既存の robustness_subsample / robustness_vix_exclusion と同じ window /
  baseline / event window / threshold を使い、fair に比較。

asset_class グルーピング (CSV の asset_class を 1 段まとめている):
  - FX        : asset_class == 'FX'                    (n=13)
  - INDEX     : asset_class == 'INDEX'                 (n=9)
  - COMMODITY : asset_class in {'METAL','ENERGY','COMMODITY'}  (n=6)
  - STOCK     : asset_class == 'STOCK'                 (n=5)
  - CRYPTO    : asset_class == 'CRYPTO'                (n=2, omit)
  - BOND      : asset_class == 'BOND'                  (n=3, omit)
  - SPECIAL   : asset_class == 'SPECIAL'               (n=2, omit)
  - ALL       : 全 40 銘柄 (baseline 比較用)

n<5 のグループは compute_gamma_timeseries と同じ最小銘柄数ルールで skip する。

tz-aware について:
  本実装は tz-naive 現状ベース (close-to-close on UTC midnight) で各 asset
  class 別の Δσ を出す。tz-aware 版 (米国 20:00 UTC に asof-resample) は
  個別 yfinance 取得が必要で重いため、現バージョンでは未実装。本スクリプトの
  目的は「どの市場が震源か」の特定であり、同じ tz-naive 系列内での相対比較
  なのでこの問題は asset class 間で等しく効く。

出力:
  data/subgraph_eventstudy.json
    - meta:          設定情報 (window, threshold, baseline/event window, ...)
    - all_40:        全銘柄 baseline (Δσ_L1, Δσ_unb, Δσ_ediv, n_baseline, n_event)
    - by_asset_class:
        FX / INDEX / COMMODITY / STOCK ごとに同じ event_response
        (omit したクラスは reason 付きで記録)
    - ranking:       Δσ_ediv が大きい順
    - finding:       震源特定の要約 (日本語)
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

# ===== パラメータ (既存 robustness と同一) =====
WINDOW = 30
THRESHOLD = 0.3
MIN_SYMBOLS = 5  # lib.compute_gamma_timeseries と同じ

RETURNS_START = "2025-01-01"
RETURNS_END = "2025-04-30"
BASELINE_START = "2025-02-15"
BASELINE_END = "2025-03-25"
EVENT_START = "2025-04-02"
EVENT_END = "2025-04-15"


# ===== asset_class グルーピング =====
def build_groups(meta: pd.DataFrame) -> dict[str, list[str]]:
    """symbol_meta.csv の asset_class から sub-graph 用グルーピングを返す."""
    g: dict[str, list[str]] = {}
    g["FX"] = meta.loc[meta["asset_class"] == "FX", "internal"].tolist()
    g["INDEX"] = meta.loc[meta["asset_class"] == "INDEX", "internal"].tolist()
    # METAL / ENERGY / COMMODITY を 1 つの COMMODITY サブグラフにまとめる
    com_mask = meta["asset_class"].isin(["METAL", "ENERGY", "COMMODITY"])
    g["COMMODITY"] = meta.loc[com_mask, "internal"].tolist()
    g["STOCK"] = meta.loc[meta["asset_class"] == "STOCK", "internal"].tolist()
    g["CRYPTO"] = meta.loc[meta["asset_class"] == "CRYPTO", "internal"].tolist()
    g["BOND"] = meta.loc[meta["asset_class"] == "BOND", "internal"].tolist()
    g["SPECIAL"] = meta.loc[meta["asset_class"] == "SPECIAL", "internal"].tolist()
    return g


# ===== 指標計算 (robustness_vix_exclusion.py と同一ロジック) =====
def compute_indicators_for_period(closes: pd.DataFrame,
                                  symbols: list[str],
                                  period_start: str,
                                  period_end: str,
                                  window: int = WINDOW,
                                  threshold: float = THRESHOLD) -> pd.DataFrame:
    """指定 symbols / period に対して日次の L1_H1, n_unb を計算する."""
    sub = closes[symbols].copy()
    returns = sub.pct_change()
    start_ts = pd.Timestamp(period_start)
    end_ts = pd.Timestamp(period_end)
    pad_days = int(window * 1.7) + 5
    pad_start = start_ts - pd.Timedelta(days=pad_days)
    returns = returns.loc[(returns.index >= pad_start) & (returns.index <= end_ts)]

    rows: list[dict] = []
    n = len(returns)
    for t_idx in range(window, n):
        date = returns.index[t_idx - 1]
        if date < start_ts or date > end_ts:
            continue
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
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


def event_response(df: pd.DataFrame) -> dict:
    """baseline window で z 化し、event window 平均 (Δσ) を返す.

    Δσ が計算できない場合は ``None`` を入れ、``degenerate_reason`` で原因を残す.
    NaN を返すと JSON シリアライズ時に標準外の ``NaN`` トークンになるため避ける.
    """
    df = df.dropna(subset=["L1_H1", "n_unb"]).copy()
    if len(df) == 0:
        return {"delta_sigma_L1": None, "delta_sigma_unb": None,
                "delta_sigma_ediv": None,
                "n_baseline": 0, "n_event": 0,
                "degenerate_reason": "empty dataframe after dropna"}
    df["date"] = pd.to_datetime(df["date"])
    bmask = (df["date"] >= BASELINE_START) & (df["date"] <= BASELINE_END)
    emask = (df["date"] >= EVENT_START) & (df["date"] <= EVENT_END)
    base = df[bmask]
    evt = df[emask]
    if len(base) < 5 or len(evt) < 3:
        return {"delta_sigma_L1": None, "delta_sigma_unb": None,
                "delta_sigma_ediv": None,
                "n_baseline": int(len(base)),
                "n_event": int(len(evt)),
                "degenerate_reason": "insufficient baseline/event rows"}

    mu_L1, sd_L1 = base["L1_H1"].mean(), base["L1_H1"].std(ddof=1)
    mu_unb, sd_unb = base["n_unb"].mean(), base["n_unb"].std(ddof=1)
    # 縮約原因: サブグラフが小さい / 同方向相関のみで n_unb が常に 0 など
    reasons: list[str] = []
    if sd_L1 == 0 or not np.isfinite(sd_L1):
        reasons.append(f"sd_L1={sd_L1} (baseline で L¹ が定数)")
    if sd_unb == 0 or not np.isfinite(sd_unb):
        reasons.append(f"sd_unb={sd_unb} (baseline で n_unb が定数; "
                       "小サブグラフは独立サイクル不足で常に 0 になりがち)")
    if reasons:
        return {"delta_sigma_L1": None, "delta_sigma_unb": None,
                "delta_sigma_ediv": None,
                "n_baseline": int(len(base)),
                "n_event": int(len(evt)),
                "mu_L1_baseline": float(mu_L1),
                "sd_L1_baseline": float(sd_L1),
                "mu_unb_baseline": float(mu_unb),
                "sd_unb_baseline": float(sd_unb),
                "degenerate_reason": "; ".join(reasons)}

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


def main() -> None:
    """全 40 + asset_class 別の sub-graph で event study を実行し JSON に保存."""
    print("=== Asset-class 別 sub-graph event study (Liberation Day 2025-04) ===")
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    meta = pd.read_csv(DATA_DIR / "symbol_meta.csv")
    print(f"Loaded ohlc_40.parquet: shape={closes.shape}")
    print(f"Loaded symbol_meta.csv: {len(meta)} symbols")

    groups = build_groups(meta)
    all_symbols = list(closes.columns)

    # ===== 全 40 銘柄 (比較 baseline) =====
    print("\n[all_40] 40 symbols (baseline for comparison) ...")
    t0 = time.time()
    df_all = compute_indicators_for_period(closes, all_symbols,
                                           RETURNS_START, RETURNS_END)
    resp_all = event_response(df_all)
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  Δσ  L1   = {resp_all['delta_sigma_L1']:+.3f}")
    print(f"  Δσ  unb  = {resp_all['delta_sigma_unb']:+.3f}")
    print(f"  Δσ  ediv = {resp_all['delta_sigma_ediv']:+.3f}")

    # ===== asset_class 別 sub-graph =====
    by_class: dict[str, dict] = {}
    for cls, syms in groups.items():
        syms_present = [s for s in syms if s in closes.columns]
        n = len(syms_present)
        print(f"\n[{cls}] n={n} symbols: {syms_present}")
        if n < MIN_SYMBOLS:
            print(f"  SKIP (n={n} < {MIN_SYMBOLS})")
            by_class[cls] = {
                "symbols": syms_present,
                "n_symbols": n,
                "skipped": True,
                "skip_reason": f"n={n} < MIN_SYMBOLS({MIN_SYMBOLS})",
            }
            continue
        t0 = time.time()
        df_cls = compute_indicators_for_period(closes, syms_present,
                                               RETURNS_START, RETURNS_END)
        resp = event_response(df_cls)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")
        if resp["delta_sigma_ediv"] is None:
            print(f"  DEGENERATE: {resp.get('degenerate_reason')}")
        else:
            print(f"  Δσ  L1   = {resp['delta_sigma_L1']:+.3f}")
            print(f"  Δσ  unb  = {resp['delta_sigma_unb']:+.3f}")
            print(f"  Δσ  ediv = {resp['delta_sigma_ediv']:+.3f}")
        by_class[cls] = {
            "symbols": syms_present,
            "n_symbols": n,
            "skipped": False,
            **resp,
        }

    # ===== ranking by Δσ_ediv (skipped / degenerate を除く) =====
    ranking = sorted(
        ((c, r["delta_sigma_ediv"]) for c, r in by_class.items()
         if not r.get("skipped") and r.get("delta_sigma_ediv") is not None
         and np.isfinite(r["delta_sigma_ediv"])),
        key=lambda x: -x[1],
    )
    print("\n=== Ranking by Δσ_ediv (Liberation Day reaction) ===")
    for cls, ds in ranking:
        n = by_class[cls]["n_symbols"]
        print(f"  {cls:10s}  n={n:2d}  Δσ_ediv = {ds:+.3f}")
    print(f"  {'ALL_40':10s}  n=40  Δσ_ediv = {resp_all['delta_sigma_ediv']:+.3f}")

    # ===== degenerate / skipped クラスの集計 =====
    degenerate = [{"asset_class": c,
                   "n_symbols": r["n_symbols"],
                   "reason": r.get("degenerate_reason", "")}
                  for c, r in by_class.items()
                  if not r.get("skipped") and r.get("delta_sigma_ediv") is None]
    skipped = [{"asset_class": c,
                "n_symbols": r["n_symbols"],
                "reason": r.get("skip_reason", "")}
               for c, r in by_class.items() if r.get("skipped")]

    # ===== finding 文 (どの市場が震源か) =====
    if ranking:
        top_cls, top_ds = ranking[0]
        same_dir = sum(1 for _, ds in ranking if ds > 0)
        n_active = len(ranking)
        if top_ds > resp_all["delta_sigma_ediv"]:
            verdict = (
                f"{top_cls} サブグラフが Δσ_ediv = {top_ds:+.3f} で "
                f"全 40 銘柄 ({resp_all['delta_sigma_ediv']:+.3f}) を上回り、震源候補。"
            )
        else:
            verdict = (
                f"全 40 銘柄が最大 (Δσ_ediv = {resp_all['delta_sigma_ediv']:+.3f}) で、"
                f"市場横断的なクロスアセット相関が主因。"
                f"{top_cls} ({top_ds:+.3f}) が単一クラスでは最大反応。"
            )
        if same_dir == n_active:
            verdict += f" {n_active}/{n_active} クラス全てで Δσ_ediv > 0 → ショックは asset class を横断して伝播。"
        else:
            verdict += f" {same_dir}/{n_active} クラスのみ正方向 → 反応は不均一。"
        if degenerate:
            deg_names = ", ".join(d["asset_class"] for d in degenerate)
            verdict += (
                f" 別途 {deg_names} は baseline で n_unb=定数のため Δσ 算出不能 "
                f"(小サブグラフ n≦6 で独立サイクル不足、これ自体が finding)。"
            )
    else:
        verdict = "全 sub-graph が n<5 でスキップされ、判定不能。"

    # ===== 出力 =====
    out = {
        "meta": {
            "window": WINDOW,
            "threshold": THRESHOLD,
            "min_symbols": MIN_SYMBOLS,
            "returns_period": [RETURNS_START, RETURNS_END],
            "baseline_window": [BASELINE_START, BASELINE_END],
            "event_window": [EVENT_START, EVENT_END],
            "event_name": "2025-04 Liberation Day (reciprocal tariff cluster)",
            "tz_aware": False,
            "tz_note": (
                "tz-naive UTC midnight close-to-close. 24h (FX/crypto) と日中銘柄を "
                "同じ日付でアライメントしているため最大 ~30h のずれは残るが、"
                "同じ tz-naive 系列内での asset class 比較は fair。"
            ),
            "grouping_rule": (
                "asset_class を 1 段まとめ: METAL+ENERGY+COMMODITY を COMMODITY、"
                "それ以外は CSV の asset_class を直接使用。"
            ),
        },
        "all_40": {
            "symbols": all_symbols,
            "n_symbols": len(all_symbols),
            "skipped": False,
            **resp_all,
        },
        "by_asset_class": by_class,
        "ranking_by_delta_sigma_ediv": [
            {"asset_class": c, "delta_sigma_ediv": ds,
             "n_symbols": by_class[c]["n_symbols"]}
            for c, ds in ranking
        ],
        "degenerate_classes": degenerate,
        "skipped_classes": skipped,
        "finding": verdict,
    }

    out_path = DATA_DIR / "subgraph_eventstudy.json"
    # allow_nan=False で NaN を JSON に書き込まないことを保証 (代わりに None を使う)
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")
    print(f"\nFinding: {verdict}")


if __name__ == "__main__":
    main()
