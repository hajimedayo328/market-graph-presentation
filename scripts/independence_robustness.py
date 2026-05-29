"""
独立性の頑健性検証 (Independence Robustness Scan)
==================================================

目的:
  「L^1 と n_unb は独立 (r=0.16)」という発見を, 設定 (window / threshold /
  市場) を変えても相関係数 r が小さい (r < 0.5, 大半は 0.1-0.3) まま保たれる
  ことを実データで示し, 独立性が「設定依存のまぐれ」ではなく頑健な性質だと
  確定させる.

検証軸:
  1. window 別 : window = 20 / 30 / 40 / 60 営業日 (threshold=0.3 固定, 5y USA)
  2. threshold 別: threshold = 0.2 / 0.3 / 0.4 (window=30 固定, 5y USA)
       - L^1 は Vietoris-Rips filtration なので threshold 非依存
       - n_unb は |corr| >= threshold エッジで構築するため threshold 依存
  3. 市場 別     : USA / EM / CN (いずれも既存 window=30 CSV)

軽量化方針:
  PH 計算は重いが 1 window あたり 5y 全期間でも約 10-15 秒程度なので, window /
  threshold の全パターンを本スクリプト内で再計算する. ただし既存 CSV
  (window=30, threshold=0.3 の USA / EM / CN) は再利用し, 二重計算を避ける.

出力:
  data/independence_robustness.json
    - window 別 r
    - threshold 別 r
    - 市場別 r
    - 全 r の min/max と「r < 0.5」判定
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

# ===== パラメータ =====
WINDOWS = [20, 30, 40, 60]
THRESHOLDS = [0.2, 0.3, 0.4]
BASE_WINDOW = 30
BASE_THRESHOLD = 0.3
MIN_SYMBOLS = 5
INPUT_FILE = "ohlc_40.parquet"  # 5y USA (40 銘柄)


def compute_timeseries(
    returns: pd.DataFrame,
    window: int,
    threshold: float,
) -> pd.DataFrame:
    """ローリング窓で L1_H1, n_unb を日次計算する.

    compute_gamma_timeseries.main() と同一ロジック (同じ窓・同じ閾値計算) を
    DataFrame 返却用に切り出したもの.

    Args:
        returns: pct_change 済みのリターン DataFrame.
        window: ローリング窓幅 (営業日).
        threshold: n_unb 用エッジ閾値 (|corr| >= threshold).

    Returns:
        date / n_symbols / L1_H1 / n_unb 列を持つ DataFrame.
    """
    n = len(returns)
    rows: list[dict] = []
    for t_idx in range(window, n):
        date = returns.index[t_idx - 1]
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
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": L1, "n_unb": float(n_unb)})
        except Exception as e:  # noqa: BLE001
            rows.append({"date": date, "n_symbols": win_clean.shape[1],
                         "L1_H1": np.nan, "n_unb": np.nan})
            print(f"    ! {date.date()} failed: {e}")
    return pd.DataFrame(rows)


def pearson_r(df: pd.DataFrame) -> tuple[float, int]:
    """L1_H1 と n_unb の Pearson 相関と有効サンプル数を返す."""
    sub = df[["L1_H1", "n_unb"]].dropna()
    # n_unb が定数だと相関が NaN になるためガード
    if len(sub) < 3 or sub["L1_H1"].std() == 0 or sub["n_unb"].std() == 0:
        return float("nan"), len(sub)
    r = float(sub["L1_H1"].corr(sub["n_unb"]))
    return r, len(sub)


def r_from_existing_csv(path: Path) -> tuple[float, int]:
    """既存 gamma CSV から corr(L1_H1, n_unb) を読み出す."""
    df = pd.read_csv(path)
    return pearson_r(df)


def main() -> None:
    """window / threshold / 市場別に独立性の頑健性を検証して JSON 保存する."""
    print("=== Independence Robustness Scan ===")
    closes = pd.read_parquet(DATA_DIR / INPUT_FILE)
    # 既存 gamma CSV (compute_gamma_timeseries.py) と完全に揃えるため,
    # pct_change はデフォルトの fill_method='pad' (前方補完) を使う.
    # fill_method=None にすると各銘柄の散発 NaN が dropna(how='any') で
    # 全列脱落を招き, window=30 既存 CSV と再現性が取れなくなる.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        returns = closes.pct_change()
    print(f"Loaded {INPUT_FILE}: {closes.shape}, "
          f"range {closes.index.min().date()} -> {closes.index.max().date()}")

    all_r: list[float] = []

    # ---------- 1. window 別 (threshold=0.3 固定) ----------
    print("\n[1] window scan (threshold=0.3 fixed)")
    window_results = {}
    for w in WINDOWS:
        t0 = time.time()
        if w == BASE_WINDOW:
            # 既存 CSV (USA, window=30, threshold=0.3) を再利用
            csv_path = DATA_DIR / "gamma_timeseries_w30.csv"
            r, n = r_from_existing_csv(csv_path)
            src = "existing:gamma_timeseries_w30.csv"
        else:
            df = compute_timeseries(returns, window=w, threshold=BASE_THRESHOLD)
            r, n = pearson_r(df)
            src = "recomputed"
        el = time.time() - t0
        print(f"  window={w:>3d}  r={r:+.4f}  n={n:>5d}  ({src}, {el:.1f}s)")
        window_results[str(w)] = {
            "window": w, "threshold": BASE_THRESHOLD,
            "pearson_r": round(r, 4), "n": n, "source": src,
        }
        if np.isfinite(r):
            all_r.append(abs(r))

    # ---------- 2. threshold 別 (window=30 固定) ----------
    print("\n[2] threshold scan (window=30 fixed)")
    threshold_results = {}
    for thr in THRESHOLDS:
        t0 = time.time()
        if abs(thr - BASE_THRESHOLD) < 1e-9:
            csv_path = DATA_DIR / "gamma_timeseries_w30.csv"
            r, n = r_from_existing_csv(csv_path)
            src = "existing:gamma_timeseries_w30.csv"
        else:
            df = compute_timeseries(returns, window=BASE_WINDOW, threshold=thr)
            r, n = pearson_r(df)
            src = "recomputed"
        el = time.time() - t0
        print(f"  threshold={thr:.1f}  r={r:+.4f}  n={n:>5d}  ({src}, {el:.1f}s)")
        threshold_results[f"{thr:.1f}"] = {
            "window": BASE_WINDOW, "threshold": thr,
            "pearson_r": round(r, 4), "n": n, "source": src,
        }
        if np.isfinite(r):
            all_r.append(abs(r))

    # ---------- 3. 市場別 (既存 window=30 CSV) ----------
    print("\n[3] market scan (existing window=30 CSVs)")
    market_files = {
        "USA": "gamma_timeseries_w30.csv",
        "EM": "gamma_em_timeseries_w30.csv",
        "CN": "gamma_cn_timeseries_w30.csv",
    }
    market_results = {}
    for mkt, fn in market_files.items():
        path = DATA_DIR / fn
        if not path.exists():
            print(f"  {mkt:>4}  (missing: {fn})")
            continue
        r, n = r_from_existing_csv(path)
        print(f"  {mkt:>4}  r={r:+.4f}  n={n:>5d}  ({fn})")
        market_results[mkt] = {
            "window": BASE_WINDOW, "threshold": BASE_THRESHOLD,
            "pearson_r": round(r, 4), "n": n, "source": fn,
        }
        if np.isfinite(r):
            all_r.append(abs(r))

    # ---------- 集計 ----------
    abs_r = np.array(all_r, dtype=float)
    summary = {
        "n_settings": int(len(abs_r)),
        "max_abs_r": round(float(abs_r.max()), 4),
        "min_abs_r": round(float(abs_r.min()), 4),
        "mean_abs_r": round(float(abs_r.mean()), 4),
        "all_below_0.5": bool((abs_r < 0.5).all()),
        "n_below_0.3": int((abs_r < 0.3).sum()),
        "fraction_in_0.1_0.3": round(
            float(((abs_r >= 0.1) & (abs_r < 0.3)).mean()), 3),
    }

    out = {
        "description": (
            "L^1 (H1 persistence の L^1 ノルム) と n_unb (符号付きサイクルの不整合数) "
            "の Pearson 相関 r を, window (20/30/40/60), threshold (0.2/0.3/0.4), "
            "市場 (USA/EM/CN) を変えて再計算した頑健性検証. 全設定で |r| < 0.5 なら "
            "独立性は設定非依存で頑健と判断する. window=30 & threshold=0.3 の USA/EM/CN "
            "は既存 gamma CSV を再利用, それ以外は ohlc_40.parquet (5y) から再計算した."
        ),
        "method": (
            "compute_gamma_timeseries.main() と同一ロジック (同じローリング窓・"
            "同じ |corr|>=threshold エッジ構築) で L1_H1, n_unb を日次計算し, "
            "両者の Pearson 相関を取る. L^1 は Vietoris-Rips filtration なので "
            "threshold 非依存, n_unb は threshold 依存."
        ),
        "input_file_5y": INPUT_FILE,
        "window_scan": window_results,
        "threshold_scan": threshold_results,
        "market_scan": market_results,
        "regional_subset_note": (
            "地域サブグラフ (日本中心 / 欧州中心) も検討したが, 40 銘柄を地域純粋に "
            "分けると地域 INDEX は欧州 3 (GER40/UK100/FRA40) ・アジア 2 (JP225/CHINA50) "
            "しか取れず, 制約 n>=9 を満たす純地域サブグラフが構成できないため見送った. "
            "代わりに市場全体が異なる EM (新興国 40 銘柄) ・ CN (中国 40 銘柄) で代替検証した."
        ),
        "summary": summary,
    }

    out_path = DATA_DIR / "independence_robustness.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nSaved: {out_path}")

    print("\n=== Summary ===")
    print(f"  settings tested : {summary['n_settings']}")
    print(f"  |r| range       : {summary['min_abs_r']} .. {summary['max_abs_r']}")
    print(f"  mean |r|        : {summary['mean_abs_r']}")
    print(f"  all |r| < 0.5   : {summary['all_below_0.5']}")
    print(f"  |r| < 0.3 count : {summary['n_below_0.3']} / {summary['n_settings']}")


if __name__ == "__main__":
    main()
