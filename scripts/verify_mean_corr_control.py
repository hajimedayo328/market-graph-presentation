"""「L¹は平均相関の言い換えでは?」への回答実験.

各30日窓の平均相関 (mean r / mean |r|) を計算し、
  1. L¹・n_unb が平均相関でどれだけ説明されるか
  2. 平均相関を統制(偏相関)した後も、L¹とn_unbの独立性の結論が変わらないか
  3. 平均相関を回帰で除いた残差L¹が、元のL¹とどれだけ違うか
を測る。出力: data/mean_corr_control.json
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path(__file__).parent.parent
WINDOW = 30


def partial_corr(rxy: float, rxz: float, ryz: float) -> float:
    """z を統制した x-y の偏相関."""
    den = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return (rxy - rxz * ryz) / den if den > 1e-12 else np.nan


def main() -> None:
    ohlc = pd.read_parquet(ROOT / "data" / "ohlc_40_20y.parquet")
    rets = ohlc.pct_change()
    ind = (pd.read_csv(ROOT / "data" / "gamma_timeseries_20y_w30.csv",
                       parse_dates=["date"])
           .dropna(subset=["L1_H1", "n_unb"]).set_index("date"))

    # 各窓の平均相関 (指標と同じ窓取り: [t-30, t), 日付は窓の最終日)
    rows = []
    for t in range(WINDOW, len(rets)):
        win = rets.iloc[t - WINDOW:t].dropna(axis=1, how="any")
        if win.shape[1] < 5:
            continue
        c = win.corr().values
        iu = np.triu_indices(c.shape[0], 1)
        pair = c[iu]
        rows.append({"date": rets.index[t - 1],
                     "mean_r": float(np.nanmean(pair)),
                     "mean_abs_r": float(np.nanmean(np.abs(pair)))})
    mc = pd.DataFrame(rows).set_index("date")

    df = ind.join(mc, how="inner").dropna(subset=["mean_abs_r"])
    print(f"結合後: {len(df)}日 ({df.index.min().date()} - {df.index.max().date()})")

    out: dict = {"description": "L1/n_unbが平均相関でどれだけ説明されるか、統制後に独立性が残るかの検証",
                 "n_days": len(df), "window": WINDOW,
                 "date_range": [str(df.index.min().date()), str(df.index.max().date())]}

    periods = {"20y_full": df, "recent_5y": df[df.index.year >= 2021]}
    for name, d in periods.items():
        L1, nu, mabs = d["L1_H1"], d["n_unb"], d["mean_abs_r"]
        r_l1_nu = float(L1.corr(nu))
        r_l1_m = float(L1.corr(mabs))
        r_nu_m = float(nu.corr(mabs))
        pc = float(partial_corr(r_l1_nu, r_l1_m, r_nu_m))
        # 平均相関を回帰で除いた残差L1
        a, b = np.polyfit(mabs, L1, 1)
        res_l1 = L1 - (a * mabs + b)
        out[name] = {
            "corr_L1_nunb_raw": round(r_l1_nu, 4),
            "corr_L1_meanAbsR": round(r_l1_m, 4),
            "corr_nunb_meanAbsR": round(r_nu_m, 4),
            "partial_corr_L1_nunb_given_meanAbsR": round(pc, 4),
            "R2_L1_explained_by_meanAbsR": round(r_l1_m**2, 4),
            "corr_residualL1_vs_L1": round(float(res_l1.corr(L1)), 4),
            "corr_residualL1_vs_nunb": round(float(res_l1.corr(nu)), 4),
        }
        o = out[name]
        print(f"\n===== {name} (n={len(d)}) =====")
        print(f"  corr(L1, n_unb)          = {o['corr_L1_nunb_raw']}   (生)")
        print(f"  corr(L1, mean|r|)        = {o['corr_L1_meanAbsR']}  → R2={o['R2_L1_explained_by_meanAbsR']}")
        print(f"  corr(n_unb, mean|r|)     = {o['corr_nunb_meanAbsR']}")
        print(f"  偏相関(L1,n_unb | mean|r|) = {o['partial_corr_L1_nunb_given_meanAbsR']}")
        print(f"  残差L1 vs 元L1           = {o['corr_residualL1_vs_L1']}")
        print(f"  残差L1 vs n_unb          = {o['corr_residualL1_vs_nunb']}")

    with open(ROOT / "data" / "mean_corr_control.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nsaved: data/mean_corr_control.json")


if __name__ == "__main__":
    main()
