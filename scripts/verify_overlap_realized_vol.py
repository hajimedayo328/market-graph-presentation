"""学会後の検証の締め: e_div は「直近20日の実現ボラ」の言い換えか、別物か.

前進検証（train 3年 → test 1年、fold 毎に同じ現金化率）で
  ・e_div ルール
  ・実現ボラ20日ルール
が降りた日を日次で並べ、
  1) 同じ日に降りているか（重なり率）
  2) 組み合わせたら良くなるか（どちらかで降りる OR / 両方で降りる AND）
を 20y / 15y / 10y で出す。しきい値の決め方・コスト・翌日約定は backtest_oos_random_vix と同じ。
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import backtest_oos_random_vix as bo
from backtest_v2 import apply_hysteresis, HYSTERESIS_DAYS

ROOT = Path(__file__).parent.parent
periods = sys.argv[1:] if len(sys.argv) > 1 else ["20y", "15y", "10y"]


def walkforward_positions(df: pd.DataFrame, rets: pd.Series) -> pd.DataFrame:
    """fold 毎に train でしきい値を決め、test の日次ポジション(現金=True)を連結して返す."""
    train_days, test_days = int(bo.TRAIN_YEARS * 252), int(bo.TEST_YEARS * 252)
    rows = []
    start = 0
    while start + train_days + test_days <= len(df):
        train = df.iloc[start: start + train_days]
        test = df.iloc[start + train_days: start + train_days + test_days]
        tr_e = train["e_div"].dropna()
        if len(tr_e) < 30:
            start += test_days
            continue
        thr_e = float(np.percentile(tr_e, bo.PERCENTILE))
        sig_e = apply_hysteresis((test["e_div"] >= thr_e).fillna(False), HYSTERESIS_DAYS)
        pos_e = sig_e.shift(1).fillna(False).astype(bool)
        cash_rate = float(pos_e.mean())
        tr_v = train["RV"].dropna()
        thr_v = float(tr_v.quantile(1 - cash_rate))
        sig_v = apply_hysteresis((test["RV"] >= thr_v).fillna(False), HYSTERESIS_DAYS)
        pos_v = sig_v.shift(1).fillna(False).astype(bool)
        rows.append(pd.DataFrame({"pos_e": pos_e, "pos_v": pos_v}, index=test.index))
        start += test_days
    return pd.concat(rows)


def metrics_of(pos: pd.Series, rets: pd.Series) -> dict:
    m = bo._metrics(bo._strat_rets_from_signal(pos, rets))
    return {k: round(v, 4) for k, v in m.items()}


out_all = {}
for period in periods:
    df = bo.load_indicators_with_vix(bo.CSV_MAP[period], bo.PARQUET_MAP[period])
    px = pd.read_parquet(ROOT / "data" / bo.PARQUET_MAP[period])["SP500"].dropna()
    rets_all = px.pct_change()
    df["RV"] = (rets_all.rolling(20).std() * np.sqrt(252)).reindex(df.index).ffill()
    common = px.index.intersection(df.index)
    df = df.loc[common]
    rets = px.loc[common].pct_change().fillna(0)

    P = walkforward_positions(df, rets)
    r = rets.loc[P.index]
    e, v = P["pos_e"], P["pos_v"]
    both, either = e & v, e | v
    n_e, n_v, n_both, n_either = int(e.sum()), int(v.sum()), int(both.sum()), int(either.sum())
    res = {
        "period": period, "oos_span": f"{P.index.min().date()}〜{P.index.max().date()}", "oos_days": int(len(P)),
        "cash_days": {"ediv": n_e, "rv20": n_v, "both": n_both, "either": n_either},
        "overlap": {
            "jaccard": round(n_both / n_either, 3) if n_either else None,
            "share_of_ediv_cash_days_also_rv": round(n_both / n_e, 3) if n_e else None,
            "share_of_rv_cash_days_also_ediv": round(n_both / n_v, 3) if n_v else None,
            "expected_jaccard_if_independent": round((e.mean() * v.mean()) / (e.mean() + v.mean() - e.mean() * v.mean()), 3),
        },
        "corr_daily_ediv_vs_rv": round(float(df.loc[P.index, ["e_div", "RV"]].corr().iloc[0, 1]), 3),
        "strategies": {
            "buy_hold": metrics_of(pd.Series(False, index=P.index), r),
            "ediv": metrics_of(e, r),
            "rv20": metrics_of(v, r),
            "either_OR": metrics_of(either, r),
            "both_AND": metrics_of(both, r),
        },
    }
    out_all[period] = res
    print(f"\n=== {period}  OOS {res['oos_span']}  {res['oos_days']}日 ===")
    print(f"現金化日数  e_div {n_e} / 実現ボラ {n_v} / 両方 {n_both} / どちらか {n_either}")
    o = res["overlap"]
    print(f"重なり(Jaccard) {o['jaccard']}  （無関係なら {o['expected_jaccard_if_independent']}）"
          f"  e_div現金日のうち実現ボラも降りていた {o['share_of_ediv_cash_days_also_rv']:.0%}")
    print(f"日次の相関 e_div vs 実現ボラ: {res['corr_daily_ediv_vs_rv']}")
    print(f"{'戦略':<14}{'Return':>9}{'Sharpe':>8}{'MaxDD':>9}")
    for k, m in res["strategies"].items():
        print(f"{k:<14}{m['total_return']*100:>+8.1f}%{m['sharpe']:>+8.2f}{m['max_drawdown']*100:>+8.1f}%")

(ROOT / "data" / "overlap_realized_vol.json").write_text(
    json.dumps({"description": "前進検証での e_div ルールと実現ボラ20日ルールの現金化日の重なりと組み合わせ(OR/AND)",
                "results": out_all}, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nsaved: data/overlap_realized_vol.json")
