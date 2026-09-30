"""学会質問2「指標にこの2つ以外を使うのは?」への中身の検証.

「指標を差し替えたら結果はどのくらい変わるか、どこまで同じか」を2通りで出す。

A) 20年・前進検証（train 3年 → test 1年、train の80パーセンタイルで現金化、ヒステリシス5日、翌日約定）
   で、シグナルだけ差し替えて成績と「e_div と同じ日に休んだ割合」を比べる。
   20年で使える指標: L¹、n_unb、均衡率、辺数、VIX、直近20日実現ボラ。
B) 直近5年（11指標が揃う期間）で、55通りの差分系列が e_div とどれだけ同じ動きか（相関、上位20%の日の重なり）。
"""
from __future__ import annotations
import sys, json, itertools, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import backtest_oos_random_vix as bo
from backtest_v2 import apply_hysteresis, HYSTERESIS_DAYS

ROOT = Path(__file__).parent.parent
MP = 30


def ez(s: pd.Series) -> pd.Series:
    """expanding z-score（過去のみ）."""
    return (s - s.expanding(min_periods=MP).mean()) / s.expanding(min_periods=MP).std()


def walkforward_pos(sig: pd.Series) -> pd.Series:
    """train の80パーセンタイルをしきい値に、test の現金化ポジション(True=現金)を連結."""
    train_days, test_days = int(bo.TRAIN_YEARS * 252), int(bo.TEST_YEARS * 252)
    out, start = [], 0
    while start + train_days + test_days <= len(sig):
        tr = sig.iloc[start: start + train_days].dropna()
        te = sig.iloc[start + train_days: start + train_days + test_days]
        if len(tr) >= 30:
            thr = float(np.percentile(tr, bo.PERCENTILE))
            pos = apply_hysteresis((te >= thr).fillna(False), HYSTERESIS_DAYS).shift(1).fillna(False).astype(bool)
            out.append(pos)
        start += test_days
    return pd.concat(out)


# ---------------- A) 20年・前進検証でシグナル差し替え ----------------
period = "20y"
df = bo.load_indicators_with_vix(bo.CSV_MAP[period], bo.PARQUET_MAP[period])
px = pd.read_parquet(ROOT / "data" / bo.PARQUET_MAP[period])["SP500"].dropna()
common = px.index.intersection(df.index)
df = df.loc[common]
rets = px.loc[common].pct_change().fillna(0)
rv20 = (px.pct_change().rolling(20).std() * np.sqrt(252)).reindex(df.index).ffill()

signals = {
    "e_div = 矛盾 − 穴（基準）": df["e_div"],
    "穴だけ（L¹ が小さい）": -df["z_L1"],
    "矛盾だけ（n_unb）": df["z_unb"],
    "不均衡率だけ（1−均衡率）": ez(1 - df["balance_rate"]),
    "不均衡率 − 穴": ez(1 - df["balance_rate"]) - df["z_L1"],
    "辺の数だけ": ez(df["n_edges"]),
    "辺の数 − 穴": ez(df["n_edges"]) - df["z_L1"],
    "VIX": df["VIX"],
    "直近20日の値動きの大きさ": rv20,
}
pos = {k: walkforward_pos(s) for k, s in signals.items()}
idx = pos["e_div = 矛盾 − 穴（基準）"].index
base = pos["e_div = 矛盾 − 穴（基準）"]
r = rets.loc[idx]
bh = bo._metrics(r)

rows_a = []
for k, p in pos.items():
    p = p.reindex(idx).fillna(False)
    m = bo._metrics(bo._strat_rets_from_signal(p, r))
    both = int((p & base).sum())
    rows_a.append({
        "signal": k, "total_return": round(m["total_return"], 3), "sharpe": round(m["sharpe"], 2),
        "max_drawdown": round(m["max_drawdown"], 3), "cash_rate": round(float(p.mean()), 2),
        "share_of_ediv_cash_days": round(both / int(base.sum()), 2),
        "jaccard_with_ediv": round(both / int((p | base).sum()), 2),
    })

print(f"=== A) 20年・前進検証  OOS {idx.min().date()}〜{idx.max().date()} ({len(idx)}日) ===")
print(f"{'シグナル':<22}{'儲け':>9}{'成績':>7}{'最大下落':>9}{'休む率':>7}{'e_divと同じ日':>12}")
print(f"{'持ち続け':<22}{bh['total_return']*100:>+8.0f}%{bh['sharpe']:>7.2f}{bh['max_drawdown']*100:>+8.1f}%{'—':>7}{'—':>12}")
for x in rows_a:
    print(f"{x['signal']:<22}{x['total_return']*100:>+8.0f}%{x['sharpe']:>7.2f}{x['max_drawdown']*100:>+8.1f}%{x['cash_rate']:>7.0%}{x['share_of_ediv_cash_days']:>12.0%}")

# ---------------- B) 直近5年・55通りが e_div とどこまで同じか ----------------
mi = pd.read_csv(ROOT / "data" / "multi_indicators_w30.csv", parse_dates=["date"]).set_index("date").sort_index()
inds = ["L1", "L2", "Linf", "nH1", "meanP", "entropy", "n_unb_total", "n_unb_3", "n_unb_4", "n_unb_5plus", "balance_rate"]
z = pd.DataFrame({c: ez(mi[c]) for c in inds}).dropna()
ediv5 = z["n_unb_total"] - z["L1"]
top_e = ediv5 >= ediv5.quantile(0.8)
rows_b = []
for a, b in itertools.combinations(inds, 2):
    d = z[a] - z[b]
    c = float(np.corrcoef(d, ediv5)[0, 1])
    d_aligned = d if c >= 0 else -d                      # 向きを e_div に揃える
    top_d = d_aligned >= d_aligned.quantile(0.8)
    rows_b.append({"pair": f"{a} - {b}", "abs_corr_with_ediv": round(abs(c), 2),
                   "top20_overlap": round(float((top_d & top_e).sum() / top_e.sum()), 2)})
rb = pd.DataFrame(rows_b).sort_values("abs_corr_with_ediv", ascending=False)
ac = rb["abs_corr_with_ediv"]
print(f"\n=== B) 直近5年 {z.index.min().date()}〜{z.index.max().date()}  55通りと e_div の近さ ===")
print(f"|相関| ≥0.8: {(ac>=0.8).sum()}通り / 0.5〜0.8: {((ac>=0.5)&(ac<0.8)).sum()} / 0.2〜0.5: {((ac>=0.2)&(ac<0.5)).sum()} / <0.2: {(ac<0.2).sum()}   中央値 {ac.median():.2f}")
print("近い順 上位8:"); print(rb.head(8).to_string(index=False))
print("遠い順 下位5:"); print(rb.tail(5).to_string(index=False))
for p in ["L2 - n_unb_3", "Linf - entropy"]:
    print("事件での1位:", rb[rb["pair"] == p].to_dict("records"))

out = {"description": "質問2の検証: 指標を差し替えたら結果はどのくらい変わるか、どこまで同じか",
       "A_walkforward_20y": {"oos_span": f"{idx.min().date()}〜{idx.max().date()}", "buy_hold": {k: round(v, 3) for k, v in bh.items()}, "signals": rows_a},
       "B_similarity_5y": {"span": f"{z.index.min().date()}〜{z.index.max().date()}", "pairs": rb.to_dict("records")}}
(ROOT / "data" / "other_indicators_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nsaved: data/other_indicators_comparison.json")
