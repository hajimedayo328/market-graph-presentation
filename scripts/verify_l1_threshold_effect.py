"""L1 の半径打ち切り(√2)が結果に影響するかの検証.

仮説: 負相関(d>√2)まで半径を伸ばす頃には、近いペアは全部繋がって
      三角形で埋まっている → √2以降に新しい穴は生まれず、打ち切っても影響しない

検証: 全期間で thresh=√2(現行) と thresh=最大距離(切らない) の L1 を比較。
      特に「birth > √2 の H1 バー」(=√2以降に生まれた穴)が存在するかを数える。
"""
import sys, warnings, json
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/hajim/dev/market-graph-presentation/scripts/lib")
import numpy as np, pandas as pd
from persistent_homology import correlation_to_distance, _MAX_EDGE_LENGTH
from ripser import ripser

REPO = r"C:/Users/hajim/dev/market-graph-presentation"
SQRT2 = _MAX_EDGE_LENGTH
STEP = 5  # 5営業日おき

ohlc = pd.read_parquet(f"{REPO}/data/ohlc_40.parquet")
rets = ohlc.pct_change()

rows = []
n_born_after = 0        # √2 より後に生まれた穴の総数
n_born_after_windows = 0  # そういう穴が1本でもある窓の数
for t in range(30, len(rets), STEP):
    win = rets.iloc[t - 30:t].dropna(axis=1, how="any")
    if win.shape[1] < 5:
        continue
    corr = win.corr()
    D = correlation_to_distance(corr)
    dmax = float(D.max())

    # 現行: thresh=√2、∞は√2で置換
    h1a = ripser(D, maxdim=1, thresh=SQRT2, distance_matrix=True)["dgms"][1]
    da = np.where(np.isfinite(h1a[:, 1]), h1a[:, 1], SQRT2)
    L1_cut = float((da - h1a[:, 0]).sum())

    # 切らない: 全距離、∞は最大距離で置換
    h1b = ripser(D, maxdim=1, distance_matrix=True)["dgms"][1]
    db = np.where(np.isfinite(h1b[:, 1]), h1b[:, 1], dmax)
    L1_full = float((db - h1b[:, 0]).sum())

    born_after = int((h1b[:, 0] > SQRT2).sum()) if len(h1b) else 0
    n_born_after += born_after
    if born_after > 0:
        n_born_after_windows += 1

    rows.append({
        "date": str(rets.index[t - 1].date()),
        "L1_cut": L1_cut, "L1_full": L1_full,
        "n_bars_cut": len(h1a), "n_bars_full": len(h1b),
        "born_after_sqrt2": born_after,
        "dmax": dmax,
    })

df = pd.DataFrame(rows)
r = float(df["L1_cut"].corr(df["L1_full"]))
diff = df["L1_full"] - df["L1_cut"]
rel = (diff / df["L1_cut"].replace(0, np.nan)).abs()

print(f"検証窓数: {len(df)}  ({df['date'].iloc[0]} 〜 {df['date'].iloc[-1]}, {STEP}営業日おき)")
print(f"\n=== 仮説の核心: √2 より後に生まれた穴はあるか ===")
print(f"  √2以降に生まれた穴の総数: {n_born_after} 本")
print(f"  そういう穴がある窓: {n_born_after_windows} / {len(df)} 窓 ({n_born_after_windows/len(df)*100:.1f}%)")

print(f"\n=== L1: 打ち切りあり vs なし ===")
print(f"  相関 r = {r:.4f}")
print(f"  L1_cut  平均 {df['L1_cut'].mean():.4f}  中央 {df['L1_cut'].median():.4f}")
print(f"  L1_full 平均 {df['L1_full'].mean():.4f}  中央 {df['L1_full'].median():.4f}")
print(f"  差(full-cut) 平均 {diff.mean():+.4f}  中央 {diff.median():+.4f}  最大 {diff.max():+.4f}")
print(f"  相対差 中央 {rel.median()*100:.1f}%  90%点 {rel.quantile(0.9)*100:.1f}%  最大 {rel.max()*100:.1f}%")
print(f"\n  H1バー本数: 打ち切り 平均{df['n_bars_cut'].mean():.1f}  切らない 平均{df['n_bars_full'].mean():.1f}")

out = {
    "description": "L1の半径打ち切り(√2=相関0)が結果に影響するかの検証。thresh=√2(現行) vs 切らない場合の比較",
    "n_windows": len(df), "step_bdays": STEP,
    "date_range": [df["date"].iloc[0], df["date"].iloc[-1]],
    "holes_born_after_sqrt2": {"total": n_born_after, "windows_with_any": n_born_after_windows},
    "L1_correlation_cut_vs_full": round(r, 4),
    "L1_cut": {"mean": round(float(df["L1_cut"].mean()), 4), "median": round(float(df["L1_cut"].median()), 4)},
    "L1_full": {"mean": round(float(df["L1_full"].mean()), 4), "median": round(float(df["L1_full"].median()), 4)},
    "diff_full_minus_cut": {"mean": round(float(diff.mean()), 4), "median": round(float(diff.median()), 4), "max": round(float(diff.max()), 4)},
    "relative_diff_pct": {"median": round(float(rel.median() * 100), 2), "p90": round(float(rel.quantile(0.9) * 100), 2), "max": round(float(rel.max() * 100), 2)},
    "n_bars": {"cut_mean": round(float(df["n_bars_cut"].mean()), 2), "full_mean": round(float(df["n_bars_full"].mean()), 2)},
}
with open(f"{REPO}/data/l1_threshold_effect.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\nsaved: data/l1_threshold_effect.json")
