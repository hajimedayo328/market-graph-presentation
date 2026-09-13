"""L1（穴の総量）が高い日は、市場で何が起きているのか。

危機時にL1は上がるのか下がるのか？ 平常時とどう違うのか？
"""
import warnings; warnings.simplefilter("ignore")
import pandas as pd, numpy as np

REPO = str(__import__("pathlib").Path(__file__).parent.parent)
ind = (pd.read_csv(f"{REPO}/data/gamma_timeseries_20y_w30.csv", parse_dates=["date"])
       .dropna(subset=["L1_H1","n_unb"]).set_index("date").sort_index())
ohlc = pd.read_parquet(f"{REPO}/data/ohlc_40_20y.parquet")
sp = ohlc["SP500"].reindex(ind.index).ffill()
ret = sp.pct_change()

# 20日先までの下落率（forward）と、直近20日の実現ボラ
fwd20 = sp.shift(-20)/sp - 1
vol20 = ret.rolling(20).std()*np.sqrt(252)

df = pd.DataFrame({"L1": ind["L1_H1"], "n_unb": ind["n_unb"],
                   "fwd20": fwd20, "vol20": vol20}).dropna()

print("=== L1 と各指標の関係 ===")
print(f"  corr(L1, 今後20日リターン) = {df.L1.corr(df.fwd20):+.3f}")
print(f"  corr(L1, 直近20日ボラ)     = {df.L1.corr(df.vol20):+.3f}")
print(f"  corr(n_unb, 今後20日リターン)= {df.n_unb.corr(df.fwd20):+.3f}")

# L1を5分位に分けて、その後の下落を見る
df["q"] = pd.qcut(df.L1, 5, labels=["最低","低","中","高","最高"])
print("\n=== L1の水準別: その後20日のリターン / 同時期のボラ ===")
g = df.groupby("q", observed=True).agg(
    L1平均=("L1","mean"), 今後20日=("fwd20","mean"),
    下落確率=("fwd20", lambda x: (x<0).mean()), ボラ=("vol20","mean"), n=("L1","size"))
print(g.round(3).to_string())

# 危機期のL1
print("\n=== 危機期 vs 平常期のL1 ===")
crises = {"2008 金融危機": ("2008-09-01","2008-12-31"),
          "2020 コロナ": ("2020-02-20","2020-04-30"),
          "2022 利上げ": ("2022-01-01","2022-06-30")}
base = df.L1.mean()
print(f"  全期間平均: {base:.3f}")
for name,(s,e) in crises.items():
    sub = df.loc[s:e, "L1"]
    if len(sub): print(f"  {name:<14}: {sub.mean():.3f}  ({(sub.mean()-base)/base*100:+.0f}%)")
