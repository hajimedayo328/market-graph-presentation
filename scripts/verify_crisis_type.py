"""危機時に n_unb(矛盾)は増えるのか減るのか。
「全部一緒に落ちる＝正相関＝矛盾が減る」なら n_unb は下がるはず。"""
import warnings; warnings.simplefilter("ignore")
import pandas as pd, numpy as np

REPO = str(__import__("pathlib").Path(__file__).parent.parent)
ind = (pd.read_csv(f"{REPO}/data/gamma_timeseries_20y_w30.csv", parse_dates=["date"])
       .dropna(subset=["L1_H1","n_unb"]).set_index("date").sort_index())
ohlc = pd.read_parquet(f"{REPO}/data/ohlc_40_20y.parquet")
sp = ohlc["SP500"].reindex(ind.index).ffill()
vol20 = sp.pct_change().rolling(20).std()*np.sqrt(252)

# 平均相関と負相関の割合も再計算
rets = ohlc.pct_change()
rows=[]
for t in range(30, len(rets)):
    win = rets.iloc[t-30:t].dropna(axis=1, how="any")
    if win.shape[1] < 5: continue
    c = win.corr().values; iu=np.triu_indices(c.shape[0],1); p=c[iu]
    rows.append({"date": rets.index[t-1], "mean_r": np.nanmean(p),
                 "neg_frac": np.mean(p<0), "strong_neg": np.mean(p<=-0.3)})
mc = pd.DataFrame(rows).set_index("date")
df = ind.join(mc).join(vol20.rename("vol20")).dropna()

base = df.mean()
print(f"{'':<16} {'L1':>6} {'n_unb':>6} {'平均r':>6} {'負の割合':>7} {'強い負(<-0.3)':>10}")
print(f"{'全期間平均':<16} {base.L1_H1:>6.2f} {base.n_unb:>6.1f} {base.mean_r:>6.3f} {base.neg_frac:>7.1%} {base.strong_neg:>10.1%}")
crises = {"2008 金融危機":("2008-09-01","2008-12-31"),
          "2020 コロナ":("2020-02-20","2020-04-30"),
          "2022 利上げ":("2022-01-01","2022-06-30"),
          "2025 関税":("2025-03-20","2025-04-30")}
for name,(s,e) in crises.items():
    sub = df.loc[s:e].mean()
    print(f"{name:<16} {sub.L1_H1:>6.2f} {sub.n_unb:>6.1f} {sub.mean_r:>6.3f} {sub.neg_frac:>7.1%} {sub.strong_neg:>10.1%}")

print(f"\ncorr(n_unb, 平均r)     = {df.n_unb.corr(df.mean_r):+.3f}")
print(f"corr(n_unb, 負の割合)   = {df.n_unb.corr(df.neg_frac):+.3f}")
print(f"corr(n_unb, 強い負の割合)= {df.n_unb.corr(df.strong_neg):+.3f}")
print(f"corr(n_unb, ボラ)       = {df.n_unb.corr(df.vol20):+.3f}")
