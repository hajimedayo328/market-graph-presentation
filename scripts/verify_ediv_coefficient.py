"""e_div は「回帰残差」なのか、それとも別物か。

z(n_unb) - k*z(L1) の係数 k を変えて、
  k=0    : n_unb だけ
  k=0.15 : 回帰残差（独立性から見た「純粋な矛盾」）
  k=1.0  : 実装の e_div
を比較する。バックテスト性能（MaxDD/Sharpe）で見る。
"""
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd
import reproduce_poster_numbers as m

REPO = str(__import__("pathlib").Path(__file__).parent.parent)
ind = (pd.read_csv(f"{REPO}/data/gamma_timeseries_20y_w30.csv", parse_dates=["date"])
       .dropna(subset=["L1_H1", "n_unb"]).set_index("date").sort_index())
px = m.load_sp500("ohlc_40_20y.parquet")
common = px.index.intersection(ind.index)
ind, px = ind.loc[common], px.loc[common]

z_l1 = m.expanding_z(ind["L1_H1"])
z_unb = m.expanding_z(ind["n_unb"])
rho = float(ind["L1_H1"].corr(ind["n_unb"]))
bh = m.buy_and_hold(px)

print(f"corr(L1, n_unb) = {rho:.4f}  → 回帰残差の係数はこれ")
print(f"B&H: MaxDD={bh['max_drawdown']*100:+.1f}%  Sharpe={bh['sharpe']:.2f}\n")
print(f"{'係数k':>7} | {'意味':<22} | {'MaxDD':>7} | {'Sharpe':>6} | {'現金化':>6}")
print("-" * 66)

rows = []
for k, label in [(0.0, "n_unb だけ"), (rho, "回帰残差(純粋な矛盾)"),
                 (0.3, "中間"), (0.5, "中間"), (0.7, "中間"),
                 (1.0, "実装の e_div"), (1.5, "引きすぎ")]:
    sig_raw = (z_unb - k * z_l1)
    # 各系列を同じ現金化率にするため、自身の80パーセンタイルで発火
    thr = sig_raw.quantile(0.8)
    sig = (sig_raw >= thr).fillna(False)
    r = m.simulate(px, sig)
    rows.append((k, label, r))
    print(f"{k:>7.2f} | {label:<22} | {r['max_drawdown']*100:>6.1f}% | "
          f"{r['sharpe']:>6.2f} | {r['cash_pct']*100:>5.0f}%")

best = max(rows, key=lambda x: x[2]["sharpe"])
print(f"\nSharpe最良: k={best[0]:.2f} ({best[1]})")
print(f"MaxDD最良: k={max(rows, key=lambda x: x[2]['max_drawdown'])[0]:.2f}")
