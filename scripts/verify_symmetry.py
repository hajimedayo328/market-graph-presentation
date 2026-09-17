"""学会で受けた質問「下落に強いなら上昇にも強いのでは（対称性）」の検証.

e_div シグナルで現金化している日と、株を持っている日で、
S&P500 の日次リターンの分布がどう違うかを見る。
  - 現金化日の平均リターンが負 → 方向性あり（下落を選択的に避けている）
  - 平均≈0 で分散だけ大きい → 対称（荒れている日を避けているだけ）
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import reproduce_poster_numbers as m

ROOT = Path(__file__).parent.parent
ind = (pd.read_csv(ROOT / "data" / "gamma_timeseries_20y_w30.csv", parse_dates=["date"])
       .dropna(subset=["L1_H1", "n_unb"]).set_index("date").sort_index())
px = m.load_sp500("ohlc_40_20y.parquet")
common = px.index.intersection(ind.index)
ind, px = ind.loc[common], px.loc[common]

ediv = m.expanding_z(ind["n_unb"]) - m.expanding_z(ind["L1_H1"])
sig = m.apply_hysteresis((ediv >= 0.8).fillna(False), 5)
cash = sig.shift(1).fillna(False).astype(bool)          # 翌日執行 → その日は現金
ret = px.pct_change().fillna(0)

on, off = ret[cash], ret[~cash]
print(f"期間 {common.min().date()}〜{common.max().date()}  現金化率 {cash.mean()*100:.1f}%")
print(f"\n{'':<14}{'現金化日':>10}{'保有日':>10}")
print(f"{'日数':<14}{len(on):>10}{len(off):>10}")
print(f"{'平均リターン':<14}{on.mean()*100:>+9.3f}%{off.mean()*100:>+9.3f}%")
print(f"{'標準偏差':<14}{on.std()*100:>9.2f}%{off.std()*100:>9.2f}%")
print(f"{'上昇日の割合':<14}{(on>0).mean()*100:>9.1f}%{(off>0).mean()*100:>9.1f}%")
print(f"{'+2%超の日':<14}{(on>0.02).mean()*100:>9.1f}%{(off>0.02).mean()*100:>9.1f}%")
print(f"{'-2%超の日':<14}{(on<-0.02).mean()*100:>9.1f}%{(off<-0.02).mean()*100:>9.1f}%")

# 現金化で「避けた」リターンの内訳
avoided_up = on[on > 0].sum() * 100
avoided_dn = on[on < 0].sum() * 100
print(f"\n現金化で取り逃した上昇の合計 {avoided_up:+.0f}%  /  避けた下落の合計 {avoided_dn:+.0f}%")
print(f"→ 差し引き {avoided_up + avoided_dn:+.0f}%  （負なら下落回避が勝る）")

# 対称性の判定: 現金化日の平均が0と有意に違うか（ざっくり t）
t = on.mean() / (on.std() / np.sqrt(len(on)))
print(f"\n現金化日の平均リターンの t 値 = {t:.2f}  （|t|<2 なら 0 と区別できない ＝ 方向性なし）")
