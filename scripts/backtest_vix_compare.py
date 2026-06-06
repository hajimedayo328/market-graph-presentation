"""
e_div vs VIX vs ランダム — 「ボラ連動なら何でも同じか?」の決着.

問い: e_div の暴落回避は「符号構造ならでは」か、それとも
      「ボラが高い時に避ける」なら VIX でも同じ結果が出るのか?

方法: 3 つの戦略を「同じ現金化率」で公平に比較
- e_div:  e_div >= 0.8 で現金化 (本物)
- VIX:    VIX が上位 X% の日に現金化 (X は e_div と同じ現金化率になるよう調整)
- ランダム: 同頻度でランダム現金化 (前スクリプトと同じ)

全て翌日寄付き約定 + コスト 0.05% + ヒステリシス。look-ahead 排除。
VIX は当日終値で判定 → 翌日約定なので look-ahead なし。
ただし VIX の「上位 X%」閾値は期間全体の分布で決める (これは弱い look-ahead だが
e_div と条件を揃えるための簡易版。expanding 版は future work)。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from backtest_v2 import (fetch_spy_ohlc, simulate_v2, apply_hysteresis,
                         HYSTERESIS_DAYS)

ROOT = HERE.parent
DATA_DIR = ROOT / "data"


def load_indicators_with_vix(csv_name: str, mp: int = 30) -> pd.DataFrame:
    """e_div (expanding z) + VIX を結合."""
    df = pd.read_csv(DATA_DIR / csv_name, parse_dates=["date"])
    df = df.dropna(subset=["L1_H1", "n_unb"]).set_index("date")
    df["z_L1"] = (df["L1_H1"] - df["L1_H1"].expanding(min_periods=mp).mean()) / df["L1_H1"].expanding(min_periods=mp).std()
    df["z_unb"] = (df["n_unb"] - df["n_unb"].expanding(min_periods=mp).mean()) / df["n_unb"].expanding(min_periods=mp).std()
    df["e_div"] = df["z_unb"] - df["z_L1"]
    # VIX を ohlc から結合
    ohlc40 = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    if "VIX" in ohlc40.columns:
        vix = ohlc40["VIX"].reindex(df.index, method="ffill")
        df["VIX"] = vix
    return df


def main():
    period = sys.argv[1] if len(sys.argv) > 1 else "5y"
    csv_map = {"5y": "gamma_timeseries_w30.csv",
               "10y": "gamma_timeseries_10y_w30.csv",
               "15y": "gamma_timeseries_15y_w30.csv",
               "20y": "gamma_timeseries_20y_w30.csv"}
    df = load_indicators_with_vix(csv_map[period])
    df = df.dropna(subset=["VIX"])
    start = df.index.min().strftime("%Y-%m-%d")
    end = (df.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    ohlc = fetch_spy_ohlc(start, end)
    common = ohlc.index.intersection(df.index)
    df = df.loc[common]; ohlc = ohlc.loc[common]

    # --- e_div S1 short ---
    ediv_sig = (df["e_div"] >= 0.8)
    ediv = simulate_v2(ohlc, ediv_sig, "short")
    cash_rate = ediv["signal_pct"]

    # --- VIX 戦略: e_div と同じ現金化率になる閾値で「VIX 高い日に現金化」---
    # cash_rate と同じ割合だけ VIX 上位を現金化
    vix_threshold = float(df["VIX"].quantile(1 - cash_rate))
    vix_sig = (df["VIX"] >= vix_threshold)
    vix = simulate_v2(ohlc, vix_sig, "short")

    # --- ランダム (前スクリプトの値を再利用 or 簡易再計算) ---
    # ヒステリシス適用後のブロック構造でランダム
    ediv_sig_h = apply_hysteresis(ediv_sig.reindex(ohlc.index, method="ffill").fillna(False).astype(bool), HYSTERESIS_DAYS)
    arr = ediv_sig_h.values.astype(int)
    changes = np.diff(np.concatenate([[0], arr, [0]]))
    starts = np.where(changes == 1)[0]; ends = np.where(changes == -1)[0]
    n_blocks = len(starts); block_len = int(np.mean(ends - starts)) if n_blocks else 5
    rng = np.random.default_rng(20260606)
    rand_dds, rand_sharpes = [], []
    n = len(ediv_sig_h)
    for _ in range(500):
        a = np.zeros(n, dtype=bool); placed = 0; att = 0
        while placed < n_blocks and att < n_blocks * 50:
            att += 1; s = int(rng.integers(0, max(1, n - block_len)))
            if not a[s:s+block_len].any():
                a[s:s+block_len] = True; placed += 1
        r = simulate_v2(ohlc, pd.Series(a, index=ediv_sig_h.index), "short", hysteresis=0)
        rand_dds.append(r["max_drawdown"]); rand_sharpes.append(r["sharpe"])
    rand_dd_mean = float(np.mean(rand_dds)); rand_sharpe_mean = float(np.mean(rand_sharpes))

    result = {
        "period": period,
        "cash_rate_ediv": round(cash_rate, 3),
        "cash_rate_vix": round(float(vix_sig.mean()), 3),
        "vix_threshold": round(vix_threshold, 2),
        "ediv":   {"sharpe": round(ediv["sharpe"], 3), "max_drawdown": round(ediv["max_drawdown"], 4),
                   "total_return": round(ediv["total_return"], 4)},
        "vix":    {"sharpe": round(vix["sharpe"], 3), "max_drawdown": round(vix["max_drawdown"], 4),
                   "total_return": round(vix["total_return"], 4)},
        "random": {"sharpe": round(rand_sharpe_mean, 3), "max_drawdown": round(rand_dd_mean, 4)},
    }
    out = DATA_DIR / f"backtest_vix_compare_{period}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"=== {period}: e_div vs VIX vs ランダム (同じ現金化率 ~{cash_rate*100:.0f}%) ===")
    print(f"{'戦略':<10} {'Sharpe':>8} {'MaxDD':>9} {'Return':>9}")
    print("-" * 40)
    print(f"{'e_div':<10} {ediv['sharpe']:>+8.2f} {ediv['max_drawdown']*100:>+8.1f}% {ediv['total_return']*100:>+8.1f}%")
    print(f"{'VIX':<10} {vix['sharpe']:>+8.2f} {vix['max_drawdown']*100:>+8.1f}% {vix['total_return']*100:>+8.1f}%")
    print(f"{'ランダム':<8} {rand_sharpe_mean:>+8.2f} {rand_dd_mean*100:>+8.1f}%")
    print()
    # 判定
    if ediv["sharpe"] > vix["sharpe"] and ediv["max_drawdown"] > vix["max_drawdown"]:
        print("=> e_div は VIX に両方で勝つ = 符号構造ならではの独自性あり")
    elif abs(ediv["sharpe"] - vix["sharpe"]) < 0.1:
        print("=> e_div と VIX はほぼ互角 = 「ボラ連動」で説明できる部分が大きい")
    else:
        print("=> 混在: 一方の指標で勝ち、他方で負け")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
