"""
e_div シグナル vs ランダム現金化 (同頻度) の比較検証.

問い: 「S1 の MaxDD が浅いのは e_div が暴落を予知してるからか、
       それとも単に半分の期間 現金化してて株式エクスポージャーが低いからか?」

方法:
- e_div の S1 short と同じ「現金化率」「取引回数」になるよう、
  ランダムに現金化日を選んだ戦略を 1000 回シミュレーション
- e_div の MaxDD/Sharpe が、ランダム分布のどこに位置するか (percentile) を見る
- e_div が分布の「良い端」(MaxDD が浅い・Sharpe が高い) にいれば → シグナルの手柄
- 分布の真ん中なら → 単なるエクスポージャー減少の副作用

look-ahead 排除 (expanding z-score) は backtest_v2 と同じ。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from backtest_v2 import (load_indicators, fetch_spy_ohlc, simulate_v2,
                         apply_hysteresis, TRANSACTION_COST, HYSTERESIS_DAYS)

ROOT = HERE.parent
DATA_DIR = ROOT / "data"
N_RANDOM = 1000
SEED = 20260606


def random_signal_like(template_sig: pd.Series, n_blocks: int, block_len: int,
                       rng: np.random.Generator) -> pd.Series:
    """template と似た現金化率・ブロック数のランダムシグナルを作る.

    e_div シグナルは連続したブロックで ON になる (ヒステリシス 5 日)。
    同じ block 数・block 長で、開始位置だけランダムにする。
    """
    n = len(template_sig)
    arr = np.zeros(n, dtype=bool)
    placed = 0
    attempts = 0
    while placed < n_blocks and attempts < n_blocks * 50:
        attempts += 1
        start = int(rng.integers(0, max(1, n - block_len)))
        if not arr[start:start + block_len].any():  # 重複回避
            arr[start:start + block_len] = True
            placed += 1
    return pd.Series(arr, index=template_sig.index)


def main():
    period = sys.argv[1] if len(sys.argv) > 1 else "5y"
    csv_map = {
        "5y": "gamma_timeseries_w30.csv",
        "10y": "gamma_timeseries_10y_w30.csv",
        "15y": "gamma_timeseries_15y_w30.csv",
        "20y": "gamma_timeseries_20y_w30.csv",
    }
    df = load_indicators_period(csv_map[period])
    start = df.index.min().strftime("%Y-%m-%d")
    end = (df.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    ohlc = fetch_spy_ohlc(start, end)
    common = ohlc.index.intersection(df.index)
    df = df.loc[common]; ohlc = ohlc.loc[common]

    # --- e_div S1 short (本物) ---
    ediv_raw = (df["e_div"] >= 0.8)
    ediv_sig = ediv_raw.reindex(ohlc.index, method="ffill").fillna(False).astype(bool)
    ediv_sig_h = apply_hysteresis(ediv_sig, HYSTERESIS_DAYS)
    real = simulate_v2(ohlc, ediv_raw, "short")
    real_dd = real["max_drawdown"]
    real_sharpe = real["sharpe"]
    real_ret = real["total_return"]
    cash_rate = real["signal_pct"]      # 現金化してる割合

    # シグナルのブロック構造を測る (連続 ON の塊の数と平均長)
    arr = ediv_sig_h.values.astype(int)
    changes = np.diff(np.concatenate([[0], arr, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    n_blocks = len(starts)
    block_len = int(np.mean(ends - starts)) if n_blocks else 5

    # --- ランダム現金化 1000 回 ---
    rng = np.random.default_rng(SEED)
    rand_dds, rand_sharpes, rand_rets = [], [], []
    for _ in range(N_RANDOM):
        rsig = random_signal_like(ediv_sig_h, n_blocks, block_len, rng)
        # ランダムは既にブロック化済みなので hysteresis=0 で渡す
        r = simulate_v2(ohlc, rsig, "short", hysteresis=0)
        rand_dds.append(r["max_drawdown"])
        rand_sharpes.append(r["sharpe"])
        rand_rets.append(r["total_return"])
    rand_dds = np.array(rand_dds); rand_sharpes = np.array(rand_sharpes); rand_rets = np.array(rand_rets)

    # --- e_div が分布のどこにいるか (percentile) ---
    # MaxDD: 浅い(0に近い)ほど良い → e_div より浅いランダムの割合
    dd_pct = float((rand_dds < real_dd).mean())   # e_div より浅い(良い)ランダムの割合
    # = e_div が「下位 dd_pct」。dd_pct 小 = e_div は浅い側=良い
    sharpe_pct = float((rand_sharpes < real_sharpe).mean())  # e_div より低いランダムの割合 = e_div の順位
    # ランダムが e_div より「良い」値を出す確率 (= e_div がまぐれである確率)
    # MaxDD は負なので「浅い (良い) = 値が大きい」→ rand > real が e_div に勝つ
    p_dd = float((rand_dds > real_dd).mean())       # ランダムが e_div より浅い(良い) MaxDD を出す割合
    p_sharpe = float((rand_sharpes >= real_sharpe).mean())  # ランダムが e_div 以上の sharpe を出す割合

    result = {
        "period": period,
        "n_random": N_RANDOM,
        "cash_rate": round(cash_rate, 3),
        "n_blocks": n_blocks,
        "block_len": block_len,
        "ediv": {"max_drawdown": round(real_dd, 4), "sharpe": round(real_sharpe, 3),
                 "total_return": round(real_ret, 4)},
        "random_maxdd": {"mean": round(float(rand_dds.mean()), 4),
                          "best": round(float(rand_dds.max()), 4),
                          "worst": round(float(rand_dds.min()), 4)},
        "random_sharpe": {"mean": round(float(rand_sharpes.mean()), 3),
                           "best": round(float(rand_sharpes.max()), 3)},
        "random_return": {"mean": round(float(rand_rets.mean()), 4)},
        "ediv_maxdd_percentile": round(dd_pct, 3),      # 小さいほど e_div が浅い側(良い)
        "ediv_sharpe_percentile": round(sharpe_pct, 3),  # 大きいほど e_div が高い側(良い)
        "p_random_beats_ediv_maxdd": round(p_dd, 4),    # ランダムが e_div 以上に浅い確率
        "p_random_beats_ediv_sharpe": round(p_sharpe, 4),
    }

    out = DATA_DIR / f"backtest_random_benchmark_{period}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"=== {period}: e_div S1 vs ランダム現金化 {N_RANDOM} 回 ===")
    print(f"現金化率: {cash_rate*100:.0f}%  / ブロック数: {n_blocks} / 平均ブロック長: {block_len}日")
    print()
    print(f"{'指標':<12} {'e_div(本物)':>14} {'ランダム平均':>14} {'ランダム最良':>14}")
    print("-" * 60)
    print(f"{'MaxDD':<12} {real_dd*100:>+13.1f}% {rand_dds.mean()*100:>+13.1f}% {rand_dds.max()*100:>+13.1f}%")
    print(f"{'Sharpe':<12} {real_sharpe:>+14.2f} {rand_sharpes.mean():>+14.2f} {rand_sharpes.max():>+14.2f}")
    print(f"{'Return':<12} {real_ret*100:>+13.1f}% {rand_rets.mean()*100:>+13.1f}%")
    print()
    print(f"MaxDD: ランダムが e_div より浅い(良い)確率 = {p_dd*100:.1f}%")
    print(f"Sharpe: ランダムが e_div 以上に良い確率 = {p_sharpe*100:.1f}%")
    print()
    if p_dd < 0.2 and p_sharpe < 0.2:
        print("=> e_div は両方でランダムを明確に上回る = シグナルの手柄 (暴落予知が本物)")
    elif p_dd > 0.4 and p_sharpe > 0.4:
        print("=> e_div はランダムと同程度 = 単なるエクスポージャー減少の副作用")
    else:
        print("=> 部分的: e_div はランダムをやや上回るが決定的ではない")
    print(f"Saved: {out}")


def load_indicators_period(csv_name: str, mp: int = 30) -> pd.DataFrame:
    """指定 CSV から expanding z-score で e_div を計算 (look-ahead 排除)."""
    df = pd.read_csv(DATA_DIR / csv_name, parse_dates=["date"])
    df = df.dropna(subset=["L1_H1", "n_unb"]).set_index("date")
    df["z_L1"] = (df["L1_H1"] - df["L1_H1"].expanding(min_periods=mp).mean()) / df["L1_H1"].expanding(min_periods=mp).std()
    df["z_unb"] = (df["n_unb"] - df["n_unb"].expanding(min_periods=mp).mean()) / df["n_unb"].expanding(min_periods=mp).std()
    df["e_div"] = df["z_unb"] - df["z_L1"]
    return df


if __name__ == "__main__":
    main()
