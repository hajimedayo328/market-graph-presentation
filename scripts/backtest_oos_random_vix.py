"""
Walk-forward OUT-OF-SAMPLE 検証: e_div vs ランダム vs VIX.

問い (研究の穴):
  既存の「e_div がランダム 1000 回・VIX に勝つ」は in-sample
  (過去全部を最適化した固定閾値 0.8) の結果。閾値が過去最適化されてる
  だけかもしれない。**未来データ (OOS) でも同じ優位が出るか?** を検証する。

設計 (look-ahead 厳密):
  1. walk-forward:
     - train (例 3 年) で e_div の閾値を 80 percentile で決定
     - 次の test (例 1 年) でその閾値を使い S1 short 戦略を実行
     - 1 年ずつ転がして全 test 期間の e_div 戦略 OOS リターンを連結
     - z-score は expanding window (過去のみ、look-ahead 排除)
     - 閾値は **train 期間の e_div のみ** で決定 (test を覗かない)

  2. 同じ test 期間でランダム比較:
     - 各 test 期間で、e_div と同じ現金化率・ブロック数のランダム現金化を
       N_RANDOM 回生成し、test の OHLC で実行
     - 各乱数 seed ごとに「全 test 期間のランダム OOS リターン」を連結
     - 連結 OOS の e_div がランダム分布のどこか (percentile)
     - 「OOS でも e_div がランダムに勝つか」

  3. VIX 比較 (VIX データがある test 期間のみ):
     - 各 test 期間の VIX 現金化閾値は **train 期間の VIX 分布** から
       「e_div と同じ現金化率」になる分位点で決定 (look-ahead 排除)
     - その閾値で test の VIX 現金化を実行 → 全 test を連結
     - OOS で e_div が VIX に勝つか
     - VIX が無い (まだ観測が無い) 初期 fold は VIX 列だけ欠測扱いにして
       「VIX 比較は X fold のみ」と正直に記録

評価指標: OOS total return / Sharpe / MaxDD。
  暴落回避の本質は MaxDD (浅い = 良い) と Sharpe。

結論:
  - OOS でも e_div がランダム/VIX に勝つ → 過去最適化じゃない、本物
  - OOS では勝てない → in-sample の優位は過去最適化だった (正直に)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from backtest_v2 import apply_hysteresis, TRANSACTION_COST, HYSTERESIS_DAYS
from backtest_random_benchmark import random_signal_like  # 既存ロジック再利用

import yfinance as yf

ROOT = HERE.parent
DATA_DIR = ROOT / "data"

N_RANDOM = 500          # 各 walk-forward 全体に対するランダム連結シミュ回数
SEED = 20260606
TRAIN_YEARS = 3
TEST_YEARS = 1
PERCENTILE = 80

CSV_MAP = {
    "5y":  "gamma_timeseries_w30.csv",
    "10y": "gamma_timeseries_10y_w30.csv",
    "15y": "gamma_timeseries_15y_w30.csv",
    "20y": "gamma_timeseries_20y_w30.csv",
}
PARQUET_MAP = {
    "5y":  "ohlc_40.parquet",
    "10y": "ohlc_40_10y.parquet",
    "15y": "ohlc_40_15y.parquet",
    "20y": "ohlc_40_20y.parquet",
}


def load_indicators_with_vix(csv_name: str, parquet_name: str,
                             mp: int = 30) -> pd.DataFrame:
    """e_div (expanding z-score, look-ahead 排除) + VIX を結合.

    VIX は parquet (US 指数) を gamma の取引日に reindex + ffill。
    ffill は「直近過去の VIX」を埋めるだけなので look-ahead は無い。
    """
    df = pd.read_csv(DATA_DIR / csv_name, parse_dates=["date"])
    df = df.dropna(subset=["L1_H1", "n_unb"]).set_index("date")
    df["z_L1"] = (
        (df["L1_H1"] - df["L1_H1"].expanding(min_periods=mp).mean())
        / df["L1_H1"].expanding(min_periods=mp).std()
    )
    df["z_unb"] = (
        (df["n_unb"] - df["n_unb"].expanding(min_periods=mp).mean())
        / df["n_unb"].expanding(min_periods=mp).std()
    )
    df["e_div"] = df["z_unb"] - df["z_L1"]
    ohlc40 = pd.read_parquet(DATA_DIR / parquet_name)
    if "VIX" in ohlc40.columns:
        # reindex してから ffill (内部 NaN も直近過去で埋める, look-ahead なし)
        df["VIX"] = ohlc40["VIX"].reindex(df.index).ffill()
    else:
        df["VIX"] = np.nan
    return df


def fetch_spy_ohlc(start: str, end: str) -> pd.DataFrame:
    print(f"Fetching ^GSPC OHLC ({start} to {end})...")
    df = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.dropna(subset=["Open", "Close"])


def _strat_rets_from_signal(sig_pos: pd.Series, rets_test: pd.Series) -> pd.Series:
    """ポジション (翌日寄付き約定済み bool) + コストから戦略リターンを作る (short)."""
    strat = rets_test * (~sig_pos).astype(float)
    pos_change = (sig_pos.astype(int).diff().abs() > 0)
    strat = strat - pos_change.astype(float) * TRANSACTION_COST
    return strat


def _metrics(strat_rets: pd.Series) -> dict:
    """連結 OOS リターン系列から total/CAGR/Sharpe/MaxDD を計算."""
    eq = (1 + strat_rets).cumprod()
    n_years = len(strat_rets) / 252
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol = float(strat_rets.std() * np.sqrt(252))
    sharpe = float(cagr / vol) if vol > 1e-9 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe,
            "max_drawdown": dd}


def walkforward_oos(df: pd.DataFrame, ohlc: pd.DataFrame,
                    train_years: int, test_years: int,
                    percentile: float, n_random: int,
                    seed: int) -> dict:
    """e_div walk-forward OOS + 同 test 期間でのランダム/VIX 連結比較."""
    common = ohlc.index.intersection(df.index)
    df_c = df.loc[common]
    ohlc_c = ohlc.loc[common]
    rets = ohlc_c["Close"].pct_change().fillna(0)

    train_days = int(train_years * 252)
    test_days = int(test_years * 252)
    n = len(df_c)

    rng = np.random.default_rng(seed)

    # 連結用: e_div / VIX の OOS リターンを入れる器
    oos_ediv = pd.Series(0.0, index=ohlc_c.index)
    oos_mask = pd.Series(False, index=ohlc_c.index)        # e_div / random が乗る区間
    oos_vix = pd.Series(0.0, index=ohlc_c.index)
    oos_vix_mask = pd.Series(False, index=ohlc_c.index)    # VIX が使えた区間のみ
    # ランダムは N_RANDOM 本の連結曲線を作る -> seed ごとに oos_mask 区間へ書き込む
    oos_rand = [pd.Series(0.0, index=ohlc_c.index) for _ in range(n_random)]

    folds = []
    fold_id = 0
    n_folds_vix = 0

    start = 0
    while start + train_days + test_days <= n:
        train = df_c.iloc[start: start + train_days]
        test = df_c.iloc[start + train_days: start + train_days + test_days]

        train_ediv = train["e_div"].dropna()
        if len(train_ediv) < 30:
            start += test_days
            continue

        # === e_div 戦略: 閾値は train のみで決定 (look-ahead 排除) ===
        thr = float(np.percentile(train_ediv, percentile))
        sig_test = (test["e_div"] >= thr).fillna(False)
        sig_test_h = apply_hysteresis(sig_test, HYSTERESIS_DAYS)
        sig_pos = sig_test_h.shift(1).fillna(False).astype(bool)
        rets_test = rets.loc[test.index]
        ediv_strat = _strat_rets_from_signal(sig_pos, rets_test)
        oos_ediv.loc[test.index] = ediv_strat
        oos_mask.loc[test.index] = True

        # e_div の test 期間ブロック構造 (現金化率・ブロック数・長さ) を測る
        arr = sig_test_h.values.astype(int)
        changes = np.diff(np.concatenate([[0], arr, [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        n_blocks = len(starts)
        block_len = int(np.mean(ends - starts)) if n_blocks else 5
        cash_rate_fold = float(sig_pos.mean())

        # === ランダム: 同 test 期間で n_blocks/block_len を真似た現金化 ===
        for k in range(n_random):
            rsig = random_signal_like(sig_test_h, n_blocks, block_len, rng)
            r_pos = rsig.shift(1).fillna(False).astype(bool)
            rand_strat = _strat_rets_from_signal(r_pos, rets_test)
            oos_rand[k].loc[test.index] = rand_strat

        # === VIX: 閾値は train 期間の VIX 分布から (look-ahead 排除) ===
        vix_fold_ret = None
        vix_used = False
        train_vix = train["VIX"].dropna()
        test_vix = test["VIX"]
        # train/test 両方に十分な VIX 観測がある fold のみ
        if len(train_vix) >= 60 and test_vix.notna().mean() > 0.9:
            # e_div と同じ現金化率になる VIX 上位の閾値を train から決める
            vix_thr = float(train_vix.quantile(1 - cash_rate_fold))
            vix_sig = (test_vix >= vix_thr).fillna(False)
            vix_sig_h = apply_hysteresis(vix_sig, HYSTERESIS_DAYS)
            vix_pos = vix_sig_h.shift(1).fillna(False).astype(bool)
            vix_strat = _strat_rets_from_signal(vix_pos, rets_test)
            oos_vix.loc[test.index] = vix_strat
            oos_vix_mask.loc[test.index] = True
            vix_fold_ret = float((1 + vix_strat).prod() - 1)
            vix_used = True
            n_folds_vix += 1

        ediv_fold_ret = float((1 + ediv_strat).prod() - 1)
        bh_fold_ret = float((1 + rets_test).prod() - 1)
        folds.append({
            "fold_id": fold_id,
            "train_period": f"{train.index.min().date()}〜{train.index.max().date()}",
            "test_period": f"{test.index.min().date()}〜{test.index.max().date()}",
            "threshold": round(thr, 4),
            "cash_rate": round(cash_rate_fold, 3),
            "n_blocks": n_blocks,
            "ediv_OOS_return": round(ediv_fold_ret, 4),
            "bh_return": round(bh_fold_ret, 4),
            "vix_used": vix_used,
            "vix_OOS_return": round(vix_fold_ret, 4) if vix_fold_ret is not None else None,
        })

        fold_id += 1
        start += test_days

    # ---- 連結 OOS 評価 (e_div / ランダム) ----
    ediv_rets = oos_ediv[oos_mask]
    ediv_m = _metrics(ediv_rets)
    bh_rets = rets[oos_mask]
    bh_eq = (1 + bh_rets).cumprod()
    bh_total = float(bh_eq.iloc[-1] - 1)
    bh_dd = float((bh_eq / bh_eq.cummax() - 1).min())
    bh_vol = float(bh_rets.std() * np.sqrt(252))
    bh_cagr = float(bh_eq.iloc[-1] ** (1 / (len(bh_rets) / 252)) - 1)
    bh_sharpe = float(bh_cagr / bh_vol) if bh_vol > 1e-9 else 0.0

    rand_total, rand_sharpe, rand_dd = [], [], []
    for k in range(n_random):
        rr = oos_rand[k][oos_mask]
        m = _metrics(rr)
        rand_total.append(m["total_return"])
        rand_sharpe.append(m["sharpe"])
        rand_dd.append(m["max_drawdown"])
    rand_total = np.array(rand_total)
    rand_sharpe = np.array(rand_sharpe)
    rand_dd = np.array(rand_dd)

    # e_div が連結ランダム分布のどこか
    # MaxDD: 浅い (値が大きい=0 に近い) ほど良い
    p_rand_beats_dd = float((rand_dd > ediv_m["max_drawdown"]).mean())     # ランダムが e_div より浅い割合
    p_rand_beats_sharpe = float((rand_sharpe >= ediv_m["sharpe"]).mean())  # ランダムが e_div 以上の sharpe
    p_rand_beats_ret = float((rand_total >= ediv_m["total_return"]).mean())
    # percentile (e_div の順位): 大きいほど e_div が上位
    ediv_dd_pct = float((rand_dd < ediv_m["max_drawdown"]).mean())   # e_div より深い(悪い)ランダムの割合 = e_div は上位 pct
    ediv_sharpe_pct = float((rand_sharpe < ediv_m["sharpe"]).mean())

    # ---- 連結 OOS 評価 (VIX, 使えた区間のみ) ----
    vix_block = None
    if oos_vix_mask.sum() > 0:
        vix_rets = oos_vix[oos_vix_mask]
        vix_m = _metrics(vix_rets)
        # 同区間の e_div を再評価 (公平比較: VIX が使えた区間だけで揃える)
        ediv_on_vix_span = oos_ediv[oos_vix_mask]
        ediv_vm = _metrics(ediv_on_vix_span)
        vix_bh = rets[oos_vix_mask]
        vix_bh_eq = (1 + vix_bh).cumprod()
        vix_block = {
            "n_folds_with_vix": n_folds_vix,
            "vix_span_start": str(oos_vix[oos_vix_mask].index.min().date()),
            "vix_span_end": str(oos_vix[oos_vix_mask].index.max().date()),
            "vix": {k: round(v, 4) for k, v in vix_m.items()},
            "ediv_same_span": {k: round(v, 4) for k, v in ediv_vm.items()},
            "ediv_beats_vix_sharpe": bool(ediv_vm["sharpe"] > vix_m["sharpe"]),
            "ediv_beats_vix_maxdd": bool(ediv_vm["max_drawdown"] > vix_m["max_drawdown"]),
            "ediv_beats_vix_return": bool(ediv_vm["total_return"] > vix_m["total_return"]),
            "bh_total_return_same_span": round(float(vix_bh_eq.iloc[-1] - 1), 4),
        }

    return {
        "params": {
            "train_years": train_years, "test_years": test_years,
            "percentile": percentile, "n_random": n_random,
            "transaction_cost": TRANSACTION_COST,
            "hysteresis_days": HYSTERESIS_DAYS,
            "execution": "next-day open", "zscore": "expanding (look-ahead排除)",
            "vix_threshold_source": "train period only (look-ahead排除)",
        },
        "n_folds": fold_id,
        "n_folds_with_vix": n_folds_vix,
        "oos_span": f"{ediv_rets.index.min().date()}〜{ediv_rets.index.max().date()}",
        "oos_n_days": int(len(ediv_rets)),
        "ediv": {k: round(v, 4) for k, v in ediv_m.items()},
        "buy_hold": {"total_return": round(bh_total, 4), "sharpe": round(bh_sharpe, 4),
                     "max_drawdown": round(bh_dd, 4)},
        "random": {
            "total_return_mean": round(float(rand_total.mean()), 4),
            "total_return_best": round(float(rand_total.max()), 4),
            "sharpe_mean": round(float(rand_sharpe.mean()), 4),
            "sharpe_best": round(float(rand_sharpe.max()), 4),
            "maxdd_mean": round(float(rand_dd.mean()), 4),
            "maxdd_best": round(float(rand_dd.max()), 4),  # 最も浅い (良い)
        },
        # ランダムが e_div に勝つ確率 (小さいほど e_div の手柄)
        "p_random_beats_ediv_maxdd": round(p_rand_beats_dd, 4),
        "p_random_beats_ediv_sharpe": round(p_rand_beats_sharpe, 4),
        "p_random_beats_ediv_return": round(p_rand_beats_ret, 4),
        # e_div の percentile (大きいほど e_div が上位 = 良い)
        "ediv_maxdd_percentile": round(ediv_dd_pct, 4),
        "ediv_sharpe_percentile": round(ediv_sharpe_pct, 4),
        "vix_comparison": vix_block,
        "folds": folds,
    }


def main():
    periods = sys.argv[1:] if len(sys.argv) > 1 else ["5y", "10y", "15y", "20y"]
    out = {"description": "Walk-forward OOS: e_div vs random vs VIX (look-ahead厳密)",
           "n_random": N_RANDOM, "seed": SEED,
           "config": f"train={TRAIN_YEARS}y test={TEST_YEARS}y percentile={PERCENTILE}",
           "results": {}}

    for period in periods:
        if period not in CSV_MAP:
            print(f"skip unknown period: {period}")
            continue
        print(f"\n{'='*70}\n=== Period: {period} ===")
        df = load_indicators_with_vix(CSV_MAP[period], PARQUET_MAP[period])
        vix_n = int(df["VIX"].notna().sum())
        print(f"indicators: {df.shape}, {df.index.min().date()} -> {df.index.max().date()}, VIX days={vix_n}")
        start = df.index.min().strftime("%Y-%m-%d")
        end = (df.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        ohlc = fetch_spy_ohlc(start, end)

        res = walkforward_oos(df, ohlc, TRAIN_YEARS, TEST_YEARS,
                              PERCENTILE, N_RANDOM, SEED)
        out["results"][period] = res

        print(f"\n--- {period} OOS (folds={res['n_folds']}, vix folds={res['n_folds_with_vix']}) ---")
        print(f"  OOS span: {res['oos_span']} ({res['oos_n_days']} 営業日)")
        e = res["ediv"]; b = res["buy_hold"]; r = res["random"]
        print(f"  {'戦略':<14} {'Return':>9} {'Sharpe':>8} {'MaxDD':>9}")
        print(f"  {'e_div (OOS)':<14} {e['total_return']*100:>+8.1f}% {e['sharpe']:>+8.2f} {e['max_drawdown']*100:>+8.1f}%")
        print(f"  {'Buy&Hold':<14} {b['total_return']*100:>+8.1f}% {b['sharpe']:>+8.2f} {b['max_drawdown']*100:>+8.1f}%")
        print(f"  {'Random mean':<14} {r['total_return_mean']*100:>+8.1f}% {r['sharpe_mean']:>+8.2f} {r['maxdd_mean']*100:>+8.1f}%")
        print(f"  {'Random best':<14} {r['total_return_best']*100:>+8.1f}% {r['sharpe_best']:>+8.2f} {r['maxdd_best']*100:>+8.1f}%")
        print(f"  ランダムが e_div に勝つ確率: MaxDD={res['p_random_beats_ediv_maxdd']*100:.1f}%  "
              f"Sharpe={res['p_random_beats_ediv_sharpe']*100:.1f}%  Return={res['p_random_beats_ediv_return']*100:.1f}%")
        if res["vix_comparison"]:
            vc = res["vix_comparison"]
            print(f"  VIX 比較 ({vc['n_folds_with_vix']} folds, {vc['vix_span_start']}〜{vc['vix_span_end']}):")
            print(f"    e_div(同区間): Sharpe={vc['ediv_same_span']['sharpe']:+.2f} MaxDD={vc['ediv_same_span']['max_drawdown']*100:+.1f}% Ret={vc['ediv_same_span']['total_return']*100:+.1f}%")
            print(f"    VIX:           Sharpe={vc['vix']['sharpe']:+.2f} MaxDD={vc['vix']['max_drawdown']*100:+.1f}% Ret={vc['vix']['total_return']*100:+.1f}%")
            print(f"    e_div が VIX に勝つ: Sharpe={vc['ediv_beats_vix_sharpe']} MaxDD={vc['ediv_beats_vix_maxdd']} Return={vc['ediv_beats_vix_return']}")
        else:
            print("  VIX 比較: 該当 fold なし")

    out_path = DATA_DIR / "backtest_oos_random_vix.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
