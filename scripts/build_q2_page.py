"""学会質問2「指標にこの2つ以外を使うのは?」の検証ページ (q2.html) を作る.

出すもの（20年 / 15年 / 10年）:
  A) 前進検証（train 3年 → test 1年、train の80パーセンタイルで休む）でシグナルだけ差し替えた成績
  B) 休む率を変えたときの成績の曲線（しきい値を 50〜95 パーセンタイルで動かす）。
     シグナルごとに休む率が違う、という不公平を、同じ休む率で読み比べられる形にして解消する
  C) 各シグナルが休んだ日（帯）と、e_div と同じ日に休んだ割合
  D) e_div と同じ休み方（回数・長さ）ででたらめに休んだ場合との比較
  E) 直近5年・11指標の55通りが e_div とどこまで同じ動きか

  python scripts/build_q2_page.py      → data/q2_page_data.json と q2.html（テンプレートがあれば）
"""
from __future__ import annotations
import sys, json, itertools, warnings, datetime
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import backtest_oos_random_vix as bo
from backtest_v2 import apply_hysteresis, HYSTERESIS_DAYS, TRANSACTION_COST
from backtest_random_benchmark import random_signal_like

ROOT = Path(__file__).parent.parent
MP = 30
N_RANDOM = 500                # しきい値80での でたらめ本数（backtest_oos_random_vix と同じ）
N_RANDOM_FRONTIER = 200       # 休む率を動かした曲線用の でたらめ本数
STEP = 5                      # 図用の間引き（営業日）
PERIODS = ["20y", "15y", "10y"]
PCTS = [50, 60, 70, 75, 80, 85, 90, 95]
SIGS = [
    ("ediv", "e_div（矛盾 − 穴）"),
    ("unb", "矛盾だけ"),
    ("l1", "穴だけ"),
    ("ubr_l1", "不均衡率 − 穴"),
    ("ubr", "不均衡率だけ"),
    ("edges", "辺の数だけ"),
    ("edges_l1", "辺の数 − 穴"),
    ("vix", "VIX（恐怖指数）"),
    ("rv", "直近20日の値動きの大きさ"),
]
CHART_KEYS = ["ediv", "rv", "unb", "l1"]


def ez(s: pd.Series) -> pd.Series:
    """expanding z-score（過去のみ）."""
    return (s - s.expanding(min_periods=MP).mean()) / s.expanding(min_periods=MP).std()


def runs(mask: np.ndarray) -> list[list[int]]:
    """True の連続区間を [start, end) の添字で返す."""
    ch = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return [[int(a), int(b)] for a, b in zip(np.where(ch == 1)[0], np.where(ch == -1)[0])]


def block_shape(sig_h: pd.Series) -> tuple[int, int]:
    arr = sig_h.values.astype(int)
    ch = np.diff(np.concatenate([[0], arr, [0]]))
    st, en = np.where(ch == 1)[0], np.where(ch == -1)[0]
    return len(st), (int(np.mean(en - st)) if len(st) else 5)


def run_period(period: str) -> dict:
    df = bo.load_indicators_with_vix(bo.CSV_MAP[period], bo.PARQUET_MAP[period])
    px_all = pd.read_parquet(ROOT / "data" / bo.PARQUET_MAP[period])["SP500"].dropna()
    common = px_all.index.intersection(df.index)
    df = df.loc[common]
    rets = px_all.loc[common].pct_change().fillna(0)
    rv20 = (px_all.pct_change().rolling(20).std() * np.sqrt(252)).reindex(df.index).ffill()
    ubr = ez(1 - df["balance_rate"])
    edges = ez(df["n_edges"])
    sig = {
        "ediv": df["e_div"], "unb": df["z_unb"], "l1": -df["z_L1"],
        "ubr_l1": ubr - df["z_L1"], "ubr": ubr, "edges": edges, "edges_l1": edges - df["z_L1"],
        "vix": df["VIX"], "rv": rv20,
    }

    # fold の切り方（e_div の train が30点以上ある fold だけ）
    train_days, test_days = int(bo.TRAIN_YEARS * 252), int(bo.TEST_YEARS * 252)
    folds, start = [], 0
    while start + train_days + test_days <= len(df):
        tr = slice(start, start + train_days)
        te = slice(start + train_days, start + train_days + test_days)
        if len(sig["ediv"].iloc[tr].dropna()) >= 30:
            folds.append((tr, te))
        start += test_days
    idx = df.index[folds[0][1]].append([df.index[te] for _, te in folds[1:]])
    r = rets.loc[idx]
    n = len(idx)

    def fold_signal(key: str, pct: float, tr: slice, te: slice) -> tuple[pd.Series, pd.Series]:
        """(ヒステリシス後のシグナル, 翌日約定のポジション)."""
        tr_s, te_s = sig[key].iloc[tr].dropna(), sig[key].iloc[te]
        if len(tr_s) < 30:
            f = pd.Series(False, index=df.index[te])
            return f, f
        thr = float(np.percentile(tr_s, pct))
        h = apply_hysteresis((te_s >= thr).fillna(False), HYSTERESIS_DAYS)
        return h, h.shift(1).fillna(False).astype(bool)

    def positions(key: str, pct: float) -> pd.Series:
        return pd.concat([fold_signal(key, pct, tr, te)[1] for tr, te in folds]).astype(bool)

    def metrics_of(p: pd.Series) -> dict:
        return bo._metrics(bo._strat_rets_from_signal(p, r))

    # ---- しきい値80（ポスターと同じ手順） ----
    pos80 = {k: positions(k, bo.PERCENTILE) for k, _ in SIGS}
    base = pos80["ediv"]

    # でたらめ500本（backtest_oos_random_vix と同じ作り方・同じ乱数順）
    rng = np.random.default_rng(bo.SEED)
    rand_parts = [[] for _ in range(N_RANDOM)]
    for tr, te in folds:
        h, _ = fold_signal("ediv", bo.PERCENTILE, tr, te)
        nb, bl = block_shape(h)
        r_te = rets.loc[df.index[te]]
        for j in range(N_RANDOM):
            rp = random_signal_like(h, nb, bl, rng).shift(1).fillna(False).astype(bool)
            rand_parts[j].append(bo._strat_rets_from_signal(rp, r_te).values)
    rand_m = [bo._metrics(pd.Series(np.concatenate(parts))) for parts in rand_parts]
    rand_dd = np.array([m["max_drawdown"] for m in rand_m])
    rand_sh = np.array([m["sharpe"] for m in rand_m])

    pick = list(range(STEP - 1, n, STEP))
    if pick[-1] != n - 1:
        pick.append(n - 1)
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    year_ends = {}
    for i, d in enumerate(idx):
        year_ends[d.year] = i
    ye = sorted(year_ends.items())

    def curve(strat: pd.Series) -> dict:
        eq = (1 + strat).cumprod().values
        dd = eq / np.maximum.accumulate(eq) - 1
        edges_ = [0] + [p + 1 for p in pick]
        dd_min = [float(dd[a:b].min()) for a, b in zip(edges_[:-1], edges_[1:])]
        return {"eq": [round(float(eq[p]), 4) for p in pick], "dd": [round(x, 4) for x in dd_min],
                "eq_year_end": [round(float(eq[i]), 3) for _, i in ye]}

    hold_m = bo._metrics(r)
    out = {"period": period, "n_folds": len(folds), "oos_start": dates[0], "oos_end": dates[-1], "n_days": n,
           "dates": dates, "pick": pick, "years": [int(y) for y, _ in ye],
           "hold": {"total_return": round(hold_m["total_return"], 4), "sharpe": round(hold_m["sharpe"], 3),
                    "max_drawdown": round(hold_m["max_drawdown"], 4), **curve(r)},
           "random": {"n": N_RANDOM, "maxdd": [round(float(x), 4) for x in np.sort(rand_dd)],
                      "sharpe": [round(float(x), 3) for x in np.sort(rand_sh)],
                      "maxdd_mean": round(float(rand_dd.mean()), 4), "sharpe_mean": round(float(rand_sh.mean()), 3)}}

    rows = {}
    for k, label in SIGS:
        p = pos80[k]
        strat = bo._strat_rets_from_signal(p, r)
        mt = bo._metrics(strat)
        both, either = int((p & base).sum()), int((p | base).sum())
        row = {"key": k, "label": label,
               "total_return": round(mt["total_return"], 4), "sharpe": round(mt["sharpe"], 3),
               "max_drawdown": round(mt["max_drawdown"], 4), "cash_rate": round(float(p.mean()), 4),
               "share_of_ediv": round(both / max(1, int(base.sum())), 4), "jaccard": round(both / max(1, either), 4),
               "rand_worse_maxdd": round(float((rand_dd < mt["max_drawdown"]).mean()), 4),
               "rand_worse_sharpe": round(float((rand_sh < mt["sharpe"]).mean()), 4),
               "runs": runs(p.values)}
        if k in CHART_KEYS:
            row.update(curve(strat))
        rows[k] = row
    out["signals"] = rows

    # ---- 休む率を動かした曲線 ----
    frontier, near = {}, {}
    target = rows["ediv"]["cash_rate"]
    for k, label in SIGS:
        pts = []
        for pct in PCTS:
            p = pos80[k] if pct == bo.PERCENTILE else positions(k, pct)
            mt = metrics_of(p)
            both, either = int((p & base).sum()), int((p | base).sum())
            pts.append({"pct": pct, "cash_rate": round(float(p.mean()), 4), "max_drawdown": round(mt["max_drawdown"], 4),
                        "sharpe": round(mt["sharpe"], 3), "total_return": round(mt["total_return"], 4),
                        "share_of_ediv": round(both / max(1, int(base.sum())), 4), "jaccard": round(both / max(1, either), 4)})
        frontier[k] = pts
        near[k] = min(pts, key=lambda x: abs(x["cash_rate"] - target))
    out["frontier"] = frontier
    out["near_budget"] = near

    rng2 = np.random.default_rng(bo.SEED + 1)
    fr_rand = []
    for pct in PCTS:
        parts = [[] for _ in range(N_RANDOM_FRONTIER)]
        cash = []
        for tr, te in folds:
            h, p_e = fold_signal("ediv", pct, tr, te)
            cash.append(p_e.values)
            nb, bl = block_shape(h)
            r_te = rets.loc[df.index[te]]
            for j in range(N_RANDOM_FRONTIER):
                rp = random_signal_like(h, nb, bl, rng2).shift(1).fillna(False).astype(bool)
                parts[j].append(bo._strat_rets_from_signal(rp, r_te).values)
        ms = [bo._metrics(pd.Series(np.concatenate(x))) for x in parts]
        dd = np.array([m["max_drawdown"] for m in ms]); sh = np.array([m["sharpe"] for m in ms])
        fr_rand.append({"pct": pct, "cash_rate": round(float(np.concatenate(cash).mean()), 4),
                        "dd_q10": round(float(np.quantile(dd, .1)), 4), "dd_q50": round(float(np.quantile(dd, .5)), 4),
                        "dd_q90": round(float(np.quantile(dd, .9)), 4), "sh_q10": round(float(np.quantile(sh, .1)), 3),
                        "sh_q50": round(float(np.quantile(sh, .5)), 3), "sh_q90": round(float(np.quantile(sh, .9)), 3)})
    out["frontier_random"] = fr_rand

    e = rows["ediv"]
    print(f"[{period}] folds={len(folds)} OOS {dates[0]}〜{dates[-1]}  でたらめが e_div より浅い割合 {float((rand_dd > e['max_drawdown']).mean()):.3f}")
    print(f"  {'しきい値80':<16} {'儲け':>8} {'成績':>5} {'最大下落':>7} {'休む率':>5} {'同じ日':>5} | 休む率を近づけた時: しきい値 休む率 最大下落 成績 同じ日")
    for k, label in SIGS:
        x, nb = rows[k], near[k]
        print(f"  {label:<16} {x['total_return']:+8.0%} {x['sharpe']:5.2f} {x['max_drawdown']:7.1%} {x['cash_rate']:5.0%} {x['share_of_ediv']:5.0%} | "
              f"{nb['pct']:>3} {nb['cash_rate']:5.0%} {nb['max_drawdown']:7.1%} {nb['sharpe']:5.2f} {nb['share_of_ediv']:5.0%}")
    print("  でたらめ(休む率別) 中央値:", [(x["pct"], f"{x['cash_rate']:.0%}", f"{x['dd_q50']:.1%}", x["sh_q50"]) for x in fr_rand])
    return out


def similarity_5y() -> dict:
    mi = pd.read_csv(ROOT / "data" / "multi_indicators_w30.csv", parse_dates=["date"]).set_index("date").sort_index()
    inds = ["L1", "L2", "Linf", "nH1", "meanP", "entropy", "n_unb_total", "n_unb_3", "n_unb_4", "n_unb_5plus", "balance_rate"]
    short = {"L1": "L¹", "L2": "L²", "Linf": "L∞", "nH1": "穴の数", "meanP": "平均寿命", "entropy": "ばらつき",
             "n_unb_total": "矛盾（全部）", "n_unb_3": "矛盾（三角形）", "n_unb_4": "矛盾（四角形）",
             "n_unb_5plus": "矛盾（5角形〜）", "balance_rate": "均衡率"}
    shape = {"L1", "L2", "Linf", "nH1", "meanP", "entropy"}
    z = pd.DataFrame({c: ez(mi[c]) for c in inds}).dropna()
    ediv = z["n_unb_total"] - z["L1"]
    top_e = ediv >= ediv.quantile(0.8)
    scan = json.loads((ROOT / "data" / "robustness_ediv_pair_scan.json").read_text(encoding="utf-8"))
    ranks = {}
    for cat, s in scan["shock_event_study"].items():
        order = sorted(s["per_pair"].items(), key=lambda kv: -abs(kv[1]["mean_dsigma"]))
        ranks[cat] = {k: i + 1 for i, (k, _) in enumerate(order)}
    rows = []
    for a, b in itertools.combinations(inds, 2):
        d = z[a] - z[b]
        c = float(np.corrcoef(d, ediv)[0, 1])
        da = d if c >= 0 else -d
        top_d = da >= da.quantile(0.8)
        kind = "形 − 符号" if (a in shape) != (b in shape) else ("形 − 形" if a in shape else "符号 − 符号")
        key = f"{a}__minus__{b}"
        rows.append({"pair": f"{short[a]} − {short[b]}", "kind": kind, "abs_corr": round(abs(c), 3),
                     "top20_overlap": round(float((top_d & top_e).sum() / top_e.sum()), 3),
                     "is_ediv": key == "L1__minus__n_unb_total",
                     "rank_trade": ranks["trade_policy"].get(key), "rank_market": ranks["market_structure"].get(key),
                     "rank_war": ranks["war"].get(key)})
    rows.sort(key=lambda x: -x["abs_corr"])
    others = [x for x in rows if not x["is_ediv"]]
    ac = np.array([x["abs_corr"] for x in others])
    hist = [int(((ac >= lo / 10) & ((ac < (lo + 1) / 10) | (lo == 9))).sum()) for lo in range(10)]
    by_kind = {}
    for kd in ("形 − 符号", "形 − 形", "符号 − 符号"):
        v = [x["abs_corr"] for x in others if x["kind"] == kd]
        by_kind[kd] = {"n": len(v), "median": round(float(np.median(v)), 2), "min": round(min(v), 2), "max": round(max(v), 2)}
    print(f"[5y] {z.index.min().date()}〜{z.index.max().date()} n={len(z)}  e_div 以外の54通り: |corr|>=0.8: {(ac >= 0.8).sum()} / >=0.5: {(ac >= 0.5).sum()} / 中央値 {np.median(ac):.2f}")
    print("  種類別:", by_kind)
    return {"start": str(z.index.min().date()), "end": str(z.index.max().date()), "n_days": int(len(z)),
            "hist": hist, "n_others": len(others), "n_ge_08": int((ac >= 0.8).sum()), "n_ge_05": int((ac >= 0.5).sum()),
            "median": round(float(np.median(ac)), 2), "by_kind": by_kind, "pairs": rows}


def main() -> None:
    data = {
        "generated": datetime.date.today().isoformat(),
        "params": {"train_years": bo.TRAIN_YEARS, "test_years": bo.TEST_YEARS, "percentile": bo.PERCENTILE,
                   "hysteresis_days": HYSTERESIS_DAYS, "cost_one_way": TRANSACTION_COST, "n_random": N_RANDOM,
                   "n_random_frontier": N_RANDOM_FRONTIER, "seed": bo.SEED, "step": STEP, "pcts": PCTS},
        "signals": [{"key": k, "label": l} for k, l in SIGS], "chart_keys": CHART_KEYS,
        "periods": {p: run_period(p) for p in PERIODS},
        "similarity": similarity_5y(),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (ROOT / "data" / "q2_page_data.json").write_text(payload, encoding="utf-8")
    print(f"saved: data/q2_page_data.json ({len(payload) / 1024:.0f} KB)")
    tpl = Path(__file__).parent / "q2_template.html"
    if tpl.exists():
        html = tpl.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
        (ROOT / "q2.html").write_text(html, encoding="utf-8")
        print(f"saved: q2.html ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
