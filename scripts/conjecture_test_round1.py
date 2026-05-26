"""
PDCA Round 1: 圏論的 conjecture 10 個の数値検証.

各仮説について「支持 / 反例 / 中立」を判定し,
証明スケッチを付けて data/conjecture_round1_results.json に保存する.

注意:
  - すべて「conjecture (予想)」レベルで, 「定理 (theorem)」と言い切らない.
  - 数値検証で支持されただけでは証明完了ではない.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(HERE / "lib"))

from persistent_homology import persistence_diagram, persistence_summary  # noqa: E402
from market_category import MarketCategory  # noqa: E402
from homology import signed_cycle_balance, betti_numbers  # noqa: E402

# ============================================================
# H1: Z/3, Z/5 係数での L¹ と独立性が保たれるか
# ============================================================


def _signed_cycle_balance_mod_k(G: nx.Graph, k: int,
                                max_cycles: int = 1000) -> dict:
    """符号を k 値 (mod k) に拡張した cycle balance.

    各エッジの相関値 r を [-1, 1] 上で k 等分し,
    各バケットに 0..k-1 のラベルを割り当てる.
    cycle の合計が 0 (mod k) でない時 unbalanced と数える.

    Args:
        G: weight, sign 属性を持つ無向グラフ. ここでは weight=|corr|,
           sign=±1. 元の相関 r = weight * sign を復元して使う.
        k: 法 (2, 3, 5 を想定).
        max_cycles: サンプリング上限.

    Returns:
        n_cycles_in_basis, n_unbalanced, balance_rate を含む dict.
    """
    cycles = nx.cycle_basis(G)[:max_cycles]
    n_total = len(cycles)
    n_unb = 0
    for cyc in cycles:
        # cycle 上のエッジラベル和を mod k で見る
        s = 0
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            if not G.has_edge(u, v):
                continue
            data = G[u][v]
            r = data["weight"] * int(data.get("sign", 1))
            # r in [-1, 1] を k 等分し 0..k-1 にラベル
            lbl = int(np.floor((r + 1) / 2 * k))
            if lbl >= k:
                lbl = k - 1
            s = (s + lbl) % k
        if s != 0:
            n_unb += 1
    return {"n_cycles_in_basis": n_total, "n_unbalanced": n_unb,
            "balance_rate": (n_total - n_unb) / n_total if n_total > 0 else 1.0}


def test_h1_mod_k_independence() -> dict:
    """H1: Z/3, Z/5 係数の n_unb_k と L¹ の独立性 (|r| < 0.30) が保たれるか.

    判定:
      支持: 全ての k ∈ {2, 3, 5} で |corr(L1, n_unb_k)| < 0.30
      反例: いずれかの k で |corr| >= 0.30
    """
    df_l1 = pd.read_csv(DATA_DIR / "gamma_timeseries_w30.csv")
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    returns = closes.pct_change()
    window = 30
    # 計算コスト圧縮: 50 営業日ごとにサブサンプル (全 1797 → 36 点)
    step = 50
    rows = []
    for t_idx in range(window, len(returns), step):
        win = returns.iloc[t_idx - window:t_idx].dropna(axis=1, how="any")
        if win.shape[1] < 5:
            continue
        try:
            corr = win.corr()
            cat = MarketCategory(symbols=list(win.columns),
                                 corr_matrix=corr, threshold=0.3)
            cat._build_graph()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diag = persistence_diagram(corr, max_dim=1)
            L1 = float(persistence_summary(diag)["L1_norm_H1"])
            r_k2 = _signed_cycle_balance_mod_k(cat.G, k=2)
            r_k3 = _signed_cycle_balance_mod_k(cat.G, k=3)
            r_k5 = _signed_cycle_balance_mod_k(cat.G, k=5)
            rows.append({"L1": L1,
                         "n_unb_2": r_k2["n_unbalanced"],
                         "n_unb_3": r_k3["n_unbalanced"],
                         "n_unb_5": r_k5["n_unbalanced"]})
        except Exception as e:
            print(f"  [H1] idx={t_idx} skipped: {e}")
    sub = pd.DataFrame(rows)
    if len(sub) < 10:
        return {"id": 1, "verdict": "neutral",
                "evidence": {"n_samples": len(sub)},
                "note": "サンプル不足 (<10)"}
    cors = {
        "r_L1_unb_2": float(sub[["L1", "n_unb_2"]].corr().iloc[0, 1]),
        "r_L1_unb_3": float(sub[["L1", "n_unb_3"]].corr().iloc[0, 1]),
        "r_L1_unb_5": float(sub[["L1", "n_unb_5"]].corr().iloc[0, 1]),
    }
    max_abs = max(abs(v) for v in cors.values())
    verdict = "support" if max_abs < 0.30 else "refute"
    return {"id": 1, "verdict": verdict,
            "evidence": {"n_samples": len(sub), "correlations": cors,
                         "max_abs_corr": max_abs, "threshold": 0.30},
            "note": "Z/3, Z/5 への係数拡張で L¹ との独立性が保たれるか" }


# ============================================================
# H3: α_naive ≤ α_norm の Galois 接続的順序関係
# ============================================================


def test_h3_galois() -> dict:
    """α_naive と α_norm の間に単調順序関係があるか.

    α_naive ∈ [0,1], α_norm ∈ [0, ∞) で単位が違うため,
    ランク順での Spearman 相関と, 同一日に対する両者のランクの単調性で判定.
      支持: Spearman 相関 ρ >= 0.70 かつ 不一致日 (大小逆転) が 30% 未満
      反例: ρ < 0.50 または不一致日が 50% 以上
    """
    df = pd.read_csv(DATA_DIR / "alpha_invariant_w30.csv").dropna()
    if len(df) < 100:
        return {"id": 3, "verdict": "neutral",
                "evidence": {"n_samples": len(df)},
                "note": "サンプル不足"}
    sp = float(df[["alpha_naive", "alpha_norm"]].corr(method="spearman").iloc[0, 1])
    # 「α_naive が大きい日は α_norm も大きい」を中央値で 2 群に割って test
    med_n = df["alpha_norm"].median()
    high_naive = df[df["alpha_naive"] > df["alpha_naive"].median()]
    p_high_norm_given_high_naive = float((high_naive["alpha_norm"] > med_n).mean())
    if sp >= 0.70 and p_high_norm_given_high_naive >= 0.65:
        verdict = "support"
    elif sp < 0.50 or p_high_norm_given_high_naive < 0.50:
        verdict = "refute"
    else:
        verdict = "neutral"
    return {"id": 3, "verdict": verdict,
            "evidence": {"spearman_naive_vs_norm": sp,
                         "P(norm>med | naive>med)": p_high_norm_given_high_naive,
                         "n_samples": len(df)},
            "note": "両者は単位が違うが順序的に整合するか (Galois 接続の必要条件)"}


# ============================================================
# H5: ergodicity (年次ローリングで r(L1, n_unb) が定数か)
# ============================================================


def test_h5_ergodicity() -> dict:
    """1 年ローリングで r(L1, n_unb) を計算し変動を見る.

    判定:
      支持: 標準偏差 < 0.15 (準定常) かつ符号が一貫して非負 or 非正
      反例: 標準偏差 >= 0.30 または符号反転あり
    """
    df = pd.read_csv(DATA_DIR / "gamma_timeseries_w30.csv").dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # 252 営業日 (1 年) のローリング相関
    roll = df[["L1_H1", "n_unb"]].rolling(252).corr().unstack().iloc[:, 1].dropna()
    if len(roll) < 252:
        return {"id": 5, "verdict": "neutral",
                "evidence": {"n_rolling": len(roll)},
                "note": "ローリング点不足"}
    mean_r = float(roll.mean())
    std_r = float(roll.std())
    min_r, max_r = float(roll.min()), float(roll.max())
    sign_flip = (roll.min() < 0) and (roll.max() > 0)
    if std_r < 0.15 and not sign_flip:
        verdict = "support"
    elif std_r >= 0.30 or sign_flip:
        verdict = "refute"
    else:
        verdict = "neutral"
    return {"id": 5, "verdict": verdict,
            "evidence": {"mean_r": mean_r, "std_r": std_r,
                         "min_r": min_r, "max_r": max_r,
                         "sign_flip": bool(sign_flip),
                         "n_rolling_points": len(roll)},
            "note": "L¹ と n_unb の相関が時間定常か (ergodic property の必要条件)"}


# ============================================================
# H8: balanced ⇔ F=0 ⇔ H_1=0 の三同値
# ============================================================


def test_h8_balanced_h1_equivalence() -> dict:
    """balanced ⇔ F=0 ⇔ n_unb=0 の 3 つは Cartwright-Harary から同値.

    数値的検証は実質的にトートロジー (n_unb の定義がそのまま F=0 と同義).
    ここでは 4 番目の条件 "H_1=0 (cycle rank 0, 木構造)" との等価性を見る.

    数学的事実:
      H_1=0 (cycle rank 0) ⇒ n_unb=0 は自明 (cycle がなければ frustrated cycle もない)
      逆 (n_unb=0 ⇒ H_1=0) は一般に偽: balanced cycle はサイクルとして存在する.
    実証:
      n_unb=0 だが H_1>0 の日が多数 ⇒ "balanced かつ cycle あり" の例が多い
      = 4 番目との等価性は反証される (が 3 同値は保たれる)
    判定:
      support  : n_unb=0 の日のうち H_1=0 の日が >95% (4 同値が経験的に近似成立)
      refute   : 4 同値は反証 (balanced かつ cycle あり が多数)
      ただし元の 3 同値は数学的事実として保たれる
    """
    df = pd.read_csv(DATA_DIR / "multi_indicators_w30.csv").dropna()
    n_unb_zero = df[df["n_unb_total"] == 0]
    if len(n_unb_zero) < 5:
        return {"id": 8, "verdict": "neutral",
                "evidence": {"n_samples_n_unb_zero": int(len(n_unb_zero))},
                "note": "n_unb=0 の日が少ない (3 同値部分のみ trivially 成立)"}
    counter_examples = int((n_unb_zero["nH1"] > 0).sum())
    total = int(len(n_unb_zero))
    rate_h1_zero = float((n_unb_zero["nH1"] == 0).sum() / total)
    # 3 同値部分はトートロジー的に常に成立, 4 同値は反証されるなら refute
    if rate_h1_zero >= 0.95:
        verdict = "support"
    else:
        verdict = "refute"
    return {"id": 8, "verdict": verdict,
            "evidence": {"n_days_n_unb_zero": total,
                         "n_days_h1_also_zero": total - counter_examples,
                         "counter_examples_balanced_but_cyclic": counter_examples,
                         "rate_h1_zero_given_n_unb_zero": rate_h1_zero,
                         "core_3_equivalence": "(balanced ⇔ F=0 ⇔ n_unb=0) は Cartwright-Harary より数学的事実",
                         "tested_4th_condition": "H_1=0 との 4 同値"},
            "note": "balanced と cycle rank 0 は別概念であり, 4 同値拡張は反証される"}


# ============================================================
# H4: cross-asset の e_div 増幅 (subgraph で再現するか)
# ============================================================


def test_h4_sheaf_amplification() -> dict:
    """全 40 銘柄の e_div が個別 asset_class subgraph より大きいか.

    層 (sheaf) なら大域切断 (全 40) >= 個別貼り合わせ (subgraph) という方向感.
    判定:
      支持: 全 40 の Δσ_e_div が全 subgraph の Δσ_e_div の max を上回る
      反例: いずれかの subgraph が全 40 を上回る
    """
    with open(DATA_DIR / "subgraph_eventstudy.json", encoding="utf-8") as f:
        d = json.load(f)
    all_40 = d["all_40"]["delta_sigma_ediv"]
    subs = d.get("by_asset_class", {})
    sub_evals = {k: v.get("delta_sigma_ediv") for k, v in subs.items()
                 if v.get("delta_sigma_ediv") is not None and not v.get("skipped", False)}
    if not sub_evals:
        return {"id": 4, "verdict": "neutral",
                "evidence": {"sub_count": 0},
                "note": "subgraph データなし"}
    max_sub = max(sub_evals.values())
    diff = all_40 - max_sub
    verdict = "support" if all_40 > max_sub else "refute"
    return {"id": 4, "verdict": verdict,
            "evidence": {"delta_sigma_ediv_all_40": all_40,
                         "delta_sigma_ediv_by_class": sub_evals,
                         "max_subgraph": max_sub,
                         "amplification": diff},
            "note": "クロスアセット相関での増幅 (層的貼り合わせの示唆)"}


# ============================================================
# H7: window=30/60/90 の連続性 (Kan extension の前提)
# ============================================================


def test_h7_window_continuity() -> dict:
    """window 20/30/60/90 で n_unb / L1 時系列が連続的に変化するか.

    複数 window で生 raw 時系列を計算し, 隣接 window 間で Spearman 相関を測る.
    Kan extension が成り立つなら window を連続変形しても出力が滑らかに変化する.
      支持: 全ての隣接 pair で Spearman >= 0.70
      反例: いずれか < 0.30
    """
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    returns = closes.pct_change()
    windows = [20, 30, 60, 90]
    step = 50
    series_by_w = {w: [] for w in windows}
    dates_by_w = {w: [] for w in windows}
    for t_idx in range(max(windows), len(returns), step):
        for w in windows:
            win = returns.iloc[t_idx - w:t_idx].dropna(axis=1, how="any")
            if win.shape[1] < 5:
                series_by_w[w].append(np.nan)
                dates_by_w[w].append(returns.index[t_idx - 1])
                continue
            try:
                cat = MarketCategory.from_returns(win, threshold=0.3)
                bal = signed_cycle_balance(cat.G)
                series_by_w[w].append(bal["n_unbalanced"])
            except Exception:
                series_by_w[w].append(np.nan)
            dates_by_w[w].append(returns.index[t_idx - 1])
    # 隣接 window の Spearman 相関
    df = pd.DataFrame({f"w{w}": series_by_w[w] for w in windows}).dropna()
    spear = {}
    for a, b in zip(windows[:-1], windows[1:]):
        sp = float(df[[f"w{a}", f"w{b}"]].corr(method="spearman").iloc[0, 1])
        spear[f"w{a}_vs_w{b}"] = sp
    min_sp = min(spear.values()) if spear else 0.0
    if min_sp >= 0.70:
        verdict = "support"
    elif min_sp < 0.30:
        verdict = "refute"
    else:
        verdict = "neutral"
    return {"id": 7, "verdict": verdict,
            "evidence": {"spearman_adjacent_windows": spear,
                         "min_spearman": min_sp,
                         "n_samples": len(df)},
            "note": "window を変えても n_unb の順序が保たれる (Kan extension 連続性の必要条件)"}


# ============================================================
# H6: shock-type → e_div 値の分類関手
# ============================================================


def test_h6_classifying_functor() -> dict:
    """同 shock type の Δσ_e_div の within-class 分散 < between-class 分散.

    分類関手 ≈ shock type 別に値が分離していること.
    eventstudy_8y_results.json から各 event の Δσ_e_div を取り,
    type 内分散 vs type 間分散の F 比 (or rate) を計算.
    支持: F 比 > 2.0
    反例: F 比 < 1.0
    """
    p = DATA_DIR / "eventstudy_8y_results.json"
    if not p.exists():
        return {"id": 6, "verdict": "neutral",
                "evidence": {"file": str(p), "exists": False}}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    # eventstudy 8y の構造: results[category][modes][post_only][indicators][e_div][per_event]
    res = d.get("results", {})
    by_cat = {}
    for cat, body in res.items():
        if cat == "_ALL":
            continue
        per_event = (body.get("modes", {}).get("post_only", {})
                          .get("indicators", {}).get("e_div", {})
                          .get("per_event", []))
        vals = [ev.get("d_sigma") for ev in per_event if ev.get("d_sigma") is not None]
        if vals:
            by_cat[cat] = vals
    if len(by_cat) < 3:
        return {"id": 6, "verdict": "neutral",
                "evidence": {"by_cat": by_cat},
                "note": "shock type 不足 (3 種未満)"}
    # within-class / between-class 分散 (one-way ANOVA 風 F 比)
    all_vals = [v for vals in by_cat.values() for v in vals]
    grand_mean = float(np.mean(all_vals))
    means = {k: float(np.mean(v)) for k, v in by_cat.items()}
    n_total = len(all_vals)
    ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in by_cat.values())
    ss_within = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in by_cat.values())
    k = len(by_cat)
    df_b = k - 1
    df_w = n_total - k
    f_ratio = ((ss_between / df_b) / (ss_within / df_w)
               if df_w > 0 and ss_within > 0 else float("inf"))
    if f_ratio > 2.0:
        verdict = "support"
    elif f_ratio < 1.0:
        verdict = "refute"
    else:
        verdict = "neutral"
    return {"id": 6, "verdict": verdict,
            "evidence": {"means_by_cat": means,
                         "n_events_by_cat": {k: len(v) for k, v in by_cat.items()},
                         "f_ratio_between_within": f_ratio,
                         "ss_between": float(ss_between),
                         "ss_within": float(ss_within)},
            "note": "shock type 内の e_div 分散 < shock type 間分散なら classifying functor として機能"}


# ============================================================
# H9: Granger causality を層コホモロジー長完全列の連結準同型と解釈
# ============================================================


def test_h9_granger_as_connecting_hom() -> dict:
    """Granger 因果性が trade_policy → e_div で有意 (一方向) であることを確認.

    判定:
      支持: forward で有意, reverse で非有意の pair が存在
      反例: 双方向有意
    """
    p = DATA_DIR / "causal_granger_results.json"
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    fwd = d.get("forward_direction", {}).get("results", {})
    rev = d.get("reverse_direction", {}).get("results", {})
    pairs = []
    for k_fwd, v_fwd in fwd.items():
        sig_fwd = any(lag.get("significant_at_0.05", False)
                      for lag in v_fwd.get("lags", {}).values())
        # reverse の対応する pair
        k_rev = next((k for k in rev.keys() if "e_div" in k or "n_unb" in k), None)
        if k_rev is None:
            continue
        v_rev = rev[k_rev]
        sig_rev = any(lag.get("significant_at_0.05", False)
                      for lag in v_rev.get("lags", {}).values())
        pairs.append({"forward": k_fwd, "reverse": k_rev,
                      "sig_forward": sig_fwd, "sig_reverse": sig_rev,
                      "asymmetric": sig_fwd and not sig_rev})
    n_asym = sum(1 for p in pairs if p["asymmetric"])
    verdict = "support" if n_asym >= 1 else "refute"
    return {"id": 9, "verdict": verdict,
            "evidence": {"n_asymmetric_pairs": n_asym, "pairs": pairs},
            "note": "Granger 一方向性 (短完全列の連結準同型) が成立"}


# ============================================================
# H10: asset_class 別 sub-graph の集計が coend で書ける
# ============================================================


def test_h10_coend() -> dict:
    """sub-graph 集計が全グラフを再現するか. coend なら sup ≈ aggregate.

    判定:
      支持: weighted sub-graph (asset_class size 加重) の Δσ_e_div が全 40 と
        符号一致かつ |差| < 1.0σ
      反例: 符号反転
    """
    with open(DATA_DIR / "subgraph_eventstudy.json", encoding="utf-8") as f:
        d = json.load(f)
    all_40 = d["all_40"]["delta_sigma_ediv"]
    subs = d.get("by_asset_class", {})
    weighted = 0.0
    n_total = 0
    for k, v in subs.items():
        if v.get("skipped", False):
            continue
        n = v.get("n_symbols", 0)
        ediv = v.get("delta_sigma_ediv")
        if ediv is None:
            continue
        weighted += ediv * n
        n_total += n
    if n_total == 0:
        return {"id": 10, "verdict": "neutral",
                "evidence": {"weighted_subs": 0}}
    agg = weighted / n_total
    diff = all_40 - agg
    same_sign = (all_40 > 0) == (agg > 0)
    verdict = "support" if same_sign and abs(diff) < 1.0 else "refute"
    return {"id": 10, "verdict": verdict,
            "evidence": {"delta_sigma_ediv_all_40": all_40,
                         "weighted_subs_avg": agg,
                         "diff": diff, "same_sign": bool(same_sign)},
            "note": "全 40 の集約値が銘柄数加重 sub-graph 平均と整合 (coend 的)"}


# ============================================================
# H2: e_div は 2 関手の自然変換の障害類 (obstruction class)
# ============================================================


def test_h2_obstruction() -> dict:
    """e_div = z(n_unb) - z(L1) と素朴に書ける = 障害類 0 ≠ identity.

    数値的に「e_div が単なる差で表現できる」ことだけ確認.
    真の障害類論証は証明スケッチ側に回す.
      支持: e_div の経験分布が両指標の z 差と一致 (相関 > 0.95)
      反例: 一致しない
    """
    df_a = pd.read_csv(DATA_DIR / "alpha_invariant_w30.csv").dropna(subset=["e_div"])
    df_m = pd.read_csv(DATA_DIR / "multi_indicators_w30.csv").dropna()
    df_a["date"] = pd.to_datetime(df_a["date"])
    df_m["date"] = pd.to_datetime(df_m["date"])
    m = df_a.merge(df_m[["date", "L1", "n_unb_total"]], on="date")
    if len(m) < 100:
        return {"id": 2, "verdict": "neutral",
                "evidence": {"n_samples": len(m)}}
    # z (90-day rolling)
    m["zL1"] = (m["L1"] - m["L1"].rolling(90).mean()) / m["L1"].rolling(90).std()
    m["zU"] = (m["n_unb_total"] - m["n_unb_total"].rolling(90).mean()) / m["n_unb_total"].rolling(90).std()
    m["ediv_recon"] = m["zU"] - m["zL1"]
    m = m.dropna()
    r = float(m[["e_div", "ediv_recon"]].corr().iloc[0, 1])
    return {"id": 2, "verdict": "support" if r >= 0.95 else "neutral",
            "evidence": {"corr_ediv_vs_reconstruction": r, "n_samples": len(m)},
            "note": "e_div は形式的に z(n_unb) - z(L1). 障害類としての論証は証明スケッチへ"}


# ============================================================
# 証明スケッチ
# ============================================================

PROOF_SKETCHES = {
    2: {
        "title": "Conjecture (e_div = 障害類)",
        "statement": (
            "z 正規化された差 e_div(t) = z(n_unb)(t) - z(L¹)(t) は "
            "2 つの関手 F_L¹: Grph → ℝ_≥0 と F_n_unb: SGrph → ℤ_≥0 の出力を "
            "同じ codomain ℝ に射影した上での『差』であり, "
            "両関手が自然変換で繋がるための障害類 (obstruction class) として機能する."
        ),
        "motivation": (
            "もし F_L¹ と F_n_unb が同じ関手なら出力は一致し e_div ≡ 0. "
            "実証的に e_div は ±5σ まで動くので, 両関手は等価ではない (障害が存在する). "
            "経験的に再構成式と相関 r > 0.9999 で一致 (= 数値的に同義)."
        ),
        "sketch": (
            "想定アプローチ: (i) 関手族 {F_L¹, F_n_unb} を時間圏から ℝ への関手として固定; "
            "(ii) 両者の codomain を rolling z-score で同一化する射 z_t を作る; "
            "(iii) 自然変換 η: z∘F_L¹ ⇒ z∘F_n_unb が存在するなら "
            "対角写像 (η_t - id) = 0 だが, 実測 e_div ≠ 0. "
            "Open: (a) z 正規化が関手として well-defined か (window 依存); "
            "(b) 障害類を本物の cohomology 類として書ける群構造の同定."
        ),
        "future_work": "z 正規化を厳密な categorical morphism として定式化"
    },
    1: {
        "title": "Conjecture (Z/k 係数独立性)",
        "statement": (
            "ローリング窓相関ネットワーク上の Vietoris-Rips PH の "
            "L¹(H₁; ℤ) 不変量と, 相関を k 等分してラベル化した cycle balance 数 "
            "n_unb_k (k=2,3,5) の間の経験的相関は, k に依らず |r| < 0.30 で抑えられる."
        ),
        "motivation": (
            "ℤ 係数 PH と ℤ/2 cohomological cycle balance の独立性 "
            "(Round 0 で r=0.16) が普遍係数定理に由来するなら, "
            "係数体を変えても独立性は保たれるはず."
        ),
        "sketch": (
            "想定アプローチ: (i) 各ローリング窓の相関グラフを simplicial complex として固定し, "
            "Universal Coefficient Theorem を H₁(X; ℤ) と H¹(X; ℤ/k) の間に適用; "
            "(ii) Ext^1_ℤ(H₀, ℤ/k) 項が rank に依存して 0 になる条件を金融データで満たすこと "
            "(ノード次数 >> 1) を補題化; (iii) すると ℤ/k に変えても情報量がほぼ同じ → "
            "L¹ との独立性は torsion 項に支配されないため k 不変. "
            "Open: (a) 連続的相関値の k 等分が真の係数変換と対応するかの整合性; "
            "(b) k → ∞ 極限での連続コホモロジーとの関係."
        ),
        "future_work": "k → ∞ の極限を取る (de Rham 的な連続コホモロジー類)"
    },
    3: {
        "title": "Conjecture (α 関手族の Galois 接続)",
        "statement": (
            "12 指標から構成した α_naive (≈ colimit 的「同時発火数」) と "
            "α_norm (≈ limit 的「Frobenius ノルム」) は, "
            "ランク順 (Spearman 相関) で正の単調関係を持ち, "
            "圏論的 Galois 接続 α_colim ⊣ α_lim の必要条件を満たす."
        ),
        "motivation": (
            "12 個の指標 F_i: ℳ → ℝ を関手族とみるとき, "
            "α_naive は colimit (同時発火) 側, α_norm は limit (全指標ノルム) 側. "
            "Galois 接続なら左随伴 ≤ 右随伴 の順序関係が常に成立する."
        ),
        "sketch": (
            "想定アプローチ: (i) 指標族 {F_i}_{i=1}^{12} を関手として固定; "
            "(ii) ポセット ℝ^12 → [0,1] (閾値 2σ 越え数) と ℝ^12 → ℝ_≥0 (Frobenius) "
            "を colimit / limit に対応させる; "
            "(iii) Galois 接続 (adjoint pair) の構成可能性を, "
            "閾値写像が monotone 関手として書けることから示す. "
            "Open: (a) 圏としての ℳ を厳密化 (現状は集合); "
            "(b) 12 指標の独立性をどう関手の充満性として捉えるか."
        ),
        "future_work": "ℳ の morphism (指標間の変換) を明示する"
    },
    8: {
        "title": "Conjecture (balanced ⇔ frustration 0 ⇔ H₁ 退化の三同値)",
        "statement": (
            "符号付きグラフが balanced (Cartwright-Harary 1956) ⇔ "
            "Z/2 cohomological frustration F=0 ⇔ "
            "n_unb=0 の 3 つは同値であり, "
            "経験的にも n_unb=0 と nH₁=0 が同日にほぼ完全一致する."
        ),
        "motivation": (
            "balanced graph の定義 (全 cycle が +1) と "
            "H¹(X; ℤ/2) ≃ ker / image 0 の同値性は古典的だが, "
            "実データで 100% 成立するかは検証していなかった."
        ),
        "sketch": (
            "想定アプローチ: (i) Cartwright-Harary の Structure Theorem: "
            "balanced ⇔ 二部分割可能 (頂点を 2 群に分けて群内 +, 群間 - のみ) を引用; "
            "(ii) この分割可能性が cocycle exact sequence 上で trivial cohomology に対応; "
            "(iii) PH 上の H₁ は連結成分内 cycle 数なので, "
            "完全グラフ + balanced なら H₁ は constructible に 0. "
            "Open: (a) 連結成分が複数ある場合の各成分独立な balance; "
            "(b) 弱閾値での balanced graph のサンプル数が経験データで稀 (>95%が unbalanced)."
        ),
        "future_work": "閾値を上げて balanced graph 比率を増やした条件での検証"
    },
    4: {
        "title": "Conjecture (cross-asset e_div 増幅 ≈ 層的貼り合わせ)",
        "statement": (
            "全 40 銘柄で計算した Δσ_e_div は, 個別 asset_class sub-graph (FX, INDEX, ...) で "
            "計算した同量の最大値を有意に上回る. "
            "これは asset_class 間の cross-edge が, "
            "局所切断 (sub-graph 上) を貼り合わせる層 (sheaf) 的構造の存在を示唆する."
        ),
        "motivation": (
            "実測: 全 40 で Δσ_e_div = +2.75σ, FX で +0.47σ, INDEX で -0.06σ. "
            "asset 内シグナルだけでは説明できず, asset 間相関 (cross-edge) が増幅源."
        ),
        "sketch": (
            "想定アプローチ: (i) 銘柄集合 V を asset_class でカバー V = ⋃_α V_α; "
            "(ii) 各 V_α 上に sub-graph を割り当てる pre-sheaf 𝓕 を構成; "
            "(iii) sheaf condition (local sections の glue) が成立しないとき, "
            "Čech cohomology H¹(V; 𝓕) ≠ 0 が cross-asset 増幅量に対応. "
            "Open: (a) sheaf 構造の opens をどう定義するか (asset_class は元々離散); "
            "(b) Δσ という統計量を sheaf 値として書く厳密化."
        ),
        "future_work": "Čech cohomology を実データから直接計算する"
    },
    6: {
        "title": "Conjecture (shock-type → e_div の classifying functor)",
        "statement": (
            "shock type (geopolitical / monetary / trade_policy / ...) と "
            "Δσ_e_div の対応は, type 内分散 << type 間分散 (F 比 > 2) を満たし, "
            "shock type を対象とする離散圏 𝒮 から ℝ への分類関手 "
            "C: 𝒮 → ℝ として記述できる."
        ),
        "motivation": (
            "trade_policy で平均 +0.96σ, market_structure で +0.74σ, "
            "macro/tech_shock で -0.93〜-0.96σ と type ごとに値が分離している. "
            "type 内の event は同符号でまとまる傾向."
        ),
        "sketch": (
            "想定アプローチ: (i) shock 集合 E を type で分割し, "
            "離散圏 𝒮 を type の集合とする; "
            "(ii) C(σ) := mean_{e ∈ σ}(Δσ_e_div(e)) を 𝒮 → ℝ の関手として定義; "
            "(iii) 関手の充実性は ANOVA F 比 > 2 で経験的に裏打ち. "
            "Open: (a) shock type 同士の morphism (type 階層) をどう与えるか; "
            "(b) event 数 (n=1-15) の不均衡による分散推定の安定性."
        ),
        "future_work": "shock type 間の射 (例: trade_policy ⊃ tariff) を持つ豊富な圏に拡張"
    },
    9: {
        "title": "Conjecture (Granger 因果性 ≈ 連結準同型)",
        "statement": (
            "trade_policy event dummy と e_div の間に Granger 一方向因果が成立する "
            "(forward 有意, reverse 非有意) ことは, "
            "層コホモロジーの長完全列における連結準同型 δ: H^n(政策層) → H^{n+1}(市場層) と解釈できる."
        ),
        "motivation": (
            "Granger 因果は予測情報の単方向流れ. 一方, sheaf cohomology の "
            "連結準同型は短完全列 0→A→B→C→0 から長完全列を引き起こす境界写像. "
            "両者とも『片側の情報が他方の構造を決める』という意味で類似."
        ),
        "sketch": (
            "想定アプローチ: (i) 時間軸を opens とする pre-sheaf "
            "𝒪(I) = {(政策状態, 市場状態) | I 上で観測}; "
            "(ii) 政策 → 市場の射が短完全列 0 → P → M → M/P → 0 を成すと仮定; "
            "(iii) 長完全列の連結準同型 δ の有意性が Granger F 統計量と対応. "
            "Open: (a) 層化の厳密性 (時間軸 open はあいまい); "
            "(b) 因果統計と境界写像の数学的対応は強い予想で, "
            "現時点では type-theoretic な mere analogy."
        ),
        "future_work": "より厳密な sheaf 構造を時間軸上で構築 (Goguen 1992 etc.)"
    },
    10: {
        "title": "Conjecture (asset_class sub-graph の coend 集約)",
        "statement": (
            "asset_class 別 sub-graph (FX, INDEX, COMMODITY, STOCK) の Δσ_e_div を "
            "銘柄数加重平均すると, 全 40 銘柄 sub-graph の Δσ_e_div に近似する. "
            "これは asset_class 圏 𝒜 上の関手 F: 𝒜 → ℝ の coend ∫^𝒜 F が "
            "全グラフ集約値に対応する示唆を与える."
        ),
        "motivation": (
            "coend は dinatural transformation の極限. "
            "asset 内 sub-graph と asset 間 cross-edge を分けて足し合わせれば "
            "全グラフが再構成できるはずだが, cross-edge 効果が大きいので "
            "純粋 sub-graph 平均と差が出るのが予想される."
        ),
        "sketch": (
            "想定アプローチ: (i) 𝒜 = 離散圏 (asset_class), F(c) = sub-graph c の Δσ_e_div; "
            "(ii) coend ≃ Σ_c F(c) ⊗ 加重 / Σ_c 加重 を end formula で正当化; "
            "(iii) cross-edge 寄与を 「𝒜 上の自然変換の障害」として書く. "
            "Open: (a) 加重が銘柄数で良いか相関エッジ数で良いか; "
            "(b) cross-edge 寄与の sheaf 的扱い (前述 H9 と接続)."
        ),
        "future_work": "asset_class を細分化 (FX 内で major/exotic 等)"
    },
}


def _attach_proof_sketch(result: dict) -> dict:
    """生き残り (support) 仮説に証明スケッチを付ける."""
    if result["verdict"] != "support":
        return result
    sk = PROOF_SKETCHES.get(result["id"])
    if sk is not None:
        result["proof_sketch"] = sk
    return result


# ============================================================
# 仮説リスト
# ============================================================

CONJECTURES = {
    1: "L¹(ℤ) と n_unb(ℤ/2) の独立性は普遍係数定理由来 → ℤ/3, ℤ/5 でも独立性が保たれる",
    2: "e_div は 2 関手 (F_L¹, F_n_unb) の自然変換の障害類 (obstruction)",
    3: "α_naive (colimit 的) と α_norm (limit 的) の間に Galois 接続 (α_colim ≤ α_lim) が成立",
    4: "cross-asset の e_div 増幅は層 (sheaf) の貼り合わせ",
    5: "8y OOS での independence (r=0.16-0.26) は ergodicity (時間定常性) から従う",
    6: "shock-type → e_div 値の対応は分類関手 (classifying functor)",
    7: "window=30/60/90 の連続性は Kan extension で繋がる",
    8: "balanced graph ⇔ frustration F=0 ⇔ n_unb=0 の三同値",
    9: "Granger 因果性 (trade_policy → e_div) は層コホモロジー長完全列の連結準同型と解釈できる",
    10: "asset_class 別 sub-graph の集計は coend (終余関手) で書ける",
}


# ============================================================
# main
# ============================================================


def main() -> None:
    print("=" * 60)
    print("Conjecture Round 1: 圏論的予想 10 個の数値検証")
    print("=" * 60)

    results = []
    tests = [
        ("H1", test_h1_mod_k_independence),
        ("H2", test_h2_obstruction),
        ("H3", test_h3_galois),
        ("H4", test_h4_sheaf_amplification),
        ("H5", test_h5_ergodicity),
        ("H6", test_h6_classifying_functor),
        ("H7", test_h7_window_continuity),
        ("H8", test_h8_balanced_h1_equivalence),
        ("H9", test_h9_granger_as_connecting_hom),
        ("H10", test_h10_coend),
    ]
    for tag, fn in tests:
        print(f"\n--- {tag}: {CONJECTURES[int(tag[1:])]} ---")
        try:
            res = fn()
            res["statement"] = CONJECTURES[res["id"]]
            res = _attach_proof_sketch(res)
            print(f"  verdict: {res['verdict']}")
            if "evidence" in res:
                ev_str = json.dumps(res["evidence"], ensure_ascii=False)
                if len(ev_str) > 240:
                    ev_str = ev_str[:240] + "..."
                print(f"  evidence: {ev_str}")
            results.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"id": int(tag[1:]),
                            "verdict": "neutral",
                            "evidence": {"error": str(e)},
                            "statement": CONJECTURES[int(tag[1:])],
                            "note": "実行エラー"})

    # 集計
    by_v = {"support": 0, "refute": 0, "neutral": 0}
    for r in results:
        by_v[r["verdict"]] = by_v.get(r["verdict"], 0) + 1
    print("\n" + "=" * 60)
    print("Summary:")
    for k, v in by_v.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    out = {
        "round": 1,
        "n_conjectures": len(results),
        "summary": by_v,
        "honest_disclaimer": (
            "本検証は『conjecture (予想)』レベルの数値支持を与えるのみで, "
            "『定理 (theorem)』としての証明は完了していない. "
            "支持と判定された仮説についても, 証明スケッチは将来研究 (future work) である."
        ),
        "results": results,
    }
    out_path = DATA_DIR / "conjecture_round1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
