"""
DAG 構造推定による因果推論強化 (PC algorithm)
================================================================

目的:
  Granger 因果性検定 (scripts/causal_granger.py / commit ac1dffe) は
    「trade_policy → e_div」p=0.022 (lag=5) で一方向性まで実証した。
  しかし Granger は本質的に「予測情報の時間的非対称性」レベルで,
  隠れた共通要因 (confounder) を排除できない弱点が残る。

  本スクリプトは PC algorithm (Peter & Clark, 1991) で
  ショック変数群 + γ-指標群 + VIX の混合変数集合から
  DAG (Directed Acyclic Graph) 構造を推定し,
  Granger の方向性が confounder で説明されないことを確認する。

設定:
  - 変数 (列):
      shock_trade   : trade_policy_dummy ±2 営業日 (events_8y.json)
      shock_struct  : market_structure_dummy ±2 営業日
      shock_war     : war_dummy ±2 営業日 (もし n>=3 なら含める)
      e_div         : z(n_unb) - z(L1_H1)         (expanding z-score)
      L1            : L1_H1 raw
      n_unb         : unbalanced cycle count
      balance_rate  : 1 - n_unb / n_edges
      VIX           : ohlc_40.parquet から (内生性は Section 11.1 VIX exclusion で確認済)
  - PC algorithm:
      条件付き独立性検定: Fisher Z (デフォルト)
      有意水準 alpha = 0.05
      stable mode (順序非依存)
  - 出力 DAG は CPDAG (有向 + 無向 mix). 有向辺のみを「因果」として解釈し,
    無向辺は「方向特定不能 (markov equivalence class)」として別途報告する。

期待される DAG:
  - shock_trade → e_div                      (Granger 一致なら強化)
  - shock_struct → L1 (or n_unb)             (5y/8y OOS と一致)
  - e_div → shock_trade はないはず           (Granger 逆方向 p=0.254)
  - VIX は e_div / L1 と相関するが
       shock → VIX → e_div の経由か,
       VIX が共通親 (VIX → shock, VIX → e_div) か,
    のどちらかが見えるかを観察する。

出力:
  data/causal_dag_results.json   推定 DAG (辺リスト + p 値 + 隣接行列)
  data/fig_causal_dag.png        networkx で可視化
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from pathlib import Path

# Windows cp932 で ↔ など特殊文字を print できるよう UTF-8 に強制
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
INPUT_CSV = DATA_DIR / "gamma_timeseries_w30.csv"
EVENTS_JSON = DATA_DIR / "events_8y.json"
OHLC_PARQUET = DATA_DIR / "ohlc_40.parquet"
OUTPUT_JSON = DATA_DIR / "causal_dag_results.json"
OUTPUT_FIG = DATA_DIR / "fig_causal_dag.png"

# ===== パラメータ =====
EVENT_HALF_WIDTH_BDAYS = 2
ALPHA = 0.05
INDEP_TEST = "fisherz"  # causal-learn の文字列キー
RNG_SEED = 20260528


def expanding_zscore(s: pd.Series, min_periods: int = 60) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=1)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def build_event_dummy(idx: pd.DatetimeIndex,
                      event_dates: list[pd.Timestamp],
                      half_width_bdays: int = EVENT_HALF_WIDTH_BDAYS
                      ) -> pd.Series:
    dummy = pd.Series(0, index=idx, dtype=float)
    for d in event_dates:
        pos = idx.get_indexer([d], method="nearest")[0]
        if pos < 0:
            continue
        lo = max(0, pos - half_width_bdays)
        hi = min(len(idx), pos + half_width_bdays + 1)
        dummy.iloc[lo:hi] = 1.0
    return dummy


def jitter_constant_cols(df: pd.DataFrame, eps: float = 1e-9,
                         rng_seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Fisher Z は連続変数前提なので, バイナリ shock dummy にごく小さい
    Gaussian noise を足して分散を確保する (情報量を実質変えない).
    """
    rng = np.random.default_rng(rng_seed)
    out = df.copy()
    for col in out.columns:
        if out[col].nunique() <= 2:
            out[col] = out[col].astype(float) + rng.normal(0, eps, size=len(out))
    return out


def cpdag_to_edge_list(cg, node_names: list[str]) -> list[dict]:
    """
    causal-learn の CausalGraph (PC 出力) から辺リストを抽出する.

    causal-learn の隣接行列 cg.G.graph[i, j] の意味:
        cg.G.graph[i, j] = -1, cg.G.graph[j, i] = 1   →  i → j   (有向)
        cg.G.graph[i, j] = -1, cg.G.graph[j, i] = -1  →  i — j   (無向 / 方向不能)
        cg.G.graph[i, j] = 1,  cg.G.graph[j, i] = 1   →  i ↔ j   (bidirected / 潜在共通親)
        cg.G.graph[i, j] = 0                          →  辺なし
    """
    adj = cg.G.graph
    n = len(node_names)
    edges: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = int(adj[i, j])
            b = int(adj[j, i])
            if a == 0 and b == 0:
                continue
            key = tuple(sorted([i, j]))
            if key in seen:
                continue
            seen.add(key)
            # 有向: i → j  (i 側 -1, j 側 1)
            if a == -1 and b == 1:
                edges.append({"src": node_names[i], "dst": node_names[j],
                              "type": "directed"})
            elif a == 1 and b == -1:
                edges.append({"src": node_names[j], "dst": node_names[i],
                              "type": "directed"})
            elif a == -1 and b == -1:
                edges.append({"src": node_names[i], "dst": node_names[j],
                              "type": "undirected"})
            elif a == 1 and b == 1:
                edges.append({"src": node_names[i], "dst": node_names[j],
                              "type": "bidirected"})
            else:
                edges.append({"src": node_names[i], "dst": node_names[j],
                              "type": f"unknown({a},{b})"})
    return edges


def draw_dag(edges: list[dict], node_names: list[str], out_path: Path) -> None:
    """networkx で DAG を描画."""
    G = nx.DiGraph()
    for n in node_names:
        G.add_node(n)

    directed_edges: list[tuple[str, str]] = []
    undirected_edges: list[tuple[str, str]] = []
    bidirected_edges: list[tuple[str, str]] = []
    for e in edges:
        s, d, t = e["src"], e["dst"], e["type"]
        if t == "directed":
            directed_edges.append((s, d))
        elif t == "undirected":
            undirected_edges.append((s, d))
        elif t == "bidirected":
            bidirected_edges.append((s, d))

    for s, d in directed_edges:
        G.add_edge(s, d, kind="directed")
    for s, d in undirected_edges:
        G.add_edge(s, d, kind="undirected")
        G.add_edge(d, s, kind="undirected")
    for s, d in bidirected_edges:
        G.add_edge(s, d, kind="bidirected")
        G.add_edge(d, s, kind="bidirected")

    # ノードを論理的にグループ化して配置 (左→右の causal flow)
    pos = {}
    # ショック層 (左)
    shock_nodes = [n for n in node_names if n.startswith("shock_")]
    for i, n in enumerate(shock_nodes):
        pos[n] = (-2.4, 0.8 - i * 1.2)
    # 中間層 (中央)
    if "VIX" in node_names:
        pos["VIX"] = (0.0, 0.4)
    # γ-指標層 (右)
    gamma_nodes = ["e_div", "L1", "n_unb", "balance_rate"]
    placed = [n for n in gamma_nodes if n in node_names]
    for i, n in enumerate(placed):
        pos[n] = (2.4, 1.6 - i * 1.0)
    # 漏れがあれば自動配置
    missing = [n for n in node_names if n not in pos]
    for i, n in enumerate(missing):
        pos[n] = (0.0, -2.0 - i * 0.6)

    fig, ax = plt.subplots(figsize=(12, 8))

    # ノード色分け
    color_map = []
    for n in node_names:
        if n.startswith("shock_"):
            color_map.append("#E83929")   # 赤 (ショック)
        elif n == "VIX":
            color_map.append("#E6B422")   # 橙 (内生検証済)
        elif n == "e_div":
            color_map.append("#5B8930")   # 緑 (主要発見)
        else:
            color_map.append("#2B3C5E")   # 藍 (その他)

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=4800, ax=ax,
                           edgecolors="black", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="white",
                            font_weight="bold", ax=ax)

    # 有向辺 (矢印を見やすく)
    nx.draw_networkx_edges(G, pos, edgelist=directed_edges,
                           edge_color="#2B3C5E", width=2.4,
                           arrows=True, arrowsize=28, arrowstyle="-|>",
                           node_size=4800,
                           min_source_margin=18, min_target_margin=22,
                           connectionstyle="arc3,rad=0.10", ax=ax)
    # 無向辺
    undir_for_draw = [(s, d) for s, d in undirected_edges]
    nx.draw_networkx_edges(G, pos, edgelist=undir_for_draw,
                           edge_color="#888888", width=1.6,
                           arrows=False, style="dashed",
                           node_size=4800, ax=ax)
    # 双方向辺 (潜在共通親)
    nx.draw_networkx_edges(G, pos, edgelist=bidirected_edges,
                           edge_color="#C53D43", width=2.0,
                           arrows=True, arrowsize=24, arrowstyle="<|-|>",
                           node_size=4800,
                           connectionstyle="arc3,rad=0.20", ax=ax)

    ax.set_title("PC algorithm 推定 DAG  (alpha=0.05, Fisher Z, n=1736)\n"
                 "実線 = 有向因果 (→),  破線 = 方向不能 (—, Markov 等価類),  "
                 "赤双方向 = 潜在共通親 (↔)",
                 fontsize=12)
    ax.axis("off")
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.0, 2.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=== DAG 因果推論 (PC algorithm) ===")

    # ----- 1. γ-指標ロード -----
    df_g = (pd.read_csv(INPUT_CSV, parse_dates=["date"])
              .sort_values("date").set_index("date"))
    df_g = df_g.dropna(subset=["L1_H1", "n_unb"]).copy()
    print(f"gamma timeseries: {df_g.shape}, "
          f"range {df_g.index.min().date()} -> {df_g.index.max().date()}")

    z_L1 = expanding_zscore(df_g["L1_H1"])
    z_unb = expanding_zscore(df_g["n_unb"])
    e_div = (z_unb - z_L1).rename("e_div")
    L1 = df_g["L1_H1"].rename("L1")
    n_unb = df_g["n_unb"].rename("n_unb")
    balance_rate = df_g["balance_rate"].rename("balance_rate")

    # ----- 2. event dummy 構築 -----
    events = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    cats = {}
    for e in events:
        cats.setdefault(e["category"], []).append(pd.Timestamp(e["date"]))
    print("event categories:",
          {k: len(v) for k, v in cats.items()})

    common_idx = e_div.dropna().index
    shock_trade = build_event_dummy(common_idx,
                                    cats.get("trade_policy", [])).rename("shock_trade")
    shock_struct = build_event_dummy(common_idx,
                                     cats.get("market_structure", [])).rename("shock_struct")
    # war は n>=3 なら含める
    war_dates = cats.get("war", [])
    use_war = len(war_dates) >= 3
    if use_war:
        shock_war = build_event_dummy(common_idx, war_dates).rename("shock_war")
    else:
        print(f"  war n={len(war_dates)} < 3 -> 列に含めない")

    # ----- 3. VIX を ohlc から ロード -----
    ohlc = pd.read_parquet(OHLC_PARQUET)
    vix_close = ohlc["VIX"].rename("VIX")
    vix_close.index = pd.to_datetime(vix_close.index)
    vix_aligned = vix_close.reindex(common_idx).ffill()
    print(f"VIX coverage on common_idx: "
          f"{vix_aligned.notna().sum()}/{len(common_idx)}")

    # ----- 4. 変数 DataFrame 組み立て -----
    cols_to_use = [shock_trade, shock_struct]
    if use_war:
        cols_to_use.append(shock_war)
    cols_to_use += [e_div.reindex(common_idx),
                    L1.reindex(common_idx),
                    n_unb.reindex(common_idx),
                    balance_rate.reindex(common_idx),
                    vix_aligned]
    X = pd.concat(cols_to_use, axis=1).dropna()
    node_names = list(X.columns)
    print(f"final variable matrix: shape={X.shape}, vars={node_names}")

    # binary dummy にごく小さい noise を入れる (Fisher Z 用)
    X_num = jitter_constant_cols(X)

    # ----- 5. PC algorithm 実行 -----
    print(f"\n[PC] alpha={ALPHA}, indep_test={INDEP_TEST}, n={len(X_num)}")
    from causallearn.search.ConstraintBased.PC import pc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = pc(X_num.values, alpha=ALPHA, indep_test=INDEP_TEST,
                stable=True, show_progress=False,
                node_names=node_names)

    # ----- 6. 辺リスト抽出 -----
    edges = cpdag_to_edge_list(cg, node_names)
    print(f"\n推定された辺 (n={len(edges)}):")
    for e in edges:
        arrow = {"directed": "→",
                 "undirected": "—",
                 "bidirected": "↔"}.get(e["type"], "?")
        print(f"  {e['src']:<14} {arrow}  {e['dst']:<14}  [{e['type']}]")

    # ----- 7. 主要解釈 -----
    def has_edge(src: str, dst: str, kind: str = "directed") -> bool:
        return any(e["src"] == src and e["dst"] == dst and e["type"] == kind
                   for e in edges)

    def has_any_edge(a: str, b: str) -> bool:
        return any({e["src"], e["dst"]} == {a, b} for e in edges)

    findings: list[str] = []
    # (i) trade → e_div の有向辺
    if has_edge("shock_trade", "e_div"):
        findings.append("shock_trade → e_div の有向辺を DAG が検出 "
                        "(Granger 結果 p=0.022 と一致, 因果方向性を補強)")
    elif has_any_edge("shock_trade", "e_div"):
        findings.append("shock_trade — e_div の辺は検出されたが方向特定不能 "
                        "(Markov 等価類, Granger より弱い結論)")
    else:
        findings.append("shock_trade ↔ e_div の辺が DAG では消失 "
                        "(Granger 結果と非整合 -- 隠れた共通要因の可能性)")

    # (ii) 逆方向 e_div → shock_trade
    if has_edge("e_div", "shock_trade"):
        findings.append("e_div → shock_trade の有向辺あり (政策事前リーク示唆)")
    else:
        findings.append("e_div → shock_trade の有向辺なし "
                        "(Granger 逆方向 p=0.254 と一致, 一方向性 OK)")

    # (iii) structure → L1
    if has_edge("shock_struct", "L1") or has_edge("shock_struct", "n_unb"):
        findings.append("shock_struct → L1 / n_unb の有向辺を検出 "
                        "(5y/8y OOS の structure 効果と一致)")

    # (iv) 共通親候補
    confounders: list[str] = []
    for e in edges:
        if e["type"] == "bidirected":
            confounders.append(f"{e['src']} ↔ {e['dst']} (潜在共通親の可能性)")
    if confounders:
        findings.append("潜在 confounder 候補: " + " / ".join(confounders))
    else:
        findings.append("双方向辺 (潜在共通親) は検出されず")

    # (v) VIX の位置
    vix_in_edges = [e for e in edges
                    if e["src"] == "VIX" or e["dst"] == "VIX"]
    if vix_in_edges:
        findings.append("VIX に接続する辺: " + ", ".join(
            f"{e['src']}→{e['dst']}" if e["type"] == "directed"
            else f"{e['src']}-{e['dst']}({e['type']})"
            for e in vix_in_edges))

    print("\n[主要解釈]")
    for f in findings:
        print(f"  - {f}")

    # ----- 8. 図を出力 -----
    draw_dag(edges, node_names, OUTPUT_FIG)
    print(f"\nsaved figure: {OUTPUT_FIG}")

    # ----- 9. JSON 出力 -----
    adj_matrix = cg.G.graph.tolist()
    output = {
        "meta": {
            "input_csv": str(INPUT_CSV.name),
            "events_json": str(EVENTS_JSON.name),
            "ohlc_parquet": str(OHLC_PARQUET.name),
            "n_obs": int(len(X_num)),
            "date_range": [str(X_num.index.min().date()),
                           str(X_num.index.max().date())],
            "alpha": ALPHA,
            "indep_test": INDEP_TEST,
            "stable": True,
            "event_half_width_bdays": EVENT_HALF_WIDTH_BDAYS,
            "war_included": use_war,
        },
        "node_names": node_names,
        "adjacency_matrix": adj_matrix,
        "adjacency_legend": {
            "(i,j)=-1, (j,i)=1": "i → j (directed)",
            "(i,j)=-1, (j,i)=-1": "i — j (undirected)",
            "(i,j)=1, (j,i)=1": "i ↔ j (bidirected / latent common cause)",
            "(i,j)=0": "no edge",
        },
        "edges": edges,
        "findings": findings,
        "comparison_with_granger": {
            "granger_main_pair": "trade_policy → e_div",
            "granger_p": 0.022,
            "granger_p_reverse": 0.254,
            "dag_has_trade_to_ediv_directed": has_edge("shock_trade", "e_div"),
            "dag_has_trade_to_ediv_undirected": (
                has_any_edge("shock_trade", "e_div")
                and not has_edge("shock_trade", "e_div")
                and not has_edge("e_div", "shock_trade")
            ),
            "dag_has_ediv_to_trade_directed": has_edge("e_div", "shock_trade"),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"saved JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
