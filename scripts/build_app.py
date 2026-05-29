"""
実装デモタブ (app.html) を独立に生成. presentation リポ側で動く.

研究リポへの依存を断ち切るため、各種データはすべて data/ から読む.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(HERE / "lib"))


# 既存イベントリスト (presentation 側にハードコード)
EVENTS_EXTENDED = [
    ("2023-08-02", "Fitch 米国格下げ", "macro", "USA"),
    ("2023-10-07", "ハマス・イスラエル戦争", "geopolitical", "ISR"),
    ("2024-01-31", "FOMC タカ派サプライズ", "monetary", "USA"),
    ("2024-04-13", "イラン・イスラエル攻撃", "geopolitical", "IRN"),
    ("2024-08-05", "円キャリー巻き戻し", "market_structure", "JPN"),
    ("2024-09-18", "FOMC 50bp 利下げ", "monetary", "USA"),
    ("2024-12-18", "FOMC タカ派サプライズ", "monetary", "USA"),
    ("2025-08-01", "サマー・ボラショック", "market_structure", "USA"),
    ("2026-01-27", "DeepSeek AI 投資見直し", "tech_shock", "CHN"),
    ("2025-04-02", "Liberation Day 相互関税", "trade_policy", "USA"),
    ("2025-04-08", "関税180日停止・株価反発", "trade_policy", "USA"),
    ("2026-03-04", "関税再発動", "trade_policy", "USA"),
    ("2022-08-09", "米CHIPS法成立", "trade_policy", "USA"),
    ("2022-10-07", "対中先端半導体輸出規制", "trade_policy", "USA"),
    ("2024-05-14", "バイデン 対中EV 100%関税", "trade_policy", "USA"),
    ("2025-04-04", "中国34%報復+希土類規制", "trade_policy", "CHN"),
    ("2025-04-09", "関税エスカレート145%", "trade_policy", "USA"),
    ("2025-04-15", "Nvidia H20輸出規制", "trade_policy", "USA"),
]


def build_monthly_snapshots(window: int = 30, threshold: float = 0.3,
                             every: int = 60, seed: int = 42) -> list:
    closes = pd.read_parquet(DATA_DIR / "ohlc_40.parquet")
    returns = closes.pct_change()
    meta = pd.read_csv(DATA_DIR / "symbol_meta.csv")
    sector_map = dict(zip(meta["internal"], meta["sector"]))
    snapshots = []
    indices = list(range(window, len(returns), every))
    print(f"Generating {len(indices)} monthly snapshots...")
    for t_idx in indices:
        win = returns.iloc[t_idx - window:t_idx]
        win_clean = win.dropna(axis=1, how="any")
        date = returns.index[t_idx - 1]
        if win_clean.shape[1] < 5:
            continue
        corr = win_clean.corr()
        G = nx.Graph()
        syms = list(win_clean.columns)
        for s in syms:
            G.add_node(s)
        edges = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                r = corr.iloc[i, j]
                if not np.isfinite(r): continue
                if abs(r) >= threshold:
                    G.add_edge(syms[i], syms[j], weight=abs(r), sign=int(np.sign(r)))
                    edges.append({"u": syms[i], "v": syms[j],
                                  "w": float(round(abs(r), 3)), "s": int(np.sign(r))})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pos = nx.spring_layout(G, k=0.6, iterations=120, seed=seed, weight="weight")
        nodes = [{"id": s, "x": float(round(pos[s][0], 3)),
                  "y": float(round(pos[s][1], 3)),
                  "sector": sector_map.get(s, "OTHER"),
                  "deg": G.degree(s)} for s in syms]
        n_holes = max(0, G.number_of_edges() - G.number_of_nodes() +
                       nx.number_connected_components(G))
        snapshots.append({
            "date": str(date.date()),
            "nodes": nodes, "edges": edges,
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "n_holes": n_holes,
        })
    return snapshots


def shock_classifier(e_div: float, l1: float, unb: float) -> dict:
    if e_div >= 0.8:
        return {"label": "政策ショック型", "color": "#c0392b",
                "desc": "符号反転を伴う構造変化 (関税連鎖など)。不整合サイクル数が突出。"}
    if e_div <= -0.5:
        return {"label": "強さ変化型", "color": "#e67e22",
                "desc": "L¹ のみ上昇、符号は安定。中規模一方的政策・市場構造ショック。"}
    if l1 >= 0.7:
        return {"label": "一般ボラ上昇", "color": "#7f8c8d",
                "desc": "両指標が同方向。VIX スパイク的な単純ボラ拡大。"}
    return {"label": "平常", "color": "#1b7e3e",
            "desc": "両指標とも平常域内。構造的に安定。"}


def main():
    gamma = pd.read_csv(DATA_DIR / "gamma_timeseries_w30.csv", parse_dates=["date"])
    gamma = gamma.dropna(subset=["L1_H1", "n_unb"])
    gamma["z_L1"]  = (gamma["L1_H1"] - gamma["L1_H1"].mean()) / gamma["L1_H1"].std()
    gamma["z_unb"] = (gamma["n_unb"] - gamma["n_unb"].mean()) / gamma["n_unb"].std()
    gamma["e_div"] = gamma["z_unb"] - gamma["z_L1"]

    last = gamma.iloc[-1]
    last_classification = shock_classifier(last["e_div"], last["L1_H1"], last["n_unb"])
    gamma_w = gamma.iloc[::3].copy()

    snapshots = build_monthly_snapshots(every=60)
    print(f"Built {len(snapshots)} monthly snapshots")
    events = [{"date": e, "label": l, "type": t, "country": c}
              for e, l, t, c in EVENTS_EXTENDED]

    # バックテスト結果 (look-ahead 完全排除版 v2 を一次ソースとする)
    bt_path = DATA_DIR / "backtest_v2_results.json"
    if not bt_path.exists():
        bt_path = DATA_DIR / "backtest_results.json"
    if bt_path.exists():
        bt = json.loads(bt_path.read_text(encoding="utf-8"))
    else:
        bt = {"summary": {}, "equity_curves": {}, "common_dates": [], "as_of": "—"}

    DATA = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M JST"),
        "data_range": {
            "start": str(gamma["date"].min().date()),
            "end": str(gamma["date"].max().date()),
            "n_days": int(len(gamma)),
        },
        "last": {
            "date": str(last["date"].date()),
            "L1": float(round(last["L1_H1"], 3)),
            "n_unb": int(last["n_unb"]),
            "z_L1": float(round(last["z_L1"], 3)),
            "z_unb": float(round(last["z_unb"], 3)),
            "e_div": float(round(last["e_div"], 3)),
            "classification": last_classification,
        },
        "ts": {
            "dates": gamma_w["date"].dt.strftime("%Y-%m-%d").tolist(),
            "L1": gamma_w["L1_H1"].round(3).tolist(),
            "n_unb": gamma_w["n_unb"].astype(int).tolist(),
            "z_L1": gamma_w["z_L1"].round(3).tolist(),
            "z_unb": gamma_w["z_unb"].round(3).tolist(),
            "e_div": gamma_w["e_div"].round(3).tolist(),
        },
        "snapshots": snapshots,
        "events": events,
        "backtest": bt,
    }

    # snapshot の指標値マージ
    gamma_idx = gamma.set_index(gamma["date"].dt.date)
    for snap in snapshots:
        try:
            d = pd.Timestamp(snap["date"]).date()
            if d in gamma_idx.index:
                row = gamma_idx.loc[d]
                snap["L1"]    = float(round(row["L1_H1"], 3))
                snap["z_L1"]  = float(round(row["z_L1"], 2))
                snap["z_unb"] = float(round(row["z_unb"], 2))
                snap["e_div"] = float(round(row["e_div"], 2))
                snap["classification"] = shock_classifier(row["e_div"], row["L1_H1"], row["n_unb"])
        except Exception:
            pass

    # HTML テンプレート読み込み
    tpl_path = HERE / "templates" / "app.html"
    template = tpl_path.read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(DATA, ensure_ascii=False))
    out = ROOT / "app.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}  ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
