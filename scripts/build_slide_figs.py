"""
発表スライド用のシンプルな図を生成する (初学者向け、数字最小、Cミニマル配色).
出力: figs/slide_*.png (slides.html から相対参照)

既存 Pages の専門的な図とは別。発表では「一目で分かる」ことを最優先。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

# 日本語フォント (Windows: Meiryo / Yu Gothic)
for fname in ["Meiryo", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP"]:
    try:
        font_manager.findfont(fname, fallback_to_default=False)
        plt.rcParams["font.family"] = fname
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# Cミニマル配色
INK = "#1a1a1f"; ACCENT = "#2f6df6"; GREEN = "#2a9d5c"
RED = "#d24b4b"; ORANGE = "#e08a2b"; MUTED = "#8a8d93"; GRID = "#e8eaed"


def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#cfd3da")
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_scatter():
    """発見①: 穴の量 vs 不均衡の数 が無相関 (バラけてる)."""
    df = pd.read_csv(DATA / "gamma_timeseries_w30.csv").dropna(subset=["L1_H1", "n_unb"])
    # expanding z-score
    mp = 90
    zL1 = (df["L1_H1"] - df["L1_H1"].expanding(mp).mean()) / df["L1_H1"].expanding(mp).std()
    zU = (df["n_unb"] - df["n_unb"].expanding(mp).mean()) / df["n_unb"].expanding(mp).std()
    m = zL1.notna() & zU.notna()
    x, y = zL1[m], zU[m]
    r = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=150)
    ax.scatter(x, y, s=14, c=ACCENT, alpha=0.35, edgecolors="none")
    style_ax(ax)
    ax.set_xlabel("穴の量  (大きい →)", fontsize=13, color=INK)
    ax.set_ylabel("不均衡の数  (大きい →)", fontsize=13, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8, ls="--"); ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_title(f"バラけている = 連動しない  (相関 {r:.2f})",
                 fontsize=14, color=INK, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIGS / "slide_scatter.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"slide_scatter.png  (相関 r={r:.3f}, n={m.sum()})")


def fig_scatter_raw():
    """3.3 独立性: 穴 L1 と 不均衡 n_unb の生値散布図 (相関 0.15 = 独立).

    ポスターに縮小配置されるため、軸ラベル/凡例/目盛りを大きめに描く。
    """
    df = pd.read_csv(DATA / "gamma_timeseries_w30.csv").dropna(subset=["L1_H1", "n_unb"])
    x, y = df["L1_H1"].values, df["n_unb"].values
    r = float(np.corrcoef(x, y)[0, 1])
    a, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=150)
    ax.scatter(x, y, s=16, c=ACCENT, alpha=0.30, edgecolors="none", label="日次のデータ点")
    ax.plot(xs, a * xs + b, color=RED, lw=3.4, ls="--", label=f"回帰直線（相関 r={r:.2f}）")
    style_ax(ax)
    ax.tick_params(labelsize=17)
    ax.set_xlabel("穴の指標 $L^1$（0〜2くらい）", fontsize=21, color=INK)
    ax.set_ylabel("不均衡の数 $n_{unb}$（0〜80くらい）", fontsize=21, color=INK)
    ax.legend(loc="upper left", fontsize=18, frameon=True, framealpha=0.95, edgecolor="#cccccc")
    ax.set_title("穴 と 不均衡 は バラけている ＝ 連動しない（独立）",
                 fontsize=21, color=INK, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIGS / "slide_scatter_raw.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"slide_scatter_raw.png  (相関 r={r:.3f}, n={len(x)})")


def fig_hole():
    """3.1: Vietoris-Rips で穴が現れて消える3コマ.

    各点に半径 r の円を描き、2円が重なった (中心間距離 <= 2r) 対を辺で結ぶ。
    r を大きくすると輪(穴 H1)が現れ、三角形が埋まって消える。
    """
    from matplotlib.patches import Circle, Polygon as MPoly
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2), dpi=150)
    titles = ["小：辺なし", "中：外周→穴", "大：埋まる"]
    radii = [0.34, 0.62, 1.02]
    n = 6
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pts = np.c_[np.cos(theta), np.sin(theta)]
    for ax, title, r in zip(axes, titles, radii):
        ax.set_title(title, fontsize=31, color=INK, pad=10)
        for (x, y) in pts:
            ax.add_patch(Circle((x, y), r, color=ACCENT, alpha=0.12, ec=ACCENT, lw=0.5))
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if np.hypot(*(pts[i] - pts[j])) <= 2 * r + 1e-9]
        eset = set(edges)
        tris = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)
                if (i, j) in eset and (j, k) in eset and (i, k) in eset]
        dense = len(edges) > n           # 全結合コマ = 埋まりを面で見せる
        lw = 0.8 if dense else 1.8
        tri_alpha = 0.16 if dense else 0.10
        for (i, j, k) in tris:
            ax.add_patch(MPoly(pts[[i, j, k]], closed=True, color=ACCENT,
                               alpha=tri_alpha, ec="none", zorder=1))
        for (i, j) in edges:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    color=ACCENT, lw=lw, zorder=3)
        ax.scatter(pts[:, 0], pts[:, 1], s=66, c=INK, zorder=5)
        if len(edges) >= n and not tris:   # 外周が閉じたが面はまだ = 穴
            ax.text(0, 0, "穴", fontsize=46, color=RED, ha="center",
                    va="center", weight="bold", zorder=6)
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-2.05, 2.05)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.5, 0.02, "2円が重なった対を辺で結ぶ ｜ 円の半径を大きくする →",
             fontsize=27, color=MUTED, ha="center")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGS / "slide_hole.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("slide_hole.png")


def fig_lifetime():
    """3.1: 穴の寿命バー (本物=長い青, ノイズ=短い灰). 文字大きめ."""
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=150)
    # (start, length, is_real)
    bars = [
        (0.04, 0.62, True),
        (0.10, 0.42, True),
        (0.13, 0.06, False),
        (0.19, 0.07, False),
        (0.27, 0.07, False),
        (0.33, 0.05, False),
        (0.40, 0.05, False),
    ]
    y = len(bars)
    for (s, ln, real) in bars:
        y -= 1
        c = ACCENT if real else "#9aa0a8"
        h = 0.44 if real else 0.30
        ax.barh(y, ln, left=s, height=h, color=c)
    ax.text(0.70, len(bars) - 1, "← 本物の穴（長く残る＝市場の構造）",
            fontsize=31, color=ACCENT, va="center", weight="bold")
    ax.text(0.46, len(bars) - 5, "← ノイズ（すぐ消える）",
            fontsize=28, color=MUTED, va="center")
    ax.annotate("", xy=(1.02, -0.7), xytext=(0.0, -0.7),
                arrowprops=dict(arrowstyle="->", color=INK, lw=2.2))
    ax.text(0.5, -1.15, "円の半径を大きくする →", fontsize=28, color=INK, ha="center")
    ax.set_xlim(-0.02, 1.4)
    ax.set_ylim(-1.6, len(bars))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGS / "slide_lifetime.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("slide_lifetime.png")


def fig_equity():
    """発見③: 暴落で戦略の方が浅い (ただ持つ vs 不均衡の日に避ける)."""
    d = json.load(open(DATA / "backtest_v2_results.json", encoding="utf-8"))
    dates = pd.to_datetime(d["common_dates"])
    bh = np.array(d["equity_curves"]["Z_buy_and_hold"])
    s1 = np.array(d["equity_curves"]["S1_ediv_high_short"])
    # 100 を起点に正規化
    bh = bh / bh[0] * 100
    s1 = s1 / s1[0] * 100

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    ax.plot(dates, bh, color=MUTED, lw=2.2, label="ただ持ち続ける")
    ax.plot(dates, s1, color=ACCENT, lw=2.4, label="不均衡の日に避ける (守り)")
    style_ax(ax)
    ax.set_ylabel("資産 (100 から開始)", fontsize=12, color=INK)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    # 暴落局面を矢印で (2025-04 関税)
    try:
        idx = (dates >= "2025-03-20") & (dates <= "2025-05-10")
        if idx.any():
            lo = min(bh[idx].min(), s1[idx].min())
            xpos = dates[idx][np.argmin(bh[idx])]
            ax.annotate("暴落時:\n戦略の方が浅い", xy=(xpos, lo),
                        xytext=(xpos, lo - 22), fontsize=11, color=RED, ha="center",
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    except Exception:
        pass
    ax.set_title("お金は増えないが、暴落で沈みにくい",
                 fontsize=14, color=INK, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIGS / "slide_equity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("slide_equity.png")


def fig_source():
    """発見②: 暴落の震源 — 何回の暴落で「上位5の震源」に入ったか (一貫性)."""
    d = json.load(open(DATA / "loo_multi_shock.json", encoding="utf-8"))
    n_shocks = len(d["shocks"])
    # 各ショックで上位5に入った回数をカウント
    from collections import Counter
    count = Counter()
    for sk, sv in d["shocks"].items():
        loo = sorted(sv.get("leave_one_out", []),
                     key=lambda x: abs(x.get("contribution", 0)), reverse=True)
        for r in loo[:5]:
            count[r["removed"]] += 1
    top = sorted(count.items(), key=lambda kv: kv[1], reverse=True)[:8]
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    # 一貫性で色分け: 4回=真の共通(赤) / 3回=準(オレンジ) / 2回以下=event依存(グレー)
    def col(v):
        return RED if v >= 4 else (ORANGE if v == 3 else "#c3c8cf")
    colors = [col(v) for v in vals]

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=150)
    bars = ax.barh(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.05, b.get_y() + b.get_height()/2, f"{v}/{n_shocks}",
                va="center", fontsize=11, color=INK)
    style_ax(ax)
    ax.set_xlim(0, n_shocks + 0.6)
    ax.set_xlabel(f"{n_shocks} 個の暴落のうち、何回「震源」になったか", fontsize=12, color=INK)
    ax.set_title("天然ガス (赤) だけが全暴落で共通の震源\n銅・中国 (橙/灰) は暴落の種類で変わる",
                 fontsize=13, color=INK, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIGS / "slide_source.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"slide_source.png  (NGAS={count['NGAS']}, COPPER={count['COPPER']}, CHINA50={count['CHINA50']})")


def fig_network():
    """道具導入: 40銘柄を点、似た動きを線で結ぶ (実データの相関ネットワーク)."""
    import networkx as nx
    ohlc = pd.read_parquet(DATA / "ohlc_40.parquet")
    rets = ohlc.pct_change().dropna(how="all")
    win = rets.tail(30).dropna(axis=1, how="any")
    corr = win.corr()
    syms = list(corr.columns)

    G = nx.Graph()
    G.add_nodes_from(syms)
    TH = 0.4
    for i, a in enumerate(syms):
        for b in syms[i+1:]:
            c = corr.loc[a, b]
            if abs(c) >= TH:
                G.add_edge(a, b, weight=c, sign=1 if c > 0 else -1)

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=150)
    pos = nx.spring_layout(G, seed=42, k=0.55)
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] < 0]
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color="#9fc0f5", width=1.4, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color="#f0b3b3", width=1.4,
                           style="dashed", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=INK, node_size=260, ax=ax)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=INK, markersize=15, label='点 ＝ 銘柄'),
        Line2D([0], [0], color='#9fc0f5', lw=3.5, label='実線 ＝ 正相関'),
        Line2D([0], [0], color='#f0b3b3', lw=3.5, ls='--', label='破線 ＝ 負相関'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18,
              frameon=True, framealpha=0.95, edgecolor='#cccccc')
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGS / "slide_network.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"slide_network.png  (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")


def fig_balance():
    """3.2: 構造的均衡 (正相関＋/負相関−、符号の積で均衡/不均衡) 3例."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2), dpi=150)
    A = (0.0, 0.9); B = (-0.8, -0.6); C = (0.8, -0.6)
    edges_xy = {"AB": (A, B), "AC": (A, C), "BC": (B, C)}
    cases = [
        ("全て ＋ ＝ 均衡", {"AB": +1, "AC": +1, "BC": +1}, GREEN),
        ("− が偶数 ＝ 均衡", {"AB": +1, "AC": -1, "BC": -1}, GREEN),
        ("− が奇数 ＝ 不均衡", {"AB": +1, "AC": -1, "BC": +1}, RED),
    ]
    gx, gy = 0.0, -0.1
    for ax, (title, signs, tcol) in zip(axes, cases):
        ax.set_title(title, fontsize=26, color=tcol, pad=12, weight="bold")
        for name, s in signs.items():
            p, q = edges_xy[name]
            col = ACCENT if s > 0 else RED
            ls = "-" if s > 0 else (0, (6, 4))
            ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=3, ls=ls, zorder=2)
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            dx, dy = mx - gx, my - gy
            nrm = np.hypot(dx, dy) or 1.0
            lx, ly = mx + dx / nrm * 0.30, my + dy / nrm * 0.30
            ax.text(lx, ly, "＋" if s > 0 else "−", fontsize=35, color=col,
                    ha="center", va="center", weight="bold", zorder=4)
        for lab, (x, y) in [("A", A), ("B", B), ("C", C)]:
            ax.add_patch(plt.Circle((x, y), 0.19, fc="white", ec=INK, lw=2, zorder=5))
            ax.text(x, y, lab, fontsize=23, ha="center", va="center",
                    weight="bold", color=INK, zorder=6)
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.25, 1.4)
        ax.set_aspect("equal"); ax.axis("off")
    fig.text(0.5, 0.02,
             "実線 ＝ 正相関（＋）　破線 ＝ 負相関（−）　｜　符号の積が −1（負相関が奇数）＝ 不均衡サイクル",
             fontsize=21, color="#374151", ha="center")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGS / "slide_balance.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("slide_balance.png")


if __name__ == "__main__":
    fig_scatter()
    fig_equity()
    fig_source()
    fig_network()
    fig_balance()
    fig_hole()
    print("Done.")
