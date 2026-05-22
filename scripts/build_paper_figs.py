"""論文用 高品質 figure 生成 (300 DPI, vector format)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "docs" / "paper" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication quality settings
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def fig1_independence_scatter():
    """L1 vs n_unb 散布図 + 周辺分布 (3 市場)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    datasets = [
        ("data/gamma_timeseries_w30.csv", "USA 40 (5y)", r"$r = +0.16$"),
        ("data/gamma_cn_timeseries_w30.csv", "China 52 (5y)", r"$r = +0.41$"),
        ("data/gamma_em_timeseries_w30.csv", "EM basket 40 (5y)", r"$r = +0.17$"),
    ]
    for ax, (path, title, label) in zip(axes, datasets):
        df = pd.read_csv(ROOT / path).dropna(subset=["L1_H1", "n_unb"])
        ax.scatter(df["L1_H1"], df["n_unb"], s=3, alpha=0.3,
                   color="#2c5aa0", edgecolor="none")
        r = df[["L1_H1", "n_unb"]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f"{label}\n$n = {len(df)}$\nactual $r = {r:+.3f}$",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                          edgecolor="gray"), fontsize=9)
        ax.set_xlabel(r"$\|H_1\|_{L^1}$ (magnitude)")
        if ax is axes[0]:
            ax.set_ylabel(r"$n_{\mathrm{unb}}$ (sign)")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_independence_scatter.pdf")
    plt.savefig(FIG_DIR / "fig1_independence_scatter.png")
    plt.close()
    print(f"Saved: fig1_independence_scatter.pdf/png")


def fig2_event_study_categorical():
    """カテゴリ別 Δσ バー (L1 vs n_unb)."""
    cat_results = json.loads((DATA_DIR / "gamma_extended_w30.json").read_text(encoding="utf-8"))
    cats = ["geopolitical", "market_structure", "trade_policy",
            "tech_shock", "macro", "monetary"]
    L1_d = [cat_results[c]["L1_H1"]["observed_delta_sigma"] for c in cats]
    unb_d = [cat_results[c]["n_unb"]["observed_delta_sigma"] for c in cats]
    L1_p = [cat_results[c]["L1_H1"]["p_perm_random"] for c in cats]
    unb_p = [cat_results[c]["n_unb"]["p_perm_random"] for c in cats]
    n_evs = [cat_results[c]["n_events"] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(cats))
    width = 0.36
    b1 = ax.bar(x - width/2, L1_d, width, label=r"$\|H_1\|_{L^1}$ (magnitude)",
                color="#c0392b", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width/2, unb_d, width, label=r"$n_{\mathrm{unb}}$ (sign)",
                color="#2c5aa0", edgecolor="black", linewidth=0.6)
    for i, (bv, p) in enumerate(zip(L1_d, L1_p)):
        if p < 0.05:
            ax.text(i - width/2, bv + 0.05 if bv > 0 else bv - 0.10, "*",
                    ha="center", fontsize=12, fontweight="bold")
    for i, (bv, p) in enumerate(zip(unb_d, unb_p)):
        if p < 0.05:
            ax.text(i + width/2, bv + 0.05 if bv > 0 else bv - 0.10, "*",
                    ha="center", fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n($n = {n}$)" for c, n in zip(cats, n_evs)],
                       fontsize=9)
    ax.set_ylabel(r"$\Delta\sigma$ (pre-event window [-15, -1])")
    ax.set_title("Event Study by Shock Category", fontsize=11)
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_event_study_categorical.pdf")
    plt.savefig(FIG_DIR / "fig2_event_study_categorical.png")
    plt.close()
    print(f"Saved: fig2_event_study_categorical.pdf/png")


def fig3_ediv_classifier():
    """e_div グループ別バー (ショックタイプ判別)."""
    div_results = json.loads((DATA_DIR / "gamma_divergence_index.json").read_text(encoding="utf-8"))
    groups = ["large_reciprocal_2025", "trade_policy_ALL", "vix_auto", "mid_unilateral"]
    labels = ["2025-04 LD\ncluster ($n = 5$)",
              "All trade\\_policy\n($n = 36$)",
              "VIX-spike\n($n = 25$)",
              "Mid-unilateral\n($n = 7$)"]
    ediv = [div_results[g]["e_div_delta_sigma"] for g in groups]
    p_perm = [div_results[g]["p_perm"] for g in groups]
    colors = ["#c0392b" if p < 0.05 else "#e67e22" if p < 0.10 else "#7f8c8d"
              for p in p_perm]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(groups)), ediv, color=colors,
                  edgecolor="black", linewidth=0.6)
    for i, (v, p) in enumerate(zip(ediv, p_perm)):
        ax.text(i, v + (0.05 if v > 0 else -0.10),
                f"{v:+.2f}\n($p={p:.3f}$)",
                ha="center", va="bottom" if v > 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(+0.8, color="#c0392b", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(-0.5, color="#e67e22", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.text(3.4, +0.8, "Policy threshold", color="#c0392b", fontsize=8,
            va="bottom", ha="right")
    ax.text(3.4, -0.5, "Magnitude threshold", color="#e67e22", fontsize=8,
            va="top", ha="right")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\Delta\sigma$ of $e_{\mathrm{div}} = z_{\mathrm{unb}} - z_{L^1}$")
    ax.set_title("Divergence Index as Shock-Type Classifier", fontsize=11)
    ax.set_ylim(-1, 2.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_ediv_classifier.pdf")
    plt.savefig(FIG_DIR / "fig3_ediv_classifier.png")
    plt.close()
    print(f"Saved: fig3_ediv_classifier.pdf/png")


def fig4_three_indicator_correlation():
    """3 指標 (n_unb, K, F) 相関ヒートマップ."""
    bench = pd.read_csv(DATA_DIR / "benchmark_indicators_w30.csv")
    valid = bench.dropna(subset=["n_unb", "K", "F"])
    corr = valid[["n_unb", "K", "F"]].corr()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = [r"$n_{\mathrm{unb}}$ (ours)", r"$K$ (Ferreira)", r"$F$ (Aref)"]
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=10)
    for i in range(3):
        for j in range(3):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    ax.set_title("Three Sign-Balance Indicator Correlation", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.05, shrink=0.85)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_three_indicator_correlation.pdf")
    plt.savefig(FIG_DIR / "fig4_three_indicator_correlation.png")
    plt.close()
    print(f"Saved: fig4_three_indicator_correlation.pdf/png")


def fig5_equity_curve():
    """全戦略の equity curve."""
    bt = json.loads((DATA_DIR / "backtest_v2_results.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 4))
    dates = bt["common_dates"]
    pick = ["Z_buy_and_hold", "S1_ediv_high_short", "S2_ediv_low_long",
            "S6_zL1_AND_zunb_short"]
    colors = {"Z_buy_and_hold": "black",
              "S1_ediv_high_short": "#c0392b",
              "S2_ediv_low_long": "#27ae60",
              "S6_zL1_AND_zunb_short": "#8e44ad"}
    labels = {"Z_buy_and_hold": "Buy \\& Hold (benchmark)",
              "S1_ediv_high_short": r"$S_1$: $e_{\mathrm{div}} \geq 0.8$ (short)",
              "S2_ediv_low_long":  r"$S_2$: $e_{\mathrm{div}} \leq -0.5$ (long)",
              "S6_zL1_AND_zunb_short": r"$S_6$: $z_{L^1} \wedge z_{\mathrm{unb}}$ (short)"}
    for k in pick:
        curve = bt["equity_curves"][k]
        ax.plot(dates, curve, label=labels[k], color=colors[k],
                linewidth=1.8 if k == "Z_buy_and_hold" else 1.2,
                linestyle="-" if k == "Z_buy_and_hold" else "-")
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    n = len(dates)
    ax.set_xticks([dates[i] for i in [0, n//4, n//2, 3*n//4, n-1]])
    ax.set_xticklabels([dates[i][:7] for i in [0, n//4, n//2, 3*n//4, n-1]],
                       fontsize=9)
    ax.set_ylabel(r"Cumulative return (\$1 $\to$ ?)")
    ax.set_xlabel("Date")
    ax.set_title("Backtest v2: Equity Curves (5y in-sample)", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_equity_curve.pdf")
    plt.savefig(FIG_DIR / "fig5_equity_curve.png")
    plt.close()
    print(f"Saved: fig5_equity_curve.pdf/png")


def fig6_drawdown_curve():
    """全戦略の drawdown curve."""
    bt = json.loads((DATA_DIR / "backtest_v2_results.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 4))
    dates = bt["common_dates"]
    pick = ["Z_buy_and_hold", "S1_ediv_high_short", "S6_zL1_AND_zunb_short"]
    colors = {"Z_buy_and_hold": "black",
              "S1_ediv_high_short": "#c0392b",
              "S6_zL1_AND_zunb_short": "#8e44ad"}
    labels = {"Z_buy_and_hold": "Buy \\& Hold",
              "S1_ediv_high_short": r"$S_1$ (short)",
              "S6_zL1_AND_zunb_short": r"$S_6$ AND (short)"}
    for k in pick:
        curve = np.array(bt["equity_curves"][k])
        peak = np.maximum.accumulate(curve)
        dd = (curve / peak - 1) * 100
        ax.fill_between(range(len(dates)), dd, 0, alpha=0.25, color=colors[k])
        ax.plot(range(len(dates)), dd, label=labels[k], color=colors[k],
                linewidth=1.4 if k == "Z_buy_and_hold" else 1.0)
    n = len(dates)
    ax.set_xticks([0, n//4, n//2, 3*n//4, n-1])
    ax.set_xticklabels([dates[i][:7] for i in [0, n//4, n//2, 3*n//4, n-1]],
                       fontsize=9)
    ax.set_ylabel("Drawdown (\\%)")
    ax.set_xlabel("Date")
    ax.set_title("Backtest v2: Drawdown Curves", fontsize=11)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_drawdown_curve.pdf")
    plt.savefig(FIG_DIR / "fig6_drawdown_curve.png")
    plt.close()
    print(f"Saved: fig6_drawdown_curve.pdf/png")


def fig7_walkforward_alpha():
    """Walk-forward 15y fold ごとの alpha bar chart."""
    wf = json.loads((DATA_DIR / "backtest_walkforward_15y.json").read_text(encoding="utf-8"))
    key = "15y_train3y_test1y_pct80_short"
    if key not in wf:
        print(f"  skip fig7: {key} not found")
        return
    folds = wf[key]["folds"]
    fold_ids = [f["fold_id"] for f in folds]
    test_periods = [f["test_period"][:11] for f in folds]
    alphas = [f["alpha"] * 100 for f in folds]
    sharpes = [f["sharpe"] for f in folds]
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#27ae60" if a > 0 else "#c0392b" for a in alphas]
    bars = ax.bar(fold_ids, alphas, color=colors, edgecolor="black", linewidth=0.6)
    for i, (a, s) in enumerate(zip(alphas, sharpes)):
        ax.text(i, a + (1 if a > 0 else -3), f"S={s:+.2f}",
                ha="center", fontsize=8, color="black")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(fold_ids)
    ax.set_xticklabels([f"f{i}\n{tp}" for i, tp in zip(fold_ids, test_periods)],
                       fontsize=8)
    ax.set_ylabel(r"$\alpha$ vs Buy \& Hold (\%)")
    ax.set_xlabel("Walk-forward fold (test year)")
    ax.set_title("Out-of-Sample Alpha by Year (Walk-Forward 15y, $S_1$ short)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_walkforward_alpha.pdf")
    plt.savefig(FIG_DIR / "fig7_walkforward_alpha.png")
    plt.close()
    print(f"Saved: fig7_walkforward_alpha.pdf/png")


if __name__ == "__main__":
    fig1_independence_scatter()
    fig2_event_study_categorical()
    fig3_ediv_classifier()
    fig4_three_indicator_correlation()
    fig5_equity_curve()
    fig6_drawdown_curve()
    fig7_walkforward_alpha()
    print(f"\nAll figures saved to: {FIG_DIR}")
