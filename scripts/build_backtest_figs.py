"""バックテスト 3 種類 (長期/asset/段階) を 1 枚のサマリー画像にする."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

C_BH = "#888888"
C_S1 = "#c0392b"

# 既存報告から取得した B&H 値 (各期間)
BH_BY_PERIOD = {
    "5y":  {"sharpe": 0.71, "max_drawdown": -0.254},
    "10y": {"sharpe": 0.75, "max_drawdown": -0.339},
    "15y": {"sharpe": 0.71, "max_drawdown": -0.339},
    "20y": {"sharpe": 0.48, "max_drawdown": -0.568},
}

def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("バックテスト 3 視点サマリー (S1: e_div ≥ +0.8 short, expanding z-score, cost 0.05%/leg)",
                 fontsize=14, y=0.995)

    # --- (1) 4 期間の Sharpe 比較 ---
    ax = axes[0, 0]
    multi = json.loads((DATA / "backtest_v2_multi_period.json").read_text(encoding="utf-8"))
    periods = ["5y", "10y", "15y", "20y"]
    bh_sharpe = [BH_BY_PERIOD[p]["sharpe"] for p in periods]
    s1_sharpe = [multi["periods"][p]["strategies"]["S1_ediv_high_short"]["sharpe"] for p in periods]
    x = np.arange(len(periods))
    w = 0.35
    ax.bar(x - w/2, bh_sharpe, w, label="B&H", color=C_BH)
    ax.bar(x + w/2, s1_sharpe, w, label="S1", color=C_S1)
    ax.set_xticks(x); ax.set_xticklabels(periods, fontsize=11)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("(1) 4 期間 in-sample Sharpe — 全期間で S1 が B&H 超え")
    ax.axhline(0, color="#000", lw=0.5)
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
    for i, (bh, s1) in enumerate(zip(bh_sharpe, s1_sharpe)):
        ax.text(i - w/2, bh + 0.02, f"{bh:+.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, s1 + 0.02, f"{s1:+.2f}", ha="center", fontsize=9, fontweight="bold", color=C_S1)

    # --- (2) 4 期間の MaxDD 比較 ---
    ax = axes[0, 1]
    bh_dd = [BH_BY_PERIOD[p]["max_drawdown"] * 100 for p in periods]
    s1_dd = [multi["periods"][p]["strategies"]["S1_ediv_high_short"]["max_drawdown"] * 100 for p in periods]
    ax.bar(x - w/2, bh_dd, w, label="B&H", color=C_BH)
    ax.bar(x + w/2, s1_dd, w, label="S1", color=C_S1)
    ax.set_xticks(x); ax.set_xticklabels(periods, fontsize=11)
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("(2) 4 期間 MaxDD — S1 は全期間 -20% 圏内、B&H は 20y で -57% まで沈む")
    ax.axhline(0, color="#000", lw=0.5)
    ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0.3)
    for i, (bh, s1) in enumerate(zip(bh_dd, s1_dd)):
        ax.text(i - w/2, bh - 2.5, f"{bh:.0f}%", ha="center", fontsize=9)
        ax.text(i + w/2, s1 - 2.5, f"{s1:.0f}%", ha="center", fontsize=9, fontweight="bold", color=C_S1)

    # --- (3) asset class 除外バックテスト ---
    ax = axes[1, 0]
    asset = json.loads((DATA / "backtest_by_asset_exclusion.json").read_text(encoding="utf-8"))
    table = asset["table"]
    keys = []; vals = []; base_sharpe = None
    for row in table:
        if row["excluded"] == "(baseline)":
            base_sharpe = row["sharpe"]
        else:
            keys.append(row["excluded"]); vals.append(row["sharpe"])
    order = np.argsort(vals)  # 弱い順 (上から悪い)
    keys = [keys[i] for i in order]; vals = [vals[i] for i in order]
    colors = [C_S1 if v < 0.3 else "#e67e22" if v < 0.7 else "#2c5aa0" for v in vals]
    ax.barh(keys, vals, color=colors)
    if base_sharpe is not None:
        ax.axvline(base_sharpe, color="#000", lw=1.5, ls="--",
                   label=f"baseline (40) = {base_sharpe:+.2f}")
    ax.set_xlabel("S1 Sharpe Ratio")
    ax.set_title("(3) Asset class 除外 — INDEX を抜くと戦略崩壊 (+0.16)")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:+.2f}", va="center", fontsize=9)

    # --- (4) 段階縮小バックテスト ---
    ax = axes[1, 1]
    sub = json.loads((DATA / "backtest_by_subsampling.json").read_text(encoding="utf-8"))
    sizes_str = sorted(sub["summary_by_size"].keys(), key=int, reverse=True)
    sizes = [int(s) for s in sizes_str]
    means = [sub["summary_by_size"][s]["sharpe"]["mean"] for s in sizes_str]
    stds  = [sub["summary_by_size"][s]["sharpe"]["std"]  for s in sizes_str]
    # baseline (40) を頭に追加
    sizes = [40] + sizes
    means = [sub["baseline_40"]["s1_sharpe"]] + means
    stds  = [0] + stds
    x_s = np.arange(len(sizes))
    ax.errorbar(x_s, means, yerr=stds, fmt="o-", color=C_S1, capsize=5, lw=2, ms=8,
                ecolor="#e67e22")
    ax.set_xticks(x_s); ax.set_xticklabels([f"N={s}" for s in sizes], fontsize=11)
    ax.set_ylabel("S1 Sharpe (mean ± std, 3 trials)")
    ax.set_title("(4) 段階縮小 — N=40 だけ高 Sharpe (+1.00)、N≤35 で +0.5 前後にダウン")
    ax.axhline(means[0], color="#000", lw=0.8, ls="--", alpha=0.5,
               label=f"baseline (N=40) = {means[0]:+.2f}")
    ax.axhline(0.71, color="#888", lw=0.8, ls=":", alpha=0.7, label="B&H = +0.71")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + max(s, 0.05) + 0.04, f"{m:+.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = DATA / "fig_backtest_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
