"""8 パターンの銘柄構成で Liberation Day event study を実行し、
Δe_div / Δ L¹ / Δn_unb を比較する。

P0: 現状 40 (baseline)
P1: 個別株増 50 (40 + 個別株 10)
P2: 暗号増   45 (40 + 暗号 5)
P3: 地域指数増 45 (40 + 地域指数 5)
P4: 全部入り 60 (40 + 全 20)
P5: minimal 15 (主要のみ)
P6: 単一資産 INDEX のみ 9
P7: 単一資産 FX のみ 13

出力: data/symbol_pattern_results_extended.json + data/fig_symbol_patterns.png
"""
from __future__ import annotations
import sys
import json
import time
import warnings
import io
from pathlib import Path

# Windows / cp932 環境でも Δ や L¹ を安全に表示
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 日本語フォント (Windows): Meiryo → Noto Sans JP → MS Gothic を順に試す
for _font in ("Meiryo", "Noto Sans JP", "Yu Gothic", "MS Gothic"):
    try:
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# event_study / compute_day の関数を流用
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))

from test_symbol_variations import event_study, EVENT_DATE  # type: ignore  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "data"


# 既存 40 銘柄
BASE_40 = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP", "USDTRY", "USDHUF", "EURTRY",
    "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "USOUSD", "UKOUSD",
    "SP500", "NAS100", "DJ30", "RUS2000", "GER40", "UK100", "FRA40", "JP225", "CHINA50",
    "DXY", "VIX",
    "AAPL", "MSFT", "GOOG", "META", "TSLA",
    "US10Y", "EUB10Y", "UKGILT",
    "COPPER", "NGAS",
]
ADD_STOCKS = ["NVDA", "AMZN", "JPM", "WMT", "V", "JNJ", "KO", "XOM", "BA", "MA"]
ADD_CRYPTO = ["SOLUSD", "BNBUSD", "XRPUSD", "DOGEUSD", "ADAUSD"]
ADD_INDEX  = ["KS11", "BSESN", "BVSP", "STI", "HSI"]

# P5: minimal 15
MINIMAL_15 = [
    # FX 3
    "EURUSD", "USDJPY", "GBPUSD",
    # INDEX 4
    "SP500", "NAS100", "GER40", "JP225",
    # COMMODITY 3
    "XAUUSD", "USOUSD", "COPPER",
    # CRYPTO 1
    "BTCUSD",
    # BOND 2
    "US10Y", "EUB10Y",
    # SPECIAL 2
    "VIX", "DXY",
]

# P6: 既存 INDEX 9
INDEX_ONLY_9 = ["SP500", "NAS100", "DJ30", "RUS2000",
                "GER40", "UK100", "FRA40", "JP225", "CHINA50"]

# P7: 既存 FX 13
FX_ONLY_13 = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD",
              "EURJPY", "GBPJPY", "EURGBP", "USDTRY", "USDHUF", "EURTRY"]


PATTERNS = [
    ("P0_baseline_40", "現状 40 (baseline)", BASE_40),
    ("P1_stocks_50",   "個別株増 50 (40+株 10)", BASE_40 + ADD_STOCKS),
    ("P2_crypto_45",   "暗号増 45 (40+暗号 5)", BASE_40 + ADD_CRYPTO),
    ("P3_index_45",    "地域指数増 45 (40+指数 5)", BASE_40 + ADD_INDEX),
    ("P4_all_60",      "全部入り 60 (40+全 20)", BASE_40 + ADD_STOCKS + ADD_CRYPTO + ADD_INDEX),
    ("P5_minimal_15",  "minimal 15 (主要のみ)", MINIMAL_15),
    ("P6_index_only_9","単一資産: 既存 INDEX 9", INDEX_ONLY_9),
    ("P7_fx_only_13",  "単一資産: 既存 FX 13", FX_ONLY_13),
]


def run_pattern(closes_all: pd.DataFrame, name: str, label: str,
                symbols: list[str]) -> dict:
    present = [s for s in symbols if s in closes_all.columns]
    missing = [s for s in symbols if s not in closes_all.columns]
    if len(present) < 5:
        return {"name": name, "label": label, "n": len(present),
                "missing": missing, "skipped": "n < 5", "delta_e_div": None}
    closes = closes_all[present]
    r = event_study(closes)
    return {
        "name": name,
        "label": label,
        "n": len(present),
        "missing": missing,
        "delta_L1": r["delta_L1"],
        "delta_n_unb": r["delta_n_unb"],
        "delta_e_div": r["delta_e_div"],
    }


def make_figure(results: list[dict], out_path: Path) -> None:
    """8 パターンの Δe_div / Δ L¹ / Δn_unb 比較バーチャート."""
    valid = [r for r in results if r.get("delta_e_div") is not None]
    labels = [r["label"] + f"\n(n={r['n']})" for r in valid]
    de = [r["delta_e_div"] for r in valid]
    dl = [r["delta_L1"] for r in valid]
    du = [r["delta_n_unb"] for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

    def style_ax(ax, vals, ylabel, color_pos, color_neg, baseline_val):
        colors = [color_pos if v >= 0 else color_neg for v in vals]
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="#222",
                      linewidth=0.8)
        ax.axhline(0, color="#444", linewidth=0.7)
        if baseline_val is not None:
            ax.axhline(baseline_val, color="#888", linestyle="--",
                       linewidth=1.0, label=f"P0 baseline = {baseline_val:+.2f}σ")
            ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.04 if v >= 0 else -0.08),
                    f"{v:+.2f}", ha="center", fontsize=9,
                    color="#111", fontweight="bold")

    base_e = next((r["delta_e_div"] for r in valid if r["name"] == "P0_baseline_40"), None)
    base_l = next((r["delta_L1"] for r in valid if r["name"] == "P0_baseline_40"), None)
    base_u = next((r["delta_n_unb"] for r in valid if r["name"] == "P0_baseline_40"), None)

    style_ax(axes[0], de, "Δe_div (post − pre, σ)", "#2E7D5B", "#B23A48", base_e)
    axes[0].set_title("Δe_div: 構造的不整合シグナル", fontsize=13, fontweight="bold")

    style_ax(axes[1], dl, "Δ L¹ (post − pre, σ)", "#1E6091", "#B23A48", base_l)
    axes[1].set_title("Δ L¹: 持続ホモロジー強度", fontsize=13, fontweight="bold")

    style_ax(axes[2], du, "Δ n_unb (post − pre, σ)", "#7A5C1E", "#B23A48", base_u)
    axes[2].set_title("Δ n_unb: 符号不整合サイクル数", fontsize=13, fontweight="bold")

    fig.suptitle("Liberation Day (2025-04-02) を 8 種類の銘柄構成で実証\n"
                 "— 銘柄を入れ替えても本質は変わらない (構造シグナルの頑健性)",
                 fontsize=14, fontweight="bold")

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    t0 = time.time()
    print("=" * 64)
    print("8 パターン銘柄構成 event study (Liberation Day 2025-04-02)")
    print("=" * 64)

    closes_all = pd.read_parquet(DATA / "ohlc_60_extended.parquet")
    print(f"Loaded: {closes_all.shape}")
    print(f"Period: {closes_all.index.min().date()} ~ {closes_all.index.max().date()}")
    print(f"Event: {EVENT_DATE.date()}")
    print()

    results = []
    print(f"{'name':<22} {'n':>4} {'Δe_div':>9} {'Δ L¹':>9} {'Δn_unb':>9}")
    print("-" * 64)
    for name, label, syms in PATTERNS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_pattern(closes_all, name, label, syms)
        results.append(r)
        if r.get("delta_e_div") is None:
            print(f"{name:<22} {r['n']:>4}  SKIP ({r.get('skipped','?')})")
        else:
            print(f"{name:<22} {r['n']:>4} "
                  f"{r['delta_e_div']:>+9.3f} "
                  f"{r['delta_L1']:>+9.3f} "
                  f"{r['delta_n_unb']:>+9.3f}")

    # 比較解釈テキスト
    base = next(r for r in results if r["name"] == "P0_baseline_40")
    base_e = base["delta_e_div"]
    print()
    print("--- baseline P0 との差 (Δe_div) ---")
    for r in results:
        if r.get("delta_e_div") is None:
            continue
        diff = r["delta_e_div"] - base_e
        kept = "強まる" if diff > 0.2 else ("弱まる" if diff < -0.2 else "ほぼ不変")
        print(f"  {r['name']:<22} Δe_div={r['delta_e_div']:+.3f}  vs P0 {diff:+.3f}σ  ({kept})")

    out_json = DATA / "symbol_pattern_results_extended.json"
    out_json.write_text(json.dumps(
        {"event": str(EVENT_DATE.date()), "n_patterns": len(results),
         "patterns": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_json}")

    out_png = DATA / "fig_symbol_patterns.png"
    make_figure(results, out_png)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
