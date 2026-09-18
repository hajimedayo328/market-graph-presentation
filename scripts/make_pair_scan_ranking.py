"""robustness_ediv_pair_scan.json を人が読める順位表 (docs/PAIR_SCAN_RANKING.md) にする.

11指標から2つ選ぶ55通りの差分 z(A) − z(B) について、事件の種類ごとに
|Δσ|（事件前後30日で指標が標準偏差の何個分動いたか）で順位をつける。
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "robustness_ediv_pair_scan.json"
OUT = ROOT / "docs" / "PAIR_SCAN_RANKING.md"

NAMES = {
    "L1": "穴の寿命の合計 L¹", "L2": "穴の寿命の二乗和 L²", "Linf": "最長の穴の寿命 L∞",
    "nH1": "穴の数", "meanP": "穴の平均寿命", "entropy": "寿命のばらつき（エントロピー）",
    "n_unb_total": "矛盾した輪の数（全部）", "n_unb_3": "矛盾した三角形の数",
    "n_unb_4": "矛盾した四角形の数", "n_unb_5plus": "矛盾した5角形以上の数",
    "balance_rate": "均衡している輪の割合",
}
CAT_JA = {"trade_policy": "関税", "market_structure": "市場構造", "war": "戦争"}
EDIV_KEY = "L1__minus__n_unb_total"   # = −e_div


def pair_label(key: str) -> str:
    a, b = key.split("__minus__")
    return f"{NAMES[a]} − {NAMES[b]}"


def table(rows: list[tuple[str, float, dict]], event_dates: list[str]) -> list[str]:
    """rows: (pair_key, mean_dsigma, per_event) を |mean| 降順で並べて Markdown 表にする."""
    rows = sorted(rows, key=lambda r: -abs(r[1]))
    head = "| 順位 | 組み合わせ（A − B） | Δσ 平均 | " + " | ".join(event_dates) + " |"
    sep = "|---|---|---|" + "---|" * len(event_dates)
    out = [head, sep]
    for i, (k, m, pe) in enumerate(rows, 1):
        mark = " **← e_div（符号は逆）**" if k == EDIV_KEY else ""
        cells = " | ".join(f"{pe[d]:+.2f}" if d in pe else "—" for d in event_dates)
        out.append(f"| {i} | {pair_label(k)}{mark} | {m:+.2f} | {cells} |")
    return out


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    meta = d["meta"]
    lines = [
        "# 55通りの指標の組み合わせ — 事件前後の反応の順位",
        "",
        f"元データ: [`data/robustness_ediv_pair_scan.json`](../data/robustness_ediv_pair_scan.json)"
        f"（生成: `scripts/make_pair_scan_ranking.py`）",
        "",
        "## 読み方",
        "",
        "- 11種類の指標から2つ選び、差分 z(A) − z(B) を作る。組み合わせは55通り。",
        f"- **Δσ** = 事件の前{meta['event_pre']}日と後{meta['event_post']}日で、その差分が標準偏差の何個分動いたか。"
        "プラスは事件後に上がった、マイナスは下がった。",
        "- 順位は |Δσ| の大きい順。**符号は問わない**（大きく動けば上位）。",
        f"- e_div = z(矛盾した輪の数) − z(穴の寿命の合計) は、表の「穴の寿命の合計 L¹ − 矛盾した輪の数（全部）」の**符号を逆にしたもの**。",
        "- 事件が5件しかないので、順位は目安。事件の種類で1位が変わる。",
        "",
        "## 事件一覧",
        "",
        "| 種類 | 日付 | 内容 |",
        "|---|---|---|",
    ]
    all_events: list[tuple[str, str]] = []
    for cat, ev in ((c, s["events"]) for c, s in d["shock_event_study"].items()):
        for e in ev:
            lines.append(f"| {CAT_JA[cat]} | {e['date']} | {e['label']} |")
            all_events.append((e["date"], cat))
    lines.append("")

    # e_div の順位まとめ
    lines += ["## e_div の順位（要約）", "", "| 事件の種類 | e_div の順位 | 1位の組み合わせ |", "|---|---|---|"]
    for cat, r in d["ediv_ranks"].items():
        top = r["top10_by_abs"][0]["pair"].replace(" - ", "__minus__")
        lines.append(f"| {CAT_JA[cat]}（{len(meta['events'][cat])}件） | {r['rank_by_abs']}位 / {r['n_pairs']} | {pair_label(top)} |")
    lines.append("")

    # 種類別の全55行
    for cat, s in d["shock_event_study"].items():
        dates = [e["date"] for e in s["events"]]
        rows = [(k, v["mean_dsigma"], v["per_event"]) for k, v in s["per_pair"].items()]
        lines += [f"## {CAT_JA[cat]}（{len(dates)}件）", ""] + table(rows, dates) + [""]

    # 5件すべての平均（派生）
    dates_all = [dt for dt, _ in all_events]
    merged: dict[str, dict] = {}
    for cat, s in d["shock_event_study"].items():
        for k, v in s["per_pair"].items():
            merged.setdefault(k, {}).update(v["per_event"])
    rows = [(k, sum(pe.values()) / len(pe), pe) for k, pe in merged.items()]
    lines += ["## 5件すべての平均（この表だけ、ここで計算した派生値）", ""] + table(rows, dates_all) + [""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {OUT.relative_to(ROOT)}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
