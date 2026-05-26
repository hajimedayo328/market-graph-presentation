"""
VPS の live_status.json (Pages 配布版, 直近 90 日) を集計し、
月次・累積の運用ログサマリーを data/live_summary.json に書き出す.

目的: 「実運用で 1 ヶ月どれだけ判定が当たったか」を可視化する素材.
データ少ない初期 (運用開始 2026-05-23) でも落ちないこと.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"

LIVE_PATH = DATA_DIR / "live_status.json"
OUT_PATH = DATA_DIR / "live_summary.json"

# 運用開始日 (note 表示用; CLAUDE 指示)
OPERATION_START = "2026-05-23"

CLASS_KEYS = ["政策ショック型", "強さ変化型", "一般ボラ上昇", "平常"]


def _safe_float(v: object) -> float | None:
    """値を float に変換する. 変換不可 or NaN は None を返す."""
    try:
        f = float(v)
        if f != f:  # NaN チェック
            return None
        return f
    except (TypeError, ValueError):
        return None


def _month_key(date_str: str) -> str | None:
    """'2026-05-23' -> '2026-05'. 失敗時 None."""
    if not date_str or len(date_str) < 7:
        return None
    return date_str[:7]


def _empty_summary(generated_at: str | None = None) -> dict:
    """データ未蓄積時の空サマリ dict を返す."""
    return {
        "generated_at": generated_at or datetime.now().isoformat(),
        "operation_start": OPERATION_START,
        "data_available": False,
        "note": "VPS 運用開始直後のためデータ蓄積中。月次集計は 1 ヶ月分の判定が貯まり次第表示。",
        "monthly": [],
        "cumulative": {
            "n_observed_days": 0,
            "n_classification_changes": 0,
            "n_trades": 0,
            "most_frequent_classification": None,
            "max_e_div": None,
            "max_e_div_date": None,
            "min_e_div": None,
            "min_e_div_date": None,
            "classification_distribution": {k: 0 for k in CLASS_KEYS},
        },
    }


def aggregate(live: dict) -> dict:
    """live_status.json の内容を月次・累積サマリに集計する."""
    gamma_rows = live.get("gamma_daily_recent") or []
    changes = live.get("classification_changes") or []
    trades = live.get("trades") or []
    generated_at = live.get("generated_at")

    if not gamma_rows:
        return _empty_summary(generated_at)

    # --- 月別バケット ---
    monthly_buckets: dict[str, dict] = defaultdict(lambda: {
        "month": "",
        "n_days": 0,
        "classification_distribution": {k: 0 for k in CLASS_KEYS},
        "e_div_values": [],
        "n_changes": 0,
        "n_trades": 0,
    })

    # gamma_daily から月別判定分布 + e_div を集計
    for row in gamma_rows:
        date = row.get("date")
        m = _month_key(date)
        if not m:
            continue
        bucket = monthly_buckets[m]
        bucket["month"] = m
        bucket["n_days"] += 1
        cls = row.get("classification")
        if cls in bucket["classification_distribution"]:
            bucket["classification_distribution"][cls] += 1
        else:
            # 未知判定はスキップ (将来増えても落ちないように)
            pass
        ediv = _safe_float(row.get("e_div"))
        if ediv is not None:
            bucket["e_div_values"].append((date, ediv))

    # classification_changes 月別カウント
    for ch in changes:
        date = ch.get("date") or (ch.get("changed_at") or "")[:10]
        m = _month_key(date)
        if not m:
            continue
        monthly_buckets[m]["n_changes"] += 1

    # trades 月別カウント (executed_at の先頭 7 文字)
    for t in trades:
        ts = t.get("executed_at") or ""
        m = _month_key(ts)
        if not m:
            continue
        # gamma が無い月でもトレードがあれば計上できるよう、bucket を初期化
        monthly_buckets[m]["n_trades"] += 1
        monthly_buckets[m]["month"] = m

    # 月別配列を仕上げ (e_div 統計を確定 / temp フィールドを除去)
    monthly = []
    for m, b in sorted(monthly_buckets.items()):
        evals = [v for (_d, v) in b["e_div_values"]]
        e_avg = round(mean(evals), 4) if evals else None
        e_max = round(max(evals), 4) if evals else None
        e_min = round(min(evals), 4) if evals else None
        monthly.append({
            "month": m,
            "n_days": b["n_days"],
            "classification_distribution": b["classification_distribution"],
            "e_div_avg": e_avg,
            "e_div_max": e_max,
            "e_div_min": e_min,
            "n_changes": b["n_changes"],
            "n_trades": b["n_trades"],
        })

    # --- 累積 ---
    cls_dist = {k: 0 for k in CLASS_KEYS}
    for row in gamma_rows:
        c = row.get("classification")
        if c in cls_dist:
            cls_dist[c] += 1

    most_frequent = None
    if any(cls_dist.values()):
        most_frequent = Counter(cls_dist).most_common(1)[0][0]

    # 全期間 e_div 最大/最小 (日付付き)
    all_ediv = []
    for row in gamma_rows:
        ediv = _safe_float(row.get("e_div"))
        if ediv is not None:
            all_ediv.append((row.get("date"), ediv))

    max_pair = max(all_ediv, key=lambda x: x[1]) if all_ediv else (None, None)
    min_pair = min(all_ediv, key=lambda x: x[1]) if all_ediv else (None, None)

    cumulative = {
        "n_observed_days": len(gamma_rows),
        "n_classification_changes": len(changes),
        "n_trades": len(trades),
        "most_frequent_classification": most_frequent,
        "max_e_div": round(max_pair[1], 4) if max_pair[1] is not None else None,
        "max_e_div_date": max_pair[0],
        "min_e_div": round(min_pair[1], 4) if min_pair[1] is not None else None,
        "min_e_div_date": min_pair[0],
        "classification_distribution": cls_dist,
    }

    # データ蓄積中フラグ: 観測日数 30 未満は "まだ少ない" 扱い
    data_available = True
    note = None
    if cumulative["n_observed_days"] < 30:
        note = (
            f"運用開始 {OPERATION_START} 直後のためデータ蓄積中 "
            f"(現在 {cumulative['n_observed_days']} 日分)。"
            "1 ヶ月分が貯まると月次集計の信頼度が上がる。"
        )

    return {
        "generated_at": generated_at or datetime.now().isoformat(),
        "operation_start": OPERATION_START,
        "data_available": data_available,
        "note": note,
        "monthly": monthly,
        "cumulative": cumulative,
    }


def main():
    """live_status.json を読み込み live_summary.json に書き出す."""
    if not LIVE_PATH.exists():
        print(f"[aggregate] live_status.json not found: {LIVE_PATH}")
        # 空サマリーでも書いて Pages 側で落ちないようにする
        summary = _empty_summary()
    else:
        try:
            live = json.loads(LIVE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[aggregate] failed to parse live_status.json: {e}")
            summary = _empty_summary()
        else:
            summary = aggregate(live)

    OUT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[aggregate] wrote: {OUT_PATH}")
    print(f"  monthly entries     : {len(summary['monthly'])}")
    print(f"  observed days       : {summary['cumulative']['n_observed_days']}")
    print(f"  classification changes: {summary['cumulative']['n_classification_changes']}")
    print(f"  trades              : {summary['cumulative']['n_trades']}")
    print(f"  most freq class     : {summary['cumulative']['most_frequent_classification']}")
    print(f"  max e_div           : {summary['cumulative']['max_e_div']} ({summary['cumulative']['max_e_div_date']})")


if __name__ == "__main__":
    main()
