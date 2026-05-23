"""
VPS の SQLite DB → JSON エクスポート + GitHub Contents API で push.

毎日の vps_daily.py 実行後に呼ばれ、Pages に最新データを反映.

GitHub PAT が .env に必要:
  GITHUB_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
  GITHUB_OWNER=hajimedayo328
  GITHUB_REPO=market-graph-presentation
"""
from __future__ import annotations

import base64
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import requests

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}
    env = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [publish] {msg}"
    print(line)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "vps_publish.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def export_db_to_json() -> dict:
    """DB から live_status.json 用のデータを抽出."""
    db_path = DATA_DIR / "market_graph.db"
    if not db_path.exists():
        log(f"DB not found: {db_path}")
        return {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 直近 90 日の gamma_daily
    c.execute("""
    SELECT date, L1_H1, n_unb, n_edges, balance_rate, z_L1, z_unb, e_div, classification, computed_at
    FROM gamma_daily
    ORDER BY date DESC LIMIT 90
    """)
    gamma_rows = [dict(r) for r in c.fetchall()]

    # 最新
    latest = gamma_rows[0] if gamma_rows else None

    # 判定変化履歴
    c.execute("""
    SELECT changed_at, date, prev_class, new_class, e_div, L1_H1, n_unb
    FROM classification_changes
    ORDER BY changed_at DESC LIMIT 50
    """)
    changes_rows = [dict(r) for r in c.fetchall()]

    # トレード履歴
    c.execute("""
    SELECT trade_id, executed_at, symbol, side, volume, classification_trigger, mt5_ticket, note
    FROM trades
    ORDER BY executed_at DESC LIMIT 100
    """)
    trades_rows = [dict(r) for r in c.fetchall()]
    n_trades = len(trades_rows)

    conn.close()
    return {
        "generated_at": datetime.now().isoformat(),
        "latest": latest,
        "gamma_daily_recent": gamma_rows,
        "classification_changes": changes_rows,
        "trades": trades_rows,
        "summary": {
            "n_days": len(gamma_rows),
            "n_changes": len(changes_rows),
            "n_trades": n_trades,
        },
    }


def push_to_github(content_json: dict, env: dict, target_path: str = "data/live_status.json") -> bool:
    """GitHub Contents API でファイルを更新 (新規作成 or upsert)."""
    pat = env.get("GITHUB_PAT")
    owner = env.get("GITHUB_OWNER", "hajimedayo328")
    repo = env.get("GITHUB_REPO", "market-graph-presentation")
    if not pat:
        log("GITHUB_PAT not set in .env, skipping GitHub push")
        return False
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{target_path}"
    headers = {
        "Authorization": f"token {pat}",
        "Accept": "application/vnd.github+json",
    }
    # 既存ファイル取得 (SHA 必要)
    sha = None
    r = requests.get(api_url, headers=headers, timeout=15)
    if r.status_code == 200:
        sha = r.json().get("sha")
        log(f"Existing file found, sha={sha[:7]}")
    elif r.status_code == 404:
        log("File not found, creating new")
    else:
        log(f"GET failed: {r.status_code} {r.text[:200]}")
        return False

    content_b64 = base64.b64encode(
        json.dumps(content_json, ensure_ascii=False, indent=1).encode("utf-8")
    ).decode("ascii")
    payload = {
        "message": f"chore(live): VPS publish {datetime.now().strftime('%Y-%m-%d %H:%M JST')}",
        "content": content_b64,
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(api_url, headers=headers, json=payload, timeout=30)
    if r.status_code in (200, 201):
        commit = r.json().get("commit", {}).get("sha", "?")[:7]
        log(f"GitHub push OK: commit {commit}")
        return True
    else:
        log(f"PUT failed: {r.status_code} {r.text[:300]}")
        return False


def main():
    log("=" * 60)
    log("VPS publish job started")
    env = load_env()
    data = export_db_to_json()
    if not data:
        log("No data to publish")
        return
    log(f"Exported: {data['summary']}")
    # ローカルにも保存 (デバッグ用)
    local_out = DATA_DIR / "live_status.json"
    local_out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Saved local: {local_out}")
    # GitHub push
    pushed = push_to_github(data, env)
    log(f"Publish {'succeeded' if pushed else 'skipped'}")
    log("=" * 60)


if __name__ == "__main__":
    main()
