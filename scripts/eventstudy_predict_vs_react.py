"""
e_div は暴落を「予知」しているのか、「反応」しているだけなのか
================================================================

研究の核心的な穴を検証する:
  e_div (= z_unb - z_L1, 矛盾サイクル超過度) が高い局面は確かに
  S&P500 の下落と重なる。だが因果の向きが問題:
    (a) 下落の「前」に e_div が立ち上がる  -> 前兆 (予知) 指標
    (b) 下落と「同時/後」に e_div が立ち上がる -> 反応 (初動検知) 指標

  リスク管理ツールとしての価値は (a) と (b) で大きく変わる。
  本スクリプトは「e_div の立ち上がり日」と「下落の起点日」の
  時間差 (ラグ) を複数イベントで測り、分布を集計する。

ラグの定義:
  lag = (e_div が +0.8 を最初に超えた日) − (下落開始日)
    lag < 0  ... 下落より前に e_div 上昇 = 予知 (lead)
    lag ~ 0  ... 同時 = 反応の初動 (coincident)
    lag > 0  ... 下落が始まってから上昇 = 純粋な反応 (lag)

look-ahead 厳密排除:
  e_div の z-score は過去のみ expanding window (min_periods=90)。
  「下落起点」は forward 20d リターンで定義するが、これは
  分析側 (ex-post) の評価ラベルであり e_div の計算には一切入らない。

下落イベントの定義 (2 系統):
  (1) data-driven: forward 20d リターンが下位 DRAWDOWN_PCTILE 以下に
      落ちる「トリガー日」を検出し、連続するトリガー日を 1 イベントに
      まとめる (gap > MERGE_GAP_BDAYS で分割)。各イベントの
        - 下落開始日 = クラスタ内で最初のトリガー日
        - 下落の底  = 開始日から FWD_HORIZON 日以内の最安値日
  (2) known: 既知の暴落 (COVID, 2022 利上げ, 2025 関税 等) を
      明示リストで併記し、data-driven と相互検証する。

出力:
  data/eventstudy_predict_vs_react.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

OUT_JSON = DATA_DIR / "eventstudy_predict_vs_react.json"

# ---- パラメータ ----
EXPANDING_MIN_PERIODS = 90      # z-score の最小サンプル (look-ahead 回避)
EDIV_THRESHOLD = 0.8            # e_div の「立ち上がり」しきい値 (+0.8σ)
FWD_HORIZON = 20               # forward リターンのホライズン (営業日)
DRAWDOWN_PCTILE = 0.10         # 下位 10% を「下落局面」と定義
MERGE_GAP_BDAYS = 15           # トリガー日の間隔がこれ以下なら同一イベント

# --- エピソード結合パラメータ (far-left バイアス回避) ---
# e_div は +0.8 超えが全体の ~24% と高頻度のため、単純な「窓内最初の超過日」
# だと無関係な過去のエピソードを拾い、見かけ上の lead を量産する。
# そこで「下落近傍でエピソードがアクティブか」を先に判定し、アクティブな
# 場合のみその塊を過去へ遡って起点を取る方式に変更。
NEAR_BDAYS = 5                 # 下落開始の前後この範囲に超過があれば「アクティブ」
FWD_REACT_BDAYS = 20           # 近傍に無い場合、下落開始後この範囲で初超過 (反応)
MAX_LOOKBACK_BDAYS = 30        # エピソード起点を遡る上限 (1.5ヶ月)
ALLOW_GAP = 3                  # エピソード遡り中に許容する閾値割れ日数
PEAK_LOOKBACK_BDAYS = 25       # data-driven start 直前の局所ピーク探索範囲

# 既知の暴落 (下落開始日 = 一般に知られる起点。底は data から自動算出)
KNOWN_CRASHES = [
    {"label": "COVID クラッシュ", "start": "2020-02-20"},
    {"label": "2022 利上げ下落 (年初)", "start": "2022-01-04"},
    {"label": "2022 夏〜秋 弱気相場", "start": "2022-08-16"},
    {"label": "2023 地銀危機 (SVB)", "start": "2023-03-08"},
    {"label": "2024-08 円キャリー巻き戻し", "start": "2024-07-16"},
    {"label": "2025-04 相互関税 (Liberation Day)", "start": "2025-02-19"},
]


def expanding_zscore(s: pd.Series, min_periods: int = EXPANDING_MIN_PERIODS) -> pd.Series:
    """過去のみ expanding window の z-score (look-ahead なし)。"""
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=1)
    return (s - mu) / sd


def load_ediv(gamma_csv: Path) -> pd.Series:
    """gamma timeseries から e_div = z_unb - z_L1 を構築。"""
    df = pd.read_csv(gamma_csv, parse_dates=["date"]).sort_values("date").set_index("date")
    df = df.dropna(subset=["L1_H1", "n_unb"]).copy()
    z_L1 = expanding_zscore(df["L1_H1"])
    z_unb = expanding_zscore(df["n_unb"])
    e_div = (z_unb - z_L1).dropna()
    e_div.name = "e_div"
    return e_div


def load_sp500(parquet: Path, start: str | None = None) -> pd.Series:
    """SP500 終値 (^GSPC) を取り、実取引日のみ (NaN 除去)。"""
    df = pd.read_parquet(parquet)
    sp = df["SP500"].dropna().sort_index()
    if start is not None:
        sp = sp[sp.index >= pd.Timestamp(start)]
    sp.name = "SP500"
    return sp


def episode_cross_date(e_div: pd.Series,
                       anchor: pd.Timestamp,
                       threshold: float,
                       near_bdays: int,
                       fwd_bdays: int,
                       max_lookback_bdays: int,
                       allow_gap: int) -> tuple[pd.Timestamp | None, str]:
    """
    下落開始日 (anchor) に「結びつく」e_div 上昇エピソードの起点
    (threshold を最初に超えた日) を返す。far-left バイアスを避けるため、
    無関係な過去のエピソードは拾わない。

    手順:
      1. anchor の前後 near_bdays 以内に e_div >= threshold の日があるか調べる
         (= 下落の近傍でエピソードが「アクティブ」か)。
      2. アクティブなら、その超過日の塊から anchor に最も近いものを起点に
         して「過去方向」へ連続超過 (allow_gap 営業日までの小欠落許容) を
         遡り、エピソード起点 (最初の上抜けエッジ) を確定する。
         遡りは max_lookback_bdays までに制限。
      3. anchor 近傍にエピソードが無い場合のみ、anchor から fwd_bdays
         先までを見て「下落が始まってから」初めて超える日を探す
         (= 反応 / 遅行)。これも無ければ no_cross。

    戻り値: (cross_date or None, mode)
      mode: "episode" (近傍エピソードの起点を遡って確定)
            "forward" (下落開始後に初めて超過)
            "none"    (近傍にも前方にも超過なし)
    """
    s = e_div.dropna()
    if anchor < s.index.min() or anchor > s.index.max():
        return None, "none"
    n = len(s)
    pos = s.index.get_indexer([anchor], method="nearest")[0]
    above = (s >= threshold).to_numpy()

    # --- 1. anchor 近傍 (±near_bdays) でエピソードがアクティブか ---
    lo_near = max(0, pos - near_bdays)
    hi_near = min(n, pos + near_bdays + 1)
    near_idx = [i for i in range(lo_near, hi_near) if above[i]]

    if near_idx:
        # anchor に最も近い超過日を起点に、過去方向へエピソードを遡る
        seed = min(near_idx, key=lambda i: abs(i - pos))
        j = seed
        gap = 0
        onset = seed
        limit = max(0, pos - max_lookback_bdays)
        while j - 1 >= limit:
            if above[j - 1]:
                onset = j - 1
                j -= 1
                gap = 0
            else:
                gap += 1
                if gap > allow_gap:
                    break
                j -= 1
        # onset を「上抜けエッジ」に丸める: onset 直前が threshold 未満であること
        # (onset が above の連続塊の先頭になっているはず)
        return s.index[onset], "episode"

    # --- 2. 近傍に無し -> 下落開始後 fwd_bdays 先で初超過を探す (反応) ---
    hi_fwd = min(n, pos + fwd_bdays + 1)
    for i in range(pos, hi_fwd):
        if above[i]:
            return s.index[i], "forward"

    return None, "none"


def trough_date(sp: pd.Series, start: pd.Timestamp, horizon: int) -> tuple[pd.Timestamp | None, float | None]:
    """下落開始日から horizon 営業日以内の最安値日とそのドローダウン%。"""
    s = sp.dropna()
    if start < s.index.min() or start > s.index.max():
        return None, None
    pos = s.index.get_indexer([start], method="nearest")[0]
    end = min(pos + horizon, len(s))
    seg = s.iloc[pos:end]
    if len(seg) < 2:
        return None, None
    tdate = seg.idxmin()
    dd = float(seg.min() / s.iloc[pos] - 1.0)
    return tdate, dd


def bday_lag(e_div: pd.Series, d1: pd.Timestamp, d0: pd.Timestamp) -> int:
    """
    e_div の取引日 index 上で d1 − d0 の営業日数を返す (符号付き)。
    両方を index 上の最近傍位置に丸めてから差を取る。
    """
    idx = e_div.dropna().index
    p1 = idx.get_indexer([d1], method="nearest")[0]
    p0 = idx.get_indexer([d0], method="nearest")[0]
    return int(p1 - p0)


def detect_drawdown_events(sp: pd.Series,
                           pctile: float,
                           horizon: int,
                           merge_gap: int) -> list[dict]:
    """
    forward horizon リターンが下位 pctile 以下になる「トリガー日」を検出し、
    連続するトリガー日を 1 イベントにまとめる。
    各イベント: 下落開始日 (最初のトリガー日) + 底 + ドローダウン。
    """
    s = sp.dropna()
    fwd = s.shift(-horizon) / s - 1.0
    fwd = fwd.dropna()
    thr = fwd.quantile(pctile)
    trig = fwd[fwd <= thr]
    if trig.empty:
        return []
    # 取引日 index 上の位置でクラスタ化
    idx = s.index
    pos_map = {d: i for i, d in enumerate(idx)}
    trig_pos = sorted(pos_map[d] for d in trig.index)
    clusters: list[list[int]] = [[trig_pos[0]]]
    for p in trig_pos[1:]:
        if p - clusters[-1][-1] <= merge_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    events = []
    for cl in clusters:
        start = idx[cl[0]]
        spos = cl[0]
        # 真の起点 (= 直近の局所ピーク) を start 直前 PEAK_LOOKBACK_BDAYS 内で取る。
        # forward-return ベースの start は下落の途中を指しがちで、これが
        # 見かけ上の lead を生む。ピーク基準だと公平な反応/予知判定ができる。
        lo_peak = max(0, spos - PEAK_LOOKBACK_BDAYS)
        pre = s.iloc[lo_peak:spos + 1]
        peak_date = pre.idxmax()
        tdate, dd = trough_date(s, start, horizon + (cl[-1] - cl[0]))
        # ピーク起点のドローダウン (ピーク→底)
        dd_from_peak = None
        if tdate is not None:
            dd_from_peak = round(float(s.loc[tdate] / s.loc[peak_date] - 1.0) * 100, 2)
        events.append({
            "start": str(start.date()),
            "peak": str(peak_date.date()),
            "peak_to_start_bdays": int(spos - idx.get_indexer([peak_date], method="nearest")[0]),
            "n_trigger_days": len(cl),
            "trough": (str(tdate.date()) if tdate is not None else None),
            "drawdown_pct": (round(dd * 100, 2) if dd is not None else None),
            "drawdown_from_peak_pct": dd_from_peak,
            "fwd_ret_at_start_pct": round(float(fwd.loc[start]) * 100, 2),
        })
    return events


def measure_event(e_div: pd.Series,
                  sp: pd.Series,
                  start_str: str,
                  label: str | None,
                  trough_str: str | None = None,
                  dd_pct: float | None = None) -> dict:
    """1 イベントの e_div 立ち上がり日とラグを測定。"""
    start = pd.Timestamp(start_str)
    cross, cross_mode = episode_cross_date(
        e_div, start, EDIV_THRESHOLD,
        near_bdays=NEAR_BDAYS, fwd_bdays=FWD_REACT_BDAYS,
        max_lookback_bdays=MAX_LOOKBACK_BDAYS, allow_gap=ALLOW_GAP)
    s = e_div.dropna()
    pos = s.index.get_indexer([start], method="nearest")[0]
    out: dict = {
        "label": label,
        "drawdown_start": start_str,
        "trough": trough_str,
        "drawdown_pct": dd_pct,
        "ediv_at_start": round(float(s.iloc[pos]), 3),
        "ediv_cross_date": (str(cross.date()) if cross is not None else None),
        "cross_mode": cross_mode,
    }
    if cross is None:
        out["lag_bdays"] = None
        out["classification"] = "no_cross"
        return out
    lag = bday_lag(e_div, cross, start)
    out["lag_bdays"] = lag
    out["ediv_at_cross"] = round(float(s.loc[cross]), 3)
    if lag <= -3:
        cls = "lead"        # 予知
    elif lag <= 3:
        cls = "coincident"  # 同時
    else:
        cls = "lag"         # 反応
    out["classification"] = cls
    return out


def summarize(events: list[dict]) -> dict:
    lags = [e["lag_bdays"] for e in events if e.get("lag_bdays") is not None]
    n_total = len(events)
    n_cross = len(lags)
    n_nocross = n_total - n_cross
    if not lags:
        return {
            "n_events": n_total, "n_with_cross": 0, "n_no_cross": n_nocross,
            "lag_median": None, "lag_mean": None,
            "n_lead": 0, "n_coincident": 0, "n_lag": 0, "verdict": "insufficient",
        }
    arr = np.array(lags, dtype=float)
    n_lead = sum(1 for e in events if e.get("classification") == "lead")
    n_coin = sum(1 for e in events if e.get("classification") == "coincident")
    n_lag = sum(1 for e in events if e.get("classification") == "lag")
    # 判定 (過大主張を避けるため、明確な優勢がある時だけ predictive/reactive)
    #   - lead/lag の差が cross 件数の 1/3 以上、かつ中央値の符号が一致したときのみ断定
    #   - それ以外 (中央値 ~0、lead と lag が拮抗) は mixed
    med = float(np.median(arr))
    margin = max(2, n_cross // 3)   # 拮抗とみなさない最小差 (最低 2 件)
    if (n_lead - n_lag) >= margin and med <= -2:
        verdict = "predictive"      # 予知型優勢
    elif (n_lag - n_lead) >= margin and med >= 2:
        verdict = "reactive"        # 反応型優勢
    else:
        verdict = "mixed"           # 混在 (拮抗・中央値ほぼ0)
    return {
        "n_events": n_total,
        "n_with_cross": n_cross,
        "n_no_cross": n_nocross,
        "lag_median": round(med, 1),
        "lag_mean": round(float(np.mean(arr)), 1),
        "lag_min": int(arr.min()),
        "lag_max": int(arr.max()),
        "n_lead": n_lead,
        "n_coincident": n_coin,
        "n_lag": n_lag,
        "verdict": verdict,
    }


def run_one(gamma_csv: Path, sp_parquet: Path, sp_start: str | None,
            label_tag: str) -> dict:
    e_div = load_ediv(gamma_csv)
    sp = load_sp500(sp_parquet, start=sp_start)
    # e_div と SP500 の共通有効レンジに揃える
    lo = max(e_div.index.min(), sp.index.min())
    hi = min(e_div.index.max(), sp.index.max())
    print(f"[{label_tag}] e_div {e_div.index.min().date()}->{e_div.index.max().date()} "
          f"({e_div.notna().sum()}), SP500 {sp.index.min().date()}->{sp.index.max().date()}; "
          f"common {lo.date()}->{hi.date()}")

    # ---- (1) data-driven イベント ----
    # start (forward-return ベース) と peak (真の局所ピーク) の 2 アンカーで測る。
    # start アンカーは下落途中を指しがちで見かけ上 lead を膨らませるため、
    # peak アンカーの方が公平な反応/予知判定になる。
    dd_events_raw = detect_drawdown_events(sp[(sp.index >= lo) & (sp.index <= hi)],
                                           DRAWDOWN_PCTILE, FWD_HORIZON, MERGE_GAP_BDAYS)
    dd_measured = []          # start アンカー
    dd_peak_measured = []     # peak アンカー (公平)
    for ev in dd_events_raw:
        m = measure_event(e_div, sp, ev["start"], None,
                          trough_str=ev["trough"], dd_pct=ev["drawdown_pct"])
        m["peak"] = ev["peak"]
        m["peak_to_start_bdays"] = ev["peak_to_start_bdays"]
        m["n_trigger_days"] = ev["n_trigger_days"]
        m["fwd_ret_at_start_pct"] = ev["fwd_ret_at_start_pct"]
        dd_measured.append(m)

        mp = measure_event(e_div, sp, ev["peak"], None,
                           trough_str=ev["trough"], dd_pct=ev["drawdown_from_peak_pct"])
        mp["forward_return_start"] = ev["start"]
        mp["n_trigger_days"] = ev["n_trigger_days"]
        dd_peak_measured.append(mp)

    # ---- (2) known crashes ----
    known_measured = []
    for kc in KNOWN_CRASHES:
        start = pd.Timestamp(kc["start"])
        if start < lo or start > hi:
            continue
        tdate, dd = trough_date(sp, start, 90)  # 既知暴落は底まで長め
        m = measure_event(e_div, sp, kc["start"], kc["label"],
                          trough_str=(str(tdate.date()) if tdate is not None else None),
                          dd_pct=(round(dd * 100, 2) if dd is not None else None))
        known_measured.append(m)

    return {
        "data_driven": {
            "params": {
                "fwd_horizon_bdays": FWD_HORIZON,
                "drawdown_pctile": DRAWDOWN_PCTILE,
                "merge_gap_bdays": MERGE_GAP_BDAYS,
            },
            "anchor_note": ("start = forward-return ベースの起点 (下落途中を指しがち)、"
                            "peak = その直前の局所価格ピーク (真の起点・公平)。"),
            "n_events": len(dd_measured),
            "events": dd_measured,
            "summary": summarize(dd_measured),
            "events_peak_anchored": dd_peak_measured,
            "summary_peak_anchored": summarize(dd_peak_measured),
        },
        "known_crashes": {
            "n_events": len(known_measured),
            "events": known_measured,
            "summary": summarize(known_measured),
        },
        "data_range": [str(lo.date()), str(hi.date())],
    }


def main() -> None:
    print("=== e_div: 予知 vs 反応 イベントスタディ ===\n")

    runs = {}
    # 5y
    runs["5y"] = run_one(
        DATA_DIR / "gamma_timeseries_w30.csv",
        DATA_DIR / "ohlc_40.parquet",
        None, "5y")
    # 20y (universe が揃う 2017-11 以降に限定)
    runs["20y_oos"] = run_one(
        DATA_DIR / "gamma_timeseries_20y_w30.csv",
        DATA_DIR / "ohlc_40_20y.parquet",
        "2017-11-01", "20y")

    # ---- 全体集計 ----
    # (A) 公平版 (headline): peak アンカーの data-driven + known crashes。
    #     真の起点 (価格ピーク / 市場が認識する起点) で測るため、これが本命の答え。
    # (B) 楽観版 (参考): start アンカー (forward-return ベース) の data-driven。
    #     下落途中を起点にするため lead 寄りに偏る。透明性のため併記。
    fair_events = []
    optimistic_events = []
    for run in runs.values():
        fair_events += run["data_driven"]["events_peak_anchored"]
        fair_events += run["known_crashes"]["events"]
        optimistic_events += run["data_driven"]["events"]
    overall_fair = summarize(fair_events)
    overall_optimistic = summarize(optimistic_events)

    out = {
        "meta": {
            "title": "e_div は暴落を予知しているか、反応しているだけか",
            "ediv_definition": "e_div = z_unb - z_L1 (両者とも過去のみ expanding z-score, min_periods=90)",
            "lookahead_note": ("e_div の z-score は過去のみ expanding window で計算し look-ahead を排除。"
                               "下落起点は forward 20d リターン (ex-post ラベル) で定義するが、"
                               "これは評価用であり e_div の計算には一切入らない。"),
            "lag_definition": ("lag = (下落近傍でアクティブな e_div 上昇エピソードの起点) − (下落開始日), 単位=営業日。"
                               "e_div は +0.8 超えが全体の約24%と高頻度なため、単純な窓内最初超過では"
                               "無関係な過去エピソードを拾い見かけ上の lead を量産する。これを避けるため "
                               "(a) 下落開始の前後 NEAR_BDAYS にエピソードがアクティブか判定し、"
                               "(b) アクティブな場合のみその塊を過去へ MAX_LOOKBACK_BDAYS まで遡って起点を取り、"
                               "(c) 近傍に無い場合のみ下落開始後 FWD_REACT_BDAYS 先で初超過を探す。"),
            "classification_rule": "lag <= -3: lead(予知) / -3<lag<=3: coincident(同時) / lag>3: lag(反応)",
            "ediv_threshold": EDIV_THRESHOLD,
            "expanding_min_periods": EXPANDING_MIN_PERIODS,
            "near_bdays": NEAR_BDAYS,
            "fwd_react_bdays": FWD_REACT_BDAYS,
            "max_lookback_bdays": MAX_LOOKBACK_BDAYS,
            "allow_gap_bdays": ALLOW_GAP,
        },
        "runs": runs,
        "overall_summary_fair": overall_fair,
        "overall_summary_optimistic": overall_optimistic,
        "overall_summary_note": ("fair = 真の起点 (価格ピーク + 既知暴落の市場起点) で測った本命の結論。"
                                 "optimistic = forward-return ベースの起点 (下落途中) で測った参考値で lead 寄りに偏る。"),
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {OUT_JSON}")

    # コンソール要約
    def pr(tag, s):
        print(f"[{tag}] n={s['n_events']} cross={s['n_with_cross']} "
              f"median={s['lag_median']} mean={s['lag_mean']} "
              f"lead={s['n_lead']} coin={s['n_coincident']} lag={s['n_lag']} "
              f"-> {s['verdict']}")
    print("\n--- 集計 (per run) ---")
    for k, run in runs.items():
        pr(f"{k}/data_driven(start)", run["data_driven"]["summary"])
        pr(f"{k}/data_driven(peak) ", run["data_driven"]["summary_peak_anchored"])
        pr(f"{k}/known_crashes     ", run["known_crashes"]["summary"])
    print("\n--- OVERALL ---")
    pr("FAIR (peak+known)      ", overall_fair)
    pr("OPTIMISTIC (start)     ", overall_optimistic)


if __name__ == "__main__":
    main()
