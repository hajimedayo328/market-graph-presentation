"""PCA 第1主成分寄与率と window 別相関の再計算検証スクリプト.

各期間の gamma timeseries CSV から L1_H1 と n_unb を取り出し,
標準化 (z-score) して 2x2 相関行列を作り, 固有値分解で第1主成分の
寄与率 (PC1 variance ratio) を計算する.
sklearn が無い環境でも動くよう np.linalg.eigh を使用する.

window 別相関は window=30 の gamma CSV しか存在しないため,
他 window (20/40/60) は元データが無く再現不可であることを記録する.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (記載ラベル, CSVファイル名, 論文記載 PC1, 論文記載 Pearson r)
DATASETS = [
    ("5y USA", "gamma_timeseries_w30.csv", 0.58, 0.16),
    ("10y USA", "gamma_timeseries_10y_w30.csv", 0.59, 0.19),
    ("15y USA", "gamma_timeseries_15y_w30.csv", 0.62, 0.26),
    ("20y USA", "gamma_timeseries_20y_w30.csv", None, None),  # 論文Table無し(参考)
    ("5y EM", "gamma_em_timeseries_w30.csv", 0.59, 0.17),
    ("5y China", "gamma_cn_timeseries_w30.csv", 0.71, 0.41),
]


def load_pairs(path):
    """L1_H1 と n_unb の両方が有効な行だけ取り出す."""
    l1, nunb = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = row.get("L1_H1", "").strip()
            b = row.get("n_unb", "").strip()
            if a == "" or b == "":
                continue
            try:
                fa = float(a)
                fb = float(b)
            except ValueError:
                continue
            if math.isnan(fa) or math.isnan(fb):
                continue
            l1.append(fa)
            nunb.append(fb)
    return np.array(l1), np.array(nunb)


def pca_pc1_ratio(x, y):
    """標準化した2変数の第1主成分寄与率を返す.

    標準化後の共分散行列 = 相関行列 (2x2).
    固有値の最大 / 合計 が第1主成分の寄与率.
    対角和(trace)=2 なので ratio = (1 + |r|) / 2 と一致する.
    """
    # z-score 標準化 (ddof=0: PCA慣例の母分散)
    xz = (x - x.mean()) / x.std(ddof=0)
    yz = (y - y.mean()) / y.std(ddof=0)
    data = np.column_stack([xz, yz])
    cov = np.cov(data, rowvar=False, ddof=0)  # 標準化済みなので相関行列
    eigvals = np.linalg.eigvalsh(cov)  # 昇順
    eigvals = np.sort(eigvals)[::-1]   # 降順
    ratio = float(eigvals[0] / eigvals.sum())
    return ratio, cov


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main():
    results = []
    print("=" * 78)
    print(f"{'Dataset':<10} {'n':>6} {'r(calc)':>9} {'r(paper)':>9} "
          f"{'PC1(calc)':>10} {'PC1(paper)':>11} {'match':>7}")
    print("-" * 78)
    for label, fname, pc1_paper, r_paper in DATASETS:
        path = DATA_DIR / fname
        x, y = load_pairs(path)
        n = len(x)
        r_calc = pearson(x, y)
        pc1_calc, cov = pca_pc1_ratio(x, y)
        # 照合 (小数2桁丸めで一致するか)
        pc1_match = (pc1_paper is not None
                     and round(pc1_calc, 2) == round(pc1_paper, 2))
        r_match = (r_paper is not None
                   and round(r_calc, 2) == round(r_paper, 2))
        print(f"{label:<10} {n:>6} {r_calc:>+9.4f} "
              f"{('%.2f' % r_paper) if r_paper is not None else '   -':>9} "
              f"{pc1_calc:>10.4f} "
              f"{('%.2f' % pc1_paper) if pc1_paper is not None else '    -':>11} "
              f"{('OK' if (pc1_match and r_match) else ('-' if pc1_paper is None else 'DIFF')):>7}")
        results.append({
            "dataset": label,
            "file": fname,
            "n": n,
            "pearson_r_calc": round(r_calc, 4),
            "pearson_r_paper": r_paper,
            "pearson_r_match_2dp": r_match,
            "pc1_variance_calc": round(pc1_calc, 4),
            "pc1_variance_paper": pc1_paper,
            "pc1_match_2dp": pc1_match,
            "correlation_matrix": [[round(v, 4) for v in row] for row in cov.tolist()],
        })
    print("=" * 78)

    out = {
        "description": (
            "PCA 第1主成分寄与率 (PC1 variance ratio) と Pearson 相関の再計算検証. "
            "L1_H1 と n_unb を z-score 標準化し, 2x2 相関行列の固有値分解で算出. "
            "標準化2変数では PC1 = (1+|r|)/2 が成立する."
        ),
        "method": "z-score standardize (ddof=0) -> 2x2 corr matrix -> np.linalg.eigvalsh -> max/sum",
        "window_correlation_note": (
            "index.html の 'Window 4 通り (20/30/40/60 日)' のうち, "
            "data/ に存在するのは window=30 の gamma CSV のみ. "
            "window=20/40/60 の gamma timeseries は元データが無く再現不可. "
            "再現できるのは window=30 の各市場・各期間の相関 (r=0.16〜0.41) と, "
            "10年データの r=0.19 (10y USA で再現確認)."
        ),
        "results": results,
    }
    out_path = DATA_DIR / "pca_variance_verify.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=> {out_path}")


if __name__ == "__main__":
    main()
