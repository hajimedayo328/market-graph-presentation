# Market Graph Research

> 圏論 × グラフ理論で金融市場ネットワークの **位相変化** を観察し、
> ショックの種類を分類する研究プロジェクト。

## 概要

株式市場の相関ネットワークを「符号付きグラフ」と見なし、
2 種類の **位相不変量** を毎日計算する:

- **L¹ ノルム of H₁** (`L1`) — 持続ホモロジーで測る相関構造の "強さ"
- **不整合サイクル数** (`n_unb`) — Heider balance に基づく "符号" の崩れ

両者の発散指標 `e_div = z(n_unb) − z(L¹)` でショックタイプを
**3 区分 (政策 / 構造変化 / 一般ボラ)** に分類する。

## 公開 URL

| | |
|---|---|
| 研究内容 (HTML) | https://hajimedayo328.github.io/market-graph-presentation/ |
| 実装デモ (リアルタイム + バックテスト) | https://hajimedayo328.github.io/market-graph-presentation/app.html |
| Paper PDF | [./docs/paper/main.pdf](./docs/paper/main.pdf) |
| Live ステータス JSON | [./data/live_status.json](./data/live_status.json) |

## 何ができるか

1. **2 つの位相不変量を計算** — `L¹ of H₁` と `n_unb` は corr ≈ 0.16 で **独立に動く**
2. **e_div でショック分類** — 政策ショック (`e_div ≥ +0.8`) と構造変化 (`e_div ≤ −0.5`) を判別
3. **VPS で毎日自動運用** — JST 07:00 に計算 → 判定 → Vantage デモ MT5 で paper trading → Pages 更新

## リポジトリ構成

```
.
├── index.html                  # 研究内容 (説明 + 可視化)
├── app.html                    # 実装デモ (リアルタイム + バックテスト)
├── data/                       # 計算結果 (parquet / csv / json)
│   ├── ohlc_40.parquet         #   40 銘柄 OHLC (5y/10y/15y/20y)
│   ├── gamma_timeseries_w30.csv#   γ (L¹, n_unb) 時系列
│   ├── multi_indicators_w30.csv#   12 指標分解
│   ├── backtest_*_results.json #   バックテスト結果
│   ├── live_status.json        #   VPS 最新ステータス
│   └── live_summary.json       #   実運用ログ月次・累積集計
├── scripts/
│   ├── update_data.py          #   yfinance データ取得
│   ├── compute_gamma_em.py     #   新興国市場版
│   ├── backtest.py             #   6 戦略 × 2 方向
│   ├── backtest_v2.py          #   改良版 (look-ahead 修正)
│   ├── backtest_walkforward*.py#   Walk-forward OOS 検証 (5/10/15/20 年)
│   ├── vps_daily.py            #   VPS 日次パイプライン本体
│   ├── vps_publish.py          #   結果を Pages に push
│   ├── trade_executor.py       #   MT5 paper trading 実行
│   ├── aggregate_live_log.py   #   実運用ログ集計
│   ├── health_check.py         #   VPS 停止検知
│   ├── build_index.py          #   研究 HTML 生成
│   ├── build_app.py            #   デモ HTML 生成
│   ├── build_paper_figs.py     #   論文用図版生成
│   └── lib/                    #   位相不変量計算コア
│       ├── homology.py
│       ├── persistent_homology.py
│       ├── market_category.py
│       ├── compute_gamma_timeseries.py
│       └── compute_multi_indicators.py
├── docs/paper/                 # LaTeX 論文 (自動ビルド PDF)
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   └── figs/
├── tests/                      # pytest スモークテスト
├── .github/workflows/
│   ├── weekly_update.yml       #   週次データ更新 (補助)
│   ├── build_paper.yml         #   docs/paper/ 変更時に PDF 自動ビルド
│   └── test.yml                #   push 時 pytest
├── requirements.txt            # 実行依存
├── requirements-dev.txt        # テスト依存
├── CITATION.cff
└── LICENSE                     # CC BY 4.0
```

## 自動化

| 何が | いつ | 何を |
|---|---|---|
| **VPS 日次パイプライン** | 毎日 JST 07:00 | データ取得 → γ 計算 → 判定 → Vantage デモ MT5 で paper trade → `data/live_status.json` 更新 → Pages へ push |
| **LaTeX ビルド** (`build_paper.yml`) | `docs/paper/**` への push | xelatex で `main.pdf` を自動再生成しコミット |
| **pytest CI** (`test.yml`) | push / PR | `requirements-dev.txt` でテスト実行 |
| **週次更新 GitHub Actions** (`weekly_update.yml`) | 毎週月曜 JST 07:00 (補助) | yfinance 再取得 + HTML 再生成 |

VPS 停止は `scripts/health_check.py` が検出し、Pages に警告表示する。

## 再現方法 (最小手順)

```bash
git clone https://github.com/hajimedayo328/market-graph-presentation.git
cd market-graph-presentation

pip install -r requirements.txt

# 1. 40 銘柄 OHLC を取得 (5 年)
python scripts/update_data.py 5

# 2. γ (L¹, n_unb) 時系列を計算 (window=30)
python scripts/lib/compute_gamma_timeseries.py 30

# 3. バックテスト (look-ahead 完全排除済み v2: expanding-window z-score)
python scripts/backtest_v2.py

# 4. (任意) HTML を再生成して挙動確認
python scripts/build_index.py
python scripts/build_app.py
```

テストを動かす場合:

```bash
pip install -r requirements-dev.txt
pytest -v
```

## 主要発見 (要約)

1. `L¹` (強さ) と `n_unb` (符号) は **corr ≈ 0.16 で独立** (5y/10y/15y/EM/CN で再現)
2. **ショック種別で反応指標が分岐** — 戦争・市場構造 → `L¹`、関税 → `n_unb`
3. `e_div = z(n_unb) − z(L¹)` が **ショックタイプ 3 区分判別器**
4. 集約スカラーを **12 指標に分解**すると `trade_policy` で `n_unb_4` (4-cycle) が特異
5. **cross-asset 連鎖が震源** — 個別銘柄 LOO で `e_div` の源泉は EUB10Y / COPPER / CHINA50 / USDJPY / VIX に集中し、米株指数は脇役
6. **Granger 一方向因果** `trade_policy → e_div` (lag5, p = 0.022)。DAG 分析で VIX が媒介と判明
7. **α 不変量** (関手族の極限/余極限) は 20y データでも `e_div` の shock 別相対順序を保持 (絶対値は 5y より減衰)
8. **売買対象を 11 種で比較** — 米国株指数 (S&P500 / NASDAQ100 / DJ30 / Russell2000) は **4 種すべて B&H を Sharpe で上回る** (全 11 種では 6 種が B&H 超え)

詳細は Paper PDF (`./docs/paper/main.pdf`) または `index.html` 参照。

## Limitation

- バックテストの look-ahead bias は **完全排除済み** (expanding-window z-score, `min_periods=30`、シグナルは前日終値で確定し翌日始値で執行)
- 取引コストは **0.05%/leg を実装済み** (リテール口座では過小評価の可能性あり)
- 多市場再現性は **検証済み** — 8y OOS (2017-11 以降) に加え新興国 (EM)・中国 (CN) で再現を確認
- in-sample (Sharpe +0.88) に対し walk-forward OOS は Sharpe +0.45〜0.65 と低下し、過学習リスクが残る。本手法は純粋なリターン増強ではなく **リスク管理の補完** と位置づける
- `trade_policy` 反応は 2025-04 Liberation Day cluster 駆動の側面が大きく、2018-2019 米中貿易戦争では再現しない (event-chain の速度依存の可能性)

## 用語集

| 記号 | 意味 |
|---|---|
| `L¹` | 持続ホモロジー H₁ バーコードの L¹ ノルム。相関構造の総"強さ"を表す連続量 |
| `n_unb` | 符号付きグラフの不整合 (unbalanced) 3-サイクル数。Heider balance の崩れを離散量で測る |
| `e_div` | `z(n_unb) − z(L¹)`。両不変量の z-score 差で、ショックタイプを分類 |

## Citation

```bibtex
@misc{marketgraph2026,
  author       = {Hajime},
  title        = {Market Graph Research: Topological Analysis of Financial Networks
                  with Categorical and Sign-based Invariants},
  year         = {2026},
  institution  = {Tokyo City University},
  howpublished = {\url{https://github.com/hajimedayo328/market-graph-presentation}},
  note         = {GitHub Pages: \url{https://hajimedayo328.github.io/market-graph-presentation/}}
}
```

CITATION.cff を配置済みのため、GitHub の "Cite this repository" ボタンからも自動取得できる。

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](./LICENSE)

引用 (上記 BibTeX) を伴う限り、共有・改変・商用利用いずれも可。

## Contact

質問・コラボ希望は GitHub Issues 経由でお願いします:
https://github.com/hajimedayo328/market-graph-presentation/issues

---

Built with Plotly, KaTeX, networkx, ripser, yfinance, MetaTrader5.
