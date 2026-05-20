# Market Graph Research — プレゼンテーション

> 圏論×グラフ理論で金融市場ネットワークの位相変化を分析する研究。
> 東京都市大学 / Hajime / 2026-05

## 🌐 公開 URL

- **研究内容**: https://hajimedayo328.github.io/market-graph-presentation/
- **実装デモ**: https://hajimedayo328.github.io/market-graph-presentation/app.html

## 構成

```
.
├── index.html              # 研究内容 (説明 + 可視化)
├── app.html                # 実装デモ (リアルタイム + バックテスト)
├── data/                   # 日次データ (週次自動更新)
│   ├── ohlc_40.parquet     # yfinance 40 銘柄 5 年
│   ├── gamma_timeseries_w30.csv
│   ├── multi_indicators_w30.csv
│   ├── backtest_results.json
│   └── ...
├── scripts/
│   ├── update_data.py      # yfinance → 時系列
│   ├── backtest.py         # 6 戦略 × 2 方向のバックテスト
│   ├── build_index.py      # 研究内容 HTML 生成
│   ├── build_app.py        # 実装デモ HTML 生成
│   ├── templates/app.html  # app テンプレート
│   └── lib/                # 持続ホモロジー等の依存ライブラリ
├── .github/workflows/
│   └── weekly_update.yml   # 毎週月曜 JST 7:00 に自動データ更新 + HTML 再生成 + push
└── requirements.txt
```

## 自動更新

GitHub Actions が **毎週月曜 JST 07:00 (UTC 日曜 22:00)** に:

1. yfinance で最新 5 年データ取得
2. γ時系列 (L¹ + 不整合サイクル数) 再計算
3. バックテスト再実行 (S&P500 vs 6 戦略)
4. `index.html` と `app.html` 再生成
5. 自動 commit & push

手動トリガーも可: Actions タブ → "Weekly Data Update" → "Run workflow"

## 主要発見 (要約)

1. L¹ ノルム (強さ) と 不整合サイクル数 (符号) は **corr=0.16 で独立に動く**
2. **ショック種別で反応指標が分岐**: 戦争・市場構造 → L¹、関税 → 不整合サイクル数
3. **e_div = z_unb − z_L1 がショックタイプ 3 区分判別器**
4. 集約スカラーを **12 指標に分解**すると trade_policy が **n_unb_4 (4-cycle) 特異**

## バックテスト結果 (2021-2026 5 年、S&P500 対象)

| 戦略 | リターン | Sharpe | Max DD |
|---|---|---|---|
| **S1: e_div≥+0.8 で現金化** | **+85.2%** | +0.98 | -18.9% |
| Buy & Hold (ベンチ) | +75.4% | +0.71 | -25.4% |
| S2: e_div≤-0.5 で買い | +63.8% | **+1.16** | **-7.6%** |

→ 政策ショック検知でリスクオフする戦略が Buy & Hold を約 10pp 上回り、最大ドローダウンも改善。

## 研究本体

コード・解析スクリプト・データ・ドキュメント詳細は別リポジトリ (Private):
https://github.com/hajimedayo328/market-graph-research

## Limitation (正直に)

- バックテストは取引コスト 0%、look-ahead bias の疑義あり (Section 3 末参照)
- 5 年データ・S&P500 のみ
- trade_policy 反応は 2025-04 cluster 駆動の側面大
- 4 手法統合では trade_policy 前兆消失 → 手法選択依存性あり

---

🤖 Built with Plotly, KaTeX, networkx, ripser, yfinance.
