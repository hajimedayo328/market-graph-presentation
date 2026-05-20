# Market Graph Research — プレゼンテーション

> 圏論×グラフ理論で金融市場ネットワークの位相変化を分析する研究のプレゼン HTML。
> 東京都市大学 [redacted]伊陽研究室準備 / 2026-05 / Hajime

## 見る

🌐 **公開 URL**: https://hajimedayo328.github.io/market-graph-presentation/

## 内容

- ヒーロー: 5 年 × 40 銘柄の L¹ ノルム + 不整合サイクル数 時系列 (Plotly)
- Section 1: なぜこの研究か
- Section 2: 市場ネットワークとは
- Section 3: 位相不変量 1 — L¹ ノルム (代数的構造の説明含む)
- Section 4: 位相不変量 2 — 不整合サイクル数 (代数的構造の説明含む)
- Section 5: 発見 1 — 2 指標の独立性
- Section 6: 発見 2 — ショック種別で反応分岐
- Section 7: 発見 3 — 関税前は実際に符号反転
- Section 8: 発見 4 — 乖離インデックス e_div がショックタイプ判別器に
- Section 8.5: [redacted]フィードバック応答 — 集約スカラーを 12 指標に分解
- Section 9: 圏論的整理
- Section 10: 先行研究との位置づけ
- Section 11: これからやろうとしてること

## 研究本体

研究のコード・データ・解析スクリプトは別リポジトリで管理 (Private):
https://github.com/hajimedayo328/market-graph-research

## 主要発見 (要約)

1. L¹ ノルム (強さ) と 不整合サイクル数 (符号) は corr=0.16 で**独立に動く** (PC1=0.58)
2. **ショック種別で反応指標が分岐**: 戦争・市場構造 → L¹、関税 → 不整合サイクル数
3. 関税前 15 日で **flip_rate が +0.45σ で上昇** (p=10⁻⁴)、符号反転を直接確認
4. **乖離インデックス e_div = z_unb − z_L1 がショックタイプ 3 区分判別器**として機能
5. 集約スカラーを 12 指標に分解すると **trade_policy が n_unb_4 (4-cycle) 特異**であることが判明

## 先行研究との位置

「TDA L¹ × 符号付きサイクル整合性」を同一パイプラインで併走させる研究は無い。
3 系統 (TDA × 金融 / 符号 × 金融 / 圏論側) を橋渡しした実装と実証として位置づけ。

Wang & Xu (2025) が独立に「2024 関税で中国 LSCBM が ×3.6 倍急増」を報告 → 我々の発見と相互補強。

---

🤖 Built with Plotly, KaTeX, and a lot of pandas.
