# 真のZを観測する4変数オラクル実験

設定は `configs/latent_oracle_seed201.json`。
元データは `experiments/latent_nonlinear_normalized_seed201/`。
出力は新規の `experiments/latent_oracle_normalized_seed201/`。

## 範囲

seed201・σ=1・1000件の既存Z/Xをコピーし、`[Z,X1,X2,X3]` の順に結合する。
Zは先頭列。サンプルIDと600/200/200分割、信号の学習行規格化係数を維持する。
乱数によるデータ再生成は行わない。
真のZを普通の観測列として使うオラクル実験で、教師なし潜在因子の発見ではない。

既存のNonLinGaussANM、2層×10ユニット、leaky-ReLU、RMSpropと収束設定を再利用。
初期12方向を候補とし、Zの向きは固定しない。
新規NN学習は1回のみ。PCA/FA、Z_hat、別シード、別ノイズ、追加学習は実行しない。

## 正解グラフの分離

既存DataManagerFileはDAG1.npyを必須とするため、学習入力には4×4全ゼロの
**診断用プレースホルダー**を置く。真のZ→各Xのグラフは学習に渡さない。
元の学習関数内でのGT比較は表示・診断だけで、損失・停止条件・マスク選択には使われない。
紛らわしいグラフ指標は独自の学習コールバックCSV/JSONから除く。
元の学習ログや自動生成adjacency.pngの「正解」欄はプレースホルダーであり、実評価ではない。

全段階のモデル・グラフを保存して凍結した後、評価処理で初めて真の4変数グラフを作成する。
正解に合わせた閾値調整は行わない。CAMは元のcutoff=0.001をそのまま使う。
既存関数内のtest_dataは検証200件のこと。最終テスト200件は学習・CAMに渡さない。

## 実行経路

```bash
bash run_in_container.sh python -m unittest discover -s tests -p test_oracle_graph.py -v
bash run_in_container.sh python latent_oracle.py --stage prepare
bash run_in_container.sh python latent_oracle.py --stage train
bash run_in_container.sh python latent_oracle.py --stage evaluate
```

準備済み・学習済みのディレクトリへの上書きや再学習は拒否する。
最大反復で未収束なら診断モデルを保存して停止し、設定を変えて再学習しない。
今回、最初のワーカー起動は古いPyTorchの`any()`型互換性チェックで学習前に停止した。
最適化・学習コールバックが0回、train/が未作成であることを確認し、停止ログ・未学習モデルを
`preflight_failure/` に移して保持した。チェック式だけをuint8対応に修正し、
データや学習設定を一切変更せずに実際のNN学習を1回開始した。
これは失敗したNN学習の再試行ではなく、学習前の起動チェック修正である。

## 指標

- TP：向きまで正しい辺。
- FN：欠けた真の有向辺（反転を含む）。FP：余分な有向辺（反転を含む）。
- R：真の方向がなく、逆向きが存在する辺数。
- SHD = FN + FP − R。反転は1操作。
- 両方向が存在して真の方向も残る場合は、余分な逆向き1辺の削除として数える。
- Precision = TP/(TP+FP)、Recall = TP/(TP+FN)、F1は調和平均。分母0なら0。

反転を除いた純粋な欠落・追加もCSVに併記する。
3変数実験との比較はX同士の誤った直接辺の本数に限定した参考比較。
異なる変数集合のSHD差を、同一条件での性能改善として扱わない。

## 出力

- `report.md`：全段階の辺・指標・解釈、日本語。
- `data/input_Z_X.npy`：(1000,4)、変数順は固定。元のZ/X・分割・係数も保存。
- `training_input/data1.npy`：(800,4)、学習・検証だけ。
- `normalization_mean.npy`, `normalization_std.npy`：4変数の学習600件だけのNN前処理。
- `initial/`, `train/`, `to-dag/`, `cam-pruning/`：元のモデルと隣接行列。
- `evaluation/adjacency_*.csv`：変数名付き正解・各推定行列（行＝原因、列＝結果）。
- `evaluation/graph_metrics.csv`, `estimated_edges.csv`, `true_edge_recovery.csv`, `stage_changes.csv`。
- `evaluation/three_variable_reference.csv`：元実験の読み取り比較。
- `evaluation/graphs.png`：正しい真のグラフと推定グラフの比較。
- `preparation_status.json`, `training_status.json`, `training_artifacts_manifest.json`, `evaluation/evaluation_status.json`：整合性と実行・凍結監査。

新コードは `latent_oracle.py`, `latent_experiments/oracle.py`, `tests/test_oracle_graph.py`。
GraN-DAG本体・損失・DAG制約・既存設定・元データと実験結果は変更しない。
