# 5観測変数・複数入力MLPの探索実験

```bash
bash run_in_container.sh python -B latent_five_observed.py --stage all
```

seed201、σ=1、1000件（600/200/200）、GraN-DAG新規学習1回。
実行前に `tests/test_five_observed.py` を自動実行する。
テスト失敗時はデータ生成・NN学習を開始しない。
学習は既存の `latent_experiments.train.train_experiment` をそのまま使用し、
再試行マーカーと既存ファイル検査で2回目の学習を拒否する。

設定は `configs/latent_five_observed_seed201.json`。
既存の3変数設定の学習条件を継承し、関数tanh(Z)、sin(Z)、Z³、tanh(2Z)、sin(2Z)を使用する。
後半2関数は新モジュールの生成プロセスだけで既存生成関数のregistryへ登録し、
生成後に元へ戻す。既存のコード・データは編集しない。
信号は学習600件だけで平均0・分散1（ddof=0）に規格化し、独立N(0,1)ノイズを加える。
Zを使うのはデータ生成と最終評価。NN学習・因子抽出では保存Zの読み込みを禁止する。

出力は `experiments/latent_five_observed_seed201/`。

- `data/`: X、Z、ノイズ、信号、規格化係数、ID、分割、真の直接隣接行列。
- `random_full/`: 実際の学習前重み・初期マスク。
- `train/`, `to-dag/`, `cam-pruning/cutoff_0.001000/`: 各段階の重み・マスク・隠れ層。
- `random_train_mask/`: 初期重み＋学習後マスクの対照。
- `training.log`, `training_metrics.jsonl`, `training_status.json`: 学習ログ・収束状況。
- `single_training_attempt.json`: 一度だけの学習を保証する試行マーカー。
- `unsupervised_factors/graphs/`, `unsupervised_factors/mlp_inputs.csv`: 各段階のグラフ・入力元・入力数。
- `unsupervised_factors/models/`, `unsupervised_factors/factors/`: A〜Eの変換器・1次元因子。
- `unsupervised_factors/metrics.csv`, `unsupervised_factors/comparison_table.csv`, `unsupervised_factors/figures/`: 評価。
- `report.md`: 日本語レポート。

条件: A=元X＋PCA、B=元X＋FA、C=初期重み・学習後マスクH1＋PCA、
D=学習済みH1＋PCA、E=各MLPのPC1を標準化して最終PCA。
因子数1、学習600件のみでfit。Cのマスクは学習結果に依存するため完全に独立した未学習モデルではない。
各段階の親数はマスクを主要定義とし、get_w_adj()>0の厳密な正経路数も記録する。
PCA/FAの符号をZで反転せず、全因子を固定した後に別プロセスで評価する。

個別段階は `--stage generate/train/extract/evaluate`。既存出力は上書きしない。
学習が失敗した場合も設定変更や再学習を行わず、未実施の解析を明記して停止する。
