# 非線形データ・seed201の単一実験

今回の専用設定は `configs/latent_nonlinear_mixed_seed201.json`。
既存の `latent_nonlinear.json`（共通tanh変換、ノイズ0.5）は変更しない。

生成式は Z ~ N(0,1)、X1 = 2Z + ε1、X2 = Z² + ε2、X3 = sin(Z) + ε3。
ε1〜ε3は独立な標準正規変数。1000行、600/200/200の固定分割、全シード201、ノイズ1。
観測変数間の直接辺はなく、Zから各Xに辺がある。隣接行列の行は原因、列は結果。

生成器の `latent_functions` は観測変数ごとの変換名で、
`Xi = Σk A[k,i] f_i(Zk) + Σj B[j,i] g(Xj) + εi` を指定する。
省略時は従来の共通 `latent_function` を使い、従来の乱数列・計算を維持する。

## 実行した経路

以下は新規実験ディレクトリが存在しない状態からの手順。
すでに結果がある場合は再実行せず、保存済み結果を参照する。
特に学習コマンドを再度実行しない。

```bash
bash run_in_container.sh python -m unittest discover -s tests -p test_nonlinear_mixed_data.py -v
bash run_in_container.sh python latent_experiment.py --config configs/latent_nonlinear_mixed_seed201.json --stage generate
bash run_in_container.sh python latent_experiment.py --config configs/latent_nonlinear_mixed_seed201.json --stage train --resume
bash run_in_container.sh python latent_unsupervised.py --config configs/latent_unsupervised_nonlinear_seed201.json --stage extract
bash run_in_container.sh python latent_unsupervised.py --config configs/latent_unsupervised_nonlinear_seed201.json --stage evaluate
```

`--resume` は生成済みディレクトリと同一設定であることを確認するために使用する。
今回NNは新規に1回だけ学習する。既存の学習ワーカーが行うCAM枝刈りはNN再学習ではない。
教師ありZ回帰を含む `latent_experiment.py --stage all/evaluate` は使用しない。

## 固定した比較

全MLPを結合し、全条件で1因子のPCA/Factor Analysisを実行。

|条件|重み|入力マスク|
|---|---|---|
|X|なし|なし|
|C_L1/C_L2|学習済み|学習直後|
|B_L1/B_L2|実際の未学習重み|Cと同一|
|E_L1/E_L2|Cと同じ重み|DAG化後|
|D_L1/D_L2|Bと同じ重み|Eと同一|

DはDAG化後についても公平な未学習対照を設けるための条件であり、別のNN学習ではない。
NNの隠れ層は各 `(1000,3,10)`、連結後は `(1000,30)`、各因子は `(1000,1)`。
定数列除去・標準化・抽出器・因子からXへのOLS再構成は学習600件だけでfit。
検証・テストは変換のみ。FAの設定・因子数・第1成分は事前固定し、Zやテストで選択しない。
全18抽出器を保存・ハッシュで凍結してから、別プロセスで評価用Zを読み込む。

## 出力

`experiments/latent_nonlinear_mixed_seed201/` 以下に保存。

- `report.md`: 今回の生成式、グラフ、数値、既存線形結果との比較。
- `data/`: X、評価用Z、ノイズ、隣接行列、サンプルID、分割。
- `train/`, `to-dag/`: 学習直後/DAG化後のモデルと `hidden/hidden_layer_001.npy`, `002.npy`。
- `random_full/`, `random_train_mask/`: 学習前重みと、マスクをそろえた隠れ層。
- `training_status.json`, `training.log`: 収束、反復数、学習データ範囲。
- `unsupervised_factors/`: 抽出器、因子、再構成、評価CSV、図、監査記録。
- `unsupervised_factors/metrics.csv`: Pearson/絶対Pearson/Spearman、X再構成MSE。`split=test` が最終評価。
- `unsupervised_factors/figures/`: 相関一覧、散布図、同一マスクの学習前後、DAG化前後の比較。

ZとZ²の関係は非線形であり、Pearson相関が小さくても潜在変数との依存がないとはいえない。
因子抽出や高相関は、因果的な潜在共通原因の同定・未知因子数の検出を意味しない。
今回は1シード・1ノイズ・NN学習1回に限定し、他条件の実験は行わない。
