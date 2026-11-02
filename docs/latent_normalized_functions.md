# 規格化tanh・sin・3乗の単一実験

実験設定は `configs/latent_nonlinear_normalized_seed201.json`、抽出設定は
`configs/latent_factors_normalized_seed201.json`。
seed201・σ=1・1000件（学習600／検証200／テスト200）の1条件だけを扱う。

## 生成

`f = (tanh, sin, cubic)` として、学習行だけから
`m_j = mean(f_j(Z_train))`、`s_j = std(f_j(Z_train), ddof=0)` を計算する。
全行に同じ変換を適用し `X_j = (f_j(Z)-m_j)/s_j + ε_j` とする。
Zと各εは独立な標準正規。Zのこの用途はデータ生成だけであり、NNや抽出器のfitには渡さない。

生成器の追加指定は `signal_normalization: {"fit_split":"train", "ddof":0}`。
`save_dataset` が先に既存の分割を確定し、その学習IDを生成器に渡す。
省略時は以前と同じ信号・乱数列を維持する。
係数は `data/config.json` の `resolved_signal_normalization`、配列は
`signal_mean.npy`, `signal_std.npy`, `raw_signal.npy`, `normalized_signal.npy` に保存する。

## 実行経路

```bash
bash run_in_container.sh python -m unittest discover -s tests -p test_normalized_signal.py -v
bash run_in_container.sh python -m unittest discover -s tests -p test_function_factors.py -v
bash run_in_container.sh python latent_experiment.py --config configs/latent_nonlinear_normalized_seed201.json --stage generate
bash run_in_container.sh python latent_experiment.py --config configs/latent_nonlinear_normalized_seed201.json --stage train --resume
bash run_in_container.sh python latent_function_factors.py --stage extract
bash run_in_container.sh python latent_function_factors.py --stage evaluate
```

これは出力先がない状態からの手順。結果が存在したら再実行しない。
`--resume` は生成済みの同一設定を検証するためで、今回NNは新規に1回だけ学習する。
損失・DAG制約・RMSpropは既存のまま。学習ワーカーのCAM処理はNN再学習ではなく、
今回CAM条件の因子比較を追加していない。
教師ありZ回帰を実行する `latent_experiment.py --stage all/evaluate` は呼ばない。
継承元設定に残る旧`probe`項目も使用しない。

## 固定した18条件

- X1、X2、X3：それぞれ学習行で標準化した値（PCA/FAは適用しない）。
- X全体：PCA、FA、RBF Kernel PCA。
- 実際の未学習重み＋学習後と同一マスク：第1・第2層のPCA/FA。
- 学習済み＋学習直後マスク：第1・第2層のPCA/FA。
- 同じ学習済み重み＋DAG化後マスク：第1・第2層のPCA/FA。

各因子数は1。全隠れ層は全MLP連結。学習済みと未学習はB/Cの同一マスクで比較する。
DAG化はC/Eの同じ重みで比較する。既存のモデル・サンプルID・前処理検証を再利用する。

Kernel PCAのgammaは、学習標準化Xの全ペアの正の二乗距離中央値`d`から
`gamma = 1/(2*d)` で固定。全179700ペアを使用し、Zやテストで調整しない。
標準化、PCA、FA、Kernel PCA、因子→全3変数XのOLS＋切片は学習600件だけでfit。
検証・テストはtransformのみ。再構成MSEは線形OLSの誤差であり、非線形再構成能力そのものではない。

抽出時には`true_Z`の読み込みを禁止する。全18因子、抽出器、X再構成、
全観測変数との単純変換診断を保存・ハッシュ凍結してから、別プロセスでZを読む。
最大の単独変数Z相関は事後診断にすぎず、教師なしモデル選択として使用しない。

## 出力

`experiments/latent_nonlinear_normalized_seed201/unsupervised_factors/`：

- `report.md`：生成係数・グラフ・全18結果・比較と限界。
- `comparison_table.csv`：テスト18条件。`metrics.csv`：全分割54条件。
- `extracted_factors.csv`：全1000行×18因子のサンプルID・分割対応。Zは含まない。
- `models/`, `factors/`, `reconstructions/`：確定したモデル、各(1000,1)の因子、X再構成。
- `input_masks.csv`：原因行→結果列のマスク。
- `observed_factor_diagnostics.csv`：全因子と全XのPearson/Spearman、学習fitのX→因子一次変換のテスト相対RMSE。
- `figures/`：全条件の比較図と散布図。
- `metadata.json`, `extraction_status.json`, `evaluation_status.json`, `extraction_manifest.json`：凍結・Z分離と不変性。

親なしMLPの定数列は既存処理で除く。全列定数なら因子0として相関は未定義を報告する。
期待に反する結果でも設定変更・再学習はしない。この1条件が終了したら実験作業を終了する。
