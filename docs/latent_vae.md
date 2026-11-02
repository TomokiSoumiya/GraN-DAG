# 固定GraN-DAG隠れ層と元Xを入力するVAEの最小比較

既存 `experiments/latent_nonlinear_normalized_seed201/` のX・H・ID・分割を読み、
A（X入力）、B（学習直後H1入力）のVAEを各1回だけ学習する。
GraN-DAG学習、データ生成、グラフ構造学習は呼び出さない。

## 実行

```bash
bash run_in_container.sh python -B latent_vae.py --stage all
```

実行前に `tests/test_latent_vae.py` の全テストを自動実行する。
テスト失敗時には実験出力を作成せず、NNの学習も開始しない。
出力先が存在する場合は学習を拒否し、途中の学習も自動再試行しない。

```bash
# 学習せずに単体・保存データ検証だけ実施
bash run_in_container.sh python -B -m unittest discover -s tests -p test_latent_vae.py -v
# 両モデルの学習・推論が完了済みで評価出力がない場合のみ
bash run_in_container.sh python -B latent_vae.py --stage evaluate
```

## 条件

設定は `configs/latent_vae_seed201.json`。
Encoderは3/30→16→16→2（平均と標準偏差用パラメータ）、Decoderは1→16→16→3。
LeakyReLU slope=0.01。潜在1次元、標準正規事前分布、対角Gaussian観測分布。
潜在標準偏差はsoftplus+1e-4、変数別観測標準偏差はsoftplus+1e-3。
XとHを学習600件のみで標準化（ddof=0）。Hの定数列判定も学習のみ。

Adam、lr=0.001、batch=64、最大500epoch、patience=30、seed201、KL係数1。
同じDecoder初期値・バッチ順序・学習時epsilon乱数系列を使用する。
学習時はMC1サンプル。検証ELBOは固定64点のq分布のGauss–Hermite求積。
64点の事前分布求積で周辺対数尤度を評価し、事前固定32/96点で精度を確認する。
評価点数や設定はZとの相関で変更しない。

## ファイル

- `latent_experiments/vae.py`: VAE・解析KL・訓練ELBO・q/事前求積。
- `latent_experiments/vae_data.py`: 元モデル・H・分割の照合、train-only標準化。
- `latent_experiments/vae_train.py`: 事前テスト、A/B各1回の学習、Zなしの推論確定。
- `latent_experiments/vae_evaluate.py`: 別プロセスでZを読み、CSV・PNG・日本語レポート作成。
- `latent_vae.py`: 学習と評価を別プロセスで順に実行。

出力 `experiments/latent_vae_normalized_seed201/`:

- `report.md`, `metrics.csv`, `reference_metrics.csv`, `figures/`
- `A/`, `B/`: `initial_state.pt`, `best.pt`, `last.pt`, `history.csv`,
  `training_status.json`, `predictions.npz`, `latent_posterior.csv`
- `config.json`, `metadata.json`, `preprocessing.npz`, `sample_indices.npy`, `sample_split.npy`
- `batch_orders.npy`: 各epochの学習600件内の行位置。同じprefixを両モデルで使用。
- `preflight_tests.txt`, `source_verification.json`, `source_inventory_before.json`
- `latent_read_audit.json`, `checkpoints_finalized.json`, `training_complete.json`, `evaluation_status.json`

`predictions.npz` は全1000件を元の行順で保存する。`mu`, `std` は(1000,1)、
`x_mean_raw` は(1000,3)。`latent_posterior.csv` には真のZを含めない。
モデル・前処理・推論値のハッシュを確定するまで真のZを読み込まない。

## 解釈

Zとの相関は事後平均から計算し符号を変更しない。
MSEは非線形Decoderの事後予測平均による自己再構成。
過去のPCA等はOLS再構成であり、MSEの差を抽出法だけの差とは扱えない。
KL<0.01かつ事後平均分散<0.01をcollapseの診断フラグとする（数学的な判定ではない）。
求積精度警告時には小さな尤度差を解釈しない。
1シード・過去にも参照したテストでの探索的評価であり、因果グラフ復元の証拠ではない。
