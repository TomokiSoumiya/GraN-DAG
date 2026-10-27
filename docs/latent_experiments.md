# 潜在変数情報の保持・読み出し実験

この実験は未知の潜在変数の自動発見ではなく、正解Zが既知の人工データで、
凍結したGraN-DAGの特徴からZをどれだけ読み出せるかを測定する。
モデル推定とZの評価を別工程に分け、既存の損失・DAG制約・最適化を再利用する。

## 初期設定A

`configs/latent_linear.json` は次を指定する。

```text
Z ~ N(0, 1)
X1 = 2 Z + ε1
X2 = 3 Z + ε2
X3 = 1 Z + ε3
εi ~ N(0, 1), すべて互いに独立
```

1000サンプル、生成シード42、GraN-DAG学習シード42、回帰CVシード43。
既知の直接X間辺は0本。既知の潜在辺はZ→X1、Z→X2、Z→X3。
観測X同士には共通原因による相関が生じる。

この分布は因果十分性を満たさない。observed_adjacencyが0であることは、
観測変数が独立であることでも、真の周辺化グラフが空のDAGであることでもない。
既存DAG形式はX間の直接辺を記述的にログ出力するために渡す。
そのSHDやSIDを潜在交絡に対する正しい構造精度と解釈しない。

## 実行

準備済みホストではリポジトリ直下から次を実行する。

```bash
bash run_in_container.sh python latent_experiment.py --config configs/latent_linear.json
```

コンテナ内の `/final_log` はホストの `experiments/` 全体を指す。
設定の `experiments/...` は、このヘルパー環境では自動的に `/final_log/...` に対応する。
PyTorch/R/CDT/scikit-learnがインストールされた通常の環境では次でも実行できる。

```bash
python latent_experiment.py --config configs/latent_linear.json
```

途中工程だけの実行は `--stage generate` / `train` / `evaluate`。
既存の同一条件を再利用するときだけ `--resume` を追加する。

```bash
bash run_in_container.sh python latent_experiment.py \
  --config configs/latent_linear.json --stage evaluate --resume
```

初回の学習済み結果がある状態で全工程を再実行する場合も `--resume` が必要。
既存フォルダがあるのに `--resume` がなければ停止する。
条件が保存済みconfigと異なる場合は、`--resume` があっても停止する。
別の `output_dir` または `--output-dir` を指定して新しい実験にする。
既存の `p10-run1` は使用・上書きしない。

## 設定の変更

JSONを使用する。`extends` は相対パスで別のJSONを継承する。
辞書は再帰的に結合し、配列は丸ごと置き換える。保存するconfigは継承解決後の全条件。

| 項目 | 設定キー |
| --- | --- |
| サンプル数 | `data.n_samples` |
| 観測数D、潜在数K | `data.n_observed`, `data.n_latent` |
| 潜在分布 | `data.latent_distribution` |
| Z→Xの係数 | `data.latent_coefficients`: K行×D列 |
| X→Xの係数 | `data.observed_coefficients`: D行×D列 |
| ノイズ標準偏差 | `data.noise_std`: スカラーまたはD要素の配列 |
| 生成関数 | `data.latent_function`, `data.observed_function` |
| 生成・分割・学習・評価のseed | `data.seed`, `split.seed`, `training.seed`, `probe.seed` |
| 分割率 | `split.train`, `split.validation`, `split.test` |
| 隠れ層数・幅 | `training.num_layers`, `training.hid_dim` |
| 学習率・反復数 | `training.lr`, `training.lr_reinit`, `training.num_train_iter` |
| CAM有意水準 | `training.cam_pruning_cutoff` |
| Ridgeの候補 | `probe.ridge_alphas` |
| 非線形回帰 | `probe.nonlinear`: 現在はExtraTreesのみ |

分布は `normal`（mean/std）、`uniform`（low/high）、`laplace`（loc/scale）を実装。
各潜在因子は独立で、パラメータはスカラーまたはK要素の配列にできる。
分布間の相関や任意のユーザー定義分布は未実装。

関数は `identity`, `tanh`, `sin`, `square`, `cubic` を実装。
生成式は次の加法SEMで、観測DAGのトポロジカル順に計算する。

```text
Xi = Σk A[k,i] f(Zk) + Σj B[j,i] g(Xj) + εi
```

`A`はZ→X、`B`はX→X。いずれも行が原因、列が結果。
自己辺・観測変数間の循環はエラーにする。非線形な相互作用、因子間の積、
エッジごとに異なる関数、Z間の因果構造は現在の設定仕様には含まれない。
必要なら`latent_experiments/data.py`の生成器を拡張する。

`observed_coefficients: null` のときは `observed_graph` を使う。
`type: empty` または `erdos_renyi` を指定できる。後者はランダム順序上で辺を張るためDAGになる。
既存の観測係数を明示した場合は、その係数を優先する。

## B〜Eへの変更例

| 条件 | 設定例 | 実装の範囲 |
| --- | --- | --- |
| B: 潜在1個・非線形 | `configs/latent_nonlinear.json` | tanh(Z)とノイズ0.5 |
| C: 潜在2個以上 | `configs/latent_two_factors.json` | K=2、D=5、各Zを個別評価 |
| D: 観測間の因果辺もある | `configs/latent_observed_edges.json` | X1→X2→X3に潜在共通原因を追加 |
| E: 観測10個 | `configs/latent_ten_observed.json` | D=10のランダムDAGにZ→Xを追加 |

いずれも生成器の小規模テストを実行した。B〜EについてGraN-DAGを収束まで学習した
実験結果があることは意味しない。初回の本実験はAのみ。

既存10変数DAGを使う場合は、そのDAGの0/1構造を保ちながら適切な係数を設定し、
`observed_coefficients`へ10×10行列として指定する。既存の観測値に単純にZを足す方式ではなく、
指定したSEMから新たに全変数を生成する。

## データ漏洩防止

1. サンプルIDで最終テスト200行を先に分離する。分割にX/Zの値を使わない。
2. 残り800行を `training_input/data1.npy` に保存する。
3. 既存DataManagerFileのshuffleを再現し、その中の600行を学習、200行を検証にする。
4. GraN-DAGの標準化統計は学習600行のみから計算する。
5. 元のCAMはtrain+validationを結合するため、ここには800行しか渡さない。
6. 学習・DAG化・CAMが完了した後に、凍結モデルで全1000行を隠れ層へ変換する。
7. Zを読むのは評価工程。RidgeとExtraTreesの選択は学習600行内CVだけ。
8. 最終テスト200行はR²/MSEを計算するためだけに使う。検証200行の指標も別列に保存する。

`latent_experiments/train.py`は `true_Z.npy` を読み込まない。
training_options.jsonにはデータ生成係数やZも含めず、Xの行数・列数とNNの学習条件のみを渡す。
最終テストXの隠れ層抽出は「学習への利用」ではなく、モデル選択後の凍結推論である。
結果を見て条件を調整する場合は新しい独立テストを用意する。

## 学習の扱い

既存の `main.build_parser()` から既定オプションを取得し、既存の
`gran_dag.train.train`, `to_dag`, `cam_pruning` を直接呼ぶ。
初期設定は隠れ層2層×10ユニット、RMSprop、初期lr=0.01、以降lr=0.0001、
最大100000反復、DAG制約閾値1e-8、CAM cutoff=0.001。CPUのスレッド数は1。

元のtrain関数は制約を満たすとチェックポイントを保存する。
上限で終了した場合は `unconverged/` に明示的に保存し、既定ではその先を停止する。
この保存は損失や更新則を変えない。
`training.require_convergence=false` を明示すれば未収束モデルの診断を続行できるが、
結果には未収束と記録される。反復数を増やすときは新しい実験フォルダを使う。
`--resume`は完了段階の再利用で、オプティマイザー状態からの中断再開機能ではない。

## 隠れ層と比較条件

既存の `save_hidden_activations()` を使用し、活性化関数適用後の値を保存する。
全サンプルを元のX/Zの行順に保存し、`sample_indices.npy`で検証する。
`sample_split.npy`は0=学習、1=検証、2=最終テスト。
mean/std、グラフマスク、checkpoint SHA-256、前処理条件も併記する。
既存の `hidden_to_csv.py`でCSVも生成する。

比較条件は次の通り。

- X全列→Ridge。
- X全列→ExtraTreesによる非線形回帰。
- 学習直後／CAM後の各層から、各MLP単独または全MLP連結→Ridge。
- DAG化後（CAM前）も別条件として保存・評価し、閾値によるDAG化とCAM枝刈りを区別する。
- 初期重みを保存した未学習NNに、学習直後と同じマスクを適用→同じ評価。
- 同じ未学習NNに、CAM後と同じマスクを適用→同じ評価。
- 初期の全結合（自己辺なし）未学習NN→同じ評価。
- 学習直後／CAM後の各MLPが使う親変数だけ→Ridge。入力情報量の違いを補助的に比較。

同じマスクのランダムNNは重みを学習していないが、マスク自体はXから学習済み。
そのため、完全にデータ非依存の対照はrandom_full条件になる。
隠れ層は元Xの決定論的な変換であり、新しい情報を増やすわけではない。
高いR²は既知Zの読み出し可能性を示すが、未知Zの発見や因果同定の証明ではない。
逆にRidgeの低いR²だけで潜在情報が完全に消えたとは断定できない。
入力マスクで残るX列の和集合からのRidgeも `input_union_ridge` として保存し、
入力情報自体の削減と特徴変換の違いを分けて評価する。

## 出力

```text
experiments/latent_linear/
  config.json, config.sha256, data_manifest.json
  data/
    observed_X.npy, true_Z.npy, noise.npy
    observed_adjacency.npy, latent_adjacency.npy
    observed_coefficients.npy, latent_coefficients.npy
    sample_indices.npy, sample_split.npy
    train_indices.npy, validation_indices.npy, test_indices.npy, pool_indices.npy
    config.json
  training_input/                  # Z・最終テストXは含まない
    data1.npy, DAG1.npy, sample_indices.npy
  training_options.json, training.log, training_metrics.jsonl
  training_status.json, normalization_mean.npy, normalization_std.npy
  train/hidden/
  to-dag/hidden/
  cam-pruning/cutoff_0.001000/hidden/
  random_full/hidden/
  random_train_mask/hidden/
  random_cam_mask/hidden/
  analysis/
    metrics.csv, test_predictions.csv, parent_group_metrics.csv
    hidden_unit_statistics.csv, hidden_z_correlations.csv
    pca_coordinates.csv, pca_summary.csv
    graph_edges.csv, graph_metrics.csv
    evaluation_status.json, gaussian_reference.json
    report.md
    probes/*.pkl                    # 補助回帰モデル。NN本体とは別
    figures/*.png
```

第1・第2層の初期出力はそれぞれ `(1000, 3, 10)`、連結すると `(1000, 30)`。
全層を一緒に連結して評価するのではなく、各層ごとにMLP軸を連結する。
MLP Xjの単独特徴は `(1000, 10)`。

PCA・相関・活性統計は学習600行だけで計算する。PCAは隠れ層を平均中心化して学習し、
最終テストを投影する。色付けは評価用Zで、PCAの学習にZは使用しない。
定数表現の相関はNaN、PCA座標・次元数は0として明示する。

## テスト

```bash
bash run_in_container.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

既存の隠れ層抽出回帰テストを維持し、生成式・B〜Eの生成器・循環検出・分割と
DataManagerの一致・学習専用統計・Z回帰の指標・テストを変更してもfitが変わらないこと・
既存出力の保護を検査する。初回Aの実際の学習結果はその実験フォルダのreport.mdを参照する。

## 実行済み確認（2026-10-27）

- 初期設定Aを実際に学習。36,974反復、初回workerは428.1秒。
- 600/200/200行を分離し、学習入力はXのみ800×3。最終テストは渡していない。
- 第1・第2層は全段階で1000×3×10。独立した各MLPの順伝播計算と照合済み。
- 保存した予測CSVから62条件の最終テストR²/MSEを独立に再計算し、一致を確認。
- 全工程の`--resume`でも、主モデルのハッシュ・隠れ層配列・評価数値が同一。
- 12件のテストが成功。複数Z・定数CAM特徴の評価統合テストはダミー凍結NNを使用し、
  B〜Eの本学習を実施したものではない。
- 既存p10-run1のチェックポイントはハッシュ不変。

実結果は `experiments/latent_linear/analysis/report.md`、検証記録は
`experiments/latent_linear/analysis/verification.json`。
初回Aでは、X→Ridgeの最終テストR²は0.9257、学習直後の連結第1層は0.9220、
第2層は0.7910、CAM後の連結第1層は0.9003、第2層は0.9000。
未学習NNも同等の読み出しが可能で、この1条件では学習による優位性は確認できない。
今回のCAMはDAG化後の辺を削除しておらず、学習直後との差にはDAG化の影響が含まれる。
