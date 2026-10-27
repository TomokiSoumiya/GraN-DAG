# ノイズ強度別・複数シードの教師なし因子抽出

普段見る結果は `experiments/latent_noise_study/report.md`、
`metrics_mean_std.csv`、`figures/noise_correlations.png` の3つ。
各モデル・隠れ層・因子の原データは `runs/` に分けて保存する。

## 生成式・対応付け

```text
Z ~ N(0,1)
X1 = 2 Z + sigma epsilon1
X2 = 3 Z + sigma epsilon2
X3 =   Z + sigma epsilon3
epsilon_i ~ N(0,1), mutually independent and independent of Z
sigma in {0.5, 1, 2, 4}
```

各1000件、学習600・検証200・最終テスト200。
5seedは42、201、202、203、204。同じseedではZ、標準正規の基礎ノイズ、
分割、NN初期重みを同じにし、sigmaだけを変更する。
全推定終了後の監査で、これらの対応と生成式を配列で照合する。

因子数は1個固定で、存在・因子数の自動検出実験ではない。
教師なし抽出器は元の設定（PCA full SVD、FA lapack/max_iter5000/tol1e-4）を使用。
標準化・定数列除外・fitは学習行だけ、検証／テストはtransformのみ。
X再構成は、学習因子と学習Xだけで求めたOLS＋切片による。

GraN-DAGは元のNonLinGaussANM、2隠れ層×10ユニット、leaky-ReLU、
RMSprop lr0.01、最大100000反復、h閾値1e-8。損失・DAG制約・最適化は変更しない。
既存workerを利用するためCAMも保存するが、主要評価は学習後とDAG化後。

## 実行

```bash
bash run_in_container.sh python latent_noise_study.py
```

設定は `configs/latent_noise_study.json`。
既存フォルダへの上書きを防ぐため、同じ出力先の再実行は停止する。
新しい実験には設定の`output_dir`を新しい場所へ変更する。
各段階を分けて実行する場合は以下。

```bash
bash run_in_container.sh python latent_noise_study.py --stage prepare
bash run_in_container.sh python latent_noise_study.py --stage fit
bash run_in_container.sh python latent_noise_study.py --stage evaluate
bash run_in_container.sh python latent_noise_study.py --stage summarize
```

`prepare`は既存の生成コードで全20データを作る。
`fit`は既存のGraN-DAG学習workerと `latent_unsupervised.py --stage extract` だけを呼ぶ。
**古い教師あり評価や `latent_experiment.py --stage all` は呼ばない。**
最大8プロセス、各1スレッドで実行する。

全20run・全360抽出器の保存とマニフェスト凍結が済むまで、Z評価は開始できない。
`evaluate`はグローバル凍結を検証した後に初めてZを読み、対応監査と評価だけを行う。
失敗はfit/evaluation_jobs.jsonとlogs/に記録し、seedの差し替えや良い条件だけの集計をしない。

## モデルの再利用

sigma=1・seed42のみ、`experiments/latent_linear/` の既存モデルをコピー再利用する。
X、分割、生成・学習設定、収束状態とZ／テスト不使用を確認してからコピーする。
新しいPCA／FA抽出は今回のrun内で実行する。
他の19条件は新規にGraN-DAGを学習する。

コピーしたopt.pkl内のパスは元runのままであり、今回は固定モデルの推論のみに使用する。
再利用の出自とコピーしたチェックポイントのSHA256は各runのmodel_origin.jsonに記録する。
過去の教師あり回帰器・教師あり評価結果はコピーしない。

## 比較条件

|記号|意味|
|---|---|
|X|元の観測データ|
|C_L1/C_L2|学習済みNN・学習直後マスクの第1／第2層|
|E_L1/E_L2|Cと同じ重み・DAG化後マスクの第1／第2層|
|B_L1/B_L2|実際の初期重み・Cと同じマスクの第1／第2層|
|D_L1/D_L2|Bと同じ未学習重み・Eと同じマスクの第1／第2層|

各9条件×PCA/FA。隠れ層は全MLP連結30列。
定数列は既存の学習行だけの規則で除き、全列が定数なら因子0固定・相関未定義とする。
マスク・初期重み・サンプル行対応は既存の検証機能で確認する。

## 集計の定義

- 主指標は最終テストの絶対Pearson相関。符号付きPearson・Spearmanも保存する。
- 因子の符号をZに合わせていないため、符号付き指標の平均は相殺に注意する。
- 全5seedの平均と標本SD（ddof=1）を計算する。SDは標準誤差・信頼区間ではない。
- 5件全て定義された指標だけに平均・SDを出す。未定義値は0にせず、定義数を明記する。
- 学習の効果は同じマスクのB→CとD→E、DAGの効果は同じ重みのC→Eを比較する。
- X→各隠れ層、層間、sigma間の比較も同じseed内で差分を求め、その後に平均・SDを出す。
- sigma間の差分は4水準の全6組合せを保存する。
- 相関を使って最良手法・層・条件を選ぶことはしない。
- 代表scatterのseed201は事前固定で、結果による選別はしない。

## 保存先の見取り図

```text
experiments/latent_noise_study/
  report.md                        日本語レポート
  metrics_mean_std.csv             各sigmaの5seed平均・SD
  metrics_by_seed.csv              全20runの評価値
  paired_mean_std.csv              学習・DAG・X・層間のペア差分集計
  paired_by_seed.csv               同上のseedごとの値
  noise_paired_mean_std.csv        sigma間のペア差分集計
  noise_paired_by_seed.csv         同上のseedごとの値
  model_origins.csv                再利用／新規・反復数
  graphs_by_seed.csv               推定グラフ
  extractor_warnings.csv           PCA／FAの警告
  pairing_verification.csv         基礎乱数・分割・初期重みの一致確認
  figures/
    noise_correlations.png         sigma対|r|、平均±SD
    noise_X_reconstruction.png     sigma対X再構成MSE
    paired_learning_effects.png    同一マスクでの学習効果
    paired_DAG_effects.png         同一重みでのDAG化効果
    paired_X_effects.png           Xに対する隠れ層の差
    correlation_summary.png        全条件の相関一覧
    representative/*.png          seed201の全sigma・両抽出器の散布図
  runs/sigma_*/seed_*/
    data/, training_input/         生成データとXのみのNN入力
    train/, to-dag/, ...           NNモデルと隠れ層
    model_origin.json              モデル出自
    unsupervised_factors/          従来と同じ抽出器・因子・CSV・PNG
  logs/                            run別の実行ログ
  plan.json                        事前固定設定・コードハッシュ
  fit_jobs.json                    学習・抽出の全成否
  global_extraction_freeze.json    全推定完了のバリア
  evaluation_jobs.json             評価の全成否
  status.json                      最終検証記録
```

人工データ生成のためZを作ることと、Zを学習・選択に使うことは異なる。
抽出では既存の読み込み禁止ガードも利用し、Zと旧教師あり結果を読まない。
最終テストはNN・CAM・抽出器・再構成係数・ハイパーパラメータ選択に使わない。

## 限界

元XのFAにも同じ比較機会を与える。高相関は因果的潜在共通原因の同定ではない。
既存seed42・sigma1のテストは以前にも閲覧しているため、今回は探索的な固定条件比較。
ノイズ増加は観測Xの分散も変えるので、生のX再構成MSEのsigma間増加をそのまま
因子の品質低下と同一視しない。標準化Xの誤差も記録する。
NN表現の重複ユニットがFAの独立残差仮定に合わない可能性も考慮する。

非線形・複数潜在因子・無潜在因子対照・Zを使わない因子数選択は、この実験では行わない。

## テスト

```bash
bash run_in_container.sh env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

`tests/test_noise_study.py`で基礎乱数対応、分割、標本SD、未定義値、ペア差分、
全推定完了前のZ評価禁止、改変検出、元モデルを変更しないコピー再利用、図・レポート生成を検証する。

## 今回の実行済み結果

全20条件が収束・抽出・評価まで完了した。新規NN学習19件、既存モデルコピー再利用1件。
全360抽出器の保存後に初めてZ評価を行い、抽出器警告・失敗・未定義テスト相関はいずれも0件だった。
既存を含む35テストは実行して成功した。

元Xの絶対Pearson平均はσ=0.5→4で、PCA 0.9834→0.6610、FA 0.9903→0.6456。
学習後第1層はPCA 0.9827→0.3881、FA 0.9754→0.3931。
同一マスクの未学習NNより平均で改善したが、高ノイズで元Xより頑健な結果にはならなかった。
第1層PCAのσ=1/2では元Xより僅かに高い平均だが、一般的・統計的な優位性は検証していない。
σ=4は学習直後から全5seedがDAGであり、DAG化による追加変化はなかった。

全1080評価行、216平均・SDグループ、800条件間ペア差分、540ノイズ間ペア差分を保存。
平均・SD・ペア差分を独立に再計算し、360因子配列の形状と凍結結果の不変も確認した。
過去4実験の1943ファイルはハッシュ不変。集計PNGは代表散布図を含め14枚。
詳しい結果と集計後の解釈は新しい実験のreport.mdを参照。
