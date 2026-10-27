# 教師なし潜在因子抽出（現在の研究用入口）

正解Zへの教師あり回帰ではなく、固定済みGraN-DAG表現からPCA／因子分析で
共通変動を表す因子を抽出する。初回は既存 `experiments/latent_linear/` を使用し、
GraN-DAGは再学習しない。因子数はZの値を参照せず事前に1個へ固定する。
**これは因子数や因果的潜在共通原因の自動同定ではない。**

## 実行

準備済み環境では、リポジトリ直下から次で全工程を実行できる。
初回実行はすでに完了しているため、同じ出力先では上書きを拒否する。

```bash
bash run_in_container.sh python latent_unsupervised.py
```

既存結果を保持して再実行するには、新しい出力先を指定する。

```bash
bash run_in_container.sh python latent_unsupervised.py \
  --output-dir experiments/latent_linear/unsupervised_factors_repeat
```

Zが不明な実データでも抽出だけはできる。今回の既存runと同じX・モデル・隠れ層・
正規化・サンプルID・分割の保存形式を用意し、設定の`run_dir`と`output_dir`を変更する。
抽出と評価を別々に行う場合は以下。

```bash
bash run_in_container.sh python latent_unsupervised.py --stage extract \
  --output-dir experiments/latent_linear/unsupervised_only_new
# Zが既知の人工データに限り、全抽出完了後に評価する
bash run_in_container.sh python latent_unsupervised.py --stage evaluate \
  --output-dir experiments/latent_linear/unsupervised_only_new
```

`--stage all`（既定）は**別々のPythonプロセス**でextract→evaluateを順に実行する。
新しい出力先は元run直下に限定し、既存のファイル・出力先は上書きしない。
信頼できるこの実験のモデルpickleのみを読み込み、保存時と同じコンテナを使用する。

## 固定条件

`configs/latent_unsupervised.json` に全条件を保存する。

- 因子数1。PCAの第1主成分、FAの第1因子を使用する。Z相関による成分選択はしない。
- PCAはfull SVD、FAはlapack SVD・max_iter=5000・tol=0.0001・seed42。
- ハイパーパラメータ探索は行わず、全条件・両手法を報告する。
- 平均・標準偏差による標準化は学習行のみでfit（StandardScaler、ddof=0）。
- 列選択も学習行のみ。既存 `varying_columns()` の規則でほぼ定数の列を除く。
- 検証・テストの行には、保存済みモデルでtransformするだけ。

|記号|表現|比較するマスク|
|---|---|---|
|X|元の観測3変数|なし|
|C_L1/C_L2|学習直後の学習済み隠れ層・全MLP連結|学習直後|
|E_L1/E_L2|Cと同じ重み・DAG化後の隠れ層|DAG化後|
|B_L1/B_L2|実際の初期重みを使った未学習NN|Cと同じマスク|
|D_L1/D_L2|Bと同じ未学習NN|Eと同じマスク|

元の依頼の7条件に加え、DAG化後にも同一マスクの未学習対照を追加し、9条件とした。
9条件×2手法で18個の抽出器をfitする。初期全結合マスクと学習済み疎マスクの
不公平な比較は行わない。CAM後は今回の主比較に追加していない。

既存 `controlled_features()` を再利用して、保存済みモデル・隠れ層・マスク・
前処理・全サンプル順序を照合する。元の隠れ層は `(1000,3,10)`、連結すると30列。
列名は対象MLPと隠れユニットが分かる形で保存する。

## Zからの分離

抽出時の `fit_extractor(features_train, x_train, method, settings)` にはZ引数がない。
既存の教師あり `fit_probe()` やGraN-DAGの訓練処理は呼ばない。

extractプロセスでは、Pythonの通常のファイル読み出しをラップし、`true_Z.npy`と
Zが含まれる旧`analysis/`・`additional_analysis/`へのアクセスを禁止する。
これは誤読み込みを検知するガードであり、悪意あるコードを隔離するセキュリティ境界ではない。
保存済みモデル・全因子・再構成・抽出設定のハッシュを凍結してからプロセスを終了する。

evaluateプロセスは、凍結した全ファイルの一致と抽出完了を確認して初めてZを読む。
評価では相関と図を作るだけで、再fit・成分選択・符号合わせを行わない。
元のXとHの前処理・モデル・抽出因子は評価後も不変であることを確認する。

抽出時の旧ファイル保護は、Zをハッシュ目的でも読まないよう、全ファイルの
サイズ・mtime・ctimeによるinventory照合と、使用したX/HのSHA256で行う。
抽出完了後の評価では元run全ファイルのSHA256も比較する。

## 再構成の定義

1. **入力特徴の再構成**：PCAは逆変換、FAは事後平均因子得点×負荷量＋平均。
   FAのノイズをランダムに加えることはしない。除外した定数列は学習平均へ戻す。
2. **元Xの再構成**：因子＋切片→観測Xを、学習600行の最小二乗でfitする。
   これはZへの教師あり回帰ではなく、観測データの再構成である。候補選択はしない。

入力特徴がHの場合のH再構成誤差と、元のX再構成誤差を別々に記録する。
Hの生MSEは尺度や次元が違うため条件間で直接比較しない。
標準化した非定数特徴の再構成MSEも保存するが、異なる特徴空間の誤差である。
`x_reconstruction_raw_mse` は全条件で同じ観測Xを対象とする比較用の指標。

全入力列が定数なら抽出因子を0とし、相関は0ではなく未定義とする。
因子のスケールや符号はZで補正しない。PCA/FAで因子分散の大きさが違うことは正常で、
それ自体は情報量や因子の品質を表さない。

## 保存先

初回結果は `experiments/latent_linear/unsupervised_factors/`。

|ファイル|内容|
|---|---|
|`report.md`|日本語の全条件結果・注意事項|
|`metrics.csv`|各条件・手法・splitのPearson、絶対r、Spearman、分散、X/H再構成誤差|
|`extracted_factors.csv`|Zなし。条件、手法、サンプルID、split、factor_1。18,000行|
|`factors/*.npy`|各条件の因子 `(1000,1)`|
|`models/*.pkl`|標準化、列選択、PCA/FA、X再構成係数を一緒に保存|
|`reconstructions/*_features.npy`|元の入力特徴空間での再構成|
|`reconstructions/*_X.npy`|元の観測Xの再構成 `(1000,3)`|
|`fit_metrics.csv`|Zを読む前の、因子分散・再構成誤差等|
|`feature_selection.csv`|列名、学習平均・std、定数閾値、採用/除外|
|`comparisons.csv`|固定マスクでの学習前後、固定重みでのDAG化前後、層間の評価差|
|`metadata.json`|固定設定、入力次元、使用列、因子数、マスク、モデル、分割、FA警告等|
|`extraction_status.json`|Z読み込み試行数、全抽出完了、学習split|
|`extraction_manifest.json`|評価前に凍結した全抽出ファイルのSHA256|
|`evaluation_status.json`|評価による抽出結果の不変、旧結果不変、コードハッシュ|
|`figures/scatter_{pca,factor_analysis}_Z1.png`|全条件のテストZ対因子散布図|
|`figures/absolute_correlations_Z1.png`|全条件の絶対Pearson比較|
|`figures/X_reconstruction_Z1.png`|同じ元Xに対する再構成誤差|
|`figures/feature_reconstruction_Z1.png`|入力特徴の標準化再構成誤差|
|`figures/factor_variances.png`|学習・テストの因子分散|
|`cleanup_inventory.csv`, `cleanup_plan.md`|依存関係と未削除の整理候補|

## 初回の実行済み結果

以下は最終テスト200件の絶対Pearson相関。元の符号はmetrics.csvに保存している。

|特徴|PCA絶対r|FA絶対r|
|---|---:|---:|
|元のX|0.943347|0.962407|
|学習後・第1層|0.943159|0.943315|
|学習後・第2層|0.940944|0.942658|
|DAG化後・第1層|0.907499|0.943315|
|DAG化後・第2層|0.904385|0.718622|
|未学習・学習後マスク・第1層|0.938766|0.816893|
|未学習・学習後マスク・第2層|0.926622|0.813697|
|未学習・DAGマスク・第1層|0.899867|0.816893|
|未学習・DAGマスク・第2層|0.904511|0.613866|

同一マスクの未学習NNに対して改善する条件はあるが、元のXに対する利点は確認できなかった。
今回の線形データでは元Xの共通因子抽出だけでもZと強く相関する。
学習後の両層は近い結果だが、DAG化の影響は手法・層で異なり、FA第2層では大きく低下した。
学習済み・未学習のDAG条件では、X3のMLPの10列が定数として除外された。

元XのFA因子からのX再構成MSEは0.563297、学習後第1層FAでは0.775430。
PCAではそれぞれ0.681025、0.690828。
因子とZが相関することと、観測ノイズを含むXを再構成することは別の評価である。
全18抽出器に収束警告はなかった。条件や手法をZで選び直すことはしていない。

今回の結論は「教師なし抽出の因子が既知Zと関連した」であって、
「未知の因果的共通原因を自動で特定した」ではない。
1因子の存在・個数は自動検出していない。既に閲覧した同じテストデータの探索的解析である。

## 旧コード・結果の整理

実験結果はGit管理外であり、そのまま削除するとGitでは復元できない。
共有ヘルパーも旧回帰モジュール内にあるため、今回の削除は0件とした。
106件の任意アーカイブ候補と保持必須ファイルを、依存関係・ハッシュ・Git管理情報付きで一覧化した。
旧教師あり経路は再現用として保持するが、新しい入口は教師なし抽出を中心とし、Z回帰を実行しない。

将来整理する場合は、共通ヘルパーを分離し、Git管理外の結果を明示的にバックアップした後に行う。
GraN-DAG本体、生成器、モデル、隠れ層、分割・サンプル対応、評価用Zは必ず保持する。

## テスト

```bash
bash run_in_container.sh env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

`tests/test_unsupervised.py`はZなしのfit API、読み込み禁止ガード、学習行のみの標準化、
サンプル/分割の破損検出、定数特徴、Zを大きく変更しても因子が同じこと、
保存モデルのtransform再現、評価中のfit禁止、凍結ファイル改変検出をテストする。
全28テストは実行して成功した。Zファイルが存在しない場合も抽出できることを確認した。
実データについても全18抽出器を保存モデルからtransformし直し、54行の相関・分散・
再構成誤差を独立に再計算した。因子値の最大絶対差は約1.8e-15で、rtol/atol=1e-12以内。
既存253ファイルのSHA256不変と、評価前に凍結した抽出結果の不変を確認した。

## 次の改善案（今回未実装）

1. Zを使わない平行分析や検証尤度による因子数選択と、無潜在因子対照での誤検出率評価。
2. 多シード・ブートストラップによる因子の安定性評価、非線形・複数因子条件への展開。
3. MLP内の重複ユニットの影響を抑え、MLP間の共通性を抽出する多視点モデル等の比較。

いずれも抽出条件の選択にZを使わず、因果的な共通原因の同定は別の仮定・検証として扱う。
