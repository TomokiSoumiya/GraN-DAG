# 非線形seed201：入力変数選択だけを追加解析する

対象は保存済みの `experiments/latent_nonlinear_mixed_seed201/`。
新しいデータ生成、GraN-DAG学習、シードやノイズの追加は行わない。

```bash
bash run_in_container.sh python -m unittest discover -s tests -p test_input_selection.py -v
bash run_in_container.sh python latent_input_selection.py --stage extract
bash run_in_container.sh python latent_input_selection.py --stage evaluate
```

`--stage all` は抽出と評価を別プロセスで順番に呼ぶ省略形。
デフォルト出力は `experiments/latent_nonlinear_mixed_seed201/input_selection_analysis/`。
既に出力がある場合は上書きを拒否する。完了済みの解析を再実行する必要はない。

指定したX123、X13、X1、X3の全条件を固定し、結果にかかわらず報告する。
X123と未学習／学習済みNN第1層のPCA/FAは、既存の6抽出器・因子・再構成をコピーして再fitしない。
新規fitはX13のPCA/FAと、X1/X3単独のStandardScaler・再構成OLSだけ。
単独変数ではPCAもFAも適用せず、学習データで標準化した観測値を因子候補とする。

標準化・抽出器・因子から元の3変数XへのOLSは学習600件だけでfit。
検証200・テスト200件は変換のみ。新旧の学習行統計とOLS係数も検証する。
抽出中は既存ガードでtrue_Zの読み込みを禁止し、全因子を保存・SHA256凍結後に評価プロセスでZを開く。
既存のZ相関CSVも抽出時には読まず、評価時にコピーした6条件の値の一致確認だけに用いる。

主な結果：X13のPCAは絶対相関0.817129、全Xは0.814260、X1単独は0.897965。
未学習NN第1層PCAは0.903343、学習済みは0.904199。
X2除外だけの効果は小さく、X1の強い信号を中心にした選択・重み付けが重要と考えられる。
学習済み第1層FAの保存因子はX1のほぼ一次変換だった。
ただし、非線形変換、特徴の重複、入力配分、実効的な重み付けを今回だけで完全には分離できない。

出力：

- `report.md`: 全条件と考察、日本語。
- `comparison_table.csv`: 同じテスト200件での10条件のPearson、絶対値、Spearman、X再構成MSE。
- `metrics.csv`: 学習・検証・テストの30行。`paired_comparisons.csv`: 固定条件間の差。
- `extracted_factors.csv`: 全サンプルの因子、サンプルID、分割。Zは含まない。
- `models/`, `factors/`, `reconstructions/`: 確定した抽出器・因子・元Xの再構成。
- `figures/scatter_all_conditions.png`, `input_selection_comparison.png`: 全条件散布図と比較図。
- `metadata.json`, `extraction_status.json`, `evaluation_status.json`, `verification.json`: 再利用、fit範囲、Z利用分離、不変性の記録。

追加コードは `latent_input_selection.py`、`latent_experiments/input_selection.py`、`tests/test_input_selection.py`。
既存のPCA/FA・定数列処理・変換・相関・CSV保存・Z読み込み禁止ガードを再利用している。
既存コード、GraN-DAG本体、保存モデル・隠れ層、過去の実験出力は変更しない。
