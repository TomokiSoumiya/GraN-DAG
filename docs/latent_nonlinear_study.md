# 非線形人工データ・新規5シード実験

## 実際の生成式

`configs/latent_nonlinear.json` は線形設定を継承し、潜在変数の変換を`tanh`、
ノイズの標準偏差を0.5に変更する。

```text
Z ~ Normal(0, 1)
X1 = 2*tanh(Z) + epsilon1
X2 = 3*tanh(Z) + epsilon2
X3 =   tanh(Z) + epsilon3
epsilon_i ~ Normal(0, 0.5^2), mutually independent and independent of Z
```

真の潜在辺はZ→X1、Z→X2、Z→X3。観測変数間の直接辺は0本。
隣接行列は行が原因・列が結果。観測DAGでは潜在共通原因を表現できないため、
推定したX間辺を真の直接因果辺と解釈しない。

1000件、学習600・検証200・最終テスト200。2隠れ層、各10ユニット。
個別MLPの特徴は10次元、全MLP連結は30次元。

## 実行順序

初回のseed42は既存の設定をそのまま使用する。

```bash
bash run_in_container.sh python latent_experiment.py --config configs/latent_nonlinear.json
bash run_in_container.sh python latent_additional_analysis.py --run-dir experiments/latent_nonlinear
```

元の実行スクリプトに加え、既存の追加解析で隠れ層のExtraTrees回帰と
固定マスク・固定重みの比較を行う。GraN-DAGの損失・DAG制約・最適化は変更しない。
`train`、`to-dag`、`cam-pruning`を区別する。

新規5seedの計画は `configs/latent_nonlinear_study.json`。
新しい出力先で次の順に実行する（準備済みの場所でprepareを繰り返すと停止する）。

```bash
bash run_in_container.sh python latent_nonlinear_study.py --stage prepare
# 初回seed42の学習・隠れ層の非線形評価が成功した後に実行
bash run_in_container.sh python latent_nonlinear_study.py --stage run
bash run_in_container.sh python latent_nonlinear_study.py --stage aggregate
```

新規seedは101、102、103、104、105。生成・分割・NN学習のseedを同時に変更し、
回帰CV/木のseedはseed+1とする。したがって集計は**データと学習双方の変動**を含み、
NN初期化だけの変動ではない。3プロセスまで同時に実行するが、各プロセスは1スレッド。
既存CAMのRスクリプトは`set.seed(42)`固定であり、この処理は変更しない。
`run`は各seedで元の`latent_experiment.py`と`latent_additional_analysis.py`を順に呼び出す。

初回42は動作確認として独立に報告し、5seedの平均・SDから除く。
平均と標本標準偏差（ddof=1）を計算する。SDは標準誤差や信頼区間ではない。
同じseed内の差分を先に求め、学習やDAG化の効果についてはペア差分の平均・SDも保存する。
事前指定した全5seedの完了が必要で、良い結果だけの部分集計やseedの差し替えは行わない。

## 条件と漏洩防止

- A：未学習NN・初期マスク
- B/C：未学習/学習済みNN・同じ学習後マスク
- D/E：未学習/学習済みNN・同じDAG化後マスク
- F/G：未学習/学習済みNN・同じCAM後マスク

重みは同じ実際の初期チェックポイントまたは学習終了時チェックポイントを固定する。
元の保存済みモデルに変更を加えず、必要な場合だけコピーのマスクを変更して抽出する。
全層・全サンプルのキャッシュと固定NNの再計算を照合する。

GraN-DAGの入力フォルダに入るのは観測Xの学習・検証800件と直接辺行列のみ。
Zと最終テストXは入らない。標準化は学習600件の統計量で行う。
CAMも同じ学習・検証のXだけを利用する。
回帰器は学習600件内の5分割CVで選択し、テストはスコア算出にのみ使用する。
候補設定は元の設定と同じまま固定し、テストの結果でモデルや候補を選び直さない。

Ridgeは学習内StandardScalerとalphaのCV。ExtraTreesは100本、
max_depth=None/5/10、min_samples_leaf=2/5をCVで選択する。
両手法を併記し、テストでどちらかを採用する処理はない。

## 保存先

```text
experiments/latent_nonlinear/             # 初回seed42
  analysis/                             # 元の評価
  additional_analysis/                  # 各MLP・両層・両回帰、CSV/PNG

experiments/latent_nonlinear_multiseed/
  plan.json                             # 継承解決した条件・seed・コードハッシュ
  protected_runs.json                    # 既存線形/p10結果の全ファイルハッシュ
  configs/seed_101.json ... seed_105.json # seedごとの完全な設定
  logs/seed_101.log ... seed_105.log
  jobs.json                             # 全seedの成否・時間
  runs/seed_101/ ... seed_105/            # 各モデル・データ・隠れ層・評価
  summary/
    report.md
    metrics_by_seed.csv                  # 全5seedの各指標
    metrics_mean_std.csv                 # 各MLP/層/条件/回帰の平均・SD
    paired_by_seed.csv
    paired_mean_std.csv                  # 同じseed内の学習/マスク効果
    graphs_by_seed.csv
    run_verification.csv
    pilot_metrics.csv
    pilot_graphs.csv
    linear_vs_nonlinear.csv
    status.json
    figures/*.png
```

同じ出力先の学習や集計の再実行は拒否する。新しい実験にはstudy設定の
`output_dir`と`pilot_dir`を新しい場所にして実行する。
集計だけ再計算する場合は新しい名前を指定できる。

```bash
bash run_in_container.sh python latent_nonlinear_study.py --stage aggregate --summary-name summary_repeat
```

既存の線形・p10結果は全ファイルのハッシュ不変を確認する。
各非線形runでもサンプル対応、最終テスト除外、保存済み活性値、予測CSVからの全R²/MSE再計算を検証する。

## 解釈上の制限

線形設定はnoise_std=1、非線形設定は0.5であり、関数形とノイズが同時に変化する。
結果の差を非線形性だけの効果とはみなさない。
Zの予測性能は、情報量そのものや潜在変数の自動発見を測定したものではない。
5seedの同じ生成条件での結果を、任意の非線形SEMへ一般化しない。

## テスト

```bash
bash run_in_container.sh env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

`tests/test_nonlinear_study.py`は元設定からの変更範囲、計画の不変性、上書き防止、
標本SD、重複seed拒否、初回実験の成功確認、および既存CLIの呼び出しをテストする。

## 今回の実行済み結果

初回seed42と新規101〜105の全6件で、生成・GraN-DAG学習・DAG化・CAM・隠れ層抽出・
Ridge/ExtraTrees評価が完了した。初回は28,529反復、新規5件はそれぞれ21,274、22,282、
22,267、29,519、26,003反復で収束した。
以下は新規5件のみの平均±標本SD。特徴は全MLP連結。

|特徴|Ridge R²|Ridge MSE|ExtraTrees R²|ExtraTrees MSE|
|---|---:|---:|---:|---:|
|元のX|0.8854±0.0110|0.1101±0.0120|0.8882±0.0150|0.1076±0.0168|
|学習後 第1層|0.8904±0.0117|0.1055±0.0145|0.8829±0.0144|0.1127±0.0165|
|学習後 第2層|0.8885±0.0120|0.1073±0.0139|0.8808±0.0139|0.1147±0.0169|
|DAG化後 第1層|0.8836±0.0148|0.1119±0.0153|0.8792±0.0165|0.1162±0.0174|
|DAG化後 第2層|0.8815±0.0178|0.1138±0.0172|0.8780±0.0165|0.1174±0.0184|

同じ学習後マスクの未学習NNとの第1層連結比較では、全5seedで小幅な改善があった。
ペア差分R²はRidgeで+0.00600±0.00288、ExtraTreesで+0.00141±0.00098。
ただし元のXのExtraTreesより学習済み隠れ層のExtraTreesは低く、隠れ層の一般的優位性は主張しない。
同じ回帰手法での改善と、別の読み出し手法を含む比較を区別する。

学習直後の個別MLPでは、平均R²が最も高いのはX3の第1層だった。
RidgeはR²=0.8494±0.0315、MSE=0.1431±0.0221。
第1層・第2層で大きな差はなく、今回の新規5seedでは線形seed42のような連結第2層Ridgeの大幅低下は見られなかった。

DAG化は新規5件中4件で辺を1本除き、その結果親を失ったMLPは定数になった。
seed101は学習直後からDAGであり変化しなかった。
親の有無がseedで異なるため、個別MLPの平均と大きなSDはモデル番号そのものの優劣ではない。
全6件でCAMマスクはDAG化後と同じで、CAMによる追加変化はなかった。

追加解析の全評価の予測CSVからR²/MSEを再計算し、対応する最終テストIDとZの一致も確認した。
新規5件の570評価行から114グループの平均・SDを計算し、独立に再計算して一致を確認した。
既存線形・p10実験の計316ファイルはハッシュ不変。全22テストは実行して成功した。

詳細な各MLP・各層・各段階のR²/MSEとSDは
`experiments/latent_nonlinear_multiseed/summary/report.md`と`metrics_mean_std.csv`、
学習・マスクの効果は`paired_mean_std.csv`に保存済み。
ノイズ水準を一致させた対照実験、別の非線形生成関数、10変数条件は今回未実施。
