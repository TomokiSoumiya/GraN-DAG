# 保存済みZ_hatを追加する探索的グラフ診断

設定：`configs/latent_zhat_seed201.json`。
元データ：`experiments/latent_nonlinear_normalized_seed201/`。
オラクル参照：`experiments/latent_oracle_normalized_seed201/`。
新出力：`experiments/latent_zhat_normalized_seed201/`。

## 実行経路

```bash
bash run_in_container.sh python latent_zhat_graph.py --stage reproduce
bash run_in_container.sh python latent_zhat_graph.py --stage validate
bash run_in_container.sh python -m unittest discover -s tests -p test_zhat_graph.py -v
bash run_in_container.sh python latent_zhat_graph.py --stage train
bash run_in_container.sh python latent_zhat_graph.py --stage evaluate
```

各コマンドの完了・成功を確認してから次へ進む。完了済みの同じ出力へ上書きしない。
新規GraN-DAG学習はtrainの1回だけ。失敗・未収束でも設定変更や再学習はしない。
元の3変数モデル、PCA、オラクル、比較モデルは学習しない。

## 因子の再現とZの分離

対象は`C_L1_pca`：元の学習済みNN・学習直後マスク・第1層・PCA1成分。
元の`train/model.pkl`と`train/hidden/hidden_layer_001.npy`を全行で検証。
保存済みの定数列判定、学習600件のStandardScaler/PCAを検証しtransformのみ行う。
PCA.fitは再現処理中に呼び出し禁止としている。
元の保存因子と一致を確認し、符号・尺度・配列のバイトをそのままコピーして凍結する。
原checkpoint・標準化器/PCA・出力配列のパスとSHA256は`zhat/provenance.json`。

reproduce中はtrue_Z読み込みを禁止。因子と結合入力を凍結した後、別のvalidateプロセスで初めてZを読む。
学習・検証・テストの元指標、符号付き相関、報告値0.825139と照合する。
照合は再現性のゲートであって、Zを使った符号・成分・ハイパーパラメータの選択ではない。
一致しなければ新しいNN学習には進まない。

実際の再計算の最大差は約1.78e-15、コピーした因子は元ファイルとSHA256一致。
テストPearsonは−0.8251390260894076、絶対相関0.8251390260894076。
Z_hatの分散(ddof=1)は学習15.3973073966、検証14.7714666352、テスト14.7403798612。

## 唯一の新規学習

入力順は`[Z_hat,X1,X2,X3]`。元X、ID、600/200/200分割、信号規格化係数は変更しない。
学習用ディレクトリには学習＋検証の800行だけを置く。真のZ列はコピーしない。
元のオラクルのtraining_options.jsonをそのまま再利用する。
`latent_experiments.oracle.train_once`の数値処理を共有し、入力メタデータはZ_hatに訂正して記録する。
そのため共有関数の旧ログ文にはoracleと出るが、この新実験は真のZを入力するオラクルではない。
NN・損失・DAG制約・RMSprop・CAMは変更していない。

NonLinGaussANM・2層×10ユニット・leaky-ReLU、seed201、初期12方向。
Z_hatの向きは固定しない。ローダー必須のDAG1.npyはゼロの診断用プレースホルダー。
元の自動生成adjacency.png内の正解欄はプレースホルダーであり、因果的正解ではない。
最終テスト200件を学習・CAM・選択に使わない。

## 指標の意味

- X同士の辺は、向きに依存しない**異なるペア数**と、有向辺数の両方を保存。双方向はペア1、有向辺2。
- Z_hat―各Xiの接続有無・両方向を記録。3接続先のカバー率は探索的指標で、因果的正解率ではない。
- 形式的SHD/P/R/F1は仮想的にZ_hatをZに対応づけた星型との形状一致度にすぎない。
- 真のZを使うオラクルのSHDと、同じ因果復元指標として比較しない。

Z_hatは固定NN/PCAによるXの関数なので、Var(Z_hat|全X)=0。
単独Xとの相関が1でなくても、全Xとの決定論的依存は消えない。
正の独立ノイズを持つ通常の4変数連続分布というモデル仮定に問題があり、
真のZも依然未観測である。推定辺を真の潜在共通原因の因果的同定と解釈しない。
同じテストを過去にも閲覧した探索的解析であり、新規の未閲覧テストによる追試でもない。

## 保存物

- `zhat/Z_hat.npy`, `scaler_pca.pkl`, `provenance.json`：保存因子・変換器・元checkpoint情報。
- `reproduction_status.json`, `reproduction_validation.json`：Zなし再現と固定後のZ照合。
- `zhat_Z_evaluation.csv`, `zhat_X_correlations.csv`：全分割での相関。
- `training.log`, `training_status.json`, `train/`, `to-dag/`, `cam-pruning/`：唯一の新規NN学習と後処理。
- `evaluation/adjacency_*.csv`, `estimated_edges.csv`, `stage_changes.csv`：新グラフ各段階。
- `evaluation/graph_comparison.csv`, `X_X_pairs.csv`, `factor_connections.csv`：A/B/C参考比較。
- `evaluation/formal_shape_metrics.csv`：明確に形式的な参考形状指標。
- `evaluation/CAM_comparison.png`, `report.md`：比較図と日本語詳細レポート。

未収束の場合はCの未生成段階を明記し、再学習せず報告する。
