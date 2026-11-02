# 学習効果・特徴量重複・MLP別要約の5条件比較

既存の規格化tanh・sin・Z³、seed201、σ=1、600/200/200分割を再利用する。
出力は新しい `experiments/latent_feature_controls_seed201/`。

```bash
bash run_in_container.sh python -B -m unittest discover -s tests -p test_feature_controls.py -v
bash run_in_container.sh python -B latent_feature_controls.py --stage all
```

`configs/latent_feature_controls_seed201.json` で5条件と係数生成規則を固定する。
新しいGraN-DAG・VAE学習や人工データ生成は呼び出さない。
`extract` と `evaluate` は別プロセス。全因子・OLSを保存してから評価でのみZを読む。
既存出力がある場合は拒否する。

- A: `unsupervised_factors/factors/X_pca.npy` を再利用。
- B: `B_L1_pca.npy`（未学習重み・学習後と同じマスク）を再利用。
- C: `C_L1_pca.npy`（学習済み重み・DAG化前マスク）を再利用。
- D: 各MLP H1を標準化→PC1。その3得点を標準化→PC1。fitは学習600件だけ。
- E: 各MLPの単一入力Xから10個の線形複製。係数はRandomState(201)の標準正規30個、切片0。

A/B/Cは既存のPCA・標準化器・因子得点・OLSを再fitせず読み込み、再変換との一致を照合する。
Dの0分散ブロックは最終PCAから除外し、全ブロックが情報なしの場合は停止する。
Eは標準化で係数の大きさが消え、等しい複製数ならAと符号・倍率を除いて一致する対照。

結果は `report.md`, `metrics.csv`, `comparison_table.csv`, `extracted_factors.csv`, `figures/`。
`feature_redundancy.csv` はグループ内外の相関、rank、participation rank、PC1分散比。
`D_block_pca.csv` はMLPごとのPC1、`pca_group_loadings.csv` は最終PC1のグループ寄与。
`factor_pair_diagnostics.csv` は抽出因子間の相関とアフィン対応誤差。
各因子は `factors/A.npy`〜`E.npy`、変換器は `models/`、線形係数は `linear_coefficients.csv`。
`features/D_block_scores.npy` は(1000,3)、`features/E_linear_features.npy` は(1000,3,10)。
出力の相関行列はtrain/validation/testそれぞれの記述統計であり、選択やfitに使わない。

各MLPの入力はX1予測←X2、X2予測←X3、X3予測←X1。
1シードでの探索的比較であり、Z相関を使ったモデル選択や因果グラフ復元を意味しない。
