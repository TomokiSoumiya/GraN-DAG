"""Read Z ONLY after both models and their label-free predictions are finalized."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from latent_unsupervised import mapped
from .additional import digest, read_csv
from .evaluate import write_csv
from .unsupervised import factor_correlations, file_inventory, save_json


def number(value):
    """Undefined correlations remain missing, including in figures/reports."""
    return float('nan') if value is None else value


def convert_row(row):
    result = {}
    for key, value in row.items():
        if value in ('True', 'False'):
            result[key] = value == 'True'
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def plot_results(output, predictions, z, labels, metrics):
    figures = output/'figures'
    figures.mkdir()
    test = labels == 2
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, name in zip(axes, ('A','B')):
        row = next(r for r in metrics if r['model'] == name and r['split'] == 'test')
        ax.scatter(z[test,0], predictions[name]['mu'][test,0], s=13, alpha=.6)
        ax.set(xlabel='True Z (evaluation only)', ylabel='Posterior mean (unaltered sign)',
               title='{}: r={:.4f}, |r|={:.4f}'.format(name,number(row['pearson_r']),number(row['absolute_pearson_r'])))
    fig.tight_layout(); fig.savefig(str(figures/'test_Z_posterior.png'),dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,7))
    for name in ('A','B'):
        history = [convert_row(r) for r in read_csv(output/name/'history.csv')]
        epochs = [r['epoch'] for r in history]
        for ax, key in zip(axes.flat, ['validation_elbo','validation_reconstruction','validation_kl','train_mu_variance']):
            ax.plot(epochs,[r[key] for r in history],label=name)
            ax.set(xlabel='Epoch',ylabel=key); ax.legend()
    fig.tight_layout(); fig.savefig(str(figures/'learning_curves.png'),dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,4,figsize=(14,3.5))
    test_rows = [next(r for r in metrics if r['model'] == n and r['split']=='test') for n in ('A','B')]
    for ax,key in zip(axes,['absolute_pearson_r','x_mse_raw','elbo_raw','log_marginal_raw']):
        ax.bar(['A: X','B: H'],[number(r[key]) for r in test_rows],color=['steelblue','darkorange'])
        ax.set_title(key)
    fig.tight_layout(); fig.savefig(str(figures/'test_comparison.png'),dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,('A','B')):
        values = [next(r for r in metrics if r['model']==name and r['split']==s) for s in ('train','validation','test')]
        for order in (32,64,96):
            key = 'log_marginal_raw' if order==64 else 'log_marginal_raw_'+str(order)
            ax.plot(['train','validation','test'],[r[key] for r in values],marker='o',label=str(order)+' nodes')
        ax.set(title=name,ylabel='Prior quadrature log density (raw X)'); ax.legend()
    fig.tight_layout(); fig.savefig(str(figures/'quadrature_accuracy.png'),dpi=160); plt.close(fig)


def report(output, config, statuses, metrics, reference, verification):
    a,b = [next(r for r in metrics if r['model']==name and r['split']=='test') for name in ('A','B')]
    lines = ['# 固定GraN-DAG特徴と元Xを入力するVAEの最小比較実験', '',
        '## 実施範囲', '',
        'A（X入力）・B（固定H入力）のVAEを各1回だけ学習した。GraN-DAG再学習、人工データ再生成、グラフ学習、接続ゲート実装は行っていない。',
        'seed201、σ=1、既存の規格化tanh・sin・3乗データ、学習600・検証200・テスト200。既存のテストは過去にも参照済みのため探索的な継続評価である。', '',
        '## 実装・前処理', '',
        'A: Encoder 3→16→16→(μ,s)、B: Encoder 30→16→16→(μ,s)。隠れ層はLeakyReLU(0.01)。',
        '共通Decoder: 1→16→16→3。Uの事前分布はN(0,1)。観測分布は3変数の独立Gaussianで、変数別・サンプル共通の標準偏差を学習する。',
        '事後標準偏差はsoftplus+1e-4、観測標準偏差はsoftplus+1e-3。観測ノイズの初期標準偏差は標準化単位で1。全計算float64・CPU 1スレッド。',
        'X・Hの標準化は学習600件のみ、標準偏差ddof=0。元GraN-DAGのH再現には元のddof=1の保存済みX前処理を使用した。Hの学習時定数列は0、使用次元30。',
        '保存モデルの全パラメータを固定してHの一致を検証し、その後は保存済み配列だけをVAEへ入力。VAE OptimizerにGraN-DAGは含まれない。',
        '入力マスク（行が原因・列が予測対象）: '+str(verification['mask'])+'。DAG化前の特徴であり、生成モデルのグラフとしては使用しない。', '',
        '## 学習・モデル確定', '',
        'Adam lr=0.001、batch=64、最大500 epoch、patience=30、min_delta=0、KL係数1。訓練時は潜在サンプリング1回。検証ELBOは固定64点のq分布のGauss–Hermite求積。',
        'A・Bは同一Decoder初期重み、同じバッチ順序、同じ訓練epsilon乱数系列の対応するprefixを使用。Encoderの入力次元が異なるためパラメータ数は異なる。',
        '|モデル|最良epoch|実行epoch|終了理由|学習秒|Encoderパラメータ|Decoder＋ノイズ|合計|',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for n in ('A','B'):
        s=statuses[n]
        lines.append('|{}|{}|{}|{}|{:.3f}|{}|{}|{}|'.format(n,s['best_epoch'],s['epochs_completed'],s['stop_reason'],s['elapsed_seconds'],s['encoder_parameters'],s['decoder_parameters'],s['total_parameters']))
    lines += ['', '早期終了は最適解への収束を保証しない。A・Bとも条件変更・再学習なし。Bの表の時間には過去のGraN-DAG学習時間を含まない。', '',
        '## Z混入と保存データの保全', '',
        '学習前の単体・統合テストはpreflight_tests.txtに保存。学習と推論のプロセスではtrue_Z読み込み禁止ガードを使用し、禁止読み込み試行は0。',
        '両チェックポイントを確定してからテストを含む推論値を保存し、そのハッシュをtraining_complete.jsonへ記録。評価は別プロセスでハッシュ確認後に初めてZを読み込んだ。符号反転・Zによる選択なし。',
        'サンプルID・分割・Hの元チェックポイント・マスクを照合。元実験の全ファイルのサイズ・更新時刻・変更時刻が処理前後で一致し、主要入力のハッシュも記録した。', '',
        '## 全分割の評価', '',
        'ELBO・再構成対数尤度は3変数を合計しサンプルで平均。下表のELBO・周辺対数尤度はXの標準化のヤコビアンを補正した元単位、nats/sample。大きい方が良い。MSEは小さい方が良い。',
        '|モデル|分割|Pearson|絶対相関|Spearman|X再構成MSE（元単位）|ELBO（元単位）|KL|周辺対数尤度64点（元単位）|μ分散|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        lines.append('|{}|{}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|'.format(
            r['model'],r['split'],r['pearson_r'] if r['pearson_r'] is not None else float('nan'),
            r['absolute_pearson_r'] if r['absolute_pearson_r'] is not None else float('nan'),
            r['spearman_r'] if r['spearman_r'] is not None else float('nan'),
            r['x_mse_raw'],r['elbo_raw'],r['kl'],r['log_marginal_raw'],r['mu_variance']))
    lines += ['', '再構成値はE_q[f(U)]を64点求積した事後予測平均。観測ノイズを追加したサンプルやf(E_q[U])ではない。自己再構成性能であり外部情報からの予測性能ではない。',
        '再構成対数尤度・各XのMSE・標準化単位の値・全ノイズ分散はmetrics.csvに保存。', '',
        '## 数値精度・posterior collapse', '',
        '周辺対数尤度は共通の事前分布を積分しEncoderを使用しない。主値64点、精度確認32・96点を学習前の設定に固定。追加点数の値によるモデル・チェックポイント・主値の変更はしない。',
        '|モデル|テスト32点|テスト64点|テスト96点|64対96のサンプル別絶対差平均|最大差|精度警告|',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in (a,b):
        lines.append('|{}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{}|'.format(r['model'],r['log_marginal_raw_32'],r['log_marginal_raw'],r['log_marginal_raw_96'],r['quadrature_mean_absolute_delta_96'],r['quadrature_max_absolute_delta_96'],r['quadrature_accuracy_warning']))
    lines += ['', '精度警告は64対96の平均絶対差が0.01 natsを超える場合。求積値は近似であり、警告がある場合、小さなモデル間差を尤度改善と断定しない。q期待値の32・96点ELBOもmetrics.csvに記録。',
        'collapseの事前固定診断は平均KL<0.01かつμ分散<0.01。閾値による便宜的診断であり、真のZの復元を保証する指標ではない。']
    for r in (a,b):
        lines.append('- {}: collapse flag={}、KL={:.6f}、μ分散={:.6f}、元単位ノイズ分散={}。'.format(r['model'],r['posterior_collapse_flag'],r['kl'],r['mu_variance'],[r['noise_variance_raw_X'+str(j+1)] for j in range(3)]))
    lines += ['', '## A・Bの比較', '',
        'B−Aのテスト差: 絶対相関={:+.6f}、再構成MSE={:+.6f}、ELBO={:+.6f}、周辺対数尤度64点={:+.6f}。'.format(
            number(b['absolute_pearson_r'])-number(a['absolute_pearson_r']), b['x_mse_raw']-a['x_mse_raw'],b['elbo_raw']-a['elbo_raw'],b['log_marginal_raw']-a['log_marginal_raw']),
        '潜在表現のZとの対応、Xの自己再構成、観測密度への適合を個別に判断する。HはXの決定論的変換で新しい観測情報を増やさない。Bの違いは表現・最適化・Encoder容量・既存事前学習の効果を含む。1回の比較から一般的優位性や学習安定性の分布は判断できない。', '',
        '## 既存結果との参考比較', '',
        '|既存条件|手法|テスト絶対相関|既存OLSによるX再構成MSE|', '|---|---|---:|---:|']
    for r in reference:
        lines.append('|{}|{}|{:.6f}|{:.6f}|'.format(r['condition'],r['method'],float(r['absolute_pearson_r']),float(r['x_reconstruction_raw_mse'])))
    lines += ['', '既存PCA/FA/Kernel PCAは再学習せずCSVを読み込んだ。以前の再構成は学習600件のOLS、今回は非線形Gaussian Decoderの事後予測平均なので、再構成MSEの違いを抽出法だけの効果とは解釈できない。', '',
        '## 限界と次の課題', '',
        '今回はUを1次元の根として全Xを生成する構造を仮定した。潜在接続先・直接因果辺・潜在因子数は推定していない。Zとの相関、生成分布の適合、因果グラフ復元は別の評価である。',
        '将来の接続ゲートには、U→Xの疎性、ゲートとNN重みのスケール不定性、観測直接辺との説明の競合、DAGを保つ正規化された生成分布、因果的識別可能性の追加仮定を整理する必要がある。',
        '事後分布を単一Gaussianに限定した近似誤差、潜在崩壊、求積精度、600件での過学習を先に確認する。今回の結果に応じた追加学習・ハイパーパラメータ変更は実施しない。', '',
        '## 再現コマンド', '', '```bash',
        'bash run_in_container.sh python -B latent_vae.py --stage all', '```',
        '既存出力があれば学習は拒否される。評価だけは --stage evaluate（評価出力が未作成の場合）。全モデル・前処理・分割・バッチ順序・乱数設定・コードハッシュを保存。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def evaluate(config_path):
    requested = json.loads(Path(config_path).read_text())
    output = mapped(requested['output_dir']).resolve()
    config = json.loads((output/'config.json').read_text())
    if requested != config:
        raise ValueError('Configuration changed after fitting')
    root = mapped(config['source_dir']).resolve()
    for filename in ('metrics.csv','report.md','evaluation_status.json','figures'):
        if (output/filename).exists():
            raise FileExistsError('Refusing to overwrite evaluation: '+str(output/filename))
    complete = json.loads((output/'training_complete.json').read_text())
    if complete['status'] != 'complete' or complete['Z_read'] or complete['new_vae_training_attempts'] != 2:
        raise ValueError('Both label-free training runs must be finalized')
    for filename, expected in complete['artifacts'].items():
        if digest(output/filename) != expected:
            raise ValueError('Frozen artifact changed: '+filename)
    before = file_inventory(root)
    ids, labels = np.load(str(output/'sample_indices.npy')), np.load(str(output/'sample_split.npy'))
    np.testing.assert_array_equal(ids, np.load(str(root/'data/sample_indices.npy')))
    np.testing.assert_array_equal(labels, np.load(str(root/'data/sample_split.npy')))
    predictions = {}
    for name in ('A','B'):
        with np.load(str(output/name/'predictions.npz')) as saved:
            predictions[name] = {k: saved[k] for k in saved.files}
        np.testing.assert_array_equal(predictions[name]['sample_ids'], ids)
        np.testing.assert_array_equal(predictions[name]['split'], labels)
    # FIRST true-Z read in the entire experimental workflow, after frozen hashes.
    z = np.load(str(root/'data/true_Z.npy'))
    if z.shape != (1000,1) or not np.isfinite(z).all():
        raise ValueError('Invalid evaluation labels')
    metrics = [convert_row(r) for r in read_csv(output/'label_free_metrics.csv')]
    for row in metrics:
        code = {'train':0,'validation':1,'test':2}[row['split']]
        idx = labels == code
        row.update(factor_correlations(predictions[row['model']]['mu'][idx,0],z[idx,0]))
    write_csv(output/'metrics.csv',metrics)
    wanted = {('X','pca'),('X','factor_analysis'),('X','kernel_pca'),('C_L1','pca')}
    reference = [r for r in read_csv(root/'unsupervised_factors/metrics.csv')
                 if r['split']=='test' and (r['condition'],r['method']) in wanted]
    write_csv(output/'reference_metrics.csv', reference)
    plot_results(output,predictions,z,labels,metrics)
    statuses = json.loads((output/'checkpoints_finalized.json').read_text())
    verification = json.loads((output/'source_verification.json').read_text())
    report(output,config,statuses,metrics,reference,verification)
    if before != file_inventory(root):
        raise AssertionError('Existing source experiment changed during evaluation')
    save_json(output/'evaluation_status.json',dict(completed=True,evaluated_at=time.time(),
        separate_evaluation_process=True, all_artifact_hashes_verified_before_Z=True,
        Z_used_only_for_evaluation=True, sign_flips=0, source_files_unchanged=True,
        new_training_runs_in_evaluation=0))
    print('Evaluation complete: '+str(output/'report.md'),flush=True)
    for row in metrics:
        if row['split']=='test': print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    evaluate(parser.parse_args().config)
