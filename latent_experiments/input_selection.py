"""Predeclared variable subsets, label-free extraction, then separate Z evaluation.

Reuses the original PCA/FA, scalers, reconstruction definition, masks, row IDs
and frozen NN factors. No GraN-DAG training or supervised Z regression route.
"""
import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from latent_unsupervised import mapped
from .unsupervised import (SPLITS, digest, factor_correlations, file_inventory,
                           fit_extractor, forbid_latent_reads, fresh_output,
                           read_csv, save_json, transform_extractor, write_csv)


# Fixed before opening Z. Reuse all-X and NN candidates without refitting them.
CANDIDATES = [
    ('X123', 'pca', [0, 1, 2], 'X'),
    ('X123', 'factor_analysis', [0, 1, 2], 'X'),
    ('X13', 'pca', [0, 2], None),
    ('X13', 'factor_analysis', [0, 2], None),
    ('X1', 'standardized_observed', [0], None),
    ('X3', 'standardized_observed', [2], None),
    ('B_L1', 'pca', None, 'B_L1'),
    ('B_L1', 'factor_analysis', None, 'B_L1'),
    ('C_L1', 'pca', None, 'C_L1'),
    ('C_L1', 'factor_analysis', None, 'C_L1'),
]


def fit_single_observed(features_train, x_train):
    """A standardized observation, not PCA/FA; only training X accepted."""
    values, x = np.asarray(features_train), np.asarray(x_train)
    if (values.ndim != 2 or values.shape[1] != 1 or x.ndim != 2 or
            len(values) != len(x) or not np.isfinite(values).all() or not np.isfinite(x).all()):
        raise ValueError('Aligned finite training X and a single observed column required')
    scaler = StandardScaler().fit(values)
    factors = scaler.transform(values)
    design = np.column_stack([np.ones(len(x)), factors])
    return dict(method='standardized_observed', scaler=scaler, n_train=len(x),
                x_decoder=np.linalg.lstsq(design, x, rcond=None)[0],
                constant_column=bool(values.std() <= 1e-10*max(1., float(np.abs(values).max()))),
                warnings=[])


def apply_candidate(bundle, features):
    if bundle['method'] != 'standardized_observed':
        factors, _, xhat = transform_extractor(bundle, features)
        return factors, xhat
    factors = bundle['scaler'].transform(features)
    design = np.column_stack([np.ones(len(factors)), factors])
    return factors, design @ bundle['x_decoder']


def load_observed(root):
    config = json.loads((root/'config.json').read_text())
    if (config['data']['seed'] != 201 or config['training']['seed'] != 201 or
            config['split']['seed'] != 201 or config['data']['noise_std'] != 1 or
            config['data'].get('latent_functions') != ['identity', 'square', 'sin']):
        raise ValueError('Only the existing mixed nonlinear seed201, sigma=1 run is in scope')
    x = np.load(str(root/'data/observed_X.npy'))
    ids = np.load(str(root/'data/sample_indices.npy'))
    labels = np.load(str(root/'data/sample_split.npy'))
    if x.shape != (1000, 3) or not np.isfinite(x).all():
        raise ValueError('Expected 1000 aligned observations of three variables')
    np.testing.assert_array_equal(ids, np.arange(1000))
    np.testing.assert_array_equal(np.bincount(labels), [600, 200, 200])
    for split, code in SPLITS:
        np.testing.assert_array_equal(np.sort(np.load(str(root/'data'/(split+'_indices.npy')))), ids[labels==code])
    return x, ids, labels


def extract(root, output):
    root, output = Path(root).resolve(), fresh_output(root, output)
    before = file_inventory(root)
    output.mkdir()
    for folder in ('models', 'factors', 'reconstructions', 'figures'):
        (output/folder).mkdir()
    with forbid_latent_reads(root) as audit:
        x, ids, labels = load_observed(root)
        train = labels == 0
        source = root/'unsupervised_factors'
        previous = json.loads((source/'metadata.json').read_text())
        previous_status = json.loads((source/'extraction_status.json').read_text())
        if not previous_status['completed'] or previous_status['z_used_in_fit'] or previous_status['z_read_attempts']:
            raise ValueError('Source factors must already have clean label-free provenance')
        old_manifest = json.loads((source/'extraction_manifest.json').read_text())
        for name, expected in previous['input_sha256'].items():
            if digest(root/name) != expected:
                raise ValueError('Source X/hidden/checkpoint metadata changed: '+name)
        source_hashes = {str((source/name).relative_to(root)):digest(source/name)
                         for name in ('metadata.json', 'extraction_status.json', 'extraction_manifest.json')}
        for name in ('data/observed_X.npy', 'data/sample_indices.npy', 'data/sample_split.npy', 'config.json'):
            source_hashes[name] = digest(root/name)
        lookup = {entry['id']:entry for entry in previous['feature_conditions']}
        model_lookup = {(entry['condition'],entry['method']):entry for entry in previous['extractors']}
        for stage in ('train', 'random_train_mask'):
            mask = np.load(str(root/stage/'hidden/adjacency.npy'))
            if stage == 'train':
                learned_mask = mask
            else:
                np.testing.assert_array_equal(mask, learned_mask)
        if learned_mask[1].any():
            raise ValueError('This analysis assumes the saved NN has no X2 input to any MLP')
        if lookup['B_L1']['mask_sha256'] != lookup['C_L1']['mask_sha256']:
            raise ValueError('Random/learned NN masks must match')
        details, factors_csv, fit_rows = [], [], []
        for condition, method, columns, reuse in CANDIDATES:
            key = condition+'_'+method
            values = x[:,columns] if columns is not None else np.load(str(root/lookup[reuse]['source'])).reshape(1000,-1)
            if reuse is not None:
                entry = model_lookup[(reuse,method)]
                relative = [entry['model_file'], entry['factor_file'],
                            'reconstructions/'+reuse+'_'+method+'_X.npy']
                for name in relative:
                    if digest(source/name) != old_manifest[name]:
                        raise ValueError('Frozen source artifact changed: '+name)
                    source_hashes[str((source/name).relative_to(root))] = old_manifest[name]
                with (source/entry['model_file']).open('rb') as stream:
                    bundle = pickle.load(stream)
                factors, xhat = apply_candidate(bundle, values)
                np.testing.assert_allclose(factors, np.load(str(source/entry['factor_file'])), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(xhat, np.load(str(source/relative[2])), rtol=1e-12, atol=1e-12)
                for old, new in zip(relative, ['models/'+key+'.pkl', 'factors/'+key+'.npy', 'reconstructions/'+key+'_X.npy']):
                    shutil.copyfile(str(source/old), str(output/new))
            else:
                bundle = (fit_single_observed(values[train],x[train]) if len(columns)==1 else
                          fit_extractor(values[train],x[train],method,previous['settings']))
                factors, xhat = apply_candidate(bundle, values)
                with (output/'models'/(key+'.pkl')).open('wb') as stream:
                    pickle.dump(bundle, stream)
                np.save(str(output/'factors'/(key+'.npy')), factors)
                np.save(str(output/'reconstructions'/(key+'_X.npy')), xhat)
            assert factors.shape == (1000,1) and xhat.shape == (1000,3) and bundle['n_train']==600
            # Independent train-only replay checks, also for reused models.
            kept = bundle.get('keep', np.ones(values.shape[1],dtype=bool))
            np.testing.assert_allclose(bundle['scaler'].mean_,values[train][:,kept].mean(0),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(bundle['scaler'].var_,values[train][:,kept].var(0),rtol=1e-12,atol=1e-12)
            design = np.column_stack([np.ones(600),factors[train]])
            np.testing.assert_allclose(bundle['x_decoder'],np.linalg.lstsq(design,x[train],rcond=None)[0],rtol=1e-12,atol=1e-12)
            details.append(dict(condition=condition,method=method,key=key,observed_columns=columns,
                input_dimension=values.shape[1],used_dimension=int(kept.sum()),factor_shape=[1000,1],
                origin='reused_frozen' if reuse else 'new_train_only_fit',source_condition=reuse,
                warnings=bundle['warnings'],model_file='models/'+key+'.pkl',factor_file='factors/'+key+'.npy'))
            for i in range(1000):
                factors_csv.append(dict(condition=condition,method=method,sample_index=int(ids[i]),
                                        split=int(labels[i]),factor_1=float(factors[i,0])))
            for split, code in SPLITS:
                chosen = labels==code
                fit_rows.append(dict(condition=condition,method=method,split=split,n_samples=int(chosen.sum()),
                    input_dimension=values.shape[1],origin=details[-1]['origin'],
                    factor_variance=float(factors[chosen,0].var(ddof=1)),
                    x_reconstruction_raw_mse=float(np.mean((x[chosen]-xhat[chosen])**2))))
            print('Frozen candidate:',condition,method,details[-1]['origin'],flush=True)
        write_csv(output/'extracted_factors.csv', factors_csv)
        write_csv(output/'fit_metrics.csv', fit_rows)
        save_json(output/'metadata.json',dict(seed=201,noise_std=1,n_factors=1,n_samples=1000,
            splits={'train':600,'validation':200,'test':200},source_run=str(root),candidates=details,
            input_sha256=source_hashes,settings=previous['settings'],nn_training_calls=0,
            all_conditions_predeclared=True,no_component_or_condition_selection=True,
            z_used_in_fit=False,test_used_in_fit=False,train_only_scaler_and_OLS_verified=True,
            nn_mask=learned_mask.tolist(),nn_same_mask_verified=True,
            reconstruction='OLS + intercept from train factors to all three raw X columns; no Z',
            singleton_rule='train-standardized observed value, no PCA or Factor Analysis',
            source_code_sha256={name:digest(name) for name in
                ('latent_experiments/input_selection.py','latent_input_selection.py','latent_experiments/unsupervised.py')}))
        if before != file_inventory(root,excluded=output):
            raise RuntimeError('Existing outputs changed during extraction')
        save_json(output/'preserved_source_inventory.json',before)
        save_json(output/'extraction_status.json',dict(completed=True,z_read_attempts=audit['blocked_attempts'],
            all_factors_frozen_before_Z=True,new_extractors=4,reused_extractors=6,
            nn_training_calls=0,original_files_unchanged=True))
        save_json(output/'extraction_manifest.json',
                  {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})


def evaluate(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if any((output/name).exists() for name in ('metrics.csv','report.md','evaluation_status.json')):
        raise FileExistsError('Evaluation already exists; never overwrite results')
    before = file_inventory(root,excluded=output)
    metadata = json.loads((output/'metadata.json').read_text())
    status = json.loads((output/'extraction_status.json').read_text())
    if not status['completed'] or status['z_read_attempts'] or not status['all_factors_frozen_before_Z']:
        raise ValueError('Finish all label-free extraction before reading Z')
    frozen = json.loads((output/'extraction_manifest.json').read_text())
    for name, expected in frozen.items():
        if digest(output/name) != expected:
            raise ValueError('Frozen extraction changed: '+name)
    for name, expected in metadata['input_sha256'].items():
        if digest(root/name) != expected:
            raise ValueError('Source changed: '+name)
    # First permitted label read in this analysis. No fitting follows this line.
    z = np.load(str(root/'data/true_Z.npy'))
    manifest = json.loads((root/'data_manifest.json').read_text())
    if z.shape != (1000,1) or digest(root/'data/true_Z.npy') != manifest['data_files']['true_Z.npy']:
        raise ValueError('Evaluation Z differs from the original aligned data')
    x, ids, labels = load_observed(root)
    factors = {entry['key']:np.load(str(output/entry['factor_file']))[:,0] for entry in metadata['candidates']}
    original_metrics = read_csv(root/'unsupervised_factors/metrics.csv')
    factor_rows = read_csv(output/'extracted_factors.csv')
    for entry in metadata['candidates']:
        saved = [r for r in factor_rows if r['condition']==entry['condition'] and r['method']==entry['method']]
        np.testing.assert_array_equal([int(r['sample_index']) for r in saved],ids)
        np.testing.assert_array_equal([int(r['split']) for r in saved],labels)
        np.testing.assert_array_equal([float(r['factor_1']) for r in saved],factors[entry['key']])
    rows = []
    for row in read_csv(output/'fit_metrics.csv'):
        key = row['condition']+'_'+row['method']
        selected = labels==dict(SPLITS)[row['split']]
        result = dict(row,**factor_correlations(factors[key][selected],z[selected,0]))
        entry = next(e for e in metadata['candidates'] if e['key']==key)
        if entry['source_condition']:
            old = next(r for r in original_metrics if r['condition']==entry['source_condition'] and
                       r['method']==row['method'] and r['split']==row['split'])
            for field in ('pearson_r','absolute_pearson_r','spearman_r','x_reconstruction_raw_mse'):
                np.testing.assert_allclose(float(result[field]),float(old[field]),rtol=1e-12,atol=1e-12)
        rows.append(result)
    write_csv(output/'metrics.csv',rows)
    test_rows = [r for r in rows if r['split']=='test']
    write_csv(output/'comparison_table.csv',test_rows)
    lookup = {(r['condition'],r['method']):r for r in test_rows}
    comparisons = []
    for method in ('pca','factor_analysis'):
        for a,b,purpose in [('X123','X13','variable_selection'),('X13','B_L1','random_feature_transform'),
                            ('X13','C_L1','learned_feature_transform'),('B_L1','C_L1','weight_learning_same_mask')]:
            first, second = lookup[(a,method)],lookup[(b,method)]
            comparisons.append(dict(before=a,after=b,method=method,purpose=purpose,
                delta_absolute_pearson_r=second['absolute_pearson_r']-first['absolute_pearson_r'],
                delta_x_reconstruction_mse=float(second['x_reconstruction_raw_mse'])-float(first['x_reconstruction_raw_mse'])))
    write_csv(output/'paired_comparisons.csv',comparisons)
    make_figures(output,test_rows,factors,z[:,0],labels==2)
    make_report(output,test_rows,comparisons)
    assert before == file_inventory(root,excluded=output)
    assert all(digest(output/name)==expected for name,expected in frozen.items())
    save_json(output/'evaluation_status.json',dict(completed=True,nn_training_calls=0,
        original_files_unchanged=True,frozen_extraction_unchanged=True,reused_metrics_match_original=True,
        z_read_only_after_all_factors_frozen=True,z_used_for_selection=False,n_test=200,
        all_predeclared_conditions_reported=True,metric_rows=len(rows),test_rows=len(test_rows)))
    print('Evaluation complete; no NN training:',output,flush=True)


def make_figures(output, rows, factors, z, test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [r['condition']+'_'+r['method'] for r in rows]
    fig, axes = plt.subplots(2,5,figsize=(19,8))
    for ax, key, row in zip(axes.flat,keys,rows):
        ax.scatter(z[test],factors[key][test],s=10,alpha=.6)
        ax.set_title(key+'\nr={:.4f}'.format(row['pearson_r']))
        ax.set_xlabel('True Z (evaluation only)'); ax.set_ylabel('Frozen factor (unaligned sign)')
    fig.tight_layout(); fig.savefig(str(output/'figures/scatter_all_conditions.png'),dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(14,9))
    for ax,field,title in [(axes[0],'absolute_pearson_r','Final-test |Pearson r|'),
                           (axes[1],'x_reconstruction_raw_mse','Final-test raw X reconstruction MSE')]:
        ax.bar(range(len(rows)),[float(row[field]) for row in rows])
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(keys,rotation=35,ha='right')
        ax.set_title(title)
        if field=='absolute_pearson_r': ax.set_ylim(0,1)
    fig.tight_layout(); fig.savefig(str(output/'figures/input_selection_comparison.png'),dpi=150); plt.close(fig)


def make_report(output, rows, comparisons):
    lookup = {(r['condition'],r['method']):r for r in rows}
    r = lambda c,m='pca':lookup[(c,m)]['absolute_pearson_r']
    lines = ['# 入力変数選択の追加解析：非線形seed201・σ=1', '',
        'GraN-DAGの再学習なし。既存1000件の同じ学習600・検証200・テスト200件を使用。',
        'Z~N(0,1)、X1=2Z+ε1、X2=Z²+ε2、X3=sin(Z)+ε3。ノイズは独立な標準正規。', '',
        '## 抽出と評価の分離', '',
        '- 指定4変数条件とNN第1層対照を事前固定し、全条件を報告する。相関で条件を選ばない。',
        '- X123とNNのPCA/FAの計6モデル・因子・OLS再構成を凍結結果からコピーし、再fitしなかった。',
        '- X13に1因子PCA/FAを適用。X1/X3単独は学習行で標準化した値で、PCA/FAを実施しない。',
        '- 新規fitはX13の抽出器2件と単独標準化2件のみ。全標準化・抽出・OLSは学習600行だけ。',
        '- OLS＋切片は因子から元のX3変数すべてを再構成する。MSEは全200×3要素の生スケール平均。',
        '- 全10因子候補を保存してハッシュ凍結した後、別プロセスでZと既存評価CSVを読む。',
        '- 符号をZへ合わせない。検証・テストで再fit、ハイパーパラメータ調整はしない。',
        '- 学習済み・未学習NNは同じ保存済み3辺マスク。どちらもX2を入力に使わない。', '',
        '## テスト200件の全結果', '',
        '|条件|方法|Pearson r|絶対r|Spearman|X再構成MSE|', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('|{}|{}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|'.format(row['condition'],row['method'],
            row['pearson_r'],row['absolute_pearson_r'],row['spearman_r'],float(row['x_reconstruction_raw_mse'])))
    lines += ['', '## 固定条件間の差分', '',
              '|比較|方法|絶対rの差(after−before)|X-MSEの差|','|---|---|---:|---:|']
    for row in comparisons:
        lines.append('|{}→{}|{}|{:+.6f}|{:+.6f}|'.format(row['before'],row['after'],row['method'],
                     row['delta_absolute_pearson_r'],row['delta_x_reconstruction_mse']))
    lines += ['', '## 研究上の質問への回答', '',
        '1. X13のPCAは|r|={:.6f}、全XのPCAは{:.6f}。差は{:+.6f}。'.format(r('X13'),r('X123'),r('X13')-r('X123')),
        '2. X13のPCAと未学習NN第1層PCA（{:.6f}）との差は{:+.6f}。'.format(r('B_L1'),r('X13')-r('B_L1')),
        '3. X13のPCAと学習済みNN第1層PCA（{:.6f}）との差は{:+.6f}。'.format(r('C_L1'),r('X13')-r('C_L1')),
        '4. X1単独は|r|={:.6f}、X3単独は{:.6f}。変数を除く効果と、残した変数の重み付けは分けて解釈する。'.format(
            r('X1','standardized_observed'),r('X3','standardized_observed')),
        '5. 未学習NNと学習済みNNのPCA差は{:+.6f}。この差だけから非線形性そのものの寄与は特定できない。'.format(r('C_L1')-r('B_L1')), '',
        '## 解釈の限界', '',
        '- マスクによる入力除外、MLPごとの入力集合、ユニット重複、実効的な重み付け、非線形変換は別要因。',
        '- 2変数を各々標準化したPCAは高いノイズを持つ変数にも分散1を与えるため、変数数を増やしてもZとの相関が上がるとは限らない。',
        '- 2観測変数の1因子FAは負荷量・独自分散の一般的な一意同定ができず、結果はアルゴリズムの1つの解として扱う。',
        '- 非線形変換の固有効果を調べるには、同じ入力配分・重複構造の線形対照等が必要だが、今回は追加していない。',
        '- 元のXの再構成も線形OLSであり、Z²を含むX全体の非線形再構成能力を直接評価するものではない。',
        '- 既存研究で閲覧済みの同一テストを再利用した探索的解析。単一条件の差を一般的な有意差・優位性とみなさない。',
        '- 高い相関は共通変動の要約を示すが、因果的な潜在共通原因の自動同定を示さない。', '',
        '## 保存物', '',
        '`comparison_table.csv`: テスト10条件、`metrics.csv`: 全分割30条件、`paired_comparisons.csv`: 固定比較差分。',
        '`extracted_factors.csv`: サンプルID・分割付き因子10000行。`models/`, `factors/`, `reconstructions/`: 保存済み抽出器と推論結果。',
        '`figures/`: 全条件散布図と相関・X-MSE比較。`metadata.json`, `extraction_status.json`, `evaluation_status.json`: 学習・Z利用分離・不変性の記録。', '']
    warnings = [(e['key'],e['warnings']) for e in json.loads((output/'metadata.json').read_text())['candidates'] if e['warnings']]
    for key, messages in warnings:
        lines.append('警告 {}: {}'.format(key,'; '.join(messages)))
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('experiments/latent_nonlinear_mixed_seed201'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--stage',choices=['all','extract','evaluate'],default='all')
    args = parser.parse_args()
    root = mapped(args.run_dir).resolve()
    output = mapped(args.output_dir).resolve() if args.output_dir else root/'input_selection_analysis'
    if args.stage=='all':
        for stage in ('extract','evaluate'):
            subprocess.check_call([sys.executable,'latent_input_selection.py','--run-dir',str(root),
                                   '--output-dir',str(output),'--stage',stage])
    elif args.stage=='extract':
        extract(root,output)
    else:
        evaluate(root,output)
