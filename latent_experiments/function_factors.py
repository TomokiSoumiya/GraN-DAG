"""One predeclared 18-condition experiment; X-only fitting precedes all Z reads."""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

from latent_unsupervised import mapped
from .input_selection import apply_candidate, fit_single_observed
from .unsupervised import (SPLITS, digest, factor_correlations, file_inventory, fit_extractor,
    forbid_latent_reads, fresh_output, load_features, read_csv, save_json, write_csv)


def fit_kernel(features_train, x_train, settings):
    """Median-distance RBF rule on training standardized X only, never Z."""
    features_train, x_train = np.asarray(features_train), np.asarray(x_train)
    if (features_train.ndim!=2 or x_train.ndim!=2 or len(features_train)!=len(x_train) or
            not np.isfinite(features_train).all() or not np.isfinite(x_train).all()):
        raise ValueError('Aligned finite training-only X required')
    scaler=StandardScaler().fit(features_train)
    scaled=scaler.transform(features_train)
    distances=pdist(scaled,metric='sqeuclidean')
    positive=distances[distances>0]
    if not len(positive):
        raise ValueError('RBF distance rule requires nonconstant training observations')
    median=float(np.median(positive))
    gamma=1/(2*median)
    if settings['kernel_pca']['kernel']!='rbf' or settings['kernel_pca']['eigen_solver']!='dense':
        raise ValueError('Fixed RBF / dense eigensolver only')
    model=KernelPCA(n_components=1,kernel='rbf',gamma=gamma,eigen_solver='dense',
                    fit_inverse_transform=False,random_state=settings['seed']).fit(scaled)
    factors=model.transform(scaled)
    design=np.column_stack([np.ones(len(x_train)),factors])
    return dict(method='kernel_pca',scaler=scaler,extractor=model,n_train=len(x_train),warnings=[],
        median_positive_squared_train_distance=median,gamma=gamma,
        n_positive_training_pairs=len(positive),x_decoder=np.linalg.lstsq(design,x_train,rcond=None)[0])


def apply_model(bundle, values):
    if bundle['method']!='kernel_pca':
        return apply_candidate(bundle,values)
    factors=bundle['extractor'].transform(bundle['scaler'].transform(values))
    return factors,np.column_stack([np.ones(len(values)),factors])@bundle['x_decoder']


def observed_diagnostics(key,factors,x,labels):
    """All observed-X diagnostics fixed in advance; no latent labels involved."""
    train,test=labels==0,labels==2
    rows=[]
    scale=float(factors[train].std())
    for j in range(x.shape[1]):
        design=np.column_stack([np.ones(int(train.sum())),x[train,j]])
        intercept,slope=np.linalg.lstsq(design,factors[train],rcond=None)[0]
        residual=factors[test]-(intercept+slope*x[test,j])
        corr=factor_correlations(factors[test],x[test,j])
        rows.append(dict(key=key,observed_variable='X'+str(j+1),
            test_pearson_with_X=corr['pearson_r'],test_spearman_with_X=corr['spearman_r'],
            affine_intercept_train=float(intercept),affine_slope_train=float(slope),
            affine_test_RMSE_over_train_factor_std=float(np.sqrt(np.mean(residual**2))/scale) if scale>1e-10 else None,
            affine_fit_split='train only',Z_used=False))
    return rows


def extract(root,output,settings):
    root,output=Path(root).resolve(),fresh_output(root,output)
    before=file_inventory(root)
    output.mkdir()
    for directory in ('models','factors','reconstructions','figures'):
        (output/directory).mkdir()
    with forbid_latent_reads(root) as audit:
        cfg=json.loads((root/'config.json').read_text())
        if (cfg['data']['seed']!=201 or cfg['training']['seed']!=201 or cfg['split']['seed']!=201 or
                cfg['data']['noise_std']!=1 or cfg['data']['latent_functions']!=['tanh','sin','cubic'] or
                cfg['data']['signal_normalization']!={'fit_split':'train','ddof':0} or settings['n_factors']!=1):
            raise ValueError('Only the specified normalized seed201 sigma1 experiment is in scope')
        features,x,ids,labels,verification=load_features(root,output,settings)
        assert x.shape==(1000,3) and [int((labels==c).sum()) for _,c in SPLITS]==[600,200,200]
        feature_map={name:(values,info) for name,values,info in features}
        candidates=[]
        for j in range(3):
            candidates.append(('X'+str(j+1),'standardized_observed',x[:,j:j+1],dict(source='data/observed_X.npy',columns=[j])))
        for method in ('pca','factor_analysis','kernel_pca'):
            candidates.append(('X',method,x,feature_map['X'][1]))
        for code in ('B','C','E'):
            for layer in (1,2):
                name=code+'_L'+str(layer)
                values,info=feature_map[name]
                for method in ('pca','factor_analysis'):
                    candidates.append((name,method,values,info))
        train=labels==0
        details,factor_rows,fit_rows,diagnostics=[],[],[],[]
        source_files={'config.json','data/observed_X.npy','data/sample_indices.npy','data/sample_split.npy',
                      'normalization_mean.npy','normalization_std.npy','training_status.json','data/config.json'}
        for condition,method,values,info in candidates:
            key=condition+'_'+method
            if method=='standardized_observed':
                bundle=fit_single_observed(values[train],x[train])
            elif method=='kernel_pca':
                bundle=fit_kernel(values[train],x[train],settings)
            else:
                bundle=fit_extractor(values[train],x[train],method,settings)
            factors,xhat=apply_model(bundle,values)
            assert factors.shape==(1000,1) and xhat.shape==(1000,3) and bundle['n_train']==600
            with (output/'models'/(key+'.pkl')).open('wb') as stream:
                pickle.dump(bundle,stream)
            np.save(str(output/'factors'/(key+'.npy')),factors)
            np.save(str(output/'reconstructions'/(key+'_X.npy')),xhat)
            source_files.add(info['source'])
            detail=dict(condition=condition,method=method,key=key,input_dimension=values.shape[1],
                used_dimension=int(bundle['keep'].sum()) if 'keep' in bundle else values.shape[1],
                model_file='models/'+key+'.pkl',factor_file='factors/'+key+'.npy',
                source=info['source'],factor_shape=[1000,1],warnings=bundle['warnings'])
            if method=='kernel_pca':
                detail.update(gamma=bundle['gamma'],median_positive_squared_train_distance=
                    bundle['median_positive_squared_train_distance'],n_positive_training_pairs=bundle['n_positive_training_pairs'])
            if method=='factor_analysis':
                detail['fa_iterations']=int(bundle['extractor'].n_iter_) if bundle['extractor'] is not None else None
            details.append(detail)
            for i in range(1000):
                factor_rows.append(dict(condition=condition,method=method,sample_index=int(ids[i]),split=int(labels[i]),factor_1=float(factors[i,0])))
            for split,code in SPLITS:
                chosen=labels==code
                fit_rows.append(dict(condition=condition,method=method,split=split,n_samples=int(chosen.sum()),
                    input_dimension=values.shape[1],used_dimension=detail['used_dimension'],
                    factor_variance=float(factors[chosen,0].var(ddof=1)),
                    x_reconstruction_raw_mse=float(np.mean((x[chosen]-xhat[chosen])**2))))
            diagnostics.extend(observed_diagnostics(key,factors[:,0],x,labels))
            print('Frozen factor:',key,'shape',factors.shape,'kept',detail['used_dimension'],flush=True)
        mask_rows=[]
        masks={}
        for code,stage in [('B','random_train_mask'),('C','train'),('E','to-dag')]:
            mask=np.load(str(root/stage/'hidden/adjacency.npy'))
            masks[code]=mask.tolist()
            source_files.add(stage+'/hidden/adjacency.npy')
            for target in range(3):
                for source in range(3):
                    mask_rows.append(dict(condition=code,source='X'+str(source+1),target='X'+str(target+1),used=bool(mask[source,target])))
        assert masks['B']==masks['C']
        write_csv(output/'input_masks.csv',mask_rows)
        write_csv(output/'observed_factor_diagnostics.csv',diagnostics)
        write_csv(output/'extracted_factors.csv',factor_rows)
        write_csv(output/'fit_metrics.csv',fit_rows)
        save_json(output/'metadata.json',dict(settings=settings,source_run=str(root),n_samples=1000,
            splits={'train':600,'validation':200,'test':200},candidates=details,masks=masks,
            source_sha256={name:digest(root/name) for name in sorted(source_files)},
            code_sha256={name:digest(name) for name in ('latent_experiments/function_factors.py',
                'latent_experiments/unsupervised.py','latent_experiments/input_selection.py','latent_experiments/data.py')},
            frozen_feature_verification=verification,z_used_in_fit=False,z_used_for_selection=False,
            n_factors_fixed=1,no_test_tuning=True,all_conditions_predeclared=True,
            singleton_max_correlation_policy='post-hoc Z-based diagnostic only, never unsupervised selection',
            preprocessing='all extractor standardization, gamma, PCA/FA/KPCA, OLS on training 600 only',
            diagnostics='all Xj -> factor affine OLS fits on train; frozen diagnostics are label-free',
            reconstruction='factor -> raw 3-column X; train-only OLS + intercept, not nonlinear reconstruction',
            factor_sign='unchanged; no Z alignment'))
        assert before==file_inventory(root,excluded=output)
        save_json(output/'preserved_source_inventory.json',before)
        save_json(output/'extraction_status.json',dict(completed=True,z_read_attempts=audit['blocked_attempts'],
            candidate_count=len(details),all_factors_saved_before_Z=True,nn_training_calls=0,
            factor_shape=[1000,1],original_files_unchanged=True))
        save_json(output/'extraction_manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})


def evaluate(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if any((output/name).exists() for name in ('metrics.csv','report.md','evaluation_status.json')):
        raise FileExistsError('Never overwrite an existing evaluation')
    status=json.loads((output/'extraction_status.json').read_text())
    if not status['completed'] or status['z_read_attempts'] or not status['all_factors_saved_before_Z']:
        raise ValueError('Freeze all factors without Z before evaluating')
    before=file_inventory(root,excluded=output)
    frozen=json.loads((output/'extraction_manifest.json').read_text())
    for name,expected in frozen.items():
        if digest(output/name)!=expected: raise ValueError('Frozen extraction changed: '+name)
    metadata=json.loads((output/'metadata.json').read_text())
    for name,expected in metadata['source_sha256'].items():
        if digest(root/name)!=expected: raise ValueError('Source changed: '+name)
    # First permitted true_Z read since factor extraction began. No fit follows.
    z=np.load(str(root/'data/true_Z.npy'))
    assert z.shape==(1000,1)
    data_manifest=json.loads((root/'data_manifest.json').read_text())
    assert digest(root/'data/true_Z.npy')==data_manifest['data_files']['true_Z.npy']
    labels=np.load(str(root/'data/sample_split.npy'))
    ids=np.load(str(root/'data/sample_indices.npy'))
    factor_map={entry['key']:np.load(str(output/entry['factor_file']))[:,0] for entry in metadata['candidates']}
    csv_factors=read_csv(output/'extracted_factors.csv')
    for entry in metadata['candidates']:
        stored=[row for row in csv_factors if row['condition']==entry['condition'] and row['method']==entry['method']]
        np.testing.assert_array_equal([int(row['sample_index']) for row in stored],ids)
        np.testing.assert_array_equal([int(row['split']) for row in stored],labels)
        np.testing.assert_array_equal([float(row['factor_1']) for row in stored],factor_map[entry['key']])
    rows=[]
    for row in read_csv(output/'fit_metrics.csv'):
        selected=labels==dict(SPLITS)[row['split']]
        result=dict(row,**factor_correlations(factor_map[row['condition']+'_'+row['method']][selected],z[selected,0]))
        rows.append(result)
    write_csv(output/'metrics.csv',rows)
    tests=[row for row in rows if row['split']=='test']
    write_csv(output/'comparison_table.csv',tests)
    lookup={(row['condition'],row['method']):row for row in tests}
    comparisons=[]
    for method in ('pca','factor_analysis'):
        for layer in (1,2):
            for a,b,purpose in [('B_L'+str(layer),'C_L'+str(layer),'learning_same_mask'),
                ('C_L'+str(layer),'E_L'+str(layer),'dag_same_weights'),('X','C_L'+str(layer),'raw_X_vs_learned')]:
                first,second=lookup[(a,method)],lookup[(b,method)]
                delta=(second['absolute_pearson_r']-first['absolute_pearson_r']
                       if first['correlation_defined'] and second['correlation_defined'] else None)
                comparisons.append(dict(before=a,after=b,method=method,purpose=purpose,
                    delta_absolute_pearson_r=delta,delta_X_MSE=float(second['x_reconstruction_raw_mse'])-float(first['x_reconstruction_raw_mse'])))
    write_csv(output/'paired_comparisons.csv',comparisons)
    make_figures(output,tests,factor_map,z[:,0],labels==2)
    make_report(root,output,metadata,tests,comparisons)
    assert before==file_inventory(root,excluded=output)
    assert all(digest(output/name)==expected for name,expected in frozen.items())
    save_json(output/'evaluation_status.json',dict(completed=True,z_used_in_fit=False,z_used_for_selection=False,
        all_18_conditions_reported=len(tests)==18,extraction_hashes_unchanged=True,original_files_unchanged=True,
        no_fit_in_evaluation=True,test_samples=200,nn_training_calls=0))
    print('All factor evaluation complete:',output,flush=True)


def fmt(value):
    return '未定義' if value in (None,'') else '{:.6f}'.format(float(value))


def make_figures(output,rows,factors,z,test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,6,figsize=(22,12))
    for ax,row in zip(axes.flat,rows):
        key=row['condition']+'_'+row['method']
        ax.scatter(z[test],factors[key][test],s=8,alpha=.6)
        ax.set_title(key+'\nr='+fmt(row['pearson_r']))
        ax.set_xlabel('True Z (evaluation only)');ax.set_ylabel('Frozen factor')
    fig.tight_layout();fig.savefig(str(output/'figures/scatter_all_conditions.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(17,10))
    keys=[row['condition']+'_'+row['method'] for row in rows]
    for ax,field in zip(axes,('absolute_pearson_r','x_reconstruction_raw_mse')):
        ax.bar(range(len(rows)),[float(row[field]) if row[field] is not None else np.nan for row in rows])
        ax.set_xticks(range(len(rows)));ax.set_xticklabels(keys,rotation=50,ha='right');ax.set_title('Final-test '+field)
        if field=='absolute_pearson_r':ax.set_ylim(0,1)
    fig.tight_layout();fig.savefig(str(output/'figures/all_conditions_comparison.png'),dpi=150);plt.close(fig)


def make_report(root,output,metadata,rows,comparisons):
    cfg=json.loads((root/'data/config.json').read_text())
    training=json.loads((root/'training_status.json').read_text())
    norm=cfg['resolved_signal_normalization']
    lines=['# 規格化したtanh・sin・3乗：単一seed201実験','',
        '## 生成と学習','',
        'Z~N(0,1)、εi~N(0,1)は独立。Xi=(fi(Z)−mi)/si+εi。f=(tanh,sin,cubic)。',
        'miとsiは学習600件だけの信号平均・標準偏差（ddof=0）。検証・テストも同じ係数を使用する。',
        'Zを生成時の規格化に使用したが、GraN-DAGや抽出器のfit・選択には与えていない。','',
        '|変数|関数|平均mi|標準偏差si|乗数1/si|切片−mi/si|','|---|---|---:|---:|---:|---:|']
    for j,name in enumerate(('tanh','sin','cubic')):
        lines.append('|X{}|{}|{:.10f}|{:.10f}|{:.10f}|{:.10f}|'.format(j+1,name,norm['mean'][j],norm['std'][j],norm['scale'][j],norm['offset'][j]))
    lines+=['','1000件、学習600・検証200・テスト200、seed201、σ=1のみ。NNは新規学習1回。',
        'NonLinGaussANM、2層×10ユニット、leaky-ReLU、既存RMSprop・DAG制約・損失を変更せず使用。',
        '収束={}、反復数={}、h={}、学習ワーカー所要時間={:.2f}秒。'.format(
            training['converged'],training['iterations'],training['final_h'],training['worker_elapsed_seconds']),
        '真の観測間直接辺はなし。潜在Zから全Xiへ辺。行が原因・列が結果。潜在交絡のため観測DAGの因果的正しさは保証されない。','',
        '## 入力マスク','',
        'B=実際の学習前重み＋学習直後マスク、C=学習済み＋学習直後、E=Cと同じ重み＋DAG化後。','']
    for code in ('B','C','E'):
        mask=np.asarray(metadata['masks'][code])
        edges=['X{}→X{}'.format(i+1,j+1) for i,j in np.argwhere(mask)]
        inputs=['X'+str(i+1) for i in np.flatnonzero(mask.any(1))]
        lines.append('- {}: 辺={}、隠れ層全体で使用する観測入力={}。'.format(code,', '.join(edges) or 'なし',', '.join(inputs) or 'なし'))
        for j in range(3):
            parents=['X'+str(i+1) for i in np.flatnonzero(mask[:,j])]
            lines.append('  - X{}予測MLPの親: {}。'.format(j+1,', '.join(parents) or 'なし（定数隠れ層）'))
    kernel=next(e for e in metadata['candidates'] if e['method']=='kernel_pca')
    lines+=['','## 抽出条件と評価の分離','',
        '- 全18条件・1次元を事前固定。単独変数は学習標準化した値。全MLPの隠れ層を結合してPCA/FA。',
        '- 定数列の除外、標準化、PCA/FA/Kernel PCA、OLS＋切片は学習600件のみでfit。',
        '- RBF Kernel PCAは学習標準化Xの正の二乗距離中央値dからgamma=1/(2d)を固定。',
        '- d={:.12g}、gamma={:.12g}、ペア数={}。第1成分を固定し、Zを使って成分・gammaを選ばない。'.format(
            kernel['median_positive_squared_train_distance'],kernel['gamma'],kernel['n_positive_training_pairs']),
        '- 全因子・抽出器・X再構成・観測変数との診断を保存して凍結した後に、別プロセスでZを読み評価。',
        '- MSEは全3変数の元Xを対象にした生スケールの再構成誤差。非線形再構成能力そのものではない。',
        '- 符号はZに合わせず保存。最大の単独変数相関はZ参照の事後的診断値で、教師なし選択法ではない。','',
        '## 最終テスト200件：全18条件','',
        '|条件|抽出器|Pearson r|絶対r|Spearman|X再構成MSE|','|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('|{}|{}|{}|{}|{}|{}|'.format(row['condition'],row['method'],fmt(row['pearson_r']),
            fmt(row['absolute_pearson_r']),fmt(row['spearman_r']),fmt(row['x_reconstruction_raw_mse'])))
    lines+=['','## 固定した比較','',
        '|比較|方法|絶対rの差|X-MSEの差|','|---|---|---:|---:|']
    for row in comparisons:
        lines.append('|{}→{}|{}|{}|{}|'.format(row['before'],row['after'],row['method'],fmt(row['delta_absolute_pearson_r']),fmt(row['delta_X_MSE'])))
    lines+=['','## 単純な観測変数の変換か','',
        'observed_factor_diagnostics.csvに全18因子×全3観測変数を保存。Zを使わない診断として抽出段階で固定した。',
        'テストの因子−各XjのPearson/Spearman、学習行のみでfitしたXj→因子の一次変換のテスト相対RMSEを記録。',
        '相対RMSEがほぼ0なら観測変数のほぼ一次変換、|Spearman|がほぼ1なら単調変換の可能性がある。',
        'これらだけで非線形依存や因果的潜在変数を同定したとはいえない。','',
        '## 限界と次の候補（今回は実施しない）','',
        '- 単一シード・単一ノイズ。相関差の有意性・一般性は検証していない。全条件を報告し最良条件のみに絞らない。',
        '- 信号分散は学習行でのみ統一。関数の形、尾部・単調性、Zとの線形相関はなお異なる。',
        '- Kernel PCA第1成分がZに対応する保証はなく、Zによる成分選択も行っていない。',
        '- 1因子は事前に既知と仮定。高相関でも潜在因子数の自動検出・因果的共通原因の同定ではない。',
        '- 次は同一マスク・同一入力配分・同一重複の線形対照で非線形性の寄与を分離し、未使用テストで確認することを提案する。',
        '- 必要なら事前登録した別シードで頑健性を評価するが、本実験では追加せず終了する。','',
        '既存ワーカーによるCAM枝刈りも保存されるがNN再学習ではなく、今回の因子比較には追加していない。',
        '隠れ層は各(1000,3,10)、結合特徴は(1000,30)、因子は各(1000,1)。','']
    for entry in metadata['candidates']:
        if entry['warnings']: lines.append('警告 {}: {}'.format(entry['key'],'; '.join(entry['warnings'])))
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/latent_factors_normalized_seed201.json'))
    parser.add_argument('--stage',choices=['all','extract','evaluate'],default='all')
    args=parser.parse_args()
    settings=json.loads(args.config.read_text())
    root,output=mapped(settings['run_dir']),mapped(settings['output_dir'])
    if args.stage=='all':
        for stage in ('extract','evaluate'):
            subprocess.check_call([sys.executable,'latent_function_factors.py','--config',str(args.config),'--stage',stage])
    elif args.stage=='extract': extract(root,output,settings)
    else: evaluate(root,output)
