"""One five-observed-variable experiment; reuse original SEM/training/extraction.

Generation is a separate process and may use Z. Training/extraction prohibit
Z reads. Only the final evaluation process reads the saved Z labels.
"""
import argparse
from contextlib import contextmanager
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analyze_hidden import correlations, varying_columns
from gran_dag.hidden import load_checkpoint_cpu
from latent_unsupervised import mapped
from . import data as original_data
from .additional import parameter_digest, verify_hidden
from .config import load_config
from .unsupervised import (SPLITS, digest, factor_correlations, file_inventory, fit_extractor,
                           forbid_latent_reads, read_csv, save_json, transform_extractor, write_csv)


NAMES = dict(A='元X＋PCA',B='元X＋FA',C='初期重み・学習後マスクH1＋PCA',
             D='学習済みH1＋PCA',E='MLP別PC1→標準化→PCA')


@contextmanager
def extra_signal_functions():
    """Extend only this generation process; leave existing source files/registry intact."""
    additions={'tanh2':lambda z:np.tanh(2*z),'sin2':lambda z:np.sin(2*z)}
    previous={key:original_data.FUNCTIONS.get(key) for key in additions}
    original_data.FUNCTIONS.update(additions)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None: original_data.FUNCTIONS.pop(key,None)
            else: original_data.FUNCTIONS[key]=value


def validate_config(config):
    d,t=config['data'],config['training']
    if ((d['n_samples'],d['n_observed'],d['n_latent'],d['seed'],d['noise_std'])!=(1000,5,1,201,1)
        or d['latent_functions']!=['tanh','sin','cubic','tanh2','sin2']
        or d['signal_normalization']!={'fit_split':'train','ddof':0}
        or config['split']!={'train':.6,'validation':.2,'test':.2,'seed':201}
        or not np.array_equal(d['observed_coefficients'],np.zeros((5,5)))
        or not np.array_equal(d['latent_coefficients'],np.ones((1,5)))):
        raise ValueError('Only the specified five-X seed201 sigma1 condition is authorized')
    expected=dict(seed=201,model='NonLinGaussANM',num_layers=2,hid_dim=10,nonlin='leaky-relu',
        optimizer='rmsprop',lr=.01,lr_reinit=.0001,num_train_iter=100000,train_batch_size=64,
        h_threshold=1e-8,normalize_data=True,cam_pruning_cutoff=.001,require_convergence=True)
    if any(t[key]!=value for key,value in expected.items()):
        raise ValueError('Original training settings must be preserved')
    a=config['analysis']
    if a['conditions']!=list('ABCDE') or a['n_factors']!=1 or a['hidden_layer']!=1 or a['hidden_stage']!='train':
        raise ValueError('Fixed five-condition, one-factor H1 analysis required')


def validate_observed(root):
    root=Path(root)
    x=np.load(str(root/'data/observed_X.npy'))
    ids=np.load(str(root/'data/sample_indices.npy'))
    labels=np.load(str(root/'data/sample_split.npy'))
    if x.shape!=(1000,5) or not np.isfinite(x).all(): raise ValueError('Expected finite X=(1000,5)')
    np.testing.assert_array_equal(ids,np.arange(1000))
    np.testing.assert_array_equal(np.bincount(labels),[600,200,200])
    for name,k in SPLITS:
        np.testing.assert_array_equal(np.sort(np.load(str(root/'data'/(name+'_indices.npy')))),ids[labels==k])
    pool=np.load(str(root/'training_input/sample_indices.npy'))
    np.testing.assert_array_equal(np.sort(pool),ids[labels!=2])
    np.testing.assert_array_equal(np.load(str(root/'training_input/data1.npy')),x[pool])
    if {p.name for p in (root/'training_input').iterdir()}!={'DAG1.npy','data1.npy','sample_indices.npy'}:
        raise ValueError('Training input must contain X only, no latent labels')
    return x,ids,labels


def fit_mlp_summary(hidden_train,x_train,settings):
    """Five-MLP block PCA; fit only the supplied training rows."""
    if (hidden_train.ndim!=3 or hidden_train.shape[1:]!=(5,10)
        or x_train.shape!=(len(hidden_train),5) or not np.isfinite(hidden_train).all()):
        raise ValueError('Aligned H=(N,5,10), X=(N,5) required')
    scores=np.zeros((len(hidden_train),5));blocks=[];details=[]
    for j in range(5):
        h=hidden_train[:,j,:];keep=varying_columns(h)
        block=dict(keep=keep,scaler=None,pca=None,informative=False)
        ratio=None
        if keep.any():
            scaler=StandardScaler().fit(h[:,keep]);scaled=scaler.transform(h[:,keep])
            pca=PCA(n_components=1,svd_solver='full').fit(scaled)
            score=pca.transform(scaled)
            if varying_columns(score).any():
                scores[:,j]=score[:,0]
                block.update(scaler=scaler,pca=pca,informative=True)
                ratio=float(pca.explained_variance_ratio_[0])
        blocks.append(block)
        details.append(dict(mlp='X'+str(j+1),nonconstant_units=int(keep.sum()),
            informative=block['informative'],pc1_explained_variance_ratio=ratio))
    if not any(b['informative'] for b in blocks): raise ValueError('No informative MLP blocks')
    return dict(blocks=blocks,final=fit_extractor(scores,x_train,'pca',settings),details=details,n_train=len(x_train))


def apply_mlp_summary(bundle,hidden):
    if hidden.ndim!=3 or hidden.shape[1:]!=(5,10) or not np.isfinite(hidden).all():
        raise ValueError('Expected finite H=(N,5,10)')
    scores=np.zeros((len(hidden),5))
    for j,b in enumerate(bundle['blocks']):
        if b['informative']:
            scores[:,j]=b['pca'].transform(b['scaler'].transform(hidden[:,j,b['keep']]))[:,0]
    factors,_,xhat=transform_extractor(bundle['final'],scores)
    return factors,xhat,scores


def mask_summary(mask,stage,weighted=None):
    if mask.shape!=(5,5) or not np.isin(mask,[0,1]).all() or np.diag(mask).any():
        raise ValueError('Binary source-row target-column 5x5 mask required')
    rows=[]
    for j in range(5):
        parents=np.flatnonzero(mask[:,j])
        effective=parents if weighted is None else np.flatnonzero((weighted[:,j]>0)&(mask[:,j]>0))
        rows.append(dict(stage=stage,target='X'+str(j+1),hidden_mlp_axis_index=j,parent_count=len(parents),
            parents=','.join('X'+str(i+1) for i in parents),multiple_inputs=bool(len(parents)>=2),
            positive_path_parent_count=len(effective),positive_path_parents=','.join('X'+str(i+1) for i in effective)))
    return rows


def redundancy(hidden,condition,split):
    rows=[]
    for j in range(5):
        h=hidden[:,j,:];keep=varying_columns(h);r=correlations(h,h)
        pairs=np.abs(r[np.triu_indices(10,1)]);pairs=pairs[np.isfinite(pairs)]
        row=dict(condition=condition,split=split,target='X'+str(j+1),nonconstant_units=int(keep.sum()),
            mean_absolute_correlation=float(pairs.mean()) if len(pairs) else None,
            fraction_absolute_r_ge_095=float((pairs>=.95).mean()) if len(pairs) else None,
            numerical_rank=0,participation_rank=0.,pc1_variance_fraction=None)
        if keep.any():
            standardized=(h[:,keep]-h[:,keep].mean(0))/h[:,keep].std(0)
            s=np.linalg.svd(standardized,compute_uv=False);v=s**2
            row.update(numerical_rank=int((s>1e-10*s[0]).sum()),participation_rank=float(v.sum()**2/(v**2).sum()),
                       pc1_variance_fraction=float(v[0]/v.sum()))
        rows.append(row)
    return rows


def generate(config,root,test_log=None):
    from .runner import run
    if root.exists(): raise FileExistsError('Never overwrite an existing experiment')
    old=file_inventory(root.parent)
    with extra_signal_functions(): run(config,root,stage='generate')
    save_json(root/'old_experiments_inventory.json',old)
    if test_log is not None: (root/'preflight_tests.txt').write_text(test_log)
    with forbid_latent_reads(root):
        validate_observed(root)
    save_json(root/'generation_status.json',dict(completed=True,new_dataset_conditions=1,
        seed=201,noise_std=1,signal_functions=['tanh(Z)','sin(Z)','Z^3','tanh(2Z)','sin(2Z)'],
        normalization_fit='train600 only, ddof=0',Z_used_only_in_generation=True))


def train_once(root):
    from .train import train_experiment
    marker=root/'single_training_attempt.json'
    if marker.exists() or (root/'random_full/model.pkl').exists():
        raise FileExistsError('The one permitted GraN-DAG training attempt has already started')
    save_json(marker,dict(attempt=1,started_at=time.time(),automatic_retry=False))
    started=time.monotonic()
    try:
        with forbid_latent_reads(root) as audit:
            validate_observed(root)
            train_experiment(root)
        save_json(root/'training_guard_audit.json',audit)
        save_json(root/'single_training_result.json',dict(completed=True,attempts=1,
            elapsed_seconds=time.monotonic()-started,Z_read_attempts=audit['blocked_attempts']))
    except Exception as error:
        save_json(root/'single_training_result.json',dict(completed=False,attempts=1,error=repr(error),
            elapsed_seconds=time.monotonic()-started,no_retry=True))
        (root/'report.md').write_text('# 5変数実験：学習処理の停止\n\n'+repr(error)+
            '\n\nGraN-DAG新規学習は1回だけ開始した。設定変更・再学習は行っていない。'+
            '未生成の段階のグラフ・因子評価は未実施。詳細はtraining.logとtraining_status.jsonを参照。\n')
        raise


def extract(config,root):
    import torch
    # Original get_w_adj() creates identity tensors using the global default;
    # the unchanged training worker and its saved checkpoints use float64.
    torch.set_num_threads(1)
    torch.set_default_tensor_type('torch.DoubleTensor')
    output=root/'unsupervised_factors'
    if output.exists(): raise FileExistsError('Never overwrite factor extraction')
    result=json.loads((root/'single_training_result.json').read_text())
    if not result['completed'] or result['attempts']!=1: raise ValueError('A completed single training run is required')
    settings=config['analysis'];output.mkdir()
    for name in ('models','factors','reconstructions','features','figures','graphs'): (output/name).mkdir()
    before=file_inventory(root,excluded=output)
    with forbid_latent_reads(root) as audit:
        x,ids,labels=validate_observed(root)
        mean=np.load(str(root/'normalization_mean.npy'));std=np.load(str(root/'normalization_std.npy'))
        models={};masks=[];stages={};hidden={}
        for stage,path in [('initial','random_full'),('train','train'),('dag','to-dag'),('cam','cam-pruning/cutoff_0.001000'),('random_train_mask','random_train_mask')]:
            model=load_checkpoint_cpu(str(root/path/'model.pkl'))
            original=parameter_digest(model)
            for p in model.parameters(): p.requires_grad=False
            h=verify_hidden(root/path/'hidden',model,x,ids,labels,mean,std,root/path/'model.pkl')[0]
            if h.shape!=(1000,5,10) or parameter_digest(model)!=original: raise ValueError('Hidden shape/weight mismatch')
            a=model.adjacency.numpy();w=model.get_w_adj().detach().numpy()
            masks.extend(mask_summary(a,stage,w))
            np.save(str(output/'graphs'/(stage+'_adjacency.npy')),a)
            np.save(str(output/'graphs'/(stage+'_weighted_adjacency.npy')),w)
            write_csv(output/'graphs'/(stage+'_adjacency.csv'),
                [dict(source='X'+str(i+1),**{'X'+str(j+1):int(a[i,j]) for j in range(5)}) for i in range(5)])
            stages[stage]=dict(path=path,edges=[['X'+str(i+1),'X'+str(j+1)] for i,j in zip(*np.where(a))],
                edge_count=int(a.sum()),multiple_input_mlps=int((a.sum(0)>=2).sum()),
                parent_counts=a.sum(0).astype(int).tolist(),model_sha256=digest(root/path/'model.pkl'),
                parameter_sha256=original,mask_sha256=digest(root/path/'hidden/adjacency.npy'))
            models[stage]=model;hidden[stage]=h
        np.testing.assert_array_equal(models['train'].adjacency.numpy(),models['random_train_mask'].adjacency.numpy())
        if parameter_digest(models['initial'])!=parameter_digest(models['random_train_mask']):
            raise ValueError('The random-weight contrast must use the exact initial weights')
        write_csv(output/'mlp_inputs.csv',masks)
        np.save(str(output/'sample_indices.npy'),ids);np.save(str(output/'sample_split.npy'),labels)
        values=dict(A=x,B=x,C=hidden['random_train_mask'].reshape(1000,50),D=hidden['train'].reshape(1000,50))
        factors={};recon={};bundles={};train=labels==0;details=[];diagnostics=[];metric_rows=[];factor_rows=[]
        for code in 'ABCD':
            method='factor_analysis' if code=='B' else 'pca'
            bundles[code]=fit_extractor(values[code][train],x[train],method,settings)
            factors[code],_,recon[code]=transform_extractor(bundles[code],values[code])
        bundles['E']=fit_mlp_summary(hidden['train'][train],x[train],settings)
        factors['E'],recon['E'],scores=apply_mlp_summary(bundles['E'],hidden['train'])
        np.save(str(output/'features/E_mlp_scores.npy'),scores)
        write_csv(output/'E_mlp_pca.csv',bundles['E']['details'])
        for code in 'ABCDE':
            bundle=bundles[code]['final'] if code=='E' else bundles[code]
            if factors[code].shape!=(1000,1) or not np.isfinite(factors[code]).all(): raise ValueError('Invalid frozen factors')
            with (output/'models'/(code+'.pkl')).open('wb') as stream: pickle.dump(bundles[code],stream)
            np.save(str(output/'factors'/(code+'.npy')),factors[code])
            np.save(str(output/'reconstructions'/(code+'_X.npy')),recon[code])
            details.append(dict(condition=code,description=NAMES[code],used_dimension=int(bundle['keep'].sum()),
                warnings=bundle['warnings'],fa_iterations=int(bundle['extractor'].n_iter_) if code=='B' else None))
            for i in range(1000): factor_rows.append(dict(condition=code,sample_id=int(ids[i]),split=int(labels[i]),factor=float(factors[code][i,0])))
            for name,k in SPLITS:
                idx=labels==k
                row=dict(condition=code,description=NAMES[code],split=name,n_samples=int(idx.sum()),
                    factor_variance=float(factors[code][idx,0].var(ddof=1)),
                    x_reconstruction_raw_mse=float(np.mean((x[idx]-recon[code][idx])**2)))
                for j in range(5):
                    c=factor_correlations(factors[code][idx,0],x[idx,j])
                    row['pearson_X'+str(j+1)]=c['pearson_r'];row['spearman_X'+str(j+1)]=c['spearman_r']
                metric_rows.append(row)
        for code,stage in [('C','random_train_mask'),('D','train')]:
            for name,k in SPLITS:
                h=hidden[stage][labels==k]
                diagnostics.extend(redundancy(h,code,name))
                np.save(str(output/'features'/(code+'_'+name+'_correlations.npy')),correlations(h.reshape(len(h),50),h.reshape(len(h),50)))
        write_csv(output/'feature_redundancy.csv',diagnostics)
        write_csv(output/'label_free_metrics.csv',metric_rows)
        write_csv(output/'extracted_factors.csv',factor_rows)
        save_json(output/'metadata.json',dict(settings=settings,stages=stages,candidates=details,
            initial_weight_contrast='exact initial weights with learned train mask; mask itself is learned from X',
            random_mask_equal_learned=True,random_parameters_equal_initial=True,
            first_hidden_shape=[1000,5,10],flattened_shape=[1000,50],axes=['sample','target MLP','unit'],
            all_fit_rows=600,OLS_fit_rows=600,sign_alignment=False,condition_selection=False))
        if before!=file_inventory(root,excluded=output): raise ValueError('Training artifacts changed during extraction')
        save_json(output/'extraction_status.json',dict(completed=True,all_factors_saved_before_Z=True,
            Z_read_attempts=audit['blocked_attempts'],new_grandag_training_calls=0,new_vae_training_calls=0))
        save_json(output/'extraction_manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    print('Five factors frozen before Z evaluation:',output,flush=True)


def evaluate(config,root):
    return evaluate_frozen(config,root)


def number(value):
    return float('nan') if value is None else value


def plots(output,metrics,factors,z,labels,metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    tests={r['condition']:r for r in metrics if r['split']=='test'}
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['absolute_pearson_r','x_reconstruction_raw_mse']):
        ax.bar(list('ABCDE'),[number(tests[c][key]) for c in 'ABCDE']);ax.set_title(key)
    fig.tight_layout();fig.savefig(str(output/'figures/test_comparison.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7));test=labels==2
    for ax,code in zip(axes.flat,'ABCDE'):
        ax.scatter(z[test,0],factors[code][test,0],s=12,alpha=.6)
        ax.set(title='{} |r|={:.4f}'.format(code,number(tests[code]['absolute_pearson_r'])),
               xlabel='True Z (evaluation only)',ylabel='Frozen factor')
    axes.flat[-1].axis('off');fig.tight_layout()
    fig.savefig(str(output/'figures/test_Z_scatter.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    positions={('X'+str(i+1)):(np.cos(2*np.pi*i/5),np.sin(2*np.pi*i/5)) for i in range(5)}
    for ax,stage in zip(axes,['train','dag','cam']):
        info=metadata['stages'][stage]
        for a,b in info['edges']:
            ax.add_patch(FancyArrowPatch(positions[a],positions[b],arrowstyle='-|>',mutation_scale=16,
                shrinkA=18,shrinkB=18,color='gray',connectionstyle='arc3,rad=.08'))
        for name,(px,py) in positions.items():
            ax.scatter([px],[py],s=700,color='lightblue',zorder=3);ax.text(px,py,name,ha='center',va='center',zorder=4)
        ax.set(xlim=(-1.35,1.35),ylim=(-1.35,1.35),title='{}: {} edges, {} multi-input MLPs'.format(
            stage,info['edge_count'],info['multiple_input_mlps']));ax.axis('off')
    fig.tight_layout();fig.savefig(str(output/'figures/graphs.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,code in zip(axes,['C','D']):
        matrix=np.load(str(output/'features'/(code+'_train_correlations.npy')))
        ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm');ax.set_title(code+' train H1 feature correlations')
    fig.tight_layout();fig.savefig(str(output/'figures/hidden_correlations.png'),dpi=160);plt.close(fig)


def report(config,root,output,metrics,metadata):
    tests={r['condition']:r for r in metrics if r['split']=='test'}
    status=json.loads((root/'training_status.json').read_text())
    generation=json.loads((root/'data/config.json').read_text())
    norm=generation['resolved_signal_normalization']
    fmt=lambda value:'未定義' if value is None else '{:.6f}'.format(value)
    delta=lambda a,b:number(tests[b]['absolute_pearson_r'])-number(tests[a]['absolute_pearson_r'])
    lines=['# 5観測変数・複数入力MLPの探索実験', '',
        'seed201、σ=1、1,000件、学習600・検証200・テスト200の1条件を実施した。GraN-DAG新規学習1回、VAE学習0回。結果に応じた再学習・設定変更・追加条件は実施していない。', '',
        '## データ生成と真の構造', '',
        'Z~N(0,1)、Xi=(fi(Z)−mi)/si+εi、各εi~N(0,1)で相互独立。mi・siは学習600件の信号だけで計算、stdはddof=0。',
        '真の構造はZ→X1, Z→X2, Z→X3, Z→X4, Z→X5で、X間の直接辺はない。観測XだけのDAGは潜在共通原因を含む全構造を表現できない。',
        '|変数|信号fi|mi|si|', '|---|---|---:|---:|']
    for j,name in enumerate(['tanh(Z)','sin(Z)','Z³','tanh(2Z)','sin(2Z)']):
        lines.append('|X{}|{}|{:.12g}|{:.12g}|'.format(j+1,name,norm['mean'][j],norm['std'][j]))
    lines+=['', 'データ生成プロセスではZを使用する。保存後のGraN-DAG学習・前処理・因子抽出にはZを読み込まず、最終評価の別プロセスだけが保存済みZを読む。分割・ID・規格化係数・データハッシュを保存した。', '',
        '## GraN-DAGの設定と収束', '',
        'NonLinGaussANM、隠れ2層×10、leaky-ReLU、RMSprop lr=0.01、lr_reinit=0.0001、batch=64、最大100000 iteration、h_threshold=1e-8、seed201、CPU 1スレッド。元の損失・DAG制約・最適化・CAM cutoff=0.001を利用した。',
        '収束={}、iteration={}、最終h={:.12g}、既存workerの学習・DAG化・CAM・隠れ層保存を含む時間={:.3f}秒。'.format(status['converged'],status['iterations'],status['final_h'],status['worker_elapsed_seconds']),
        '学習600件と検証200件を既存処理に渡し、CAMもこの800件だけを使用。最終テスト200件は学習・収束判定・CAM・モデル選択に使用していない。', '',
        '## 各段階のグラフと入力元', '',
        '隣接行列は行=入力元（原因候補）、列=予測対象。初期値と全段階のモデル・マスク・CSVを保存した。']
    rows=read_csv(output/'mlp_inputs.csv')
    for stage in ['train','dag','cam']:
        info=metadata['stages'][stage]
        lines+=['', '### '+stage, '',
            '辺数={}、複数入力（親2個以上）MLP数={}。辺: {}。'.format(info['edge_count'],info['multiple_input_mlps'],
                '、'.join(a+'→'+b for a,b in info['edges']) if info['edges'] else 'なし'),
            '|予測対象|マスク上の入力元|入力数|正の重み経路を持つ入力元|', '|---|---|---:|---|']
        for row in rows:
            if row['stage']==stage:
                lines.append('|{}|{}|{}|{}|'.format(row['target'],row['parents'] or 'なし',row['parent_count'],row['positive_path_parents'] or 'なし'))
    train_info=metadata['stages']['train']
    lines+=['', 'マスク上の入力数を主要な定義として使用した。重み経路の補足はget_w_adj()>0という厳密なゼロ判定で、事後的な閾値調整はしない。極めて小さな正の経路が実質的な影響を持つとは限らない。',
        '入力マスクはグラフ推定結果であり、真の観測変数間直接辺を意味しない。今回、推定されたX間の辺は真の生成モデルには存在しない。', '',
        '## 第1隠れ層と対照', '',
        '保存形状は(1000,5,10)、軸はサンプル・予測対象MLP（X1〜X5）・ユニット。全MLP結合は(1000,50)。各段階のhidden/metadata.jsonとunsupervised_factors/mlp_inputs.csvが対応を示す。',
        'Cは保存済み初期重みを使い、入力マスクだけを学習後と一致させた対照。マスク自体がXからのGraN-DAG学習に依存するため、完全に独立した未学習モデルではない。初期重み一致・入力マスク一致を検証した。', '',
        '## 因子抽出と全条件のテスト結果', '',
        '因子数は1。Aは元X＋PCA、Bは元X＋FA、Cは初期重み・学習後マスクH1＋PCA、Dは学習済みH1＋PCA、Eは学習済み各MLPのPC1を標準化して最終PCA。',
        '定数判定・標準化・PCA・FA・切片付きX再構成OLSは学習600件のみでfit。検証・テストはtransformのみ。Eは情報のないブロックを除外する。因子の符号をZで反転せず、Z相関で条件や主成分を選んでいない。',
        '|条件|絶対Pearson|Spearman（未反転）|因子分散(ddof=1)|OLSによる元X再構成MSE|',
        '|---|---:|---:|---:|---:|']
    for code in 'ABCDE':
        r=tests[code]
        lines.append('|{}|{}|{}|{:.6f}|{:.6f}|'.format(code,fmt(r['absolute_pearson_r']),fmt(r['spearman_r']),r['factor_variance'],r['x_reconstruction_raw_mse']))
    lines+=['', '因子の分散は任意の尺度に依存し、大きいほど情報が多いことを意味しない。全分割と各XiとのPearson/Spearmanはunsupervised_factors/metrics.csv。', '',
        '|条件|X1とのPearson|X2とのPearson|X3とのPearson|X4とのPearson|X5とのPearson|',
        '|---|---:|---:|---:|---:|---:|']
    for code in 'ABCDE':
        lines.append('|'+code+'|'+'|'.join(fmt(tests[code]['pearson_X'+str(j+1)]) for j in range(5))+'|')
    lines+=['', '## 学習効果・元Xとの差・MLP単位要約', '',
        '同じ学習後マスクでC→Dの絶対相関差={:+.6f}、X再構成MSE差={:+.6f}。'.format(delta('C','D'),tests['D']['x_reconstruction_raw_mse']-tests['C']['x_reconstruction_raw_mse']),
        '元X＋PCAのA→Dの絶対相関差={:+.6f}、A→Eの差={:+.6f}。'.format(delta('A','D'),delta('A','E')),
        'MLP単位の要約D→Eの絶対相関差={:+.6f}、X再構成MSE差={:+.6f}。'.format(delta('D','E'),tests['E']['x_reconstruction_raw_mse']-tests['D']['x_reconstruction_raw_mse']),
        '複数入力のMLPが学習直後に{}個得られた。存在しない場合もマスク調整や再学習は行わない。1条件の相関差から一般的優位性や因果的な同定を主張しない。'.format(train_info['multiple_input_mlps']), '',
        '## 特徴量重複', '',
        '各MLP内の平均絶対相関・|r|≥0.95の割合・数値rank・participation rank・PC1分散割合をfeature_redundancy.csvへ保存。学習600件の学習済みH1は以下の通り。',
        '|MLP|入力数|非定数列|平均絶対相関|数値rank|participation rank|PC1分散比|', '|---|---:|---:|---:|---:|---:|---:|']
    for row in read_csv(output/'feature_redundancy.csv'):
        if row['condition']=='D' and row['split']=='train':
            j=int(row['target'][1:])-1
            vals=[row[k] for k in ['target','nonconstant_units','mean_absolute_correlation','numerical_rank','participation_rank','pc1_variance_fraction']]
            lines.append('|{}|{}|{}|{}|{}|{}|{}|'.format(vals[0],train_info['parent_counts'][j],*vals[1:]))
    lines+=['', '重複・rankは記述統計であり、テストを使った特徴選択には使用しない。複数入力になっても、ユニットが強く相関していたり単一の入力に実質的に偏る可能性は残る。入力数だけで有効な潜在表現が形成されたとは言えない。', '',
        '## 保全・検証・限界', '',
        '事前の単体テストで新規関数の生成式・train-only信号規格化・分割・(N,5,10)隠れ層・PCA/FA・MLP要約・定数ブロック・Z読み込み禁止を確認した。ログはpreflight_tests.txt。',
        '因子・OLS・メタデータをハッシュで固定してから、別プロセスでのみtrue_Zを読み込んだ。学習と抽出のZ読み込み試行は0。元実験ファイルを変更していない。',
        '潜在Zの個体値はノイズを持つXから完全には回復できず、Z相関・X再構成・因果グラフ復元は異なる評価である。今回は因子数1を固定し、未知の潜在因子数は検出していない。',
        '抽出因子はXから作った関数なので、そのまま独立した潜在原因の観測値として追加しても因果的同定は保証されない。今後の潜在生成モデルには、潜在説明と直接辺の競合、識別可能性の追加仮定、安定した潜在推論が課題として残る。', '',
        '## 追加実装と実行', '',
        '追加: latent_five_observed.py、latent_experiments/five_observed.py、configs/latent_five_observed_seed201.json、tests/test_five_observed.py、docs/latent_five_observed.md。既存コードは編集していない。',
        '```bash','bash run_in_container.sh python -B latent_five_observed.py --stage all','```',
        '学習試行マーカーと既存出力の検査で再学習・上書きを拒否する。今回の結果を用いた追加学習や条件変更は行わない。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def evaluate_frozen(config,root):
    output=root/'unsupervised_factors'
    if any((output/name).exists() for name in ['metrics.csv','evaluation_status.json']):
        raise FileExistsError('Never overwrite evaluation')
    status=json.loads((output/'extraction_status.json').read_text())
    if not status['all_factors_saved_before_Z'] or status['Z_read_attempts']: raise ValueError('Clean frozen factors required')
    manifest=json.loads((output/'extraction_manifest.json').read_text())
    for name,sha in manifest.items():
        if digest(output/name)!=sha: raise ValueError('Frozen artifact changed: '+name)
    ids=np.load(str(output/'sample_indices.npy'));labels=np.load(str(output/'sample_split.npy'))
    np.testing.assert_array_equal(ids,np.load(str(root/'data/sample_indices.npy')))
    np.testing.assert_array_equal(labels,np.load(str(root/'data/sample_split.npy')))
    factors={c:np.load(str(output/'factors'/(c+'.npy'))) for c in 'ABCDE'}
    # First saved-Z read since generation: all models and transformations fixed.
    z=np.load(str(root/'data/true_Z.npy'))
    if z.shape!=(1000,1) or not np.isfinite(z).all(): raise ValueError('Invalid evaluation labels')
    metrics=[]
    for original in read_csv(output/'label_free_metrics.csv'):
        row={key:(value if key in ['condition','description','split'] else (float(value) if value else None))
             for key,value in original.items()}
        idx=labels==dict(SPLITS)[row['split']]
        row.update(factor_correlations(factors[row['condition']][idx,0],z[idx,0]));metrics.append(row)
    write_csv(output/'metrics.csv',metrics)
    write_csv(output/'comparison_table.csv',[r for r in metrics if r['split']=='test'])
    metadata=json.loads((output/'metadata.json').read_text())
    plots(output,metrics,factors,z,labels,metadata)
    report(config,root,output,metrics,metadata)
    before=json.loads((root/'old_experiments_inventory.json').read_text())
    if before!=file_inventory(root.parent,excluded=root): raise ValueError('Prior experiment files changed')
    for name,sha in manifest.items():
        if digest(output/name)!=sha: raise ValueError('Frozen transformation changed')
    save_json(output/'evaluation_status.json',dict(completed=True,Z_evaluation_only=True,separate_process=True,
        old_experiments_unchanged=True,all_frozen_hashes_verified=True,new_grandag_training_attempts=1,
        new_vae_training_attempts=0,additional_condition_fits=0))
    for row in metrics:
        if row['split']=='test': print(json.dumps(row,ensure_ascii=False),flush=True)
    print('Report:',root/'report.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['generate','train','extract','evaluate'])
    parser.add_argument('--config',required=True)
    args=parser.parse_args();config=load_config(args.config);validate_config(config)
    root=mapped(config['output_dir']).resolve()
    if args.stage=='generate': generate(config,root)
    elif args.stage=='train': train_once(root)
    elif args.stage=='extract': extract(config,root)
    else: evaluate(config,root)


if __name__=='__main__': main()
