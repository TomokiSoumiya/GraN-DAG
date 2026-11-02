"""Frozen H1/PCA -> Z_hat; exactly one new graph fit, with no true Z input."""
import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA

from analyze_hidden import varying_columns
from gran_dag.hidden import load_checkpoint_cpu
from latent_unsupervised import mapped
from .additional import parameter_digest,verify_hidden
from .oracle import graph_scores
from .unsupervised import (SPLITS,digest,factor_correlations,file_inventory,forbid_latent_reads,
                           read_csv,save_json,transform_extractor,write_csv)


VARIABLES=['Z_hat','X1','X2','X3']


def assert_frozen(root):
    frozen=json.loads((root/'zhat_manifest.json').read_text())
    for name,expected in frozen.items():
        if digest(root/name)!=expected:raise ValueError('Frozen Z_hat preparation changed: '+name)


def reproduce(settings):
    source,oracle,root=[mapped(settings[name]).resolve() for name in ('source_run','oracle_run','output_dir')]
    if root.exists():raise FileExistsError('Never overwrite an existing Z_hat run')
    if settings['source_condition']!='C_L1_pca' or settings['variable_order']!=VARIABLES:
        raise ValueError('Only the fixed learned/pre-DAG/layer1/PCA condition is allowed')
    before=file_inventory(source);oracle_before=file_inventory(oracle)
    root.mkdir(parents=True);(root/'data').mkdir();(root/'zhat').mkdir()
    with forbid_latent_reads(source) as audit, patch.object(PCA,'fit',side_effect=AssertionError('Saved PCA must never be refit')):
        config=json.loads((source/'config.json').read_text())
        training=json.loads((oracle/'training_options.json').read_text())
        assert config['data']['seed']==settings['seed']==201 and config['data']['noise_std']==settings['noise_std']==1
        assert training['seed']==201 and training['n_observed']==4
        x=np.load(str(source/'data/observed_X.npy'))
        ids=np.load(str(source/'data/sample_indices.npy'));labels=np.load(str(source/'data/sample_split.npy'))
        assert x.shape==(1000,3)
        np.testing.assert_array_equal(ids,np.arange(1000));np.testing.assert_array_equal(np.bincount(labels),[600,200,200])
        for name,code in SPLITS:
            np.testing.assert_array_equal(np.sort(np.load(str(source/'data'/(name+'_indices.npy')))),ids[labels==code])
        old_status=json.loads((source/'unsupervised_factors/extraction_status.json').read_text())
        assert old_status['completed'] and not old_status['z_read_attempts'] and old_status['all_factors_saved_before_Z']
        metadata=json.loads((source/'unsupervised_factors/metadata.json').read_text())
        assert not metadata['z_used_in_fit'] and not metadata['z_used_for_selection']
        entry=next(e for e in metadata['candidates'] if e['key']==settings['source_condition'])
        manifest=json.loads((source/'unsupervised_factors/extraction_manifest.json').read_text())
        for name in (entry['model_file'],entry['factor_file']):
            assert digest(source/'unsupervised_factors'/name)==manifest[name]
        checkpoint=source/'train/model.pkl';model=load_checkpoint_cpu(str(checkpoint))
        assert (model.num_vars,model.num_layers,model.hid_dim)==(3,2,10)
        params_before=parameter_digest(model)
        mean=np.load(str(source/'normalization_mean.npy'));std=np.load(str(source/'normalization_std.npy'))
        np.testing.assert_allclose(mean,x[labels==0].mean(0,keepdims=True),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(std,x[labels==0].std(0,ddof=1,keepdims=True),rtol=1e-12,atol=1e-12)
        layers=verify_hidden(source/'train/hidden',model,x,ids,labels,mean,std,checkpoint)
        assert parameter_digest(model)==params_before
        hidden=layers[0].reshape(1000,-1)
        with (source/'unsupervised_factors'/entry['model_file']).open('rb') as stream:bundle=pickle.load(stream)
        assert bundle['method']=='pca' and bundle['n_train']==600 and bundle['extractor'].n_components_==1
        np.testing.assert_array_equal(bundle['keep'],varying_columns(hidden[labels==0]))
        np.testing.assert_allclose(bundle['scaler'].mean_,hidden[labels==0][:,bundle['keep']].mean(0),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(bundle['scaler'].var_,hidden[labels==0][:,bundle['keep']].var(0),rtol=1e-12,atol=1e-12)
        replay,_,_=transform_extractor(bundle,hidden)
        saved=np.load(str(source/'unsupervised_factors'/entry['factor_file']))
        assert saved.shape==(1000,1) and np.isfinite(saved).all()
        np.testing.assert_allclose(replay,saved,rtol=1e-12,atol=1e-12)
        # Preserve the exact saved sign, score scale and bytes, not a re-fit score.
        shutil.copyfile(str(source/'unsupervised_factors'/entry['factor_file']),str(root/'zhat/Z_hat.npy'))
        shutil.copyfile(str(source/'unsupervised_factors'/entry['model_file']),str(root/'zhat/scaler_pca.pkl'))
        for name in ['observed_X.npy','sample_indices.npy','sample_split.npy','train_indices.npy',
                     'validation_indices.npy','test_indices.npy','pool_indices.npy','signal_mean.npy','signal_std.npy']:
            shutil.copyfile(str(source/'data'/name),str(root/'data'/name))
        shutil.copyfile(str(source/'data/config.json'),str(root/'data/generation_config.json'))
        input_data=np.column_stack([saved,x]);np.save(str(root/'data/input_Zhat_X.npy'),input_data)
        save_json(root/'training_options.json',training)
        save_json(root/'config.json',dict(settings,training=training,Z_is_observed=False,
            deterministic_factor_from_X=True,original_NN_retrained=False,PCA_refit=False,sign_flipped=False))
        source_names=['train/model.pkl','train/hidden/hidden_layer_001.npy','train/hidden/metadata.json',
                      'normalization_mean.npy','normalization_std.npy','data/observed_X.npy',
                      'unsupervised_factors/'+entry['model_file'],'unsupervised_factors/'+entry['factor_file']]
        save_json(root/'zhat/provenance.json',dict(checkpoint=str(checkpoint),condition='C_L1_pca',
            hidden_shape=list(layers[0].shape),flattened_shape=list(hidden.shape),kept_columns=int(bundle['keep'].sum()),
            source_sha256={name:digest(source/name) for name in source_names},parameters_sha256=params_before,
            PCA_file=str(source/'unsupervised_factors'/entry['model_file']),original_PCA_component_retained=True,
            original_NN_retrained=False,PCA_refit=False,z_used=False,sign_alignment_used=False,
            function='Z_hat = saved_PCA(saved_StandardScaler(flatten(H1_saved_NN_train_mask(normalize_train_X(X)))))'))
        save_json(root/'reproduction_status.json',dict(completed=True,z_read_attempts=audit['blocked_attempts'],
            saved_factors_frozen_before_Z=True,shape=[1000,1],finite=True,
            max_replay_error=float(np.max(np.abs(replay-saved))),
            variance_ddof1={name:float(saved[labels==code,0].var(ddof=1)) for name,code in SPLITS},
            all_rows_and_splits_verified=True,train_only_scaler_verified=True,original_extractor_training_rows=600,
            no_fit_invoked=True,no_sign_flip=True,true_Z_used=False))
        assert before==file_inventory(source) and oracle_before==file_inventory(oracle)
        save_json(root/'preserved_source_inventory.json',before)
        save_json(root/'preserved_oracle_inventory.json',oracle_before)
        save_json(root/'zhat_manifest.json',{str(p.relative_to(root)):digest(p)
                  for p in root.rglob('*') if p.is_file()})
    print('Z_hat reproduced and frozen WITHOUT true Z; next phase validates the fixed score.',flush=True)


def validate(root):
    root=Path(root).resolve();assert_frozen(root)
    if (root/'reproduction_validation.json').exists():raise FileExistsError('Validation already exists')
    config=json.loads((root/'config.json').read_text());source=mapped(config['source_run']);oracle=mapped(config['oracle_run'])
    status=json.loads((root/'reproduction_status.json').read_text())
    assert status['completed'] and not status['z_read_attempts'] and status['saved_factors_frozen_before_Z']
    provenance=json.loads((root/'zhat/provenance.json').read_text())
    assert all(digest(source/name)==expected for name,expected in provenance['source_sha256'].items())
    # Evaluation-only Z read, after the exact original PCA score is frozen.
    true_z=np.load(str(source/'data/true_Z.npy'))[:,0]
    factor=np.load(str(root/'zhat/Z_hat.npy'))[:,0];x=np.load(str(root/'data/observed_X.npy'))
    labels=np.load(str(root/'data/sample_split.npy'))
    old=read_csv(source/'unsupervised_factors/metrics.csv')
    rows=[];observed=[]
    for split,code in SPLITS:
        chosen=labels==code;corr=factor_correlations(factor[chosen],true_z[chosen])
        previous=next(r for r in old if r['condition']=='C_L1' and r['method']=='pca' and r['split']==split)
        for field in ('pearson_r','absolute_pearson_r','spearman_r'):
            np.testing.assert_allclose(corr[field],float(previous[field]),rtol=1e-12,atol=1e-12)
        rows.append(dict(split=split,n_samples=int(chosen.sum()),**corr))
        for j in range(3):
            observed.append(dict(split=split,observed_variable='X'+str(j+1),
                **factor_correlations(factor[chosen],x[chosen,j])))
    test=next(row for row in rows if row['split']=='test')
    if abs(test['absolute_pearson_r']-config['expected_test_absolute_correlation'])>config['reported_rounding_tolerance']:
        raise ValueError('Z_hat does not reproduce the predeclared reported correlation; stop before training')
    write_csv(root/'zhat_Z_evaluation.csv',rows);write_csv(root/'zhat_X_correlations.csv',observed)
    assert_frozen(root)
    # Only after the fixed-factor validation passes, prepare the single new fit.
    (root/'training_input').mkdir()
    pool=np.load(str(root/'data/pool_indices.npy'));input_data=np.load(str(root/'data/input_Zhat_X.npy'))
    assert input_data.shape==(1000,4) and not (labels[pool]==2).any()
    np.testing.assert_array_equal(input_data[:,0],factor);np.testing.assert_array_equal(input_data[:,1:],x)
    np.save(str(root/'training_input/data1.npy'),input_data[pool])
    np.save(str(root/'training_input/sample_indices.npy'),pool)
    np.save(str(root/'training_input/DAG1.npy'),np.zeros((4,4),dtype=np.int64))
    save_json(root/'preparation_status.json',dict(completed=True,input_order=VARIABLES,
        input_sha256=digest(root/'training_input/data1.npy'),pool_ids_sha256=digest(root/'training_input/sample_indices.npy'),
        no_true_Z_in_training=True,new_data_generated=False,Z_hat_sign_unchanged=True,
        true_graph_given_to_training=False,loader_graph='zero diagnostic placeholder'))
    save_json(root/'reproduction_validation.json',dict(passed=True,metrics=test,
        old_metrics_match_all_splits=True,saved_sign_preserved=True,
        true_Z_read_after_factor_frozen=True,correlation_gate='reproducibility check only, not model/parameter/sign selection',
        approved_for_one_new_training=True))
    assert file_inventory(source)==json.loads((root/'preserved_source_inventory.json').read_text())
    assert file_inventory(oracle)==json.loads((root/'preserved_oracle_inventory.json').read_text())
    print('Fixed Z_hat validated:',json.dumps(test),'No source NN/PCA refit. One new graph fit is now permitted.',flush=True)


def train_once(root):
    from .oracle import train_once as shared_four_variable_train
    root=Path(root).resolve();assert_frozen(root)
    gate=json.loads((root/'reproduction_validation.json').read_text());assert gate['passed']
    assert not (root/'data/true_Z.npy').exists()
    print('Exploratory [Z_hat,X1,X2,X3] fit using the unchanged generic 4-column training harness. Legacy worker text says oracle; true Z is NOT input.',flush=True)
    failure=None
    try:
        shared_four_variable_train(root)
    except BaseException as error:
        failure=dict(error_type=type(error).__name__,message=str(error))
        raise
    finally:
        if not (root/'training_status.json').exists() and failure:
            save_json(root/'training_status.json',dict(converged=False,training_calls=1,failure=failure))
        if (root/'training_status.json').exists():
            status=json.loads((root/'training_status.json').read_text())
            status.update(input_order=VARIABLES,Z_used_as_observed_input=False,Z_hat_used_as_observed_input=True,
                Z_hat_training_run=True,oracle=False,deterministic_factor_from_X=True,
                PCA_FA_run=False,original_three_variable_NN_retrained=False,PCA_refit=False,
                shared_harness='latent_experiments.oracle.train_once; labels corrected here; numerical training unchanged',
                zhat_pipeline_sha256=digest('latent_experiments/zhat_graph.py'))
            save_json(root/'training_status.json',status)


def evaluate(root):
    from .data import topological_order
    root=Path(root).resolve();output=root/'evaluation'
    if output.exists():raise FileExistsError('Never overwrite an existing graph diagnosis')
    assert_frozen(root)
    config=json.loads((root/'config.json').read_text());status=json.loads((root/'training_status.json').read_text())
    source,oracle=mapped(config['source_run']),mapped(config['oracle_run'])
    assert file_inventory(source)==json.loads((root/'preserved_source_inventory.json').read_text())
    assert file_inventory(oracle)==json.loads((root/'preserved_oracle_inventory.json').read_text())
    frozen=json.loads((root/'training_artifacts_manifest.json').read_text()) if (root/'training_artifacts_manifest.json').exists() else {}
    for name,expected in frozen.items():
        if digest(root/name)!=expected:raise ValueError('New training artifacts changed')
    output.mkdir()
    cutoff=config['training']['cam_pruning_cutoff']
    stages=[('train','train'),('to-dag','to-dag'),('cam','cam-pruning/cutoff_%.6f'%cutoff)]
    models=[('A_original_X',source,['X1','X2','X3']),('B_oracle_Z',oracle,['Z','X1','X2','X3']),('C_derived_Zhat',root,VARIABLES)]
    graph_rows,edge_rows,connections,pairs,formal,changes=[],[],[],[],[],[]
    matrices={};availability=[]
    shape_template=np.zeros((4,4),dtype=int);shape_template[0,1:]=1
    for model_id,directory,names in models:
        for stage,path in stages:
            location=directory/path/'DAG.npy'
            available=location.exists() and (model_id!='C_derived_Zhat' or status.get('converged',False))
            availability.append(dict(model=model_id,stage=stage,available=available))
            if not available:continue
            matrix=np.load(str(location));assert matrix.shape==(len(names),len(names)) and np.isin(matrix,[0,1]).all()
            matrices[(model_id,stage)]=matrix
            stats=structural_summary(matrix,len(names)==4)
            try:topological_order(matrix);is_dag=True
            except ValueError:is_dag=False
            graph_rows.append(dict(model=model_id,stage=stage,variable_order=';'.join(names),is_DAG=is_dag,**stats))
            offset=1 if len(names)==4 else 0
            for i,j in np.argwhere(matrix):
                edge_rows.append(dict(model=model_id,stage=stage,source=names[i],target=names[j],
                    edge_kind=('factor_to_X' if i==0 else 'X_to_factor') if offset and (i==0 or j==0) else 'X_to_X'))
            for i in range(3):
                for j in range(i+1,3):
                    a,b=bool(matrix[i+offset,j+offset]),bool(matrix[j+offset,i+offset])
                    pairs.append(dict(model=model_id,stage=stage,pair='X{} -- X{}'.format(i+1,j+1),
                        connected=a or b,forward=a,reverse=b))
            if offset:
                for j in range(1,4):
                    forward,reverse=bool(matrix[0,j]),bool(matrix[j,0])
                    connections.append(dict(model=model_id,stage=stage,factor=names[0],observed='X'+str(j),
                        connected=forward or reverse,factor_to_X=forward,X_to_factor=reverse,
                        direction='both' if forward and reverse else 'factor -> X' if forward else 'X -> factor' if reverse else 'none'))
            if model_id=='C_derived_Zhat':
                write_csv(output/('adjacency_'+stage+'.csv'),[dict(source=names[i],**{n:int(matrix[i,j]) for j,n in enumerate(names)}) for i in range(4)])
                score=graph_scores(matrix,shape_template)
                formal.append(dict(stage=stage,interpretation='formal shape mapping Z_hat to Z only; NOT causal recovery accuracy',
                                   **{'formal_'+key:value for key,value in score.items()}))
    for first,second in [('train','to-dag'),('to-dag','cam')]:
        a,b=matrices.get(('C_derived_Zhat',first)),matrices.get(('C_derived_Zhat',second))
        if a is None or b is None:continue
        for i,j in np.argwhere((a!=0)&(b==0)):
            changes.append(dict(before=first,after=second,action='removed',source=VARIABLES[i],target=VARIABLES[j]))
        for i,j in np.argwhere((a==0)&(b!=0)):
            changes.append(dict(before=first,after=second,action='added',source=VARIABLES[i],target=VARIABLES[j]))
    write_csv(output/'graph_comparison.csv',graph_rows)
    write_csv(output/'estimated_edges.csv',edge_rows)
    write_csv(output/'X_X_pairs.csv',pairs)
    write_csv(output/'factor_connections.csv',connections,['model','stage','factor','observed','connected','factor_to_X','X_to_factor','direction'])
    write_csv(output/'formal_shape_metrics.csv',formal,['stage','interpretation']+['formal_'+key for key in graph_scores(shape_template,shape_template)])
    write_csv(output/'stage_changes.csv',changes,['before','after','action','source','target'])
    write_csv(output/'stage_availability.csv',availability)
    make_figure(output,matrices)
    make_report(root,output,status,graph_rows,edge_rows,connections,formal,changes)
    assert_frozen(root)
    assert all(digest(root/name)==expected for name,expected in frozen.items())
    assert file_inventory(source)==json.loads((root/'preserved_source_inventory.json').read_text())
    assert file_inventory(oracle)==json.loads((root/'preserved_oracle_inventory.json').read_text())
    save_json(output/'evaluation_status.json',dict(completed=True,new_NN_converged=status.get('converged',False),
        source_and_oracle_unchanged=True,new_artifacts_unchanged=True,Z_hat_sign_unchanged=True,
        causal_identification_claimed=False,formal_SHD_is_shape_diagnostic_only=True,
        X_X_counting='distinct unordered pairs; directed count also reported',
        training_calls=status['training_calls'],no_comparison_model_trained=True))
    print('Exploratory Z_hat graph diagnosis complete:',output,flush=True)


def structural_summary(matrix,has_factor=True):
    p=np.asarray(matrix)
    if p.ndim!=2 or p.shape[0]!=p.shape[1] or not np.isin(p,[0,1]).all() or np.diag(p).any():
        raise ValueError('Binary square adjacency without self edges required')
    observed=p[1:,1:] if has_factor else p
    pair_count=int(np.triu((observed!=0)|(observed.T!=0),1).sum())
    return dict(n_variables=len(p),directed_edges=int(p.sum()),X_X_unordered_pairs=pair_count,
        X_X_directed_edges=int(observed.sum()),
        factor_connected_X_count=int(((p[0,1:]!=0)|(p[1:,0]!=0)).sum()) if has_factor else None,
        factor_connection_coverage=float(((p[0,1:]!=0)|(p[1:,0]!=0)).mean()) if has_factor else None,
        factor_to_X_edges=int(p[0,1:].sum()) if has_factor else None,
        X_to_factor_edges=int(p[1:,0].sum()) if has_factor else None)


def make_figure(output,matrices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,model in zip(axes,['A_original_X','B_oracle_Z','C_derived_Zhat']):
        matrix=matrices.get((model,'cam'))
        if matrix is None:ax.text(.2,.5,'CAM unavailable');ax.axis('off');continue
        names=['X1','X2','X3'] if model=='A_original_X' else ['Z','X1','X2','X3'] if model=='B_oracle_Z' else VARIABLES
        pos={0:(-.8,0),1:(0,.8),2:(.8,0)} if len(names)==3 else {0:(0,.8),1:(-.9,-.3),2:(0,-.5),3:(.9,-.3)}
        graph=nx.DiGraph();graph.add_nodes_from(range(len(names)));graph.add_edges_from(map(tuple,np.argwhere(matrix)))
        nx.draw_networkx(graph,pos=pos,labels=dict(enumerate(names)),ax=ax,node_size=1200,node_color='#ddeeff',arrows=True,arrowsize=20)
        ax.set_title(model+' (CAM)');ax.axis('off')
    fig.suptitle('Different variables: Z_hat is derived from X, NOT the true causal Z')
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(str(output/'CAM_comparison.png'),dpi=170);plt.close(fig)


def make_report(root,output,status,graph_rows,edges,connections,formal,changes):
    reproduction=json.loads((root/'reproduction_status.json').read_text())
    validation=json.loads((root/'reproduction_validation.json').read_text())
    provenance=json.loads((root/'zhat/provenance.json').read_text())
    lines=['# 抽出Z_hatを追加した探索的グラフ構造診断','',
        '## 範囲と再現性','',
        'seed201・σ=1・既存の規格化tanh/sin/Z³データ1000件、同じ学習600・検証200・テスト200を使用。再生成なし。',
        '新入力順は[Z_hat,X1,X2,X3]。Z_hatは真の因果的潜在原因ではなく、保存済みXから計算した変数。',
        '元3変数NNもPCAも再fitせず、学習済み重み・DAG化前マスク・第1層・PCA1成分を固定して再利用。',
        '元チェックポイント: '+provenance['checkpoint'],
        '元PCA/標準化器: '+provenance['PCA_file'],
        'shape=(1000,1)、有限値のみ、再推論/transformの最大差={:.12g}。'.format(reproduction['max_replay_error']),
        '因子分散(ddof=1): '+json.dumps(reproduction['variance_ddof1']),
        '保存済みの定数列除去と学習600件の標準化・PCAを検証。符号・尺度を一切変更せず元配列をコピー。',
        'Z_hat確定・ハッシュ凍結後の別プロセスでのみ評価用Zを読み込み、全分割の元指標との一致を確認。',
        'テストPearson={:.12f}、絶対相関={:.12f}、Spearman={:.12f}。'.format(validation['metrics']['pearson_r'],validation['metrics']['absolute_pearson_r'],validation['metrics']['spearman_r']),
        '0.825139との照合は保存結果の再現性確認で、符号・モデル・パラメータの選択ではない。',
        'ただし同じテストを過去にも閲覧済みで、今回の条件自体も指定済みの探索的解析。独立した未閲覧テストの確認実験とはしない。','',
        '## 唯一の新規グラフ学習','',
        'オラクルと同じ4変数用のNonLinGaussANM・2層×10・leaky-ReLU・RMSprop・DAG制約・収束設定を再利用。',
        '正解Zは新規学習入力に含めず、因子の向きも固定しない。初期マスクは自己辺以外の12方向すべて。',
        '既存oracle.train_onceは数値処理だけ共通利用し、この実験の入出力メタデータはZ_hatとして記録。',
        'ローダー必須のDAG1は全ゼロの診断用プレースホルダー。仮想的な星型形状を学習に与えない。',
        '新規学習回数={}、収束={}。'.format(status['training_calls'],status.get('converged',False)),
        '学習時間={}秒、全ワーカー時間={}秒、反復数={}、h={}。'.format(status.get('elapsed_nn_seconds'),status.get('worker_elapsed_seconds'),status.get('iterations'),status.get('final_h')),
        '最終テストは学習・CAM・選択に使わず、既存モデルや比較モデルを追加学習しない。','',
        '## 今回の各段階の有向辺','']
    for stage in ('train','to-dag','cam'):
        selected=[r for r in edges if r['model']=='C_derived_Zhat' and r['stage']==stage]
        available=any(r['model']=='C_derived_Zhat' and r['stage']==stage for r in graph_rows)
        lines+=['### '+stage,'',', '.join(r['source']+' → '+r['target'] for r in selected) if selected else '辺なし' if available else '未生成（収束/処理失敗、再学習せず）','']
    lines+=['## 全3条件の参考比較','',
        'A=元X3変数、B=真のZを追加したオラクル、C=決定論的なZ_hatを追加した診断。意味・変数数が異なる。','',
        '|条件|段階|有向辺数|X同士のペア数|X同士の有向辺数|因子の接続X数|因子→X|X→因子|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in graph_rows:
        lines.append('|{}|{}|{}|{}|{}|{}|{}|{}|'.format(r['model'],r['stage'],r['directed_edges'],r['X_X_unordered_pairs'],r['X_X_directed_edges'],r['factor_connected_X_count'],r['factor_to_X_edges'],r['X_to_factor_edges']))
    lines+=['','X同士のペアは向きに関係なく1組1本。双方向があってもペア数は1、有向辺数は2として区別する。','',
        '## Z_hatと各Xの接続','', '|段階|X|接続|方向|','|---|---|---|---|']
    for r in connections:
        if r['model']=='C_derived_Zhat':lines.append('|{}|{}|{}|{}|'.format(r['stage'],r['observed'],r['connected'],r['direction']))
    lines+=['','接続先カバー率は3個のXのうち何個に接続したか、という探索的形状指標にすぎない。因果的正解率ではない。','',
        '## 形式的にZ_hatをZへ対応づけた参考値','',
        '以下は仮想のZ_hat→各Xという星型に対する形状一致度。真の潜在原因の復元精度ではなく、オラクルSHDと同等の意味ではない。',
        '有向反転を1操作としSHD=FN+FP−反転数。Precision/Recall/F1も仮想的な有向形状に対してだけ計算。','',
        '|段階|形式的SHD|形式的Precision|形式的Recall|形式的F1|','|---|---:|---:|---:|---:|']
    for r in formal:
        lines.append('|{}|{}|{:.6f}|{:.6f}|{:.6f}|'.format(r['stage'],r['formal_shd'],r['formal_precision'],r['formal_recall'],r['formal_f1']))
    lines+=['','## 後処理での変化','']
    for r in changes:lines.append('- {}→{}: {} {}→{}。'.format(r['before'],r['after'],r['action'],r['source'],r['target']))
    lines+=['','## 決定論的依存の数値と理論','',
        '元X→保存済みNNの第1層→保存標準化器→保存PCAという固定関数で、全行の保存Z_hatを再現した。',
        'したがって条件付きでVar(Z_hat|X1,X2,X3)=0。単一Xとの相関が1でなくても、全Xへの決定論的依存はある。',
        '通常の非退化な4変数連続分布・独立な正の分散のノイズというモデルには、この追加列はそのまま整合しない。',
        '真のZは依然として未観測であり、Z_hatの追加が因果十分性を回復する保証もない。',
        '特にX→Z_hatは因子の計算方法を反映する可能性があり、Z_hat→Xも真の共通原因の証明ではない。',
        '縮退した残差・過学習・向きの不安定性の可能性があり、今回の結果は探索的な構造診断に限る。','',
        '|分割|X|Pearson(Z_hat,X)|絶対値|Spearman|','|---|---|---:|---:|---:|']
    for r in read_csv(root/'zhat_X_correlations.csv'):
        if r['split']=='test':lines.append('|test|{}|{:.6f}|{:.6f}|{:.6f}|'.format(r['observed_variable'],float(r['pearson_r']),float(r['absolute_pearson_r']),float(r['spearman_r'])))
    lines+=['','## 今後への課題（今回追加実験なし）','',
        '- 真のZとZ_hatを同じ因果変数として扱わず、観測間のペア数、因子接続先、辺の向きを別々に評価する。',
        '- オラクルにも方向誤りがあり、Z_hat実験の失敗を抽出精度だけに帰属できない。',
        '- 学習行のみで因子抽出器をfitし独立評価を維持する。クロスフィットだけでは同一行Xから作る因子の決定論的依存は解消しない。',
        '- 因果的同定には、別の測定・独立指標・追加の生成仮定等を持つモデル設計が必要。高相関や星型グラフだけでは証明できない。',
        '- 1シード・1条件の診断であり、一般的優位性を主張しない。追加比較モデルや再学習は実行せず終了する。','',
        '## 保存物と実行','',
        'zhat/Z_hat.npy、scaler_pca.pkl、provenance.json、reproduction_status.json、reproduction_validation.json：再現性と符号の監査。',
        'zhat_Z_evaluation.csv、zhat_X_correlations.csv：固定後の評価。training.logとtraining_status.json：唯一の新規学習。',
        'evaluation/adjacency_*.csv、estimated_edges.csv、graph_comparison.csv、X_X_pairs.csv、factor_connections.csv、formal_shape_metrics.csv、CAM_comparison.png。',
        '新入口latent_zhat_graph.py、設定configs/latent_zhat_seed201.json。実行順はreproduce、validate、train、evaluate。','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/latent_zhat_seed201.json'))
    parser.add_argument('--stage',choices=['reproduce','validate','train','evaluate'],required=True)
    args=parser.parse_args();settings=json.loads(args.config.read_text());root=mapped(settings['output_dir'])
    if args.stage=='reproduce':reproduce(settings)
    elif args.stage=='validate':validate(root)
    elif args.stage=='evaluate':evaluate(root)
    else:
        if (root/'training_invocation.json').exists():raise FileExistsError('The only new training invocation is already used')
        with (root/'training.log').open('x') as log:
            subprocess.check_call([sys.executable,'-m','latent_experiments.zhat_graph','--config',str(args.config),'--worker'],
                                  stdout=log,stderr=subprocess.STDOUT)


if __name__=='__main__':
    if '--worker' in sys.argv:
        p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--worker',action='store_true')
        a=p.parse_args();train_once(mapped(json.loads(a.config.read_text())['output_dir']))
    else:main()
