"""Reuse saved Z and X as four observations; ground-truth graph is evaluation-only."""
import argparse
import copy
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from latent_unsupervised import mapped
from .unsupervised import digest, file_inventory, read_csv, save_json, write_csv


VARIABLES=['Z','X1','X2','X3']


def graph_scores(prediction,truth):
    """Binary directed edge scores. Reversal replaces a missing true direction."""
    p,t=np.asarray(prediction),np.asarray(truth)
    if (p.shape!=t.shape or p.ndim!=2 or p.shape[0]!=p.shape[1] or
            not np.isin(p,[0,1]).all() or not np.isin(t,[0,1]).all() or
            np.diag(p).any() or np.diag(t).any()):
        raise ValueError('Aligned binary source-row matrices without self edges required')
    p,t=p.astype(bool),t.astype(bool)
    tp=int((p&t).sum());fn=int((t&~p).sum());fp=int((p&~t).sum())
    rev=int((t&~p&p.T).sum())
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    return dict(shd=fn+fp-rev,precision=precision,recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        predicted_edges=int(p.sum()),correct_edges=tp,missing_true_edges=fn,
        extra_directed_edges=fp,reversed_edges=rev,
        missing_without_reversal=fn-rev,extra_without_reversal=fp-rev)


def prepare(settings):
    source,root=mapped(settings['source_run']).resolve(),mapped(settings['output_dir']).resolve()
    if root.exists():raise FileExistsError('Never overwrite an existing oracle run: '+str(root))
    config=json.loads((source/'config.json').read_text())
    if (settings['variable_order']!=VARIABLES or settings['seed']!=201 or settings['noise_std']!=1 or
            config['data']['seed']!=201 or config['split']['seed']!=201 or config['training']['seed']!=201 or
            config['data']['latent_functions']!=['tanh','sin','cubic'] or config['data']['noise_std']!=1):
        raise ValueError('Only the existing normalized nonlinear seed201 sigma1 data are in scope')
    before=file_inventory(source)
    x=np.load(str(source/'data/observed_X.npy'));z=np.load(str(source/'data/true_Z.npy'))
    ids=np.load(str(source/'data/sample_indices.npy'));labels=np.load(str(source/'data/sample_split.npy'))
    assert x.shape==(1000,3) and z.shape==(1000,1)
    np.testing.assert_array_equal(ids,np.arange(1000))
    np.testing.assert_array_equal(np.bincount(labels),[600,200,200])
    input_data=np.column_stack([z,x])
    assert np.isfinite(input_data).all() and input_data.shape==(1000,4)
    for name,code in [('train',0),('validation',1),('test',2)]:
        np.testing.assert_array_equal(np.sort(np.load(str(source/'data'/(name+'_indices.npy')))),ids[labels==code])
    pool=np.load(str(source/'data/pool_indices.npy'))
    assert len(pool)==800 and not np.any(labels[pool]==2)
    generation=json.loads((source/'data/config.json').read_text())
    norm=generation['resolved_signal_normalization']
    train_ids=np.load(str(source/'data/train_indices.npy'))
    # Verify existing generation, without drawing or changing a single sample.
    raw=np.column_stack([np.tanh(z[:,0]),np.sin(z[:,0]),z[:,0]**3])
    np.testing.assert_array_equal(np.asarray(norm['mean']),raw[train_ids].mean(0))
    np.testing.assert_array_equal(np.asarray(norm['std']),raw[train_ids].std(0))
    np.testing.assert_array_equal(x,(raw-np.asarray(norm['mean']))/np.asarray(norm['std'])+
                                  np.load(str(source/'data/noise.npy')))
    root.mkdir(parents=True);(root/'data').mkdir();(root/'training_input').mkdir()
    names=['observed_X.npy','true_Z.npy','sample_indices.npy','sample_split.npy','train_indices.npy',
           'validation_indices.npy','test_indices.npy','pool_indices.npy','signal_mean.npy','signal_std.npy']
    manifest=json.loads((source/'data_manifest.json').read_text())
    hashes={}
    for name in names:
        assert digest(source/'data'/name)==manifest['data_files'][name]
        shutil.copyfile(str(source/'data'/name),str(root/'data'/name))
        hashes['data/'+name]=digest(source/'data'/name)
    shutil.copyfile(str(source/'data/config.json'),str(root/'data/generation_config.json'))
    hashes['data/config.json']=digest(source/'data/config.json')
    shutil.copyfile(str(source/'config.json'),str(root/'source_config.json'))
    np.save(str(root/'data/input_Z_X.npy'),input_data)
    np.save(str(root/'training_input/data1.npy'),input_data[pool])
    np.save(str(root/'training_input/sample_indices.npy'),pool)
    # Original DataManagerFile requires DAG1. This is NOT the true star graph.
    np.save(str(root/'training_input/DAG1.npy'),np.zeros((4,4),dtype=np.int64))
    training=dict(config['training'],n_observed=4,n_train=600,n_validation=200)
    save_json(root/'config.json',dict(settings,training=training,source_data_sha256=hashes,
        oracle=True,Z_observed=True,source_generation_parameters=norm))
    save_json(root/'training_options.json',training)
    save_json(root/'preparation_status.json',dict(completed=True,new_data_generated=False,
        saved_Z_X_rows_exact=True,variable_order=VARIABLES,split_counts=[600,200,200],
        source_normalization_preserved=True,truth_passed_to_training=False,
        loader_graph='all-zero placeholder for diagnostic logging only; not used for optimization',
        input_sha256=digest(root/'training_input/data1.npy'),pool_ids_sha256=digest(root/'training_input/sample_indices.npy'),
        source_inventory_unchanged=before==file_inventory(source)))
    save_json(root/'preserved_source_inventory.json',before)
    print('Prepared existing [Z,X1,X2,X3] data, no generation:',root,flush=True)


def train_once(root):
    import torch
    from gran_dag.data import DataManagerFile
    from gran_dag.dag_optim import compute_constraint
    from gran_dag.models.learnables import LearnableModel_NonLinGaussANM
    from gran_dag.train import train,to_dag,cam_pruning
    from .train import dump,make_options
    root=Path(root).resolve()
    # Exclusive marker persists even on failure: no retry/resume of this NN fit.
    with (root/'training_invocation.json').open('x') as stream:
        json.dump(dict(invocations=1,seed=201,started_unix=time.time()),stream)
    if (root/'train').exists():raise FileExistsError('Training artifacts already exist')
    settings=json.loads((root/'training_options.json').read_text())
    prepared=json.loads((root/'preparation_status.json').read_text())
    assert digest(root/'training_input/data1.npy')==prepared['input_sha256']
    assert digest(root/'training_input/sample_indices.npy')==prepared['pool_ids_sha256']
    torch.set_num_threads(settings['torch_num_threads']);torch.set_default_tensor_type('torch.DoubleTensor')
    torch.manual_seed(settings['seed']);np.random.seed(settings['seed'])
    opt=make_options(root,settings)
    assert opt.num_vars==4 and opt.model=='NonLinGaussANM' and opt.retrain is False
    model=LearnableModel_NonLinGaussANM(4,opt.num_layers,opt.hid_dim,nonlin=opt.nonlin,
                                      norm_prod=opt.norm_prod,square_prod=opt.square_prod)
    assert torch.equal(model.adjacency,torch.ones(4,4)-torch.eye(4))
    (root/'initial').mkdir();dump(root/'initial/model.pkl',model);dump(root/'initial/opt.pkl',opt)
    np.save(str(root/'initial/adjacency.npy'),model.adjacency.detach().cpu().numpy())
    train_data=DataManagerFile(opt.data_path,'1',600,200,normalize=opt.normalize_data,random_seed=201)
    validation=DataManagerFile(opt.data_path,'1',600,200,train=False,normalize=opt.normalize_data,
                               mean=train_data.mean,std=train_data.std,random_seed=201)
    assert not (train_data.adjacency!=0).any().item() and not (validation.adjacency!=0).any().item()
    pool=np.load(str(root/'training_input/sample_indices.npy'))
    labels=np.load(str(root/'data/sample_split.npy'))
    np.testing.assert_array_equal(pool[train_data.sample_indices],np.load(str(root/'data/train_indices.npy')))
    np.testing.assert_array_equal(pool[validation.sample_indices],np.load(str(root/'data/validation_indices.npy')))
    assert not np.any(labels[pool]==2)
    np.save(str(root/'normalization_mean.npy'),train_data.mean.cpu().numpy())
    np.save(str(root/'normalization_std.npy'),train_data.std.cpu().numpy())
    def callback(stage,step,metrics,throttle=None):
        # All built-in graph metrics compare with the zero placeholder, not truth.
        safe={key:float(value) for key,value in metrics.items() if key not in
              ('fn','fp','rev','shd','shd_cpdag','sid','fn_edges','fp_edges','rev_edges')}
        with (root/'training_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(stage=stage,step=step,metrics=safe))+'\n')
        if stage!='train' or step%1000==0: print(stage,step,'h=',safe.get('constraint_violation'),flush=True)
    started=time.time()
    trained=train(model,np.zeros((4,4)),train_data,validation,opt,callback,None)
    elapsed_nn=time.time()-started
    if trained is None:
        (root/'unconverged').mkdir();dump(root/'unconverged/model.pkl',model)
        save_json(root/'training_status.json',dict(converged=False,training_calls=1,
            iteration_limit=opt.num_train_iter,elapsed_nn_seconds=elapsed_nn,
            final_h=float(compute_constraint(model,model.get_w_adj()).item()),retraining_forbidden=True))
        raise RuntimeError('Single oracle training hit iteration limit. Saved diagnostics; no retry permitted.')
    dag=to_dag(copy.deepcopy(trained),train_data,validation,opt,callback,None)
    pruned=cam_pruning(copy.deepcopy(dag),train_data,validation,opt,callback,None,cutoff=settings['cam_pruning_cutoff'])
    with (root/'train/aug-lagrangians.pkl').open('rb') as stream:iterations=len(pickle.load(stream))
    save_json(root/'training_status.json',dict(converged=True,training_calls=1,iterations=iterations,
        elapsed_nn_seconds=elapsed_nn,worker_elapsed_seconds=time.time()-started,
        final_h=float(compute_constraint(trained,trained.get_w_adj()).item()),
        input_order=VARIABLES,Z_used_as_observed_input=True,Z_direction_fixed=False,
        true_graph_used_in_training=False,training_graph_placeholder='all zeros, diagnostic only',
        n_train=600,n_validation=200,n_test=200,final_test_used_in_training_or_pruning=False,
        cam_pruning_input_rows=800,cam_pruning_retrains_nn=False,PCA_FA_run=False,Z_hat_training_run=False,
        source_sha256={name:digest(name) for name in ('gran_dag/train.py','gran_dag/models/base_model.py',
            'gran_dag/data.py','latent_experiments/oracle.py')}))
    save_json(root/'training_artifacts_manifest.json',{str(p.relative_to(root)):digest(p)
        for stage in ('train','to-dag','cam-pruning','initial') for p in (root/stage).rglob('*') if p.is_file()})
    print('ONE oracle NN fit and existing postprocessing complete.',flush=True)


def evaluate(root):
    from .data import topological_order
    root=Path(root).resolve();output=root/'evaluation'
    if output.exists():raise FileExistsError('Never overwrite an existing oracle evaluation')
    status=json.loads((root/'training_status.json').read_text())
    if not status['converged'] or status['training_calls']!=1:
        raise ValueError('Only evaluate the single completed, converged run')
    frozen=json.loads((root/'training_artifacts_manifest.json').read_text())
    for name,expected in frozen.items():
        if digest(root/name)!=expected:raise ValueError('Training artifact changed: '+name)
    config=json.loads((root/'config.json').read_text());source=mapped(config['source_run'])
    assert file_inventory(source)==json.loads((root/'preserved_source_inventory.json').read_text())
    # The true graph is instantiated/saved only now, after every graph is frozen.
    truth=np.zeros((4,4),dtype=np.int64);truth[0,1:]=1
    output.mkdir();np.save(str(output/'true_adjacency.npy'),truth)
    def named_csv(path,matrix,names):
        write_csv(path,[dict(source=names[i],**{name:int(matrix[i,j]) for j,name in enumerate(names)}) for i in range(len(names))])
    named_csv(output/'adjacency_true.csv',truth,VARIABLES)
    stages=[('train','train'),('to-dag','to-dag'),('cam','cam-pruning/cutoff_%.6f'%config['training']['cam_pruning_cutoff'])]
    matrices={};metrics=[];edges=[];recovery=[];reference=[];transitions=[]
    for stage,path in stages:
        matrix=np.load(str(root/path/'DAG.npy'))
        assert matrix.shape==(4,4) and np.isin(matrix,[0,1]).all()
        matrices[stage]=matrix
        named_csv(output/('adjacency_'+stage+'.csv'),matrix,VARIABLES)
        try:topological_order(matrix);acyclic=True
        except ValueError:acyclic=False
        score=dict(stage=stage,**graph_scores(matrix,truth))
        score.update(is_DAG=acyclic,extra_X_to_X_edges=int(matrix[1:,1:].sum()),
                     incoming_edges_to_Z=int(matrix[1:,0].sum()))
        metrics.append(score)
        for i,j in np.argwhere(matrix):
            edges.append(dict(stage=stage,source=VARIABLES[i],target=VARIABLES[j],
                correct=bool(truth[i,j]),reverse_of_truth=bool(truth[j,i] and not matrix[j,i])))
        for j in range(1,4):
            recovery.append(dict(stage=stage,true_edge='Z -> '+VARIABLES[j],recovered=bool(matrix[0,j]),
                                 reverse_present=bool(matrix[j,0])))
        old=np.load(str(source/path/'DAG.npy'))
        assert old.shape==(3,3)
        old_edges=['{} -> {}'.format(VARIABLES[i+1],VARIABLES[j+1]) for i,j in np.argwhere(old)]
        reference.append(dict(stage=stage,three_variable_X_to_X_edges=int(old.sum()),
            oracle_X_to_X_edges=int(matrix[1:,1:].sum()),
            three_variable_edges='; '.join(old_edges),
            caution='descriptive comparison only; different variable sets, no direct SHD performance subtraction'))
    for first,second in [('train','to-dag'),('to-dag','cam')]:
        a,b=matrices[first],matrices[second]
        for i,j in np.argwhere((a!=0)&(b==0)):
            transitions.append(dict(before=first,after=second,action='removed',source=VARIABLES[i],
                                    target=VARIABLES[j],true_edge=bool(truth[i,j])))
        for i,j in np.argwhere((a==0)&(b!=0)):
            transitions.append(dict(before=first,after=second,action='added',source=VARIABLES[i],
                                    target=VARIABLES[j],true_edge=bool(truth[i,j])))
    write_csv(output/'graph_metrics.csv',metrics)
    write_csv(output/'estimated_edges.csv',edges,['stage','source','target','correct','reverse_of_truth'])
    write_csv(output/'true_edge_recovery.csv',recovery)
    write_csv(output/'three_variable_reference.csv',reference)
    write_csv(output/'stage_changes.csv',transitions,['before','after','action','source','target','true_edge'])
    make_report(root,output,config,status,metrics,matrices,reference,transitions)
    make_graph_figure(output,truth,matrices)
    assert all(digest(root/name)==expected for name,expected in frozen.items())
    assert file_inventory(source)==json.loads((root/'preserved_source_inventory.json').read_text())
    save_json(output/'evaluation_status.json',dict(completed=True,oracle_not_latent_discovery=True,
        truth_used_after_training_only=True,true_graph_used_for_model_selection=False,
        threshold_tuned_against_truth=False,training_artifacts_unchanged=True,source_unchanged=True,
        variable_order=VARIABLES,training_calls=1,PCA_FA_run=False,Z_hat_training_run=False,
        metrics_definition='TP directed intersection; FN/FP include reversal; R counts absent true edge with reverse present; SHD=FN+FP-R',
        zero_denominator_policy='precision/recall/F1 set to zero when their denominator is zero'))
    print('Oracle evaluation complete:',output,flush=True)


def make_graph_figure(output,truth,matrices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    positions={0:(0,.9),1:(-1,-.5),2:(0,-.7),3:(1,-.5)}
    for ax,(name,matrix) in zip(axes,[('truth',truth)]+list(matrices.items())):
        graph=nx.DiGraph();graph.add_nodes_from(range(4));graph.add_edges_from(map(tuple,np.argwhere(matrix)))
        colors=['#228833' if truth[i,j] else '#cc3311' for i,j in graph.edges()]
        nx.draw_networkx(graph,pos=positions,labels=dict(enumerate(VARIABLES)),ax=ax,
                         node_color='#ddeeff',node_size=1100,edge_color=colors,arrows=True,arrowsize=20)
        ax.set_title(name);ax.axis('off')
    fig.suptitle('Oracle [Z, X1, X2, X3]: green=true direction, red=extra/reversed')
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(str(output/'graphs.png'),dpi=170);plt.close(fig)


def make_report(root,output,config,status,metrics,matrices,reference,transitions):
    lines=['# 真のZを観測するオラクル因果グラフ実験','',
        '## 目的とデータ','',
        '真のZを通常の観測変数として入力したオラクル実験。教師なし潜在変数発見やZ_hatの成功ではない。',
        '保存済みの規格化tanh・sin・3乗データを再利用し、データを再生成していない。',
        'Z~N(0,1)、Xi=(fi(Z)−mi)/si+εi、f=(tanh,sin,cubic)、独立標準正規ノイズσ=1。',
        '信号規格化係数は元データの学習600件で算出した値をそのまま維持。data/generation_config.jsonに保存。',
        'seed201、1000件、学習600／検証200／最終テスト200。サンプルIDと分割は元実験と完全一致。',
        '変数順は[Z,X1,X2,X3]、隣接行列は行＝原因・列＝結果。Zは先頭列であり、方向は固定しない。','',
        '## 学習','',
        '元実験と同じNonLinGaussANM・2層×10ユニット・leaky-ReLU・RMSprop・DAG制約・収束設定。',
        '新規NN学習1回のみ。初期マスクは自己辺を除く12方向すべて。Zへの入辺も候補に含めた。',
        '収束={}、反復数={}、h={:.12g}。'.format(status['converged'],status['iterations'],status['final_h']),
        'NN学習時間={:.2f}秒、DAG化・CAMを含むワーカー時間={:.2f}秒。'.format(status['elapsed_nn_seconds'],status['worker_elapsed_seconds']),
        '標準化は学習600件のみ。最終テスト200件は学習・収束・DAG化・CAM・選択に使用しない。',
        '正解グラフは学習に与えず、必須ローダーには全ゼロの診断表示用プレースホルダーを渡した。',
        '既存学習ログ・自動生成adjacency.png内の正解欄はこのプレースホルダーであり、本評価ではない。',
        '評価用の正解グラフと正しい比較図は、全段階のグラフを保存した後にevaluation/へ作成した。',
        'CAMは元設定のcutoff={}。NN再学習や正解に合わせた閾値変更は行わない。'.format(config['training']['cam_pruning_cutoff']),
        'PCA・FA・Kernel PCA・Z_hatによる追加学習は実行しない。','',
        '## 正解と推定辺','',
        '正解：Z→X1、Z→X2、Z→X3。X同士の直接辺とZへの入辺はない。','']
    for stage,matrix in matrices.items():
        edges=['{} → {}'.format(VARIABLES[i],VARIABLES[j]) for i,j in np.argwhere(matrix)]
        lines+=['### '+stage,'',', '.join(edges) if edges else '辺なし','']
    lines+=['## グラフ指標','',
        'TPは向きまで正しい辺、FNは存在しない真の有向辺、FPは余分な有向辺。反転は通常のFN/FPの両方に数える。',
        'Rは「真の向きがなく逆向きがある」辺数。SHD=FN+FP−Rとし、反転を1操作に数える。',
        '両方向を推定して正しい向きも残る場合、逆向きの1辺削除として扱い、Rには数えない。',
        'Precision=TP/(TP+FP)、Recall=TP/(TP+FN)、F1は調和平均。分母0は0とする。','',
        '|段階|SHD|Precision|Recall|F1|TP|FN※|FP※|反転R|X同士の辺|Zへの入辺|DAG|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in metrics:
        lines.append('|{}|{}|{:.6f}|{:.6f}|{:.6f}|{}|{}|{}|{}|{}|{}|{}|'.format(row['stage'],row['shd'],row['precision'],
            row['recall'],row['f1'],row['correct_edges'],row['missing_true_edges'],row['extra_directed_edges'],
            row['reversed_edges'],row['extra_X_to_X_edges'],row['incoming_edges_to_Z'],row['is_DAG']))
    lines+=['','※FN/FPは反転を含む。反転を除いた純粋な欠落・追加もgraph_metrics.csvに保存。','',
        '## Zから各Xへの復元','',
        '|段階|Z→X1|Z→X2|Z→X3|','|---|---|---|---|']
    for stage,matrix in matrices.items():
        lines.append('|{}|{}|{}|{}|'.format(stage,*['復元' if matrix[0,j] else '未復元' for j in range(1,4)]))
    lines+=['','## DAG化・CAMでの変化','']
    if not transitions:lines.append('両後処理とも辺の変化なし。')
    for row in transitions:
        lines.append('- {}→{}: {} {}→{}（真の辺={}）。'.format(row['before'],row['after'],row['action'],row['source'],row['target'],row['true_edge']))
    lines+=['','## 既存の3変数実験との参考比較','',
        '|段階|3変数モデルのX同士の辺|オラクルのX同士の辺|','|---|---:|---:|']
    for row in reference:
        lines.append('|{}|{}|{}|'.format(row['stage'],row['three_variable_X_to_X_edges'],row['oracle_X_to_X_edges']))
    lines+=['','3変数と4変数では対象グラフ・学習問題が異なる。SHD差を同一条件の性能改善として比較しない。',
        'Zを含むことで真の交絡原因が観測されるが、有限サンプル・関数近似・最適化・枝刈りにより方向や辺の完全復元は保証されない。','',
        '## 解釈とZ_hat実験への課題','',
        '- 真のZを与えた結果は潜在変数を未知のXから発見できたことを意味しない。',
        '- 正しく復元できても、抽出Z_hatの測定誤差・符号尺度・観測Xへの依存が同じ結果を生む保証はない。',
        '- Z_hatはXの関数なので、[Z_hat,X]は本来の独立ノイズの因果生成モデルと同一ではなく、循環的な解釈に注意する。',
        '- 将来Z_hatを使う場合は学習行だけで抽出器をfitし、必要に応じクロスフィットと独立テストを事前設計する。',
        '- 単一シード・単一条件の探索的結果であり、一般的識別可能性や有意な優位性は主張しない。',
        '- 今回は保存済みデータを再利用し、Z_hatや別条件の追加学習を実行せず終了する。','',
        '## 保存物','',
        '- evaluation/adjacency_true.csv、adjacency_train.csv、adjacency_to-dag.csv、adjacency_cam.csv：変数名付き行列。',
        '- graph_metrics.csv、estimated_edges.csv、true_edge_recovery.csv、stage_changes.csv：全指標・辺・段階差。',
        '- three_variable_reference.csv：既存3変数結果の読み取り比較。graphs.png：正解と各推定グラフ。',
        '- train/、to-dag/、cam-pruning/：変更していない元処理の学習済みモデルとDAG.npy。',
        '- preparation_status.json、training_status.json、training_artifacts_manifest.json、evaluation_status.json：整合性・再学習回数・不変性。','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/latent_oracle_seed201.json'))
    parser.add_argument('--stage',choices=['prepare','train','evaluate'],required=True)
    args=parser.parse_args();settings=json.loads(args.config.read_text());root=mapped(settings['output_dir'])
    if args.stage=='prepare':prepare(settings)
    elif args.stage=='evaluate':evaluate(root)
    else:
        if (root/'training_invocation.json').exists():raise FileExistsError('The single training invocation was already used')
        with (root/'training.log').open('x') as stream:
            subprocess.check_call([sys.executable,'-m','latent_experiments.oracle','--config',str(args.config),'--worker'],
                                  stdout=stream,stderr=subprocess.STDOUT)


if __name__=='__main__':
    if '--worker' in sys.argv:
        parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--worker',action='store_true')
        args=parser.parse_args();train_once(mapped(json.loads(args.config.read_text())['output_dir']))
    else:main()
