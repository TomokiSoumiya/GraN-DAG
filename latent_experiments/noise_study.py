"""Matched base randomness across noise levels, original GraN-DAG and label-free extractors."""
import copy
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .additional import digest, read_csv, snapshot
from .config import load_config
from .evaluate import write_csv
from .runner import run as run_original
from .unsupervised import file_inventory, save_json, transform_extractor


KEYS=('condition','method','split','latent')
SCORES=('pearson_r','absolute_pearson_r','spearman_r','x_reconstruction_raw_mse',
        'x_reconstruction_standardized_mse','factor_variance')
SOURCE_FILES=('gran_dag/train.py','gran_dag/dag_optim.py','gran_dag/models/base_model.py',
              'gran_dag/hidden.py','gran_dag/utils/cam_pruning.R','latent_experiments/train.py',
              'latent_experiments/data.py','latent_experiments/unsupervised.py',
              'latent_experiments/additional.py')


def mapped(path):
    path=Path(path)
    if (not path.is_absolute() and path.parts[0]=='experiments' and
            Path.cwd()==Path('/code') and Path('/final_log').is_dir()):
        return Path('/final_log').joinpath(*path.parts[1:])
    return path


def sigma_name(sigma):
    return 'sigma_'+str(float(sigma)).replace('.','p')


def run_config(spec,sigma,seed):
    config=load_config(spec['base_config'])
    config['name']='latent_noise_{}_seed_{}'.format(sigma,seed)
    config['output_dir']=str(Path(spec['output_dir'])/'runs'/sigma_name(sigma)/('seed_'+str(seed)))
    config['data']['noise_std']=float(sigma)
    for section in ('data','split','training'):
        config[section]['seed']=seed
    return config


def expected_jobs(spec,root):
    return [dict(sigma=float(sigma),seed=seed,
                 directory=str(Path('runs')/sigma_name(sigma)/('seed_'+str(seed))))
            for sigma in spec['sigmas'] for seed in spec['seeds']]


def prepare(spec,root):
    root=Path(root).resolve()
    if root.exists():
        raise FileExistsError('Choose a NEW study directory: '+str(root))
    if spec['sigmas']!=[.5,1.,2.,4.] or len(set(spec['seeds']))!=5:
        raise ValueError('Four fixed sigmas and five distinct seeds are required')
    factor=load_config(spec['factor_config'])
    if factor['n_factors']!=1 or factor['condition_codes']!=['C','E','B','D']:
        raise ValueError('Freeze one factor and the same matched-mask feature conditions')
    root.mkdir(parents=True)
    (root/'logs').mkdir()
    # Inventory only: do not read old latent values or supervised results before extraction.
    protected={p.name:file_inventory(p) for p in sorted(root.parent.iterdir())
               if p.is_dir() and p!=root}
    save_json(root/'protected_inventory.json',protected)
    jobs=expected_jobs(spec,root)
    for job in jobs:
        directory=root/job['directory']
        config=run_config(spec,job['sigma'],job['seed'])
        run_original(config,directory,stage='generate')
        settings=copy.deepcopy(factor)
        settings.update(seed=job['seed'],run_dir=str(directory),output_dir=str(directory/'unsupervised_factors'))
        save_json(directory/'factor_settings.json',settings)
        job['config_sha256']=digest(directory/'config.json')
        job['factor_settings_sha256']=digest(directory/'factor_settings.json')
    save_json(root/'plan.json',dict(spec=spec,jobs=jobs,source_sha256={p:digest(p) for p in SOURCE_FILES},
        factor_count_fixed=1,hyperparameter_search=False,representative_seed_fixed=spec['representative_seed'],
        phase_order=['generate_all','train_and_extract_all_without_Z','verify_global_freeze',
                     'evaluate_all_with_Z','paired_summary'],
        note='seed 42 sigma=1 reuses a previously inspected dataset; all results are exploratory'))
    print('Prepared 20 matched datasets:',root,flush=True)


def validate_plan(spec,root):
    plan=json.loads((root/'plan.json').read_text())
    if plan['spec']!=spec:
        raise ValueError('Study settings changed after preparation')
    for name,sha in plan['source_sha256'].items():
        if digest(name)!=sha:
            raise ValueError('Frozen code changed: '+name)
    for job in plan['jobs']:
        directory=root/job['directory']
        if (digest(directory/'config.json')!=job['config_sha256'] or
                digest(directory/'factor_settings.json')!=job['factor_settings_sha256']):
            raise ValueError('Frozen run settings changed: '+str(directory))
    return plan


def copy_existing_if_exact(spec,job,directory):
    reuse=spec.get('reuse')
    if not reuse or job['sigma']!=reuse['sigma'] or job['seed']!=reuse['seed']:
        return False
    source=mapped(reuse['source']).resolve()
    old=json.loads((source/'config.json').read_text())
    new=json.loads((directory/'config.json').read_text())
    for section in ('data','split','training'):
        if old[section]!=new[section]:
            raise ValueError('Predeclared reuse configuration does not exactly match: '+section)
    status=json.loads((source/'training_status.json').read_text())
    if not status['converged'] or status['latent_labels_used_by_grandag'] or status['final_test_used_by_grandag']:
        raise ValueError('Existing model fails convergence/leakage checks')
    # Identity checks on X/splits only; no latent label is read to choose a model.
    for name in ('observed_X','sample_indices','sample_split','train_indices','validation_indices','test_indices'):
        if digest(source/'data'/(name+'.npy'))!=digest(directory/'data'/(name+'.npy')):
            raise ValueError('Dataset identity mismatch for reuse: '+name)
    paths=('train','to-dag','cam-pruning','random_full','random_train_mask','random_cam_mask')
    for name in paths:
        shutil.copytree(str(source/name),str(directory/name))
    for name in ('normalization_mean.npy','normalization_std.npy','training_status.json'):
        shutil.copy2(str(source/name),str(directory/name))
    save_json(directory/'model_origin.json',dict(origin='copied_existing_exact_match',source=str(source),
        no_new_nn_optimization=True,copied_paths=list(paths),
        warning='opt.pkl paths refer to the original run; copied models are used for frozen inference',
        checkpoint_sha256={str(p.relative_to(directory)):digest(p)
                           for name in paths for p in (directory/name).rglob('model.pkl')}))
    return True


def fit(spec,root):
    root=Path(root).resolve()
    plan=validate_plan(spec,root)
    if (root/'fit_jobs.json').exists():
        raise FileExistsError('Fit was already attempted; existing outputs will not be overwritten')
    for job in plan['jobs']:
        directory=root/job['directory']
        if (directory/'train').exists() or (directory/'unsupervised_factors').exists():
            raise FileExistsError('Model/extraction already exists: '+str(directory))

    def worker(job):
        directory=root/job['directory']
        started=time.time()
        result=dict(sigma=job['sigma'],seed=job['seed'],directory=job['directory'])
        print('Start train/extract',job['sigma'],job['seed'],flush=True)
        logpath=root/'logs'/('{}_seed_{}.log'.format(sigma_name(job['sigma']),job['seed']))
        try:
            with logpath.open('x') as log:
                reused=copy_existing_if_exact(spec,job,directory)
                result['model_origin']='reused' if reused else 'newly_trained'
                if not reused:
                    subprocess.check_call([sys.executable,'-m','latent_experiments.train',
                                           '--run-dir',str(directory)],stdout=log,stderr=subprocess.STDOUT)
                    save_json(directory/'model_origin.json',dict(origin='newly_trained',
                        no_new_nn_optimization=False,training_seed=job['seed']))
                # Extraction ONLY. No old supervised evaluation route is invoked.
                subprocess.check_call([sys.executable,'latent_unsupervised.py','--stage','extract',
                                       '--config',str(directory/'factor_settings.json')],
                                      stdout=log,stderr=subprocess.STDOUT)
            status=json.loads((directory/'training_status.json').read_text())
            extra=json.loads((directory/'unsupervised_factors/extraction_status.json').read_text())
            if not status['converged'] or extra['z_read_attempts'] or not extra['completed']:
                raise RuntimeError('Convergence or label-free extraction audit failed')
            if (directory/'unsupervised_factors/metrics.csv').exists():
                raise RuntimeError('Z evaluation occurred before the global extraction barrier')
            result.update(completed=True,iterations=status['iterations'],elapsed_seconds=time.time()-started)
        except Exception as error:
            result.update(completed=False,error=str(error),elapsed_seconds=time.time()-started)
        print('Train/extract result:',result,flush=True)
        return result
    results=[]
    with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
        futures=[pool.submit(worker,job) for job in plan['jobs']]
        for future in as_completed(futures):
            results.append(future.result())
            save_json(root/'fit_jobs.json',sorted(results,key=lambda r:(r['sigma'],r['seed'])))
    if not all(r['completed'] for r in results):
        raise RuntimeError('Some prescribed runs failed; see fit_jobs.json. No Z evaluation or favorable subset selection.')
    freeze=[]
    for job in plan['jobs']:
        directory=root/job['directory']/'unsupervised_factors'
        freeze.append(dict(directory=job['directory'],manifest_sha256=digest(directory/'extraction_manifest.json'),
                           status_sha256=digest(directory/'extraction_status.json')))
    save_json(root/'global_extraction_freeze.json',dict(
        completed=True,all_run_count=len(results),z_evaluation_started=False,extractors_frozen=freeze))
    print('ALL 20 runs extracted WITHOUT Z evaluation; global barrier complete.',flush=True)


def check_global_barrier(spec,root):
    """No Z values are read by this precondition check."""
    plan=validate_plan(spec,root)
    jobs=json.loads((root/'fit_jobs.json').read_text())
    expected={(j['sigma'],j['seed']) for j in plan['jobs']}
    if len(jobs)!=len(expected) or {(j['sigma'],j['seed']) for j in jobs}!=expected or not all(j['completed'] for j in jobs):
        raise RuntimeError('All 20 predeclared runs must finish; no partial/favorable subset evaluation')
    freeze=json.loads((root/'global_extraction_freeze.json').read_text())
    if not freeze['completed'] or freeze['all_run_count']!=len(expected):
        raise RuntimeError('Global extraction freeze is incomplete')
    directories={j['directory'] for j in plan['jobs']}
    if (len(freeze['extractors_frozen'])!=len(expected) or
            {r['directory'] for r in freeze['extractors_frozen']}!=directories):
        raise RuntimeError('Global freeze must contain exactly all prescribed runs')
    for record in freeze['extractors_frozen']:
        directory=root/record['directory']/'unsupervised_factors'
        if digest(directory/'extraction_manifest.json')!=record['manifest_sha256']:
            raise ValueError('Frozen extraction manifest changed')
        if digest(directory/'extraction_status.json')!=record['status_sha256']:
            raise ValueError('Extraction status changed')
        for filename,sha in json.loads((directory/'extraction_manifest.json').read_text()).items():
            if digest(directory/filename)!=sha:
                raise ValueError('Frozen factor/model changed: '+str(directory/filename))
    return plan,jobs


def verify_pairing(spec,root):
    """Generation identity audit AFTER all fits; Z is now evaluation-only."""
    rows=[]
    from gran_dag.hidden import load_checkpoint_cpu
    from .additional import parameter_digest
    for seed in spec['seeds']:
        base=None
        for sigma in spec['sigmas']:
            directory=root/'runs'/sigma_name(sigma)/('seed_'+str(seed))
            x=np.load(str(directory/'data/observed_X.npy'))
            z=np.load(str(directory/'data/true_Z.npy'))
            eps=np.load(str(directory/'data/noise.npy'))/sigma
            labels=np.load(str(directory/'data/sample_split.npy'))
            ids=np.load(str(directory/'data/sample_indices.npy'))
            initial_hash=parameter_digest(load_checkpoint_cpu(str(directory/'random_full/model.pkl')))
            np.testing.assert_allclose(x,z@np.array([[2.,3.,1.]])+sigma*eps,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(ids,np.arange(1000))
            if [int((labels==code).sum()) for code in (0,1,2)]!=[600,200,200]:
                raise ValueError('Unexpected train/validation/test size')
            if base is None:
                base=(z,eps,labels,ids,initial_hash)
            else:
                for actual,expected in zip((z,eps,labels,ids),base[:4]):
                    np.testing.assert_array_equal(actual,expected)
                if initial_hash!=base[4]:
                    raise ValueError('NN initialization differs across sigma for a matched seed')
            pool=np.load(str(directory/'training_input/sample_indices.npy'))
            if np.any(labels[pool]==2):
                raise ValueError('Final test entered GraN-DAG training/pruning')
            np.testing.assert_array_equal(np.load(str(directory/'training_input/data1.npy')),x[pool])
            rows.append(dict(sigma=sigma,seed=seed,equation_verified=True,matched_latent=True,
                             matched_base_noise=True,matched_split=True,matched_initial_weights=True,
                             final_test_excluded=True))
    write_csv(root/'pairing_verification.csv',rows)


def evaluate(spec,root):
    root=Path(root).resolve()
    plan,_=check_global_barrier(spec,root)
    if (root/'evaluation_jobs.json').exists():
        raise FileExistsError('Evaluation already attempted; old outputs will not be overwritten')
    for job in plan['jobs']:
        if (root/job['directory']/'unsupervised_factors/metrics.csv').exists():
            raise FileExistsError('Existing per-run evaluation must not be overwritten')
    # Values of Z are opened only after the all-run extraction freeze above.
    verify_pairing(spec,root)
    protected=json.loads((root/'protected_inventory.json').read_text())
    for name,expected in protected.items():
        if file_inventory(root.parent/name)!=expected:
            raise ValueError('Old experiment inventory changed: '+name)
    save_json(root/'protected_content_after_fitting.json',{
        name:snapshot(root.parent/name) for name in protected})
    results=[]
    # Four evaluation processes suffice for plotting; no fitting occurs here.
    def worker(job):
        directory=root/job['directory']
        result=dict(sigma=job['sigma'],seed=job['seed'])
        try:
            logpath=root/'logs'/('evaluate_{}_seed_{}.log'.format(sigma_name(job['sigma']),job['seed']))
            with logpath.open('x') as log:
                subprocess.check_call([sys.executable,'latent_unsupervised.py','--stage','evaluate',
                    '--config',str(directory/'factor_settings.json')],stdout=log,stderr=subprocess.STDOUT)
            result['completed']=True
        except Exception as error:
            result.update(completed=False,error=str(error))
        return result
    with ThreadPoolExecutor(max_workers=min(4,spec['workers'])) as pool:
        futures=[pool.submit(worker,job) for job in plan['jobs']]
        for future in as_completed(futures):
            result=future.result(); results.append(result)
            save_json(root/'evaluation_jobs.json',sorted(results,key=lambda r:(r['sigma'],r['seed'])))
            print('Evaluation-only Z result:',result,flush=True)
    if not all(j['completed'] for j in results):
        raise RuntimeError('Some final evaluations failed; see evaluation_jobs.json')


def summarize_rows(rows,keys,scores):
    """All five seeds retained. Undefined metrics make the full-cohort mean undefined."""
    grouped=defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    output=[]
    for key,values in sorted(grouped.items()):
        if len(values)!=5 or len({int(r['seed']) for r in values})!=5:
            raise ValueError('Exactly the five paired seeds are required in every group')
        record=dict(zip(keys,key)); record['n_seeds']=5
        for score in scores:
            a=[float(r[score]) for r in values if r.get(score) not in (None,'')]
            if not np.isfinite(a).all():
                raise ValueError('Nonfinite numeric score; use an explicit undefined value instead')
            record[score+'_n_defined']=len(a)
            record[score+'_mean']=float(np.mean(a)) if len(a)==5 else None
            record[score+'_std']=float(np.std(a,ddof=1)) if len(a)==5 else None
        output.append(record)
    return output


def paired_rows(metrics):
    """Predefined comparisons only, never choosing a condition by correlation."""
    pairs=[('X',code,'hidden_or_random_vs_X') for code in ('C_L1','C_L2','E_L1','E_L2','B_L1','B_L2','D_L1','D_L2')]
    pairs += [('B_L1','C_L1','learning_train_mask'),('B_L2','C_L2','learning_train_mask'),
              ('D_L1','E_L1','learning_dag_mask'),('D_L2','E_L2','learning_dag_mask'),
              ('C_L1','E_L1','dag_learned_weights'),('C_L2','E_L2','dag_learned_weights'),
              ('B_L1','D_L1','dag_random_weights'),('B_L2','D_L2','dag_random_weights'),
              ('C_L1','C_L2','layer'),('E_L1','E_L2','layer'),
              ('B_L1','B_L2','layer'),('D_L1','D_L2','layer')]
    lookup={(float(r['sigma']),int(r['seed']),r['condition'],r['method']):r
            for r in metrics if r['split']=='test'}
    rows=[]
    for sigma,seed,condition,method in sorted(lookup):
        if condition!='X':
            continue
        for before,after,purpose in pairs:
            a,b=(lookup[(sigma,seed,code,method)] for code in (before,after))
            record=dict(sigma=sigma,seed=seed,comparison=purpose,before=before,after=after,method=method)
            for score in ('absolute_pearson_r','x_reconstruction_raw_mse'):
                record['delta_'+score]=(float(b[score])-float(a[score])
                    if a[score] not in (None,'') and b[score] not in (None,'') else None)
            rows.append(record)
    return rows


def noise_paired_rows(metrics,sigmas):
    from itertools import combinations
    lookup={(float(r['sigma']),int(r['seed']),r['condition'],r['method']):r
            for r in metrics if r['split']=='test'}
    rows=[]
    for seed,condition,method in sorted({(k[1],k[2],k[3]) for k in lookup}):
        for before,after in combinations(sigmas,2):
            a,b=(lookup[(float(s),seed,condition,method)] for s in (before,after))
            record=dict(before_sigma=before,after_sigma=after,seed=seed,condition=condition,method=method)
            for score in ('absolute_pearson_r','x_reconstruction_raw_mse'):
                record['delta_'+score]=(float(b[score])-float(a[score])
                    if a[score] not in (None,'') and b[score] not in (None,'') else None)
            rows.append(record)
    return rows


def verify_run_metrics(directory):
    """Recompute factor transforms, correlations and reconstruction without fitting."""
    import pickle
    from scipy.stats import spearmanr
    output=directory/'unsupervised_factors'
    metadata=json.loads((output/'metadata.json').read_text())
    status=json.loads((directory/'training_status.json').read_text())
    if not status['converged'] or status['latent_labels_used_by_grandag'] or status['final_test_used_by_grandag']:
        raise ValueError('Original NN training failed convergence/input safeguards')
    if metadata['z_used_in_extraction'] or metadata['test_used_for_selection']:
        raise ValueError('Label-free extraction safeguards failed')
    for name,sha in json.loads((output/'extraction_manifest.json').read_text()).items():
        if digest(output/name)!=sha:
            raise ValueError('Saved extractor/factor changed: '+name)
    x=np.load(str(directory/'data/observed_X.npy'))
    z=np.load(str(directory/'data/true_Z.npy'))[:,0]
    labels=np.load(str(directory/'data/sample_split.npy'))
    conditions={c['id']:c for c in metadata['feature_conditions']}
    metrics=read_csv(output/'metrics.csv')
    for model in metadata['extractors']:
        values=np.load(str(directory/conditions[model['condition']]['source'])).reshape(len(x),-1)
        with (output/model['model_file']).open('rb') as stream:
            bundle=pickle.load(stream)
        factors,_,xhat=transform_extractor(bundle,values)
        np.testing.assert_allclose(factors,np.load(str(output/model['factor_file'])),rtol=1e-12,atol=1e-12)
        for split,code in [('train',0),('validation',1),('test',2)]:
            row=next(r for r in metrics if r['condition']==model['condition'] and r['method']==model['method'] and r['split']==split)
            selected=labels==code
            np.testing.assert_allclose(float(row['x_reconstruction_raw_mse']),np.mean((xhat[selected]-x[selected])**2),rtol=1e-12,atol=1e-12)
            if row['correlation_defined']=='True':
                np.testing.assert_allclose(float(row['pearson_r']),np.corrcoef(factors[selected,0],z[selected])[0,1],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(float(row['spearman_r']),spearmanr(factors[selected,0],z[selected])[0],rtol=1e-12,atol=1e-12)
    return metrics,metadata,status


def summarize(spec,root):
    root=Path(root).resolve()
    plan,fit_jobs=check_global_barrier(spec,root)
    evaluated=json.loads((root/'evaluation_jobs.json').read_text())
    if len(evaluated)!=20 or not all(r['completed'] for r in evaluated):
        raise RuntimeError('All 20 final evaluations must complete; no subset summary')
    if (root/'metrics_by_seed.csv').exists():
        raise FileExistsError('Summary already exists; it will not be overwritten')
    metrics,origins,graphs,warnings_found=[],[],[],[]
    from .nonlinear_study import graph_rows
    fit_lookup={(r['sigma'],r['seed']):r for r in fit_jobs}
    for job in plan['jobs']:
        directory=root/job['directory']
        values,metadata,status=verify_run_metrics(directory)
        metrics += [dict(r,sigma=job['sigma'],seed=job['seed']) for r in values]
        fitted=fit_lookup[(job['sigma'],job['seed'])]
        origins.append(dict(sigma=job['sigma'],seed=job['seed'],model_origin=fitted['model_origin'],
            iterations=status['iterations'],final_h=status['final_h'],elapsed_seconds=fitted['elapsed_seconds'],
            matched_masks_verified=True,extractor_count=len(metadata['extractors']),
            undefined_test_correlations=sum(r['correlation_defined']!='True' for r in values if r['split']=='test')))
        edges,_=graph_rows(directory,job['seed'])
        graphs += [dict(r,sigma=job['sigma']) for r in edges]
        for model in metadata['extractors']:
            for warning in model['warnings']:
                warnings_found.append(dict(sigma=job['sigma'],seed=job['seed'],condition=model['condition'],
                                            method=model['method'],warning=warning))
    summary=summarize_rows(metrics,('sigma',)+KEYS,SCORES)
    within=paired_rows(metrics)
    paired=summarize_rows(within,('sigma','comparison','before','after','method'),
                          ('delta_absolute_pearson_r','delta_x_reconstruction_raw_mse'))
    noise_pairs=noise_paired_rows(metrics,spec['sigmas'])
    noise_summary=summarize_rows(noise_pairs,('before_sigma','after_sigma','condition','method'),
                                 ('delta_absolute_pearson_r','delta_x_reconstruction_raw_mse'))
    write_csv(root/'metrics_by_seed.csv',metrics)
    write_csv(root/'metrics_mean_std.csv',summary)
    write_csv(root/'paired_by_seed.csv',within)
    write_csv(root/'paired_mean_std.csv',paired)
    write_csv(root/'noise_paired_by_seed.csv',noise_pairs)
    write_csv(root/'noise_paired_mean_std.csv',noise_summary)
    write_csv(root/'model_origins.csv',origins)
    write_csv(root/'graphs_by_seed.csv',graphs)
    write_csv(root/'extractor_warnings.csv',warnings_found,['sigma','seed','condition','method','warning'])
    make_figures(spec,root,summary,paired)
    make_report(spec,root,summary,paired,origins,warnings_found)
    protected=json.loads((root/'protected_content_after_fitting.json').read_text())
    for name,expected in protected.items():
        if snapshot(root.parent/name)!=expected:
            raise ValueError('Old experiment content changed: '+name)
    inventory=json.loads((root/'protected_inventory.json').read_text())
    if any(file_inventory(root.parent/name)!=expected for name,expected in inventory.items()):
        raise ValueError('Original experiment inventory changed')
    save_json(root/'status.json',dict(completed=True,run_count=20,new_models=sum(r['model_origin']=='newly_trained' for r in origins),
        reused_models=sum(r['model_origin']=='reused' for r in origins),all_seeds_reported=spec['seeds'],
        global_extraction_before_z_evaluation=True,source_runs_unchanged=list(protected),
        all_metric_rows_verified=len(metrics),mean_std_groups=len(summary),paired_groups=len(paired),
        sample_std_ddof=1,z_used_for_selection=False,n_factors_fixed=1,
        undefined_metrics_policy='full-five-seed mean and SD undefined unless all five values defined',
        representative_seed_predeclared=spec['representative_seed'],
        source_sha256={str(p):digest(p) for p in [Path(__file__),Path('latent_noise_study.py'),Path('configs/latent_noise_study.json')]}))
    print('Noise robustness study complete:',root,flush=True)


def make_figures(spec,root,summary,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=root/'figures'; figures.mkdir()
    lookup={(float(r['sigma']),r['condition'],r['method']):r for r in summary if r['split']=='test'}
    sigmas=spec['sigmas']
    methods=('pca','factor_analysis')
    def number(value):
        return float(value) if value is not None else np.nan
    for metric,filename,title in [
            ('absolute_pearson_r','noise_correlations','Final-test |Pearson r|'),
            ('x_reconstruction_raw_mse','noise_X_reconstruction','Final-test raw X reconstruction MSE')]:
        fig,axes=plt.subplots(2,2,figsize=(14,9))
        for i,method in enumerate(methods):
            for j,layer in enumerate(('1','2')):
                ax=axes[i,j]
                for code,label in [('X','raw X'),('C','learned/train mask'),('B','random/train mask'),
                                   ('E','learned/DAG mask'),('D','random/DAG mask')]:
                    condition='X' if code=='X' else code+'_L'+layer
                    rows=[lookup[(s,condition,method)] for s in sigmas]
                    ax.errorbar(sigmas,[number(r[metric+'_mean']) for r in rows],
                        yerr=[number(r[metric+'_std']) for r in rows],marker='o',capsize=3,label=label)
                ax.set_xticks(sigmas); ax.set_xlabel('Noise sigma')
                ax.set_ylabel(title); ax.set_title(method+' L'+layer+'; mean +/- sample SD, n=5')
                ax.legend(fontsize=8)
                if metric=='absolute_pearson_r':ax.set_ylim(0,1.04)
        fig.tight_layout(); fig.savefig(str(figures/(filename+'.png')),dpi=150); plt.close(fig)
    comparisons=[('learning',[('B_L1','C_L1'),('B_L2','C_L2'),('D_L1','E_L1'),('D_L2','E_L2')]),
                 ('DAG',[('C_L1','E_L1'),('C_L2','E_L2'),('B_L1','D_L1'),('B_L2','D_L2')]),
                 ('X',[('X','C_L1'),('X','C_L2'),('X','E_L1'),('X','E_L2')])]
    pair_lookup={(float(r['sigma']),r['before'],r['after'],r['method']):r for r in paired}
    for name,pairs in comparisons:
        fig,axes=plt.subplots(1,2,figsize=(14,4.5))
        for ax,method in zip(axes,methods):
            for before,after in pairs:
                rows=[pair_lookup[(s,before,after,method)] for s in sigmas]
                ax.errorbar(sigmas,[number(r['delta_absolute_pearson_r_mean']) for r in rows],
                    yerr=[number(r['delta_absolute_pearson_r_std']) for r in rows],
                    marker='o',capsize=3,label=after+' minus '+before)
            ax.axhline(0,color='black',lw=.7); ax.set_xticks(sigmas); ax.set_xlabel('Noise sigma')
            ax.set_ylabel('Paired delta |Pearson r| (mean +/- sample SD)')
            ax.set_title(method+' '+name+' comparisons'); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(str(figures/('paired_'+name+'_effects.png')),dpi=150); plt.close(fig)
    conditions=['X','C_L1','C_L2','E_L1','E_L2','B_L1','B_L2','D_L1','D_L2']
    fig,axes=plt.subplots(1,2,figsize=(13,10))
    for ax,method in zip(axes,methods):
        values=np.array([[number(lookup[(s,c,method)]['absolute_pearson_r_mean']) for s in sigmas] for c in conditions])
        ax.imshow(values,vmin=0,vmax=1,aspect='auto',cmap='viridis')
        for i,c in enumerate(conditions):
            for j,s in enumerate(sigmas):
                r=lookup[(s,c,method)]
                text=('{:.3f}\n+/- {:.3f}'.format(r['absolute_pearson_r_mean'],r['absolute_pearson_r_std'])
                      if r['absolute_pearson_r_mean'] is not None else 'undefined\nn='+str(r['absolute_pearson_r_n_defined']))
                ax.text(j,i,text,ha='center',va='center',fontsize=8,
                        bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        ax.set_xticks(range(4)); ax.set_xticklabels(sigmas); ax.set_xlabel('Noise sigma')
        ax.set_yticks(range(9)); ax.set_yticklabels(conditions); ax.set_title(method+' |r| mean +/- SD')
    fig.tight_layout(); fig.savefig(str(figures/'correlation_summary.png'),dpi=150); plt.close(fig)
    representative=figures/'representative'; representative.mkdir()
    for sigma in sigmas:
        source=root/'runs'/sigma_name(sigma)/('seed_'+str(spec['representative_seed']))/'unsupervised_factors/figures'
        for method in methods:
            shutil.copy2(str(source/('scatter_'+method+'_Z1.png')),
                         str(representative/('{}_seed_{}_{}.png'.format(sigma_name(sigma),spec['representative_seed'],method))))


def make_report(spec,root,summary,paired,origins,warnings_found):
    lookup={(float(r['sigma']),r['condition'],r['method']):r for r in summary if r['split']=='test'}
    pair_lookup={(float(r['sigma']),r['before'],r['after'],r['method']):r for r in paired}
    def fmt(value):
        return '未定義' if value is None else '{:.4f}'.format(float(value))
    def cell(row,score):
        return (fmt(row[score+'_mean'])+' ± '+fmt(row[score+'_std'])
                if row[score+'_mean'] is not None else '未定義 (定義数{}/5)'.format(row[score+'_n_defined']))
    names={'X':'元のX','C_L1':'学習後 第1層','C_L2':'学習後 第2層','E_L1':'DAG後 第1層',
           'E_L2':'DAG後 第2層','B_L1':'未学習・学習後マスク 第1層','B_L2':'未学習・学習後マスク 第2層',
           'D_L1':'未学習・DAGマスク 第1層','D_L2':'未学習・DAGマスク 第2層'}
    lines=['# ノイズ強度別・教師なし潜在因子抽出実験', '',
           '## 生成式と固定条件', '',
           'Z~N(0,1)、X1=2Z+σε1、X2=3Z+σε2、X3=Z+σε3。εiは相互独立でZからも独立な標準正規。',
           'σ=0.5,1,2,4、各1000件。学習600・検証200・最終テスト200。'
           '5seedは42,201,202,203,204。各seedのZ・基礎ノイズε・分割・NN初期化をσ間で一致させた。', '',
           'GraN-DAGはNonLinGaussANM、隠れ層2×10、leaky-ReLU、学習行で標準化、'
           '元のRMSprop(lr=0.01)、上限100000反復、h閾値1e-8。損失・DAG制約・最適化は変更していない。',
           '既存学習workerを再利用するためCAMも保存したが、主要比較は学習直後とDAG化後。CAMのR seed42は従来どおり固定。', '',
           'PCAとFAはいずれも1因子を事前固定。PCAはfull SVD、FAはlapack/max_iter5000/tol1e-4。'
           '標準化・定数列除外・fit・因子→X再構成係数は学習行のみ。検証・テストはtransformのみ。', '',
           '全20runの学習・全360抽出器の保存後にグローバル凍結を確認し、初めてZによる評価を実行した。'
           '教師ありZ回帰は実行していない。Zで因子・層・手法・条件を選んでいない。', '',
           '## 新規学習／再利用', '',
           'σ=1・seed42だけ、X・分割・生成／学習設定の完全一致を確認して既存モデルを新しい出力先へコピーした。'
           '残り19件は新規学習。既存結果を直接変更していない。', '',
           '|σ|seed|モデル|反復数|テスト相関未定義数|', '|---|---|---|---|---|']
    for r in origins:
        lines.append('|{}|{}|{}|{}|{}|'.format(r['sigma'],r['seed'],r['model_origin'],r['iterations'],r['undefined_test_correlations']))
    for score,title in [('absolute_pearson_r','最終テストの絶対Pearson相関'),
                        ('x_reconstruction_raw_mse','最終テストのX再構成MSE')]:
        lines += ['', '## '+title, '', 'セルは5seedの平均±標本標準偏差(ddof=1)。', '',
                  '|特徴|手法|σ=0.5|σ=1|σ=2|σ=4|','|---|---|---|---|---|---|']
        for condition,name in names.items():
            for method in ('pca','factor_analysis'):
                cells=[cell(lookup[(s,condition,method)],score) for s in spec['sigmas']]
                lines.append('|{}|{}|{}|'.format(name,method,'|'.join(cells)))
    lines += ['', '## ペア比較：同じseed内の差分', '',
              '正の相関差はafterでの改善。X再構成MSE差は負が改善。SDは差分をseed内で先に計算してから求めた。', '',
              '|比較|手法|σ=0.5 Δ絶対r|σ=1 Δ絶対r|σ=2 Δ絶対r|σ=4 Δ絶対r|',
              '|---|---|---|---|---|---|']
    for before,after in [('B_L1','C_L1'),('B_L2','C_L2'),('D_L1','E_L1'),('D_L2','E_L2'),
                         ('C_L1','E_L1'),('C_L2','E_L2'),('X','C_L1'),('X','C_L2'),('X','E_L1'),('X','E_L2')]:
        for method in ('pca','factor_analysis'):
            cells=[cell(pair_lookup[(s,before,after,method)],'delta_absolute_pearson_r') for s in spec['sigmas']]
            lines.append('|{}→{}|{}|{}|'.format(before,after,method,'|'.join(cells)))
    lines += ['', '## 結果の読み方・限界', '',
              '- 元Xとの比較はX→C/E、純粋なNN学習の比較は同じマスクのB→CとD→E。初期全結合マスクとの不公平な比較はしない。',
              '- グラフ・根ノードはノイズとseedで異なる。親を失ったMLPは定数となり既存のtrain-only規則で除外される。graphs_by_seed.csvも参照。',
              '- 1因子固定の頑健性実験であり、潜在因子の存在／個数を自動検出したものではない。',
              '- 符号をZに合わせていないため、signed Pearson/Spearmanのseed平均は符号の相殺があり得る。主要比較は絶対Pearson相関。',
              '- 定数因子の相関は未定義で、0には置換しない。5件全て定義されない場合、5seed平均・SDを未定義とし定義数を表示する。',
              '- X再構成は学習因子から元XへのOLS＋切片。Zは使わない。H再構成誤差とは別で、Xのノイズ自体も再構成対象に含む。',
              '- ノイズが大きいほど観測Xの生の分散も変わるため、σ間の生MSEの増加だけで因子品質を判断しない。標準化X誤差もCSVに保存。',
              '- 平均±SDは記述統計であり、信頼区間や統計的な優位性の検定ではない。全条件を併記し最高相関で手法や層を選ばない。',
              '- seed42のσ=1は既に閲覧済みのテストを再利用している。事前固定条件での探索的比較として報告する。',
              '- 統計的共通因子と因果的な潜在共通原因の同定は異なる。観測DAGは未観測共通原因Zを表現できない。',
              '- FAの独立残差仮定はNN隠れ層の重複ユニットでは満たされない可能性があり、表現・尺度・抽出モデルへの依存がある。', '',
              '## 次の検証（今回未実施）', '',
              'tanh等の非線形生成でノイズを同様に対応付ける実験、2個以上の潜在因子と固定因子数での部分空間比較、'
              '無潜在因子の陰性対照、Zを使わない因子数選択、別cohortでの確認実験を検討する。', '',
              '## 保存ファイル', '',
              '- metrics_by_seed.csv：全20runの各split・特徴・手法のPearson/Spearman/再構成/分散。',
              '- metrics_mean_std.csv：σごとの5seed平均・SD・定義数。',
              '- paired_by_seed.csv／paired_mean_std.csv：同一seedでの学習・X比較・DAG・層の差分。',
              '- noise_paired_by_seed.csv／noise_paired_mean_std.csv：同一seedのσ間差分（全組合せ）。',
              '- model_origins.csv：コピー再利用／新規学習・反復数・未定義数。graphs_by_seed.csv：推定グラフ。',
              '- pairing_verification.csv：Z・基礎ノイズ・分割・初期重みの対応確認。',
              '- figures/：平均±SD折れ線・ペア比較・一覧図。代表scatterは事前固定seed201の全σ・両手法。',
              '- runs/sigma_*/seed_*/：元データ、NN、隠れ層、抽出器、因子、評価を従来形式で保存。', '',
              '## エラー・警告', '',
              '全20条件のfit/evaluate成否はfit_jobs.jsonとevaluation_jobs.jsonに保存。失敗条件の差し替えや除外は行わない。',
              '抽出器警告は{}件。詳細はextractor_warnings.csv。'.format(len(warnings_found)), '']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
