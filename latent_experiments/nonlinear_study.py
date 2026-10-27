"""Seed orchestration and descriptive summaries; original training is untouched."""
import copy
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .additional import digest, read_csv, snapshot
from .config import load_config
from .data import topological_order
from .evaluate import write_csv


KEYS = ('condition', 'decoder', 'layer', 'scope', 'latent')
SCORES = ('test_r2', 'test_mse', 'train_r2', 'validation_r2')


def resolved_seed_config(spec, seed):
    config = load_config(spec['base_config'])
    config['name'] = 'latent_nonlinear_seed_'+str(seed)
    config['output_dir'] = str(Path(spec['output_dir'])/'runs'/('seed_'+str(seed)))
    for section in ('data', 'split', 'training'):
        config[section]['seed'] = seed
    config['probe']['seed'] = seed+1
    return config


def prepare(spec, root, pilot):
    root, pilot = Path(root).resolve(), Path(pilot).resolve()
    if root.exists():
        raise FileExistsError('Study directory already exists: '+str(root))
    seeds = spec['seeds']
    if len(seeds) != 5 or len(set(seeds)) != len(seeds) or 42 in seeds:
        raise ValueError('Use exactly five distinct fresh seeds, excluding pilot 42')
    base = load_config(spec['base_config'])
    if json.loads((pilot/'config.json').read_text()) != base:
        raise ValueError('Pilot does not match the exact base nonlinear config')
    root.mkdir(parents=True)
    (root/'configs').mkdir()
    (root/'logs').mkdir()
    (root/'runs').mkdir()
    plan = dict(spec=spec, resolved_base_config=base, seed_configs={},
                pilot_in_summary=False, std_ddof=1,
                candidate_grid_frozen_before_new_seed_results=True,
                source_sha256={name: digest(name) for name in [spec['base_config'],
                    'configs/latent_linear.json', 'gran_dag/train.py', 'gran_dag/dag_optim.py',
                    'latent_experiments/train.py', 'latent_experiments/evaluate.py',
                    'latent_experiments/additional.py']})
    for seed in seeds:
        config = resolved_seed_config(spec, seed)
        path = root/'configs'/('seed_'+str(seed)+'.json')
        path.write_text(json.dumps(config, indent=2))
        plan['seed_configs'][str(seed)] = dict(path=str(path.relative_to(root)), sha256=digest(path))
    # Protect every previously saved linear/p10 experiment, not just checkpoints.
    protected = {}
    for name in ('latent_linear', 'p10-run1'):
        directory = root.parent/name
        if directory.exists():
            protected[name] = snapshot(directory)
    (root/'protected_runs.json').write_text(json.dumps(protected, indent=2))
    (root/'plan.json').write_text(json.dumps(plan, indent=2))
    print('Prepared five frozen seed configurations:', seeds, root, flush=True)


def validate_plan(spec, root):
    plan = json.loads((root/'plan.json').read_text())
    if plan['spec'] != spec:
        raise ValueError('Study spec changed after preparation')
    for name, expected in plan['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Frozen source/config changed: '+name)
    for row in plan['seed_configs'].values():
        if digest(root/row['path']) != row['sha256']:
            raise ValueError('Seed config changed after preparation')
    return plan


def run_seeds(spec, root, pilot):
    root, pilot = Path(root).resolve(), Path(pilot).resolve()
    plan = validate_plan(spec, root)
    pilot_status = json.loads((pilot/'training_status.json').read_text())
    pilot_analysis = json.loads((pilot/'additional_analysis/status.json').read_text())
    if not pilot_status['converged'] or not pilot_analysis['completed']:
        raise RuntimeError('Complete the pilot training and hidden nonlinear evaluation first')
    if (root/'jobs.json').exists():
        raise FileExistsError('Seed execution already attempted; existing results will not be overwritten')
    for seed in spec['seeds']:
        if (root/'runs'/('seed_'+str(seed))).exists() or (root/'logs'/('seed_'+str(seed)+'.log')).exists():
            raise FileExistsError('Seed output already exists: '+str(seed))

    def worker(seed):
        started = time.time()
        directory = root/'runs'/('seed_'+str(seed))
        print('Starting fresh seed', seed, flush=True)
        config_path = root/plan['seed_configs'][str(seed)]['path']
        commands = [
            [sys.executable, 'latent_experiment.py', '--config', str(config_path)],
            [sys.executable, 'latent_additional_analysis.py', '--run-dir', str(directory)],
        ]
        try:
            with (root/'logs'/('seed_'+str(seed)+'.log')).open('x') as log:
                for command in commands:
                    subprocess.check_call(command, stdout=log, stderr=subprocess.STDOUT)
            result = dict(seed=seed, completed=True, elapsed_seconds=time.time()-started)
        except Exception as error:
            result = dict(seed=seed, completed=False, error=str(error), elapsed_seconds=time.time()-started)
        print('Fresh seed result:', result, flush=True)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
        futures = [pool.submit(worker, seed) for seed in spec['seeds']]
        for future in as_completed(futures):
            results.append(future.result())
            (root/'jobs.json').write_text(json.dumps(sorted(results, key=lambda r: r['seed']), indent=2))
    if not all(row['completed'] for row in results):
        raise RuntimeError('Some prescribed seeds failed; see jobs.json. No seed replacement/selection is performed.')


def summarize(rows, keys, scores):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        if len({row['seed'] for row in values}) != len(values):
            raise ValueError('Duplicate seed within a summary group')
        record = dict(zip(keys, key))
        record['n_seeds'] = len(values)
        for score in scores:
            a = np.array([float(row[score]) for row in values])
            if not np.isfinite(a).all():
                raise ValueError('Nonfinite metric in summary')
            record[score+'_mean'] = float(a.mean())
            record[score+'_std'] = float(a.std(ddof=1)) if len(a) > 1 else None
        result.append(record)
    return result


def verify_completed_run(directory):
    status = json.loads((directory/'training_status.json').read_text())
    extra = json.loads((directory/'additional_analysis/status.json').read_text())
    if not status['converged'] or status['latent_labels_used_by_grandag'] or status['final_test_used_by_grandag']:
        raise ValueError('Training convergence/leakage check failed: '+str(directory))
    if not extra['completed'] or extra['final_test_used_for_selection'] or extra['nn_retrained']:
        raise ValueError('Probe/leakage status failed: '+str(directory))
    for filename, expected in extra['source_sha256'].items():
        if digest(directory/filename) != expected:
            raise ValueError('Run source changed after additional analysis: '+filename)
    x = np.load(str(directory/'data/observed_X.npy'))
    z = np.load(str(directory/'data/true_Z.npy'))
    noise = np.load(str(directory/'data/noise.npy'))
    np.testing.assert_allclose(x, np.tanh(z) @ np.array([[2., 3., 1.]])+noise)
    labels = np.load(str(directory/'data/sample_split.npy'))
    ids = np.load(str(directory/'data/sample_indices.npy'))
    np.testing.assert_array_equal(ids, np.arange(len(x)))
    pool = np.load(str(directory/'training_input/sample_indices.npy'))
    np.testing.assert_array_equal(np.load(str(directory/'training_input/data1.npy')), x[pool])
    if np.any(labels[pool] == 2) or x.shape[1] != 3 or z.shape != (len(x), 1):
        raise ValueError('Final test or latent columns entered GraN-DAG input')
    metrics = read_csv(directory/'additional_analysis/metrics.csv')
    predictions = read_csv(directory/'additional_analysis/test_predictions.csv')
    by_key = defaultdict(list)
    for row in predictions:
        sample = int(row['sample_index'])
        if labels[sample] != 2 or float(row['true_z']) != z[sample, int(row['latent'][1:])-1]:
            raise ValueError('Test sample/label mismatch')
        by_key[tuple(row[key] for key in KEYS)].append(row)
    expected_ids = sorted(ids[labels == 2].tolist())
    for row in metrics:
        values = by_key[tuple(row[key] for key in KEYS)]
        if sorted(int(r['sample_index']) for r in values) != expected_ids:
            raise ValueError('Duplicated or missing final-test samples')
        truth = np.array([float(r['true_z']) for r in values])
        prediction = np.array([float(r['prediction']) for r in values])
        mse = np.mean((truth-prediction)**2)
        r2 = 1-np.sum((truth-prediction)**2)/np.sum((truth-truth.mean())**2)
        np.testing.assert_allclose([mse, r2], [float(row['test_mse']), float(row['test_r2'])], atol=1e-12)
    return status, metrics, extra['controls']


def graph_rows(directory, seed):
    config = json.loads((directory/'config.json').read_text())
    paths = [('train', 'train'), ('dag', 'to-dag'),
             ('cam', 'cam-pruning/cutoff_%.6f' % config['training']['cam_pruning_cutoff'])]
    rows, masks = [], {}
    for stage, path in paths:
        mask = np.load(str(directory/path/'hidden/adjacency.npy')) != 0
        masks[stage] = mask
        try:
            topological_order(mask)
            acyclic = True
        except ValueError:
            acyclic = False
        edges = ['X{}->X{}'.format(a+1, b+1) for a, b in np.argwhere(mask)]
        rows.append(dict(seed=seed, stage=stage, edges='; '.join(edges) or '(empty)',
                         edge_count=int(mask.sum()), is_acyclic=acyclic,
                         parent_counts=json.dumps(mask.sum(0).tolist())))
    return rows, masks


def aggregate(spec, root, pilot, summary_name='summary'):
    root, pilot = Path(root).resolve(), Path(pilot).resolve()
    validate_plan(spec, root)
    output = (root/summary_name).resolve()
    if output.parent != root or output.exists():
        raise FileExistsError('Choose a new summary directory directly below the study root')
    jobs = json.loads((root/'jobs.json').read_text())
    if sorted(r['seed'] for r in jobs) != sorted(spec['seeds']) or not all(r['completed'] for r in jobs):
        raise RuntimeError('All five predeclared seeds must complete; no favorable subset aggregation')
    output.mkdir()
    (output/'figures').mkdir()
    all_metrics, all_pairs, graphs, statuses = [], [], [], []
    masks_by_seed = {}
    for seed in spec['seeds']:
        directory = root/'runs'/('seed_'+str(seed))
        status, metrics, controls = verify_completed_run(directory)
        all_metrics.extend([dict(row, seed=seed) for row in metrics])
        all_pairs.extend([dict(row, seed=seed) for row in read_csv(directory/'additional_analysis/paired_comparisons.csv')])
        edges, masks = graph_rows(directory, seed)
        graphs.extend(edges)
        masks_by_seed[seed] = masks
        statuses.append(dict(seed=seed, converged=status['converged'], iterations=status['iterations'],
                             worker_seconds=status['original_worker_elapsed_seconds'],
                             dag_equals_cam=controls['dag_equals_cam_mask'],
                             same_learned_parameters=controls['learned_stage_parameters_equal_train'],
                             no_test_or_z_used_by_grandag=True, predictions_verified=True))
    summary = summarize(all_metrics, KEYS, SCORES)
    pair_keys = ('comparison', 'before', 'after', 'decoder', 'layer', 'scope', 'latent')
    paired = summarize(all_pairs, pair_keys, ('delta_r2', 'delta_mse'))
    if any(row['n_seeds'] != 5 for row in summary+paired):
        raise ValueError('A summary group is missing a prescribed seed')
    write_csv(output/'metrics_by_seed.csv', all_metrics)
    write_csv(output/'metrics_mean_std.csv', summary)
    write_csv(output/'paired_by_seed.csv', all_pairs)
    write_csv(output/'paired_mean_std.csv', paired)
    write_csv(output/'graphs_by_seed.csv', graphs)
    write_csv(output/'run_verification.csv', statuses)
    _, pilot_metrics, _ = verify_completed_run(pilot)
    pilot_graphs, pilot_masks = graph_rows(pilot, 42)
    write_csv(output/'pilot_metrics.csv', pilot_metrics)
    write_csv(output/'pilot_graphs.csv', pilot_graphs)
    linear = read_csv(root.parent/'latent_linear/additional_analysis/metrics.csv')
    comparison = []
    for population, rows in [('linear_pilot_42', linear), ('nonlinear_pilot_42', pilot_metrics)]:
        for row in rows:
            comparison.append(dict(population=population, **{key: row[key] for key in KEYS},
                                   n_seeds=1, r2_mean=float(row['test_r2']), r2_std=None,
                                   mse_mean=float(row['test_mse']), mse_std=None))
    for row in summary:
        comparison.append(dict(population='nonlinear_fresh_five', **{key: row[key] for key in KEYS},
                               n_seeds=5, r2_mean=row['test_r2_mean'], r2_std=row['test_r2_std'],
                               mse_mean=row['test_mse_mean'], mse_std=row['test_mse_std']))
    write_csv(output/'linear_vs_nonlinear.csv', comparison)
    protected = json.loads((root/'protected_runs.json').read_text())
    for name, expected in protected.items():
        if snapshot(root.parent/name) != expected:
            raise ValueError('Previously saved experiment changed: '+name)
    make_figures(output, summary, all_metrics, paired, pilot_masks, masks_by_seed, comparison)
    make_report(output, summary, paired, pilot_metrics, graphs, pilot_graphs)
    (output/'status.json').write_text(json.dumps(dict(
        completed=True, fresh_seeds=spec['seeds'], pilot_seed=42, pilot_in_mean_std=False,
        std_ddof=1, original_runs_unchanged=list(protected),
        final_test_used_for_selection=False, verified_all_predictions=True,
        summary_rows=len(summary), metrics_rows=len(all_metrics),
        seed_variation='data generation + split + NN initialization/minibatches + probe CV/trees',
        caution='Descriptive mean/sample-SD; five seeds do not establish statistical superiority',
        source_sha256={str(p): digest(p) for p in [Path(__file__), Path('latent_nonlinear_study.py')]}
    ), indent=2))
    print('Five-seed summary complete:', output, flush=True)
    return summary


def make_figures(output, summary, raw, paired, pilot_masks, masks_by_seed, comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup = {tuple(str(r[k]) for k in KEYS): r for r in summary}
    scopes = ['X1', 'X2', 'X3', 'concat']
    codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for decoder in ('ridge', 'extra_trees'):
        for score in ('r2', 'mse'):
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            for ax, layer in zip(axes, ('1', '2')):
                values = np.array([[lookup[(c, decoder, layer, s, 'Z1')]['test_'+score+'_mean']
                                    for c in codes] for s in scopes])
                ax.imshow(values, aspect='auto', cmap='viridis')
                ax.set_xticks(range(7)); ax.set_xticklabels(codes)
                ax.set_yticks(range(4)); ax.set_yticklabels(scopes)
                for i, scope in enumerate(scopes):
                    for j, code in enumerate(codes):
                        r = lookup[(code, decoder, layer, scope, 'Z1')]
                        ax.text(j, i, '{:.3f}\n+/- {:.3f}'.format(r['test_'+score+'_mean'], r['test_'+score+'_std']),
                                ha='center', va='center', fontsize=9,
                                bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
                ax.set_title('{} L{} test {}: fresh 5 seeds, mean +/- sample SD'.format(decoder, layer, score))
            fig.text(.5, .01, 'A random/initial; B random/train; C learned/train; D random/DAG; '
                     'E learned/DAG; F random/CAM; G learned/CAM', ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .04, 1, 1))
            fig.savefig(str(output/'figures'/('mlp_'+score+'_'+decoder+'.png')), dpi=150)
            plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, decoder in enumerate(('ridge', 'extra_trees')):
        for j, layer in enumerate(('1', '2')):
            ax = axes[i, j]
            labels, means, errors = [], [], []
            for code in ('X', 'B', 'C', 'D', 'E', 'F', 'G'):
                key = (code, decoder, '0' if code == 'X' else layer, 'all' if code == 'X' else 'concat', 'Z1')
                row = lookup[key]
                labels.append(code); means.append(row['test_r2_mean']); errors.append(row['test_r2_std'])
            ax.errorbar(range(len(labels)), means, yerr=errors, fmt='o', capsize=4)
            for seed in sorted({r['seed'] for r in raw}):
                values = []
                for code in labels:
                    values.append(next(float(r['test_r2']) for r in raw if r['seed'] == seed and
                                       r['condition'] == code and r['decoder'] == decoder and
                                       r['layer'] == ('0' if code == 'X' else layer) and
                                       r['scope'] == ('all' if code == 'X' else 'concat')))
                ax.plot(range(len(labels)), values, alpha=.2, lw=.7)
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
            ax.set_title('{} concat L{}; X=raw observations'.format(decoder, layer))
            ax.set_ylabel('Final-test R2')
    fig.tight_layout(); fig.savefig(str(output/'figures/concat_vs_X.png'), dpi=150); plt.close(fig)
    comparisons = ['learning_train_mask', 'learning_dag_mask', 'learning_cam_mask',
                   'dag_mask_learned_weights', 'cam_mask_learned_weights']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for i, decoder in enumerate(('ridge', 'extra_trees')):
        for j, layer in enumerate(('1', '2')):
            ax = axes[i, j]
            rows = [next(r for r in paired if r['comparison'] == c and r['decoder'] == decoder and
                         r['layer'] == layer and r['scope'] == 'concat') for c in comparisons]
            ax.errorbar(range(len(rows)), [r['delta_r2_mean'] for r in rows],
                        yerr=[r['delta_r2_std'] for r in rows], fmt='o', capsize=4)
            ax.axhline(0, color='black', lw=.7)
            ax.set_xticks(range(5)); ax.set_xticklabels(['C-B', 'E-D', 'G-F', 'E-C', 'G-E'])
            ax.set_title('{} L{}: paired delta R2 mean +/- SD'.format(decoder, layer))
    fig.tight_layout(); fig.savefig(str(output/'figures/paired_effects.png'), dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for j, stage in enumerate(('train', 'dag', 'cam')):
        axes[0, j].imshow(pilot_masks[stage].T, vmin=0, vmax=1, cmap='Blues')
        axes[0, j].set_title('Pilot 42 '+stage)
        frequency = np.mean([m[stage] for m in masks_by_seed.values()], axis=0)
        axes[1, j].imshow(frequency.T, vmin=0, vmax=1, cmap='Blues')
        axes[1, j].set_title('Fresh five edge frequency '+stage)
        for i in range(3):
            for k in range(3):
                axes[1, j].text(k, i, '{:.1f}'.format(frequency[k, i]), ha='center', va='center')
        for i in (0, 1):
            axes[i, j].set_xticks(range(3)); axes[i, j].set_xticklabels(['X1','X2','X3'])
            axes[i, j].set_yticks(range(3)); axes[i, j].set_yticklabels(['X1','X2','X3'])
            axes[i, j].set_xlabel('Cause'); axes[i, j].set_ylabel('Effect')
    fig.tight_layout(); fig.savefig(str(output/'figures/graphs.png'), dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    categories = [('X','0','all'), ('C','1','concat'), ('C','2','concat'), ('G','1','concat'), ('G','2','concat')]
    for ax, decoder in zip(axes, ('ridge', 'extra_trees')):
        for population, offset in [('linear_pilot_42', -.22), ('nonlinear_pilot_42', 0), ('nonlinear_fresh_five', .22)]:
            rows = [next(r for r in comparison if r['population'] == population and r['decoder'] == decoder
                         and (r['condition'],str(r['layer']),r['scope']) == key) for key in categories]
            ax.bar(np.arange(5)+offset, [r['r2_mean'] for r in rows], width=.22, label=population,
                   yerr=[r['r2_std'] or 0 for r in rows], capsize=2)
        ax.set_xticks(range(5)); ax.set_xticklabels(['X','C L1','C L2','G L1','G L2'])
        ax.set_title(decoder+' final-test R2'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(str(output/'figures/linear_vs_nonlinear.png'), dpi=150); plt.close(fig)


def make_report(output, summary, paired, pilot, graphs, pilot_graphs):
    lines = ['# 非線形潜在変数実験', '',
             'Z~N(0,1)、Xi=ai*tanh(Z)+εi、a=(2,3,1)、εi~N(0,0.5²)は独立。'
             '真の直接観測辺はなし。潜在辺はZ→X1/X2/X3。', '',
             '初回seed42を動作確認に使用し、別の新規seed101〜105のみで平均と標本標準偏差(ddof=1)を計算。'
             '生成・分割・NN学習seedを同時に変更し、回帰seedはseed+1。データと学習双方の変動を含む。', '',
             '既存CAMのRスクリプト内のset.seed(42)は変更せず固定した。', '',
             'NNはXのみを使用。学習600件・検証200件、最終テスト200件はNN学習・CAM・回帰の候補選択に不使用。'
             '回帰は元の候補を固定し、学習内5分割CVで選択。両回帰を報告しテストで採用手法を選ばない。', '',
             'A=未学習/初期マスク、B=未学習/学習後マスク、C=学習済み/学習後マスク、'
             'D=未学習/DAGマスク、E=学習済み/DAGマスク、F=未学習/CAMマスク、G=学習済み/CAMマスク。', '',
             '隠れ層は全条件で(1000,3,10)。各MLP10特徴、連結30特徴。', '',
             '## 推定グラフ', '', '|seed|段階|辺|非巡回|', '|---|---|---|---|']
    for r in pilot_graphs+graphs:
        lines.append('|{}|{}|{}|{}|'.format(r['seed'], r['stage'], r['edges'], r['is_acyclic']))
    lines += ['', '観測変数間DAGだけでは共通原因Zを表現できず、推定辺を真の直接因果とみなさない。'
              '学習終了時の二値マスクはDAG化前なので循環し得る。', '',
              '## 新規5シード：各MLP・各層のテスト結果', '',
              'セルはR²平均±標準偏差 / MSE平均±標準偏差。', '',
              '|条件|回帰|層|X1 MLP|X2 MLP|X3 MLP|全MLP連結|', '|---|---|---|---|---|---|---|']
    lookup = {tuple(str(r[k]) for k in KEYS): r for r in summary}
    for code in ('C','E','G'):
        for decoder in ('ridge','extra_trees'):
            for layer in ('1','2'):
                cells = []
                for scope in ('X1','X2','X3','concat'):
                    r = lookup[(code,decoder,layer,scope,'Z1')]
                    cells.append('{:.4f}±{:.4f} / {:.4f}±{:.4f}'.format(r['test_r2_mean'],r['test_r2_std'],
                                                                 r['test_mse_mean'],r['test_mse_std']))
                lines.append('|{}|{}|{}|{}|'.format(code,decoder,layer,'|'.join(cells)))
    lines += ['', '## 元のXと固定マスクでの比較', '']
    for decoder in ('ridge','extra_trees'):
        r = lookup[('X',decoder,'0','all','Z1')]
        lines.append('- X {}: R²={:.5f}±{:.5f}、MSE={:.5f}±{:.5f}'.format(
            decoder,r['test_r2_mean'],r['test_r2_std'],r['test_mse_mean'],r['test_mse_std']))
    lines += ['', '連結特徴でのペア差分（正のΔR²はafterの予測改善）：', '',
              '|比較|回帰|層|ΔR²平均±SD|ΔMSE平均±SD|', '|---|---|---|---|---|']
    for r in paired:
        if r['scope'] != 'concat' or r['comparison'] not in (
                'learning_train_mask','learning_dag_mask','learning_cam_mask',
                'dag_mask_learned_weights','cam_mask_learned_weights'):
            continue
        lines.append('|{}→{}|{}|{}|{:+.5f}±{:.5f}|{:+.5f}±{:.5f}|'.format(
            r['before'],r['after'],r['decoder'],r['layer'],r['delta_r2_mean'],r['delta_r2_std'],
            r['delta_mse_mean'],r['delta_mse_std']))
    lines += ['', '## 線形データとの比較と解釈の範囲', '',
              '- linear_vs_nonlinear.csvおよび対応PNGに、線形seed42、非線形seed42、非線形新規5seedを保存。',
              '- 線形はnoise_std=1、非線形は0.5。生成関数とノイズ水準が同時に変わり、差を非線形性だけに帰属できない。',
              '- tanhの飽和により大きな|Z|でXの変化が小さくなるが、実測精度の違いの原因をこの解析だけで断定しない。',
              '- Ridgeと非線形回帰の違いは読み出し方に依存する。低いRidgeスコアだけでZに関する依存性の消失とは結論しない。',
              '- 親がないMLPは入力依存性を失い一定活性になる。一定特徴のR²は0近傍または負になり得る。',
              '- 個別MLPの集計ではseedごとに親集合や根ノードが異なり得る。MLP番号だけで恒常的な優劣を解釈しない。親数はgraphs_by_seed.csvで併記。',
              '- Z予測性能の測定であり、情報量自体や潜在変数の自動発見・因果的同定を測定していない。',
              '- 5seedは独立したデータセットであるが、同じ生成条件に限定される。平均±SDは信頼区間や有意差検定ではない。',
              '- NNの構造・損失・DAG制約・最適化・CAMを変更せず、事前に定めた全5seedを報告。テストでseedを選別しない。',
              '- 任意の非線形SEMや10変数での実験、ノイズ水準を揃えた対照、潜在変数の自動発見は今回未実施。', '',
              '初回seed42の詳細は[初回レポート](../../latent_nonlinear/additional_analysis/report.md)、'
              '全条件の平均・SDはmetrics_mean_std.csv、各seed値はmetrics_by_seed.csvを参照。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
