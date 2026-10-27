"""Controlled weight/mask comparisons without changing or refitting GraN-DAG.

Reuse verified frozen hidden arrays and Ridge probes. Only missing decoder
probes are fit, using the pre-existing training-only CV specification.
All outputs go to a fresh direct child of the experiment directory.
"""
import copy
import csv
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import mean_squared_error, r2_score

from analyze_hidden import varying_columns
from gran_dag.hidden import load_checkpoint_cpu, save_hidden_activations
from .evaluate import evaluate_probe, fit_probe, write_csv


CONDITIONS = [
    ('A', 'initial', 'initial', 'random_full'),
    ('B', 'initial', 'train', 'random_train_mask'),
    ('C', 'learned', 'train', 'train'),
    ('D', 'initial', 'dag', None),
    ('E', 'learned', 'dag', 'to_dag'),
    ('F', 'initial', 'cam', 'random_cam_mask'),
    ('G', 'learned', 'cam', 'cam'),
]
COMPARISONS = [
    ('learning_train_mask', 'B', 'C'),
    ('learning_dag_mask', 'D', 'E'),
    ('learning_cam_mask', 'F', 'G'),
    ('dag_mask_learned_weights', 'C', 'E'),
    ('dag_mask_initial_weights', 'B', 'D'),
    ('cam_mask_learned_weights', 'E', 'G'),
    ('cam_mask_initial_weights', 'D', 'F'),
    ('learning_and_mask_combined', 'A', 'C'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root, excluded=None):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and (excluded is None or excluded not in p.parents)}


def fresh_output(root, output=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root/'additional_analysis'
    if output.parent != root:
        raise ValueError('Output must be a NEW direct child of run-dir')
    if output.exists():
        raise FileExistsError('Output exists; choose a new --output-dir: '+str(output))
    return output


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def parameter_digest(model):
    """All registered parameters, excluding the separately checked graph mask."""
    sha = hashlib.sha256()
    for name, tensor in model.named_parameters():
        a = tensor.detach().cpu().numpy()
        sha.update((name+str(a.shape)+str(a.dtype)).encode())
        sha.update(a.tobytes())
    architecture = [type(model).__name__, model.num_vars, model.num_layers,
                    model.hid_dim, model.nonlin]
    sha.update(json.dumps(architecture).encode())
    return sha.hexdigest()


def array_digest(a):
    return hashlib.sha256((str(a.shape)+str(a.dtype)).encode()+a.tobytes()).hexdigest()


def verify_hidden(directory, model, x, ids, labels, mean, std, checkpoint=None):
    """Prove cached activations represent these exact weights, mask and rows."""
    metadata = json.loads((directory/'metadata.json').read_text())
    if checkpoint is not None and metadata.get('checkpoint_sha256') != digest(checkpoint):
        raise ValueError('Checkpoint/activation hash mismatch: '+str(checkpoint))
    for filename, expected in [('sample_indices.npy', ids), ('sample_split.npy', labels),
                               ('mean.npy', mean), ('std.npy', std),
                               ('adjacency.npy', model.adjacency.cpu().numpy())]:
        np.testing.assert_array_equal(np.load(str(directory/filename)), expected)
    layers = [np.load(str(directory/entry['file'])) for entry in metadata['layers']]
    if len(layers) != model.num_layers or any(h.shape[:2] != x.shape for h in layers):
        raise ValueError('Invalid hidden layer dimensions')
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for start in range(0, len(x), 256):
            raw = torch.as_tensor(x[start:start+256], dtype=dtype)
            # Same operation order and dtype as the original extraction worker.
            data = (raw-torch.as_tensor(mean, dtype=dtype))/torch.as_tensor(std, dtype=dtype)
            actual = model.get_hidden_activations(data)
            for saved, current in zip(layers, actual):
                np.testing.assert_allclose(saved[start:start+len(raw)], current.cpu().numpy(),
                                           rtol=1e-12, atol=1e-12)
    return layers


def controlled_features(root, output, config, x, ids, labels):
    cam = 'cam-pruning/cutoff_%.6f' % config['training']['cam_pruning_cutoff']
    paths = dict(random_full='random_full', random_train_mask='random_train_mask',
                 train='train', to_dag='to-dag', random_cam_mask='random_cam_mask', cam=cam)
    models = {key: load_checkpoint_cpu(str(root/path/'model.pkl')) for key, path in paths.items()}
    hashes = {key: parameter_digest(model) for key, model in models.items()}
    masks = {name: models[key].adjacency.detach().cpu().numpy().copy()
             for name, key in [('initial', 'random_full'), ('train', 'train'),
                               ('dag', 'to_dag'), ('cam', 'cam')]}
    base = dict(initial=models['random_full'], learned=models['train'])
    mean = np.load(str(root/'normalization_mean.npy'))
    std = np.load(str(root/'normalization_std.npy'))
    if config['training']['normalize_data']:
        np.testing.assert_allclose(mean, x[labels == 0].mean(0, keepdims=True), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(std, x[labels == 0].std(0, ddof=1, keepdims=True), rtol=1e-12, atol=1e-12)
    cached, features, conditions = {}, {}, []
    for code, weights, mask_name, preferred in CONDITIONS:
        expected_hash = parameter_digest(base[weights])
        candidates = [key for key in models if hashes[key] == expected_hash and
                      np.array_equal(models[key].adjacency.cpu().numpy(), masks[mask_name])]
        source = preferred if preferred in candidates else (candidates[0] if candidates else None)
        if source is not None:
            directory = root/paths[source]/'hidden'
            if source not in cached:
                cached[source] = verify_hidden(directory, models[source], x, ids, labels, mean, std,
                                                root/paths[source]/'model.pkl')
            layers = cached[source]
        else:
            # For other runs a learned stage may have refit weights, or the
            # random DAG checkpoint may be absent. Apply only the desired mask
            # to the same frozen initial/train weights; never optimize anything.
            frozen = copy.deepcopy(base[weights])
            frozen.adjacency = torch.as_tensor(masks[mask_name], dtype=next(frozen.parameters()).dtype)
            directory = output/'controlled_hidden'/code
            data = torch.as_tensor(x, dtype=next(frozen.parameters()).dtype)
            data = (data-torch.as_tensor(mean, dtype=data.dtype))/torch.as_tensor(std, dtype=data.dtype)
            save_hidden_activations(frozen, data, str(directory), sample_indices=ids,
                                    sample_split=labels, mean=mean, std=std,
                                    metadata={'condition': code, 'frozen_parameters_sha256': expected_hash})
            layers = verify_hidden(directory, frozen, x, ids, labels, mean, std)
        features[code] = layers
        conditions.append(dict(condition=code, weights=weights, mask=mask_name,
                               ridge_reuse_stage=source,
                               hidden_directory=str(directory.relative_to(root)),
                               parameters_sha256=expected_hash, mask_sha256=array_digest(masks[mask_name]),
                               parent_counts=masks[mask_name].astype(bool).sum(0).tolist(),
                               shapes=[list(h.shape) for h in layers]))
    verification = dict(
        parameter_hashes=hashes,
        all_random_parameters_equal_initial=all(hashes[key] == hashes['random_full']
                                               for key in ('random_train_mask', 'random_cam_mask')),
        learned_stage_parameters_equal_train=all(hashes[key] == hashes['train'] for key in ('to_dag', 'cam')),
        learned_parameters_differ_from_initial=hashes['train'] != hashes['random_full'],
        dag_equals_cam_mask=bool(np.array_equal(masks['dag'], masks['cam'])),
        every_condition_hidden_recomputed_for_verification=True,
        preprocessing='original training-only mean and unbiased std',
        adjacency_orientation='row=source; column=target',
        sample_order_verified=True)
    return conditions, features, verification


def comparison_rows(metrics):
    lookup = {(r['condition'], r['decoder'], r['layer'], r['scope'], r['latent']): r
              for r in metrics if r['condition'] != 'X'}
    result = []
    for name, before, after in COMPARISONS:
        for key, a in sorted(lookup.items()):
            if key[0] != before:
                continue
            b = lookup[(after,)+key[1:]]
            row = dict(comparison=name, before=before, after=after,
                       decoder=key[1], layer=key[2], scope=key[3], latent=key[4])
            for metric in ('r2', 'mse'):
                row['before_'+metric] = a['test_'+metric]
                row['after_'+metric] = b['test_'+metric]
                row['delta_'+metric] = b['test_'+metric]-a['test_'+metric]
            result.append(row)
    return result


def numeric_record(row):
    return {key: float(row[key]) for key in ('cv_mse', 'train_r2', 'train_mse',
                                            'validation_r2', 'validation_mse', 'test_r2', 'test_mse')}


def score_features(root, output, config, conditions, hidden, x, z, ids, labels):
    previous = read_csv(root/'analysis/metrics.csv')
    old = {(r['stage'], r['representation'], int(r['layer']), r['scope'], r['latent']): r for r in previous}
    metrics, predictions, cache = [], [], {}
    train, test = labels == 0, labels == 2
    reused, fitted = 0, 0
    feature_sets = []
    for condition in conditions:
        for layer, values in enumerate(hidden[condition['condition']], 1):
            scopes = [('X'+str(j+1), values[:, j, :], condition['parent_counts'][j])
                      for j in range(x.shape[1])]
            scopes.append(('concat', values.reshape(len(x), -1), None))
            for scope, values, parents in scopes:
                feature_sets.append((condition, layer, scope, values, parents))
    baseline = dict(condition='X', weights='none', mask='none', ridge_reuse_stage='X')
    feature_sets.insert(0, (baseline, 0, 'all', x, None))
    for condition, layer, scope, values, parents in feature_sets:
        for decoder, nonlinear in [('ridge', False), ('extra_trees', True)]:
            representation = ('X_'+decoder if layer == 0 else 'hidden_'+decoder)
            for factor in range(z.shape[1]):
                latent = 'Z'+str(factor+1)
                stage = condition['ridge_reuse_stage']
                source_key = (stage, representation, layer, scope, latent)
                old_row = old.get(source_key)
                # Cache identical FEATURES rather than merely identical masks.
                # Equal no-parent MLPs can have different constant bias values.
                cache_key = (array_digest(values), decoder, latent)
                if cache_key in cache:
                    stats, params, pred, model_ref, origin = cache[cache_key]
                elif old_row is not None:
                    path = root/'analysis/probes'/'{}_{}_L{}_{}_{}.pkl'.format(
                        stage, representation, layer, scope, latent)
                    with path.open('rb') as stream:
                        model = pickle.load(stream)
                    stats = evaluate_probe(model, values, z[:, factor], labels)
                    for key in stats:
                        np.testing.assert_allclose(stats[key], float(old_row[key]), rtol=1e-12, atol=1e-12)
                    stats['cv_mse'] = float(old_row['cv_mse'])
                    params = old_row['best_params']
                    pred = model.predict(values[test])
                    model_ref, origin = str(path.relative_to(root)), 'existing_probe_verified'
                    cache[cache_key] = (stats, params, pred, model_ref, origin)
                    reused += 1
                else:
                    model, selected, cv_mse = fit_probe(values[train], z[train, factor],
                                                         config['probe'], nonlinear=nonlinear)
                    stats = evaluate_probe(model, values, z[:, factor], labels)
                    stats['cv_mse'] = cv_mse
                    params = json.dumps(selected, sort_keys=True)
                    pred = model.predict(values[test])
                    path = output/'probes'/'{}_{}_L{}_{}_{}.pkl'.format(
                        condition['condition'], decoder, layer, scope, latent)
                    with path.open('wb') as stream:
                        pickle.dump(model, stream)
                    model_ref, origin = str(path.relative_to(root)), 'new_training_only_cv'
                    cache[cache_key] = (stats, params, pred, model_ref, origin)
                    fitted += 1
                # Independently recompute scores from exported test predictions.
                np.testing.assert_allclose(stats['test_mse'], mean_squared_error(z[test, factor], pred), atol=1e-12)
                np.testing.assert_allclose(stats['test_r2'], r2_score(z[test, factor], pred), atol=1e-12)
                record = dict(condition=condition['condition'], weights=condition['weights'],
                              mask=condition['mask'], decoder=decoder, layer=layer, scope=scope,
                              latent=latent, parents=parents, feature_count=values.shape[1],
                              varying_features=int(varying_columns(values[train]).sum()),
                              n_train=int(train.sum()), n_validation=int((labels == 1).sum()),
                              n_test=int(test.sum()), best_params=params, probe_source=origin,
                              model_file=model_ref)
                record.update(stats)
                metrics.append(record)
                for row, prediction in zip(np.flatnonzero(test), pred):
                    predictions.append(dict(condition=condition['condition'], decoder=decoder,
                                             layer=layer, scope=scope, latent=latent,
                                             sample_index=int(ids[row]), true_z=float(z[row, factor]),
                                             prediction=float(prediction)))
        print('Probes complete:', condition['condition'], 'L'+str(layer), scope, flush=True)
    return metrics, predictions, dict(new_probe_fits=fitted, reused_existing_probes=reused,
                                     identical_feature_results_shared=True)


def save_tables(output, metrics):
    write_csv(output/'metrics.csv', metrics)
    differences = comparison_rows(metrics)
    write_csv(output/'paired_comparisons.csv', differences)
    wide = {}
    for row in metrics:
        if row['condition'] == 'X':
            continue
        key = (row['condition'], row['decoder'], row['layer'], row['latent'])
        if key not in wide:
            wide[key] = dict(zip(('condition', 'decoder', 'layer', 'latent'), key))
        for score in ('r2', 'mse'):
            wide[key][row['scope']+'_'+score] = row['test_'+score]
    write_csv(output/'mlp_comparison.csv', list(wide.values()))
    lookup = {(r['condition'], r['decoder'], r['layer'], r['scope'], r['latent']): r for r in metrics}
    layers, baselines = [], []
    for row in metrics:
        if row['layer'] == 2:
            first = lookup[(row['condition'], row['decoder'], 1, row['scope'], row['latent'])]
            layers.append(dict(condition=row['condition'], decoder=row['decoder'], scope=row['scope'],
                               latent=row['latent'], layer1_r2=first['test_r2'], layer2_r2=row['test_r2'],
                               delta_r2=row['test_r2']-first['test_r2'],
                               layer1_mse=first['test_mse'], layer2_mse=row['test_mse'],
                               delta_mse=row['test_mse']-first['test_mse']))
        if row['layer']:
            reference = lookup[('X', row['decoder'], 0, 'all', row['latent'])]
            baselines.append(dict(condition=row['condition'], decoder=row['decoder'], layer=row['layer'],
                                  scope=row['scope'], latent=row['latent'], x_r2=reference['test_r2'],
                                  hidden_r2=row['test_r2'], delta_r2=row['test_r2']-reference['test_r2'],
                                  x_mse=reference['test_mse'], hidden_mse=row['test_mse'],
                                  delta_mse=row['test_mse']-reference['test_mse']))
    if layers:
        write_csv(output/'layer_comparison.csv', layers)
    write_csv(output/'x_vs_hidden.csv', baselines)
    return differences


def make_figures(output, metrics, predictions, differences):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    codes = [c[0] for c in CONDITIONS]
    lookup = {(r['condition'], r['decoder'], r['layer'], r['scope'], r['latent']): r for r in metrics}
    scopes = sorted({r['scope'] for r in metrics if r['layer'] and r['scope'] != 'concat'})+['concat']
    layers = sorted({r['layer'] for r in metrics if r['layer']})
    figures = output/'figures'
    for latent in sorted({r['latent'] for r in metrics}):
        for decoder in ('ridge', 'extra_trees'):
            for score in ('r2', 'mse'):
                fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3.5*len(layers)), squeeze=False)
                for ax, layer in zip(axes[:, 0], layers):
                    grid = np.array([[lookup[(code, decoder, layer, scope, latent)]['test_'+score]
                                      for scope in scopes] for code in codes])
                    plot = ax.imshow(grid.T, aspect='auto', cmap='viridis')
                    ax.set_xticks(range(len(codes)))
                    ax.set_xticklabels(codes)
                    ax.set_yticks(range(len(scopes)))
                    ax.set_yticklabels(scopes)
                    ax.set_title('{} {} L{} test {}'.format(latent, decoder, layer, score.upper()))
                    for i in range(len(scopes)):
                        for j in range(len(codes)):
                            ax.text(j, i, '{:.3f}'.format(grid[j, i]), ha='center', va='center',
                                    bbox=dict(facecolor='white', alpha=.65, edgecolor='none'))
                    fig.colorbar(plot, ax=ax)
                fig.text(.5, .01, 'A: random/initial  B: random/train  C: learned/train  D: random/DAG  '
                         'E: learned/DAG  F: random/CAM  G: learned/CAM', ha='center', fontsize=8)
                fig.tight_layout(rect=(0, .04, 1, 1))
                fig.savefig(str(figures/'mlp_{}_{}_{}.png'.format(score, decoder, latent)), dpi=150)
                plt.close(fig)
        for title, comparisons in [('learning', COMPARISONS[:3]), ('mask', COMPARISONS[3:7])]:
            fig, axes = plt.subplots(2, len(layers), figsize=(12, 7), squeeze=False)
            for i, decoder in enumerate(('ridge', 'extra_trees')):
                for j, layer in enumerate(layers):
                    ax = axes[i, j]
                    width = .8/len(comparisons)
                    for k, (name, before, after) in enumerate(comparisons):
                        values = [next(r['delta_r2'] for r in differences if r['comparison'] == name and
                                       r['decoder'] == decoder and r['layer'] == layer and
                                       r['scope'] == scope and r['latent'] == latent) for scope in scopes]
                        ax.bar(np.arange(len(scopes))+(k-(len(comparisons)-1)/2)*width, values,
                               width=width, label=before+' -> '+after)
                    ax.axhline(0, color='black', lw=.7)
                    ax.set_xticks(range(len(scopes)))
                    ax.set_xticklabels(scopes)
                    ax.set_title('{} L{} delta test R2'.format(decoder, layer))
                    ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(str(figures/(title+'_effects_'+latent+'.png')), dpi=150)
            plt.close(fig)
        # Fixed a priori panel selection: all MLPs and concat, C versus E;
        # not selecting a favorable panel by final-test performance.
        for decoder in ('ridge', 'extra_trees'):
            fig, axes = plt.subplots(len(scopes), len(layers), figsize=(10, 2.8*len(scopes)), squeeze=False)
            for i, scope in enumerate(scopes):
                for j, layer in enumerate(layers):
                    ax = axes[i, j]
                    for code in ('C', 'E'):
                        rows = [p for p in predictions if p['condition'] == code and p['decoder'] == decoder
                                and p['layer'] == layer and p['scope'] == scope and p['latent'] == latent]
                        truth = np.array([p['true_z'] for p in rows])
                        ax.scatter(truth, [p['prediction'] for p in rows], s=8, alpha=.45, label=code)
                    ax.plot([truth.min(), truth.max()], [truth.min(), truth.max()], color='black', lw=.7)
                    ax.set_title('{} L{} {}'.format(scope, layer, decoder))
                    ax.set_xlabel('True '+latent)
                    ax.set_ylabel('Predicted '+latent)
                    ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(str(figures/'predictions_{}_{}.png'.format(decoder, latent)), dpi=150)
            plt.close(fig)


def make_report(output, metrics, verification, fitting):
    lookup = {(r['condition'], r['decoder'], r['layer'], r['scope'], r['latent']): r for r in metrics}
    lines = ['# 潜在変数予測の追加解析', '',
             'NNの再学習は実施せず、既存のRidge回帰と隠れ層を検証して再利用した。'
             '新規学習は不足するRidge／ExtraTrees回帰器のみ。必要な対照がない場合は固定NNで抽出する。', '',
             '## 条件', '',
             '|条件|NNの重み|入力マスク|', '|---|---|---|']
    for code, weights, mask, _ in CONDITIONS:
        lines.append('|{}|{}|{}|'.format(code, weights, mask))
    lines += ['', '全条件で同じNN構造・元の学習データのみで計算した前処理を使用。'
              '列X1等はその変数を**予測するMLP**を指す。隣接行列は原因が行、結果が列。', '',
              '## 検証', '',
              '- 学習後・DAG化後・CAM後のNNパラメータが同一: '+str(verification['learned_stage_parameters_equal_train']),
              '- DAG化後とCAM後のマスクが同一: '+str(verification['dag_equals_cam_mask']),
              '- 保存された活性値と固定NNによる再計算を全層・全サンプルで比較済み。',
              '- 新規回帰fit数: {}; 既存回帰再利用数: {}（同じ特徴は結果を共有）。'.format(
                  fitting['new_probe_fits'], fitting['reused_existing_probes']), '',
              '## テスト結果', '']
    for latent in sorted({r['latent'] for r in metrics}):
        lines += ['### '+latent, '', '|条件|回帰|層|X1等の個別MLP R²|連結R²|連結MSE|', '|---|---|---|---|---|---|']
        for code, _, _, _ in CONDITIONS:
            for decoder in ('ridge', 'extra_trees'):
                for layer in (1, 2):
                    rows = [r for r in metrics if r['condition'] == code and r['decoder'] == decoder and
                            r['layer'] == layer and r['latent'] == latent]
                    if not rows:
                        continue
                    individual = ', '.join('{}: {:.4f}'.format(r['scope'], r['test_r2'])
                                           for r in rows if r['scope'] != 'concat')
                    concat = next(r for r in rows if r['scope'] == 'concat')
                    lines.append('|{}|{}|{}|{}|{:.4f}|{:.5f}|'.format(
                        code, decoder, layer, individual, concat['test_r2'], concat['test_mse']))
        lines += ['', '観測Xの基準:', '']
        for decoder in ('ridge', 'extra_trees'):
            row = lookup[('X', decoder, 0, 'all', latent)]
            lines.append('- {}: R²={:.6f}, MSE={:.6f}'.format(decoder, row['test_r2'], row['test_mse']))
        lines += ['', '学習済みNN（C/E/G）の個別MLPで最も高かったテスト点推定値:', '']
        for decoder in ('ridge', 'extra_trees'):
            eligible = [r for r in metrics if r['condition'] in ('C', 'E', 'G') and r['scope'] != 'concat'
                        and r['latent'] == latent and r['decoder'] == decoder]
            best = max(eligible, key=lambda r: r['test_r2'])
            lines.append('- {}: 条件{} / {} / L{}: R²={:.6f}, MSE={:.6f}'.format(
                decoder, best['condition'], best['scope'], best['layer'], best['test_r2'], best['test_mse']))
        lines += ['', '学習効果（C−B：同じ学習後マスク、全MLP連結）:', '']
        for decoder in ('ridge', 'extra_trees'):
            for layer in (1, 2):
                if ('C', decoder, layer, 'concat', latent) not in lookup:
                    continue
                a, b = (lookup[(c, decoder, layer, 'concat', latent)] for c in ('B', 'C'))
                lines.append('- {} L{}: ΔR²={:+.6f}, ΔMSE={:+.6f}'.format(
                    decoder, layer, b['test_r2']-a['test_r2'], b['test_mse']-a['test_mse']))
        lines += ['', 'DAG化マスクの効果（E−C：同じ学習済み重み、全MLP連結）:', '']
        for decoder in ('ridge', 'extra_trees'):
            for layer in (1, 2):
                if ('C', decoder, layer, 'concat', latent) not in lookup:
                    continue
                a, b = (lookup[(c, decoder, layer, 'concat', latent)] for c in ('C', 'E'))
                lines.append('- {} L{}: ΔR²={:+.6f}, ΔMSE={:+.6f}'.format(
                    decoder, layer, b['test_r2']-a['test_r2'], b['test_mse']-a['test_mse']))
    lines += ['', '## 解釈上の制限', '',
              '- これはZの予測精度であり、情報量そのものの測定でも潜在変数の自動発見でもない。',
              '- 学習済みNNと未学習NNを同一マスクで比較することと、同じ重みでマスクを変更することを分離した。',
              '- 親がないMLPでは入力に依存しない一定活性となる。定数特徴からの予測は訓練Z平均となり、R²は僅かに負にもなる。',
              '- Ridgeの低精度だけではZに関する予測可能な依存性の消失を断定できない。ExtraTreesも万能ではない。',
              '- ExtraTreesの候補・乱数・5分割CVは既存設定をそのまま使用。モデルfit・前処理・候補選択は学習600件内のみ。',
              '- 最終テスト200件でモデルやハイパーパラメータは選ばない。検証200件も回帰選択には使わない。',
              '- 最終テストは先の実験で既に閲覧済み。同じテストでの追加比較は探索的解析で、独立した確認実験ではない。',
              '- 最高値の記述は点推定の要約であり、テストによるモデル選択や統計的優位性の主張ではない。',
              '- 一つの生成条件・乱数シードの結果であり、一般的なGraN-DAGの優位性は結論できない。',
              '- 潜在共通原因を含むため、観測DAGの正解復元とZ予測を同一視しない。', '',
              '詳細なR²/MSEはmetrics.csv、同一マスク・同一重みの差分はpaired_comparisons.csv、'
              '層比較はlayer_comparison.csv、Xとの比較はx_vs_hidden.csv。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def analyze(root, output=None):
    root = Path(root).resolve()
    output = fresh_output(root, output)
    before = snapshot(root)
    start = time.time()
    config = json.loads((root/'config.json').read_text())
    manifest = json.loads((root/'data_manifest.json').read_text())
    for filename, expected in manifest['data_files'].items():
        if digest(root/'data'/filename) != expected:
            raise ValueError('Generated input changed: '+filename)
    x = np.load(str(root/'data/observed_X.npy'))
    z = np.load(str(root/'data/true_Z.npy'))
    ids = np.load(str(root/'data/sample_indices.npy'))
    labels = np.load(str(root/'data/sample_split.npy'))
    np.testing.assert_array_equal(ids, np.arange(len(x)))
    if z.ndim != 2 or len(z) != len(x) or labels.shape != (len(x),) or set(labels) != {0, 1, 2}:
        raise ValueError('Data/split correspondence is invalid')
    torch.set_num_threads(1)
    output.mkdir()
    (output/'probes').mkdir()
    (output/'figures').mkdir()
    (output/'config.json').write_text(json.dumps(dict(
        source_run=str(root), source_config=config, probe_settings=config['probe'],
        unchanged_probe_hyperparameters=True, final_test_used_for_selection=False,
        analysis_type='exploratory reuse of previously inspected final test'), indent=2))
    conditions, hidden, verification = controlled_features(root, output, config, x, ids, labels)
    (output/'conditions.json').write_text(json.dumps(conditions, indent=2))
    (output/'controls_verification.json').write_text(json.dumps(verification, indent=2))
    metrics, predictions, fitting = score_features(root, output, config, conditions, hidden, x, z, ids, labels)
    differences = save_tables(output, metrics)
    write_csv(output/'test_predictions.csv', predictions)
    make_figures(output, metrics, predictions, differences)
    make_report(output, metrics, verification, fitting)
    after = snapshot(root, excluded=output)
    if before != after:
        raise RuntimeError('Existing experiment files changed during analysis')
    status = dict(completed=True, source_files_unchanged=True, source_sha256=before,
                  controls=verification, fitting=fitting, n_metric_rows=len(metrics),
                  n_prediction_rows=len(predictions), elapsed_seconds=time.time()-start,
                  final_test_used_for_selection=False, nn_retrained=False,
                  nonlinear_selection='unchanged original settings; CV within training rows only',
                  test_interpretation='exploratory; final test previously inspected',
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                sklearn=sklearn.__version__, torch=torch.__version__),
                  source_code_sha256={str(p): digest(p) for p in [Path(__file__),
                      Path('latent_additional_analysis.py'), Path('latent_experiments/evaluate.py')]})
    (output/'status.json').write_text(json.dumps(status, indent=2))
    print('Additional analysis completed:', output, fitting, flush=True)
    return status
