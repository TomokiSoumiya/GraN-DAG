"""Latent-label analysis AFTER X-only fitting. Final test never enters fit/CV."""
import csv
import hashlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_hidden import correlations, varying_columns
from .data import topological_order


def write_csv(path, rows, fieldnames=None):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_probe(x_train, z_train, settings, nonlinear=False):
    """This API accepts ONLY training data; final-test data cannot enter CV."""
    cv = KFold(n_splits=settings['cv_folds'], shuffle=True, random_state=settings['seed'])
    if nonlinear:
        cfg = settings['nonlinear']
        if cfg['type'] != 'extra_trees':
            raise ValueError('Only extra_trees nonlinear probes are implemented')
        model = ExtraTreesRegressor(n_estimators=cfg['n_estimators'],
                                     random_state=settings['seed'], n_jobs=1)
        params = {'max_depth': cfg['max_depth'], 'min_samples_leaf': cfg['min_samples_leaf']}
    else:
        model = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(solver='svd'))])
        params = {'ridge__alpha': settings['ridge_alphas']}
    search = GridSearchCV(model, params, cv=cv, scoring='neg_mean_squared_error',
                          n_jobs=1, refit=True, error_score='raise')
    search.fit(x_train, z_train)
    return search.best_estimator_, search.best_params_, float(-search.best_score_)


def evaluate_probe(model, x, z, labels):
    result = {}
    for name, code in [('train', 0), ('validation', 1), ('test', 2)]:
        rows = labels == code
        pred = model.predict(x[rows])
        result[name+'_r2'] = float(r2_score(z[rows], pred))
        result[name+'_mse'] = float(mean_squared_error(z[rows], pred))
    return result


def graph_analysis(root, stages, observed, latent, output):
    edges, metrics = [], []
    for a, b in np.argwhere(observed):
        edges.append(dict(stage='true_observed_direct', source='X'+str(a+1), target='X'+str(b+1)))
    for a, b in np.argwhere(latent):
        edges.append(dict(stage='true_latent', source='Z'+str(a+1), target='X'+str(b+1)))
    for stage in ['train', 'to-dag', stages['cam']]:
        graph = np.load(str(root/stage/'hidden/adjacency.npy')) != 0
        for a, b in np.argwhere(graph):
            edges.append(dict(stage=stage, source='X'+str(a+1), target='X'+str(b+1)))
        difference = observed.astype(int)-graph.astype(int)
        reversed_edges = int((((difference+difference.T) == 0) & (difference != 0)).sum()//2)
        missing = int((difference == 1).sum()-reversed_edges)
        added = int((difference == -1).sum()-reversed_edges)
        try:
            topological_order(graph)
            acyclic = True
        except ValueError:
            acyclic = False
        metrics.append(dict(stage=stage, edges=int(graph.sum()), is_acyclic=acyclic,
                            direct_true_edges=int(observed.sum()),
                            missing_direct_edges=missing, additional_vs_direct=added,
                            reversed_direct_edges=reversed_edges,
                            direct_edge_shd=missing+added+reversed_edges,
                            interpretation='descriptive direct-edge comparison only; latent confounding omitted'))
    write_csv(output/'graph_edges.csv', edges, ['stage', 'source', 'target'])
    write_csv(output/'graph_metrics.csv', metrics)
    return metrics


def pca_on_train(values, labels):
    mean = values[labels == 0].mean(0)
    centered = values[labels == 0]-mean
    if not varying_columns(values[labels == 0]).any():
        return np.zeros((len(values), 2)), np.zeros(2), 0
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    ratio = variance/variance.sum()
    coordinates = (values-mean) @ vectors[:2].T
    if coordinates.shape[1] < 2:
        coordinates = np.column_stack([coordinates, np.zeros(len(values))])
    return coordinates, ratio[:2], int(np.searchsorted(np.cumsum(ratio), .95)+1)


def evaluate(root, config):
    root = Path(root)
    status = json.loads((root/'training_status.json').read_text())
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    (output/'probes').mkdir(exist_ok=True)
    (output/'figures').mkdir(exist_ok=True)
    # Integrity checks bind labels and observations to the generated run.
    manifest = json.loads((root/'data_manifest.json').read_text())
    for name, digest in manifest['data_files'].items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Generated data were modified: '+name)
    x = np.load(str(root/'data/observed_X.npy'))
    z = np.load(str(root/'data/true_Z.npy'))
    ids = np.load(str(root/'data/sample_indices.npy'))
    labels = np.load(str(root/'data/sample_split.npy'))
    observed = np.load(str(root/'data/observed_adjacency.npy'))
    latent = np.load(str(root/'data/latent_adjacency.npy'))
    assert np.array_equal(ids, np.arange(len(x))) and len(z) == len(x)
    cam = 'cam-pruning/cutoff_%.6f' % config['training']['cam_pruning_cutoff']
    stage_paths = {'train': 'train', 'to_dag': 'to-dag', 'cam': cam, 'random_full': 'random_full',
                   'random_train_mask': 'random_train_mask', 'random_cam_mask': 'random_cam_mask'}
    hidden, masks = {}, {}
    from hidden_to_csv import convert_hidden_directory
    for stage, path in stage_paths.items():
        directory = root/path/'hidden'
        np.testing.assert_array_equal(np.load(str(directory/'sample_indices.npy')), ids)
        np.testing.assert_array_equal(np.load(str(directory/'sample_split.npy')), labels)
        metadata = json.loads((directory/'metadata.json').read_text())
        checkpoint = root/path/'model.pkl'
        if checkpoint.exists() and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after hidden extraction: '+str(checkpoint))
        hidden[stage] = [np.load(str(directory/layer['file'])) for layer in metadata['layers']]
        masks[stage] = np.load(str(directory/'adjacency.npy'))
        assert all(a.shape[:2] == x.shape for a in hidden[stage])
        convert_hidden_directory(directory, overwrite=True)
    np.testing.assert_array_equal(masks['train'], masks['random_train_mask'])
    np.testing.assert_array_equal(masks['cam'], masks['random_cam_mask'])

    metrics, predictions = [], []
    units, correlation_rows, pca_rows, pca_summary = [], [], [], []
    pca_concat = {}
    test = labels == 2
    feature_sets = [('X', 'X_ridge', 0, 'all', -1, x, None, False),
                    ('X', 'X_extra_trees', 0, 'all', -1, x, None, True)]
    for stage in ('train', 'to_dag', 'cam'):
        used = np.flatnonzero(masks[stage].any(axis=1))
        union_values = x[:, used] if len(used) else np.zeros((len(x), 1))
        feature_sets.append((stage, 'input_union_ridge', 0, 'all_used_inputs', -1,
                             union_values, len(used), False))
        for target in range(x.shape[1]):
            parents = np.flatnonzero(masks[stage][:, target])
            # A constant feature represents the no-parent intercept baseline.
            values = x[:, parents] if len(parents) else np.zeros((len(x), 1))
            feature_sets.append((stage, 'parents_ridge', 0, 'X'+str(target+1), target,
                                 values, len(parents), False))
    for stage, layers in hidden.items():
        for layer, values in enumerate(layers, 1):
            for target in range(x.shape[1]):
                scope = 'X'+str(target+1)
                parent_count = int(masks[stage][:, target].sum())
                h = values[:, target, :]
                varying = varying_columns(h[labels == 0])
                for unit in range(h.shape[1]):
                    unit_values = h[labels == 0, unit]
                    units.append(dict(stage=stage, layer=layer, mlp=scope, unit=unit+1,
                                      parents=parent_count, varying=bool(varying[unit]),
                                      mean=float(unit_values.mean()), std=float(unit_values.std())))
                pearson = correlations(h[labels == 0], z[labels == 0])
                pearson[~varying] = np.nan
                for unit in range(h.shape[1]):
                    for factor in range(z.shape[1]):
                        correlation_rows.append(dict(stage=stage, layer=layer, mlp=scope,
                                                      unit=unit+1, latent='Z'+str(factor+1),
                                                      pearson=float(pearson[unit, factor])))
                feature_sets.append((stage, 'hidden_ridge', layer, scope, target,
                                     h, parent_count, False))
            flat = values.reshape(len(x), -1)
            feature_sets.append((stage, 'hidden_ridge', layer, 'concat', -1,
                                 flat, None, False))
            if stage in ('train', 'cam'):
                for scope, h in [('X'+str(j+1), values[:, j, :]) for j in range(x.shape[1])] + [('concat', flat)]:
                    coords, ratios, dimensions = pca_on_train(h, labels)
                    pca_summary.append(dict(stage=stage, layer=layer, scope=scope,
                                            pc1_ratio=float(ratios[0]), pc2_ratio=float(ratios[1]),
                                            pc95_dimensions=dimensions))
                    for row in range(len(x)):
                        pca_rows.append(dict(stage=stage, layer=layer, scope=scope,
                                             sample_index=int(ids[row]), split=int(labels[row]),
                                             pc1=float(coords[row, 0]), pc2=float(coords[row, 1])))
                    if scope == 'concat':
                        pca_concat[(stage, layer)] = coords

    for stage, kind, layer, scope, target, values, parents, nonlinear in feature_sets:
        for factor in range(z.shape[1]):
            model, params, cv_mse = fit_probe(values[labels == 0], z[labels == 0, factor],
                                             config['probe'], nonlinear)
            record = dict(stage=stage, representation=kind, layer=layer, scope=scope,
                          latent='Z'+str(factor+1), parents=parents,
                          n_train=int((labels == 0).sum()), n_validation=int((labels == 1).sum()),
                          n_test=int(test.sum()), feature_count=values.shape[1],
                          varying_features=int(varying_columns(values[labels == 0]).sum()),
                          cv_mse=cv_mse, best_params=json.dumps(params, sort_keys=True))
            record.update(evaluate_probe(model, values, z[:, factor], labels))
            metrics.append(record)
            filename = '{}_{}_L{}_{}_Z{}.pkl'.format(stage, kind, layer, scope, factor+1)
            with (output/'probes'/filename).open('wb') as stream:
                pickle.dump(model, stream)
            pred = model.predict(values[test])
            for row, prediction in zip(np.flatnonzero(test), pred):
                predictions.append(dict(stage=stage, representation=kind, layer=layer, scope=scope,
                                         latent='Z'+str(factor+1), sample_index=int(ids[row]),
                                         true_z=float(z[row, factor]), prediction=float(prediction)))
        print('Probe complete:', stage, kind, layer, scope, flush=True)
    write_csv(output/'metrics.csv', metrics)
    write_csv(output/'test_predictions.csv', predictions)
    write_csv(output/'hidden_unit_statistics.csv', units)
    write_csv(output/'hidden_z_correlations.csv', correlation_rows)
    write_csv(output/'pca_coordinates.csv', pca_rows)
    write_csv(output/'pca_summary.csv', pca_summary)
    graphs = graph_analysis(root, stage_paths, observed, latent, output)
    groups = []
    for stage in ('train', 'cam', 'random_train_mask', 'random_cam_mask'):
        for factor in range(z.shape[1]):
            for layer in range(1, config['training']['num_layers']+1):
                for condition, has_parents in [('no_parents', False), ('has_parents', True)]:
                    rows = [r for r in metrics if r['stage'] == stage and r['latent'] == 'Z'+str(factor+1)
                            and r['layer'] == layer and r['scope'] != 'concat'
                            and (r['parents'] > 0) == has_parents]
                    groups.append(dict(stage=stage, latent='Z'+str(factor+1), layer=layer,
                                        group=condition, n_mlps=len(rows),
                                        mean_test_r2=float(np.mean([r['test_r2'] for r in rows])) if rows else np.nan,
                                        mean_test_mse=float(np.mean([r['test_mse'] for r in rows])) if rows else np.nan))
    write_csv(output/'parent_group_metrics.csv', groups)
    reference = gaussian_reference(config, root)
    (output/'gaussian_reference.json').write_text(json.dumps(reference, indent=2))
    evaluation_status = dict(config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                              probe_fit='train only', probe_selection='5-fold CV within train only',
                              test_used_for_selection=False,
                              python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                              metric_rows=len(metrics), final_test_rows=int(test.sum()),
                              note='Single generated dataset/model seed; exploratory comparisons, not latent discovery')
    (output/'evaluation_status.json').write_text(json.dumps(evaluation_status, indent=2))
    from .visualize import make_figures, make_report
    make_figures(output/'figures', x, z, labels, hidden, masks, observed, latent,
                 metrics, predictions, correlation_rows, pca_concat)
    make_report(root, config, metrics, graphs, units, reference)
    print('Evaluation complete:', output/'report.md', flush=True)
    return metrics


def gaussian_reference(config, root):
    data = config['data']
    dist = data['latent_distribution']
    if (dist['type'] != 'normal' or data['latent_function'] != 'identity' or
            data['observed_function'] != 'identity'):
        return {'applicable': False, 'reason': 'Reference is for the implemented linear Gaussian SEM only'}
    k, d = data['n_latent'], data['n_observed']
    std_z = np.broadcast_to(np.asarray(dist.get('std', 1.)), (k,))
    noise = np.broadcast_to(np.asarray(data['noise_std']), (d,))
    if min(std_z.min(), noise.min()) <= 0:
        return {'applicable': False, 'reason': 'Strictly positive latent/noise variances required'}
    a = np.load(str(root/'data/latent_coefficients.npy'))
    covariance = np.linalg.inv(np.diag(1/std_z**2) + (a/noise**2)@a.T)
    return {'applicable': True, 'population_posterior_covariance': covariance.tolist(),
            'population_bayes_mse_per_z': np.diag(covariance).tolist(),
            'population_bayes_r2_per_z': (1-np.diag(covariance)/std_z**2).tolist(),
            'note': 'Population reference, not measured test results; assumes known true linear SEM'}
