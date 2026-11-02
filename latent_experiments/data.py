"""Configurable independent latent variables and an observed acyclic SEM."""
import json
from pathlib import Path

import numpy as np


FUNCTIONS = {
    'identity': lambda x: x,
    'tanh': np.tanh,
    'sin': np.sin,
    'square': np.square,
    'cubic': lambda x: x**3,
}


def topological_order(coefficients):
    graph = np.asarray(coefficients) != 0
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError('observed_coefficients must be a square source-row/target-column matrix')
    if np.diag(graph).any():
        raise ValueError('Observed self-edges are not allowed')
    remaining, result = set(range(len(graph))), []
    while remaining:
        roots = sorted(j for j in remaining if not any(graph[i, j] for i in remaining))
        if not roots:
            raise ValueError('Observed coefficients contain a directed cycle')
        result.extend(roots)
        remaining.difference_update(roots)
    return result


def observed_coefficients(config):
    d = int(config['n_observed'])
    if config.get('observed_coefficients') is not None:
        matrix = np.asarray(config['observed_coefficients'], dtype=float)
    else:
        graph = config.get('observed_graph', {'type': 'empty'})
        matrix = np.zeros((d, d))
        if graph['type'] == 'erdos_renyi':
            rng = np.random.RandomState(graph.get('seed', config['seed']))
            order = rng.permutation(d)
            low, high = graph.get('weight_range', [0.3, 0.8])
            probability = graph.get('edge_probability', 0.2)
            if not 0 <= probability <= 1 or not 0 < low <= high:
                raise ValueError('Invalid random-DAG probability/weight range')
            for i in range(d):
                for j in range(i+1, d):
                    if rng.uniform() < probability:
                        matrix[order[i], order[j]] = rng.uniform(low, high) * rng.choice([-1., 1.])
        elif graph['type'] != 'empty':
            raise ValueError('observed_graph.type must be empty or erdos_renyi')
    if matrix.shape != (d, d) or not np.isfinite(matrix).all():
        raise ValueError('Observed coefficient shape/values are invalid')
    topological_order(matrix)
    return matrix


def generate_arrays(config, signal_fit_indices=None):
    n, d, k = [int(config[key]) for key in ('n_samples', 'n_observed', 'n_latent')]
    if min(n, d, k) <= 0:
        raise ValueError('Sample/observed/latent counts must be positive')
    latent = np.asarray(config['latent_coefficients'], dtype=float)
    if latent.shape != (k, d) or not np.isfinite(latent).all():
        raise ValueError('latent_coefficients must have shape (n_latent, n_observed)')
    observed = observed_coefficients(config)
    std = np.broadcast_to(np.asarray(config.get('noise_std', 1.), dtype=float), (d,))
    if not np.isfinite(std).all() or (std < 0).any():
        raise ValueError('Noise standard deviations must be finite and nonnegative')
    rng = np.random.RandomState(config['seed'])
    dist = config.get('latent_distribution', {'type': 'normal'})
    if dist['type'] == 'normal':
        z = rng.normal(loc=dist.get('mean', 0.), scale=dist.get('std', 1.), size=(n, k))
    elif dist['type'] == 'uniform':
        low, high = np.asarray(dist.get('low', -1.)), np.asarray(dist.get('high', 1.))
        if np.any(low >= high):
            raise ValueError('Uniform latent distribution requires low < high')
        z = rng.uniform(low, high, size=(n, k))
    elif dist['type'] == 'laplace':
        z = rng.laplace(dist.get('loc', 0.), dist.get('scale', 1.), size=(n, k))
    else:
        raise ValueError('Latent distribution must be normal, uniform, or laplace')
    try:
        latent_function = FUNCTIONS[config.get('latent_function', 'identity')]
        observed_function = FUNCTIONS[config.get('observed_function', 'identity')]
        # Optional target-specific transformations; absent means the original
        # shared transformation and RNG sequence remain exactly unchanged.
        names = config.get('latent_functions')
        if names is not None and (not isinstance(names, list) or len(names) != d):
            raise ValueError('latent_functions must list one function per observed variable')
        target_functions = [FUNCTIONS[name] for name in names] if names is not None else None
    except KeyError as error:
        raise ValueError('Unknown generation function: {}'.format(error))
    noise = rng.normal(size=(n, d)) * std
    x = np.zeros((n, d))
    if target_functions is None:
        latent_effect = latent_function(z) @ latent
    else:
        latent_effect = np.column_stack([
            function(z) @ latent[:, target]
            for target, function in enumerate(target_functions)])
    signal_arrays = {}
    normalization = config.get('signal_normalization')
    if normalization is not None:
        if normalization != {'fit_split': 'train', 'ddof': 0} or signal_fit_indices is None:
            raise ValueError('Signal normalization requires explicit training IDs and fit_split=train, ddof=0')
        indices = np.asarray(signal_fit_indices)
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) < 2 or
                len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= n).any()):
            raise ValueError('Invalid signal-normalization training IDs')
        raw_signal = latent_effect.copy()
        signal_mean = raw_signal[indices].mean(0)
        signal_std = raw_signal[indices].std(0, ddof=0)
        if not np.isfinite(signal_std).all() or (signal_std <= 0).any():
            raise ValueError('Signal training standard deviations must be positive and finite')
        latent_effect = (raw_signal-signal_mean)/signal_std
        signal_arrays = dict(raw_signal=raw_signal, normalized_signal=latent_effect.copy(),
                             signal_mean=signal_mean, signal_std=signal_std)
    for target in topological_order(observed):
        parents = np.flatnonzero(observed[:, target])
        x[:, target] = latent_effect[:, target] + noise[:, target]
        if len(parents):
            x[:, target] += observed_function(x[:, parents]) @ observed[parents, target]
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('Nonfinite generated data; adjust SEM coefficients/functions')
    result = {'observed_X': x, 'true_Z': z, 'noise': noise,
            'observed_adjacency': (observed != 0).astype(np.int64),
            'latent_adjacency': (latent != 0).astype(np.int64),
            'observed_coefficients': observed, 'latent_coefficients': latent,
            'sample_indices': np.arange(n, dtype=np.int64)}
    result.update(signal_arrays)
    return result


def split_indices(n, split_config, training_seed):
    fractions = [float(split_config[name]) for name in ('train', 'validation', 'test')]
    if min(fractions) <= 0 or not np.isclose(sum(fractions), 1.):
        raise ValueError('Positive train/validation/test fractions must sum to one')
    n_train, n_validation = int(n*fractions[0]), int(n*fractions[1])
    n_test = n-n_train-n_validation
    if min(n_train, n_validation, n_test) < 2:
        raise ValueError('Each split must have at least two rows')
    permutation = np.random.RandomState(split_config['seed']).permutation(n)
    final_test = np.sort(permutation[:n_test])
    pool = np.sort(permutation[n_test:])
    # Match the shuffle performed by the original DataManagerFile exactly.
    shuffled_pool = np.random.RandomState(training_seed).permutation(len(pool))
    train = pool[shuffled_pool[:n_train]]
    validation = pool[shuffled_pool[n_train:]]
    labels = np.full(n, 2, dtype=np.int8)
    labels[train], labels[validation] = 0, 1
    return {'train': train, 'validation': validation, 'test': final_test,
            'pool': pool, 'labels': labels}


def save_dataset(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    splits = split_indices(int(config['data']['n_samples']), config['split'], config['training']['seed'])
    arrays = generate_arrays(config['data'], signal_fit_indices=splits['train'])
    for name, array in arrays.items():
        np.save(str(directory / (name+'.npy')), array)
    np.save(str(directory / 'sample_split.npy'), splits['labels'])
    for name in ('train', 'validation', 'test', 'pool'):
        np.save(str(directory / (name+'_indices.npy')), splits[name])
    data_config = dict(config['data'])
    data_config['resolved_observed_coefficients'] = arrays['observed_coefficients'].tolist()
    data_config['adjacency_convention'] = 'source row -> target column; latent adjacency shape=(K,D)'
    data_config['equation'] = 'Xi=sum_k A[k,i]*f(Zk)+sum_j B[j,i]*g(Xj)+epsilon_i'
    if config['data'].get('latent_functions') is not None:
        data_config['equation'] = 'Xi=sum_k A[k,i]*f_i(Zk)+sum_j B[j,i]*g(Xj)+epsilon_i; f_i=latent_functions[i]'
    if config['data'].get('signal_normalization') is not None:
        data_config['equation'] = 'Xi=(sum_k A[k,i]*f_i(Zk)-signal_mean[i])/signal_std[i]+sum_j B[j,i]*g(Xj)+epsilon_i'
        data_config['resolved_signal_normalization'] = dict(
            mean=arrays['signal_mean'].tolist(), std=arrays['signal_std'].tolist(),
            scale=(1/arrays['signal_std']).tolist(), offset=(-arrays['signal_mean']/arrays['signal_std']).tolist(),
            fit_split='train', n_fit=len(splits['train']), ddof=0,
            fit_indices_file='train_indices.npy', Z_used_only_for_data_generation=True)
    data_config['causal_warning'] = 'Observed direct DAG omits unobserved common causes and is not the full marginal causal graph.'
    (directory / 'config.json').write_text(json.dumps(data_config, indent=2), encoding='utf-8')
    pool_dir = directory.parent / 'training_input'
    pool_dir.mkdir(exist_ok=False)
    # No Z or final-test X in this directory. DAG1 is for descriptive logging only.
    np.save(str(pool_dir / 'data1.npy'), arrays['observed_X'][splits['pool']])
    np.save(str(pool_dir / 'DAG1.npy'), arrays['observed_adjacency'])
    np.save(str(pool_dir / 'sample_indices.npy'), splits['pool'])
    return splits
