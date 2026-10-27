"""Label-free PCA/FA extraction and a strictly subsequent, read-only Z evaluation.

No GraN-DAG or latent-label regressor is trained here. Reuse frozen feature
verification from additional.py; its supervised entry points are never called.
"""
import argparse
import builtins
from contextlib import contextmanager
import io
import json
import os
import pickle
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from scipy.stats import rankdata
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

from analyze_hidden import correlations, varying_columns
from .additional import (controlled_features, digest, fresh_output, read_csv, snapshot)
from .evaluate import write_csv


METHODS = ('pca', 'factor_analysis')
SPLITS = [('train', 0), ('validation', 1), ('test', 2)]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def file_inventory(root, excluded=None):
    """Preservation check WITHOUT reading old Z or supervised prediction files."""
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and (excluded is None or excluded not in path.parents):
            stat = path.stat()
            result[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return result


@contextmanager
def forbid_latent_reads(root):
    """Fail closed on Python reads of Z or old label-containing evaluation data.

    This is an auditable guard against accidental reads, not a security sandbox.
    Extraction also runs in its own process, with no latent argument to fit APIs.
    """
    root = Path(root).resolve()
    z_path = root/'data/true_Z.npy'
    prohibited = [root/'analysis', root/'additional_analysis']
    audit = dict(blocked_attempts=[], policy='no true_Z or old supervised-result reads during extraction')
    originals = (builtins.open, io.open, os.open)

    def guard(original):
        def opened(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                resolved = Path(os.fsdecode(path)).resolve()
                if (resolved == z_path or resolved.name.startswith('true_Z.') or
                        any(p == resolved or p in resolved.parents for p in prohibited)):
                    audit['blocked_attempts'].append(str(resolved))
                    raise PermissionError('Latent/supervised data are forbidden during extraction: '+str(resolved))
            return original(path, *args, **kwargs)
        return opened
    builtins.open, io.open, os.open = [guard(original) for original in originals]
    try:
        yield audit
    finally:
        builtins.open, io.open, os.open = originals


def fit_extractor(features_train, x_train, method, settings):
    """Accepts only training features and observed X. No Z parameter exists."""
    features_train = np.asarray(features_train, dtype=np.float64)
    x_train = np.asarray(x_train, dtype=np.float64)
    if (features_train.ndim != 2 or x_train.ndim != 2 or len(features_train) != len(x_train)
            or not np.isfinite(features_train).all() or not np.isfinite(x_train).all()):
        raise ValueError('Finite aligned training matrices are required')
    if settings['n_factors'] != 1 or method not in METHODS:
        raise ValueError('This initial experiment fixes ONE factor and PCA/FA a priori')
    keep = varying_columns(features_train)
    bundle = dict(method=method, keep=keep, feature_mean=features_train.mean(0),
                  scaler=None, extractor=None, warnings=[], n_train=len(x_train), n_factors_requested=1,
                  x_mean=x_train.mean(0), x_scale=x_train.std(0))
    bundle['x_scale'][bundle['x_scale'] == 0] = 1
    if keep.any():
        scaler = StandardScaler().fit(features_train[:, keep])
        transformed = scaler.transform(features_train[:, keep])
        if method == 'pca':
            extractor = PCA(n_components=1, svd_solver=settings['pca']['svd_solver'])
        else:
            cfg = settings['factor_analysis']
            extractor = FactorAnalysis(n_components=1, svd_method=cfg['svd_method'],
                                       max_iter=cfg['max_iter'], tol=cfg['tol'], random_state=settings['seed'])
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            extractor.fit(transformed)
        bundle.update(scaler=scaler, extractor=extractor,
                      warnings=[str(message.message) for message in messages])
        factors = extractor.transform(transformed)
    else:
        # No arbitrary component is invented from constant inputs.
        factors = np.zeros((len(x_train), 1))
    design = np.column_stack([np.ones(len(x_train)), factors])
    # Reconstruction of observed X, NOT regression on Z. No CV/hyperparameter choice.
    bundle['x_decoder'] = np.linalg.lstsq(design, x_train, rcond=None)[0]
    return bundle


def transform_extractor(bundle, features):
    values = np.asarray(features, dtype=np.float64)
    keep = bundle['keep']
    if values.ndim != 2 or values.shape[1] != len(keep) or not np.isfinite(values).all():
        raise ValueError('Invalid frozen-extractor input')
    reconstructed = np.broadcast_to(bundle['feature_mean'], values.shape).copy()
    if keep.any():
        scaled = bundle['scaler'].transform(values[:, keep])
        model = bundle['extractor']
        factors = model.transform(scaled)
        # PCA inverse scores; FA posterior-mean factor reconstruction, no noise draw.
        scaled_reconstructed = factors @ model.components_ + model.mean_
        reconstructed[:, keep] = bundle['scaler'].inverse_transform(scaled_reconstructed)
    else:
        factors = np.zeros((len(values), 1))
    design = np.column_stack([np.ones(len(values)), factors])
    return factors, reconstructed, design @ bundle['x_decoder']


def load_features(root, output, settings):
    """Reuse exact model/activation/mask/normalization checks; never read Z."""
    config = json.loads((root/'config.json').read_text())
    x = np.load(str(root/'data/observed_X.npy'))
    ids = np.load(str(root/'data/sample_indices.npy'))
    labels = np.load(str(root/'data/sample_split.npy'))
    if (x.ndim != 2 or not np.isfinite(x).all() or ids.shape != (len(x),) or labels.shape != (len(x),)
            or len(np.unique(ids)) != len(ids) or set(labels.tolist()) != {0, 1, 2}):
        raise ValueError('Invalid observed samples/IDs/splits')
    # This repository's source arrays have original row IDs 0..N-1.
    np.testing.assert_array_equal(ids, np.arange(len(x)))
    for name, code in SPLITS:
        expected = np.load(str(root/'data'/(name+'_indices.npy')))
        np.testing.assert_array_equal(np.sort(expected), ids[labels == code])
    pool = np.load(str(root/'training_input/sample_indices.npy'))
    if np.any(labels[pool] == 2):
        raise ValueError('Final-test rows entered original GraN-DAG input')
    np.testing.assert_array_equal(np.load(str(root/'training_input/data1.npy')), x[pool])
    registry, hidden, verification = controlled_features(root, output, config, x, ids, labels)
    if not verification['all_random_parameters_equal_initial']:
        raise ValueError('Saved random weights do not match the actual initial model')
    lookup = {c['condition']: c for c in registry}
    for a, b in [('B','C'), ('D','E')]:
        if lookup[a]['mask_sha256'] != lookup[b]['mask_sha256']:
            raise ValueError('Random/learned masks differ')
    features = [('X', x, dict(condition='X', weights='none', mask='none', layer=0,
                              feature_kind='observed_X', source='data/observed_X.npy',
                              columns=['X'+str(j+1) for j in range(x.shape[1])]))]
    if settings['condition_codes'] != ['C', 'E', 'B', 'D']:
        raise ValueError('Fixed comparison conditions C/E/B/D are required; no Z-driven condition selection')
    for code in settings['condition_codes']:
        if len(hidden[code]) < 2:
            raise ValueError('Both first and second hidden layers are required')
        for layer in (1, 2):
            h = hidden[code][layer-1]
            info = dict(lookup[code], layer=layer, feature_kind='hidden',
                        source=lookup[code]['hidden_directory']+'/hidden_layer_%03d.npy'%layer,
                        columns=['X{}_unit{:03d}'.format(j+1,k+1)
                                 for j in range(h.shape[1]) for k in range(h.shape[2])])
            features.append((code+'_L'+str(layer), h.reshape(len(x), -1), info))
    return features, x, ids, labels, verification


def extract_experiment(root, output, settings):
    """Complete ALL fits/transforms before any true_Z read is permitted."""
    root = Path(root).resolve()
    output = fresh_output(root, output)
    before = file_inventory(root)
    output.mkdir()
    for name in ('models', 'factors', 'reconstructions', 'figures'):
        (output/name).mkdir()
    started = time.time()
    with forbid_latent_reads(root) as audit:
        features, x, ids, labels, verified = load_features(root, output, settings)
        train = labels == 0
        # Hash only X/H/model-related inputs. In particular never hash/read true_Z here.
        input_files = ['config.json', 'data/observed_X.npy', 'data/sample_indices.npy',
                       'data/sample_split.npy', 'normalization_mean.npy', 'normalization_std.npy']
        for _, _, info in features:
            input_files.append(info['source'])
        source_hashes = {name: digest(root/name) for name in sorted(set(input_files))}
        rows, fit_metrics, selections, conditions, model_details = [], [], [], [], []
        for condition, values, info in features:
            conditions.append(dict(info, id=condition, input_dimension=values.shape[1]))
            threshold = 1e-10 * max(1., float(np.abs(values[train]).max(initial=0)))
            keep = varying_columns(values[train])
            for column, name in enumerate(info['columns']):
                selections.append(dict(condition=condition, column_index=column, column=name,
                                       train_mean=float(values[train,column].mean()),
                                       train_std=float(values[train,column].std()),
                                       constant_threshold=threshold, kept=bool(keep[column])))
            for method in METHODS:
                bundle = fit_extractor(values[train], x[train], method, settings)
                factors, reconstruction, x_hat = transform_extractor(bundle, values)
                filename = condition+'_'+method
                with (output/'models'/(filename+'.pkl')).open('wb') as stream:
                    pickle.dump(bundle, stream)
                np.save(str(output/'factors'/(filename+'.npy')), factors)
                np.save(str(output/'reconstructions'/(filename+'_X.npy')), x_hat)
                np.save(str(output/'reconstructions'/(filename+'_features.npy')), reconstruction)
                for index in range(len(x)):
                    rows.append(dict(condition=condition, method=method, sample_index=int(ids[index]),
                                     split=int(labels[index]), factor_1=float(factors[index,0])))
                model = bundle['extractor']
                detail = dict(condition=condition, method=method, input_dimension=values.shape[1],
                              used_dimension=int(keep.sum()), removed_constant_columns=int((~keep).sum()),
                              n_factors_requested=1, factor_shape=list(factors.shape),
                              n_nonconstant_factors=int(varying_columns(factors[train]).sum()),
                              all_constant_fallback=not bool(keep.any()),
                              warnings=bundle['warnings'],
                              model_file='models/'+filename+'.pkl', factor_file='factors/'+filename+'.npy',
                              fa_iterations=int(model.n_iter_) if method == 'factor_analysis' and model is not None else None,
                              pca_explained_variance_ratio=float(model.explained_variance_ratio_[0])
                                  if method == 'pca' and model is not None else None)
                model_details.append(detail)
                for split, code in SPLITS:
                    selected = labels == code
                    feature_std_mse = None
                    if keep.any():
                        delta = (values[selected][:,keep]-reconstruction[selected][:,keep])/bundle['scaler'].scale_
                        feature_std_mse = float(np.mean(delta**2))
                    fit_metrics.append(dict(condition=condition, method=method, split=split,
                        feature_kind=info['feature_kind'], input_dimension=values.shape[1],
                        used_dimension=int(keep.sum()), n_factors=1, n_samples=int(selected.sum()),
                        factor_variance=float(factors[selected,0].var(ddof=1)),
                        feature_reconstruction_raw_mse=float(np.mean((values[selected]-reconstruction[selected])**2)),
                        feature_reconstruction_standardized_mse_kept=feature_std_mse,
                        x_reconstruction_raw_mse=float(np.mean((x[selected]-x_hat[selected])**2)),
                        x_reconstruction_standardized_mse=float(np.mean(((x[selected]-x_hat[selected])/bundle['x_scale'])**2))))
                print('Unsupervised extraction:', condition, method, 'shape', factors.shape,
                      'kept', int(keep.sum()), flush=True)
        write_csv(output/'extracted_factors.csv', rows)
        write_csv(output/'fit_metrics.csv', fit_metrics)
        write_csv(output/'feature_selection.csv', selections)
        save_json(output/'metadata.json', dict(
            settings=settings, source_run=str(root), n_samples=len(x),
            splits={name:int((labels == code).sum()) for name,code in SPLITS},
            sample_ids_file='data/sample_indices.npy', split_file='data/sample_split.npy',
            split_codes={'0':'train','1':'validation','2':'final_test'},
            factor_order='first component/factor only; fixed before Z evaluation',
            factor_sign='original PCA/FA sign; never aligned to Z',
            factor_variance='sample variance ddof=1; PCA/FA score scales differ',
            standardization_ddof=0, feature_conditions=conditions, extractors=model_details,
            frozen_feature_verification=verified, input_sha256=source_hashes,
            z_used_in_extraction=False, z_used_for_selection=False, test_used_for_selection=False,
            x_decoder='OLS with intercept on train factors -> train raw X; no Z',
            reconstruction_distinction='feature MSE concerns X or H as labelled; x_reconstruction MSE always raw observed X',
            no_component_or_condition_selection=True, nn_retrained=False,
            phase_order=['extract_all_conditions_without_Z', 'persist_all_models_factors_reconstructions',
                         'separate_process_final_Z_evaluation'],
            versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__)))
        if before != file_inventory(root, excluded=output):
            raise RuntimeError('Existing file inventory changed during extraction')
        save_json(output/'preserved_source_inventory.json', before)
        save_json(output/'extraction_status.json', dict(
            completed=True, phase='label_free_extraction_complete', z_read_attempts=audit['blocked_attempts'],
            z_used_in_fit=False, fitted_on_split=0, factors_saved_before_z_read=True,
            factor_csv_rows=len(rows), extractor_count=len(model_details),
            source_inventory_unchanged=True, elapsed_seconds=time.time()-started))
        artifact_hashes = {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
        save_json(output/'extraction_manifest.json', artifact_hashes)
    print('ALL extraction complete WITHOUT Z:', output, flush=True)
    return output


def factor_correlations(factor, z):
    """Evaluation only. Undefined constant-vector correlations stay undefined."""
    if not varying_columns(factor[:, None]).any() or not varying_columns(z[:, None]).any():
        return dict(pearson_r=None, absolute_pearson_r=None, spearman_r=None,
                    correlation_defined=False)
    r = float(correlations(factor[:,None], z[:,None])[0,0])
    rho = float(correlations(rankdata(factor)[:,None], rankdata(z)[:,None])[0,0])
    return dict(pearson_r=r, absolute_pearson_r=abs(r), spearman_r=rho,
                correlation_defined=True)


def evaluate_experiment(root, output):
    """No fit occurs here; verify frozen extraction BEFORE opening true Z."""
    root, output = Path(root).resolve(), Path(output).resolve()
    status = json.loads((output/'extraction_status.json').read_text())
    if not status['completed'] or not status['factors_saved_before_z_read'] or status['z_read_attempts']:
        raise ValueError('Complete clean label-free extraction before evaluation')
    for filename in ('metrics.csv', 'report.md', 'evaluation_status.json'):
        if (output/filename).exists():
            raise FileExistsError('Evaluation output already exists: '+str(output/filename))
    frozen = json.loads((output/'extraction_manifest.json').read_text())
    for filename, expected in frozen.items():
        if digest(output/filename) != expected:
            raise ValueError('Frozen extraction changed: '+filename)
    metadata = json.loads((output/'metadata.json').read_text())
    for filename, expected in metadata['input_sha256'].items():
        if digest(root/filename) != expected:
            raise ValueError('Source feature changed: '+filename)
    before = snapshot(root, excluded=output)
    # FIRST permitted true_Z read: every factor/model is already persisted/frozen.
    z = np.load(str(root/'data/true_Z.npy'))
    ids = np.load(str(root/'data/sample_indices.npy'))
    labels = np.load(str(root/'data/sample_split.npy'))
    if z.ndim != 2 or len(z) != len(ids) or not np.isfinite(z).all():
        raise ValueError('Evaluation-only Z must have shape (samples, latent variables)')
    manifest = json.loads((root/'data_manifest.json').read_text())
    if digest(root/'data/true_Z.npy') != manifest['data_files']['true_Z.npy']:
        raise ValueError('True Z changed from original data manifest')
    fit_rows = read_csv(output/'fit_metrics.csv')
    extracted = read_csv(output/'extracted_factors.csv')
    factor_map = {}
    for entry in metadata['extractors']:
        key = (entry['condition'], entry['method'])
        values = np.load(str(output/entry['factor_file']))
        if values.shape != (len(ids), 1):
            raise ValueError('Incorrect factor array shape')
        factor_map[key] = values[:,0]
        rows = [r for r in extracted if (r['condition'],r['method']) == key]
        np.testing.assert_array_equal([int(r['sample_index']) for r in rows], ids)
        np.testing.assert_array_equal([int(r['split']) for r in rows], labels)
        np.testing.assert_array_equal([float(r['factor_1']) for r in rows], values[:,0])
    metrics = []
    for old in fit_rows:
        chosen = labels == dict(SPLITS)[old['split']]
        values = factor_map[(old['condition'],old['method'])][chosen]
        for factor in range(z.shape[1]):
            record = dict(old, latent='Z'+str(factor+1))
            record.update(factor_correlations(values, z[chosen,factor]))
            metrics.append(record)
    write_csv(output/'metrics.csv', metrics)
    comparisons = []
    test_lookup = {(r['condition'],r['method'],r['latent']):r for r in metrics if r['split'] == 'test'}
    for before_code, after_code, purpose in [
            ('B_L1','C_L1','learning_train_mask'), ('B_L2','C_L2','learning_train_mask'),
            ('D_L1','E_L1','learning_dag_mask'), ('D_L2','E_L2','learning_dag_mask'),
            ('C_L1','E_L1','dag_mask_same_learned_weights'), ('C_L2','E_L2','dag_mask_same_learned_weights'),
            ('C_L1','C_L2','layer_same_learned_weights'), ('E_L1','E_L2','layer_same_learned_weights')]:
        for method in METHODS:
            for factor in range(z.shape[1]):
                latent = 'Z'+str(factor+1)
                a, b = (test_lookup[(code,method,latent)] for code in (before_code,after_code))
                delta = (b['absolute_pearson_r']-a['absolute_pearson_r']
                         if a['correlation_defined'] and b['correlation_defined'] else None)
                comparisons.append(dict(comparison=purpose, before=before_code, after=after_code,
                                         method=method, latent=latent, delta_absolute_pearson_r=delta,
                                         delta_x_reconstruction_mse=float(b['x_reconstruction_raw_mse'])-
                                         float(a['x_reconstruction_raw_mse'])))
    write_csv(output/'comparisons.csv', comparisons)
    cleanup_rows = cleanup_inventory(root, output)
    make_figures(output, metadata, factor_map, metrics, z, labels)
    make_report(output, metadata, metrics, comparisons, cleanup_rows)
    if snapshot(root, excluded=output) != before:
        raise RuntimeError('Existing experiment content changed during evaluation')
    for filename, expected in frozen.items():
        if digest(output/filename) != expected:
            raise RuntimeError('Evaluation modified a fitted model or extracted factor')
    save_json(output/'evaluation_status.json', dict(
        completed=True, z_first_opened_after_all_extractors_frozen=True, z_used_in_fit=False,
        latent_regression_trained=False, hyperparameters_selected_using_z=False,
        frozen_extraction_unchanged=True, original_experiment_files_unchanged=True,
        original_experiment_sha256=before, metric_rows=len(metrics),
        primary_split='test', test_data_previously_inspected_in_legacy_studies=True,
        caution='exploratory evaluation; one factor assumed in advance, not causal latent discovery',
        source_sha256={str(p):digest(p) for p in [Path(__file__),Path('latent_unsupervised.py'),
                                                Path('configs/latent_unsupervised.json')]}))
    print('Evaluation-only Z correlations complete:', output, flush=True)
    return metrics


def cleanup_inventory(root, output):
    """List dependencies/archive candidates. No file is moved/deleted."""
    import subprocess
    repo = Path(__file__).resolve().parent.parent
    tracked = set(subprocess.check_output(['git','ls-files'], cwd=str(repo)).decode().splitlines())
    rows = []
    source_files = {
        'latent_experiments/evaluate.py': ('shared_keep', 'unsupervised.write_csv; additional.py; runner.py; regression tests'),
        'latent_experiments/additional.py': ('shared_keep', 'unsupervised.controlled_features and preservation helpers; nonlinear_study.py; tests'),
        'analyze_hidden.py': ('shared_keep', 'unsupervised.varying_columns/correlations; evaluate.py'),
        'latent_experiment.py': ('mixed_keep', 'artificial data generation/training entry point; nonlinear study launcher'),
        'latent_experiments/runner.py': ('mixed_keep', 'generation/training orchestration plus old supervised evaluation route'),
        'latent_experiments/data.py': ('required_keep', 'artificial data and sample splits'),
        'latent_experiments/train.py': ('required_keep', 'X-only original GraN-DAG training and frozen checkpoints'),
        'gran_dag/hidden.py': ('required_keep', 'hidden export/loading; controlled feature verification'),
        'extract_hidden.py': ('required_keep', 'checkpoint hidden extraction CLI'),
        'hidden_to_csv.py': ('required_keep', 'saved activation inspection without changing original arrays'),
        'latent_additional_analysis.py': ('legacy_optional_archive', 'old supervised CLI; historical reproduction; nonlinear_study.py invokes it'),
        'latent_nonlinear_study.py': ('legacy_optional_archive', 'historical five-seed supervised reproduction'),
        'latent_experiments/nonlinear_study.py': ('legacy_optional_archive', 'historical five-seed supervised aggregation; tests'),
        'tests/test_latent_experiment.py': ('mixed_keep', 'data generation/splits and legacy regression safeguards'),
        'tests/test_latent_additional.py': ('mixed_keep', 'shared feature/mask verification plus legacy probes'),
        'tests/test_nonlinear_study.py': ('legacy_optional_archive', 'historical multi-seed safeguards'),
    }
    for filename, (category, dependencies) in source_files.items():
        path = repo/filename
        if path.exists():
            rows.append(dict(path=filename, category=category, dependencies=dependencies,
                             git_tracked=filename in tracked, bytes=path.stat().st_size,
                             action='retained; no deletion', sha256=digest(path)))
    for dirname in ('analysis','additional_analysis'):
        for path in sorted((root/dirname).rglob('*')):
            if not path.is_file():
                continue
            name = str(path.relative_to(root))
            probe_only = ('probes' in path.parts or path.name in ('metrics.csv','test_predictions.csv',
                           'mlp_comparison.csv','paired_comparisons.csv','x_vs_hidden.csv') or
                          'probe_scores' in path.name or 'latent_predictions' in path.name)
            rows.append(dict(path=str(root.name)+'/'+name,
                             category='supervised_result_optional_archive' if probe_only else 'historical_mixed_result_keep',
                             dependencies='legacy evaluation/reproduction; not required for new factor fitting',
                             git_tracked=False, bytes=path.stat().st_size,
                             action='retained; ignored experiment files are NOT recoverable via Git', sha256=digest(path)))
    write_csv(output/'cleanup_inventory.csv', rows)
    candidates = [r for r in rows if 'optional_archive' in r['category']]
    save_json(output/'cleanup_status.json', dict(deleted_files=[], moved_files=[],
        optional_archive_candidate_count=len(candidates),
        reason='Shared helpers remain in legacy modules; ignored experiment results lack Git recovery. Preserve historical controls.',
        primary_workflow='latent_unsupervised.py; no Z regression is run'))
    text = ['# 既存コード・結果の整理', '',
            '今回は削除・移動しなかった。新しい標準入口はlatent_unsupervised.py。', '',
            '## 保持する依存関係', '',
            '- additional.pyのモデル/マスク/行対応検証と、evaluate.pyのCSV保存関数を再利用している。',
            '- 同じファイル内の教師あり回帰関数は新しい実験では呼ばない。既存モジュール全体の削除は新機能も壊す。',
            '- データ生成、GraN-DAG本体、学習済みモデル、隠れ層、分割、サンプルID、true_Zは保持する。', '',
            '## 任意のアーカイブ候補（未削除）', '',
            'cleanup_inventory.csvに個別ファイル、SHA256、依存関係、Git管理有無を記録した。',
            '- 教師ありZ回帰器、教師あり比較CSV・図、旧専用CLIは将来の整理候補。',
            '- ただし既存結果はexperiments/が.gitignore対象のためGitだけでは復元できない。',
            '- Git管理済みコードでも未コミット変更があり得る。Git管理有りは現状態の復元保証ではない。',
            '- 先に明示的なバックアップと共有ヘルパーの分離を行い、旧再現経路を壊さないことを確認してから整理する。', '',
            'この一覧は削除済みファイルの一覧ではなく、安全性を確認した保持・候補一覧である。', '']
    (output/'cleanup_plan.md').write_text('\n'.join(text), encoding='utf-8')
    return rows


def make_figures(output, metadata, factors, metrics, z, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions = [r['id'] for r in metadata['feature_conditions']]
    test = labels == 2
    figures = output/'figures'
    for factor in range(z.shape[1]):
        latent = 'Z'+str(factor+1)
        lookup = {(r['condition'],r['method']):r for r in metrics if r['split'] == 'test' and r['latent'] == latent}
        for method in METHODS:
            fig, axes = plt.subplots(3, 3, figsize=(13, 11))
            for ax, condition in zip(axes.flat, conditions):
                ax.scatter(z[test,factor], factors[(condition,method)][test], s=8, alpha=.55)
                r = lookup[(condition,method)]['pearson_r']
                ax.set_title(condition+' '+method+'; r='+('undefined' if r is None else '{:.3f}'.format(r)))
                ax.set_xlabel('True '+latent+' (evaluation only)')
                ax.set_ylabel('Extracted factor (unaligned sign)')
            fig.tight_layout()
            fig.savefig(str(figures/('scatter_'+method+'_'+latent+'.png')), dpi=150)
            plt.close(fig)
        for metric, filename, title in [
                ('absolute_pearson_r','absolute_correlations_'+latent,'Final-test |Pearson r| (no Z in fitting)'),
                ('x_reconstruction_raw_mse','X_reconstruction_'+latent,'Final-test raw X reconstruction MSE'),
                ('feature_reconstruction_standardized_mse_kept','feature_reconstruction_'+latent,
                 'Final-test standardized feature MSE (different feature spaces)')]:
            fig, ax = plt.subplots(figsize=(13, 4))
            for offset, method in zip((-.18,.18), METHODS):
                values = [lookup[(c,method)][metric] for c in conditions]
                ax.bar(np.arange(len(conditions))+offset, [float(v) if v not in (None,'') else np.nan for v in values],
                       width=.36, label=method)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=25)
            ax.set_title(title)
            ax.legend()
            if metric == 'absolute_pearson_r':
                ax.set_ylim(0,1)
            fig.tight_layout()
            fig.savefig(str(figures/(filename+'.png')), dpi=150)
            plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(13,7))
    # Label-free variances; different method score scales are intentionally separate.
    rows = read_csv(output/'fit_metrics.csv')
    for ax, method in zip(axes,METHODS):
        for offset, split in zip((-.18,.18),('train','test')):
            values = [next(float(r['factor_variance']) for r in rows if r['condition']==c and
                           r['method']==method and r['split']==split) for c in conditions]
            ax.bar(np.arange(len(conditions))+offset,values,width=.36,label=split)
        ax.set_xticks(range(len(conditions))); ax.set_xticklabels(conditions,rotation=20)
        ax.set_title(method+' score variance (not a measure of latent information)'); ax.legend()
    fig.tight_layout(); fig.savefig(str(figures/'factor_variances.png'),dpi=150); plt.close(fig)


def make_report(output, metadata, metrics, comparisons, cleanup_rows):
    details = {(r['condition'],r['method']):r for r in metadata['extractors']}
    lines = ['# 教師なし潜在因子抽出：既存線形データ', '',
             'NNは再学習せず、既存の学習済みモデル・隠れ層を検証して再利用した。'
             '抽出器・標準化・定数列判定・X再構成係数は学習{}件だけでfitした。'
             '検証{}件・最終テスト{}件にはtransformのみを適用した。'.format(
                 metadata['splits']['train'],metadata['splits']['validation'],metadata['splits']['test']), '',
             '## Zを使わない抽出', '',
             '- 全条件で因子数1を事前固定。PCAは第1主成分を固定し、Zとの相関による成分選択は行わない。',
             '- PCAは標準化特徴の分散最大方向。FAは線形共通因子＋特徴ごとの独立ノイズを仮定し、事後平均の因子得点を出力する。',
             '- Zと過去の教師あり結果の読み込みを禁止した抽出プロセスで全18モデル・全因子を保存し、別プロセスで初めてtrue_Zを読む。',
             '- 符号はZに合わせず、元の推定値を保存。Pearson rと絶対値・Spearmanを併記する。',
             '- 因子数の未知推定、Zによるハイパーパラメータ探索、最良条件の自動選択は未実装／未実施。', '',
             '## 比較条件', '',
             '|記号|重み|マスク|特徴|', '|---|---|---|---|',
             '|X|なし|なし|元の観測3変数|',
             '|C_L1/L2|学習済み|学習直後|全MLP連結30次元|',
             '|E_L1/L2|Cと同じ学習済み|DAG化後|全MLP連結30次元|',
             '|B_L1/L2|実際の学習前|Cと同じ学習直後マスク|全MLP連結30次元|',
             '|D_L1/L2|Bと同じ学習前|Eと同じDAG化後マスク|全MLP連結30次元|', '',
             '## 最終テストでの相関と再構成', '',
             '|条件|抽出器|使用列/入力列|Pearson r|絶対r|Spearman|因子分散|X再構成MSE|標準化特徴再構成MSE|',
             '|---|---|---|---|---|---|---|---|---|']
    def fmt(value):
        return '未定義' if value in (None,'') else '{:.6f}'.format(float(value))
    for row in metrics:
        if row['split'] != 'test':
            continue
        lines.append('|{}|{}|{}/{}|{}|{}|{}|{}|{}|{}|'.format(
            row['condition'],row['method'],row['used_dimension'],row['input_dimension'],
            fmt(row['pearson_r']),fmt(row['absolute_pearson_r']),fmt(row['spearman_r']),
            fmt(row['factor_variance']),fmt(row['x_reconstruction_raw_mse']),
            fmt(row['feature_reconstruction_standardized_mse_kept'])))
    lines += ['', '## 同じマスク・同じ重みでの比較', '',
              '以下は評価上の差分であり、この値を使ったモデルや条件の選択はしない。', '',
              '|比較|抽出器|絶対rの差(after−before)|X再構成MSEの差|', '|---|---|---|---|']
    for row in comparisons:
        lines.append('|{}→{}|{}|{}|{}|'.format(row['before'],row['after'],row['method'],
                     fmt(row['delta_absolute_pearson_r']),fmt(row['delta_x_reconstruction_mse'])))
    lines += ['', '## 定数特徴・分散・再構成の扱い', '',
              '- 学習行のみのstdで定数／ほぼ定数列を検出し、抽出器から除く。列名・閾値はfeature_selection.csv。',
              '- 除いた列の再構成は学習平均。全列一定なら因子を0固定し、相関は未定義とする。',
              '- PCA/FAの因子得点の尺度は異なるため、因子の分散が大きいことを情報量や品質の高さとみなさない。',
              '- 特徴再構成は各抽出器の線形生成部分によるもの。FAではノイズを加えず事後平均得点から再構成。',
              '- H再構成のMSEと元のX再構成のMSEは別指標。Hの生MSEを条件間で直接比較しない。標準化特徴の誤差も異なる特徴空間の誤差。',
              '- X再構成だけは全条件で同じ元Xを対象とし、因子→Xの固定OLS＋切片を学習用Xだけで学習する。Z回帰ではない。', '',
              '## 解釈の範囲と次の改善', '',
              '- 教師なしで共通変動を要約する因子を抽出したことと、因果的な潜在共通原因を特定したことは異なる。',
              '- 真のZとの対応は抽出後の人工データ評価であり、高相関でも因果的同定や潜在変数の自動検出の証明ではない。',
              '- 元のXや未学習NNも共通変動を保持する。GraN-DAGの利点は同一マスクの対照と元のXの両方と比較して判断する。',
              '- 因子数1は既知の実験設計として事前固定した。因子が存在するか／何個かを自動判定したわけではない。',
              '- 一つの線形データ・NNシードで、既に過去の研究で閲覧されたテストを再利用している。探索的結果であり一般的優位性は断定しない。',
              '- 次はZを使わない平行分析・検証再構成／尤度による因子数選択、ブートストラップ安定性・多シード、無潜在因子対照、非線形／複数因子条件を検討する（今回未実装）。',
              '- MLP内の重複ユニットはFAの独立残差仮定を満たさない場合がある。MLP間の共通変動を分離する多視点因子モデル等も今後の候補。',
              '- 因果的共通原因の同定には、追加の生成仮定や独立性検証、必要に応じて介入等の別の根拠が必要。', '',
              '## 整理と再現', '',
              '旧ファイルは削除・移動していない。共通ヘルパーの依存とGit管理外の実験結果の復元性を考慮し、'
              'cleanup_inventory.csvとcleanup_plan.mdに保持理由・任意のアーカイブ候補を一覧化した。', '',
              'models/*.pkl、factors/*.npy、extracted_factors.csv、抽出マニフェストを保存。'
              '評価は抽出結果を一切変更せず行う。メタデータとextraction_status.jsonに分割・固定条件・Z読み込み禁止の記録を保存した。', '',
              '実行入口: latent_unsupervised.py。既存のGraN-DAG損失・DAG制約・学習アルゴリズムは変更していない。', '']
    warnings_found = [r for r in details.values() if r['warnings']]
    if warnings_found:
        lines += ['## 抽出器の警告', '']
        for row in warnings_found:
            lines.append('- {} {}: {}'.format(row['condition'],row['method'],'; '.join(row['warnings'])))
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['extract','evaluate'])
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if args.stage == 'extract':
        extract_experiment(args.run_dir,args.output_dir,settings)
    else:
        evaluate_experiment(args.run_dir,args.output_dir)


if __name__ == '__main__':
    main()
