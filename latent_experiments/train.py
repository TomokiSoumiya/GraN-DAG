"""X-only training worker. This module never reads true_Z.npy.

The unchanged original train/to_dag/cam_pruning functions see train+validation
X only. Full observed_X is opened for frozen inference AFTER fitting finishes.
"""
import argparse
import copy
import hashlib
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import torch

from main import build_parser
from gran_dag.data import DataManagerFile
from gran_dag.dag_optim import compute_constraint
from gran_dag.hidden import save_hidden_activations, load_checkpoint_cpu
from gran_dag.models.learnables import LearnableModel_NonLinGauss, LearnableModel_NonLinGaussANM
from gran_dag.train import train, to_dag, cam_pruning


def dump(path, value):
    with Path(path).open('wb') as stream:
        pickle.dump(value, stream)


def make_options(root, settings):
    opt = build_parser().parse_args([
        '--model', settings['model'], '--data-path', str(root/'training_input'),
        '--exp-path', str(root), '--i-dataset', '1', '--num-vars', str(settings['n_observed'])])
    for key in ('num_layers', 'hid_dim', 'nonlin', 'optimizer', 'lr', 'lr_reinit',
                'num_train_iter', 'train_batch_size', 'h_threshold', 'normalize_data'):
        setattr(opt, key, settings[key])
    opt.random_seed = settings['seed']
    opt.train_samples, opt.test_samples = settings['n_train'], settings['n_validation']
    opt.train, opt.to_dag, opt.cam_pruning = True, True, True
    opt.cam_pruning_cutoff = [settings['cam_pruning_cutoff']]
    opt.no_w_adjs_log = True
    return opt


def train_experiment(root):
    root = Path(root).resolve()
    settings = json.loads((root/'training_options.json').read_text())
    torch.set_num_threads(settings['torch_num_threads'])
    torch.set_default_tensor_type('torch.DoubleTensor')
    torch.manual_seed(settings['seed'])
    np.random.seed(settings['seed'])
    opt = make_options(root, settings)
    if opt.train_batch_size > opt.train_samples:
        raise ValueError('train_batch_size must not exceed the number of training rows')
    model_type = {'NonLinGaussANM': LearnableModel_NonLinGaussANM,
                  'NonLinGauss': LearnableModel_NonLinGauss}[opt.model]
    initial = model_type(opt.num_vars, opt.num_layers, opt.hid_dim, nonlin=opt.nonlin,
                         norm_prod=opt.norm_prod, square_prod=opt.square_prod)
    random_dir = root/'random_full'
    random_dir.mkdir(exist_ok=True)
    initial_path = random_dir/'model.pkl'
    if initial_path.exists():
        # A resume must preserve the actual original initial weights.
        initial = load_checkpoint_cpu(str(initial_path))
    else:
        dump(initial_path, initial)
        dump(random_dir/'opt.pkl', opt)
    model = copy.deepcopy(initial)
    train_data = DataManagerFile(opt.data_path, '1', opt.train_samples, opt.test_samples,
                                 normalize=opt.normalize_data, random_seed=opt.random_seed)
    validation_data = DataManagerFile(opt.data_path, '1', opt.train_samples, opt.test_samples,
                                      train=False, normalize=opt.normalize_data,
                                      mean=train_data.mean, std=train_data.std, random_seed=opt.random_seed)
    pool_ids = np.load(str(root/'training_input/sample_indices.npy'))
    split = np.load(str(root/'data/sample_split.npy'))
    actual_train = pool_ids[train_data.sample_indices]
    actual_validation = pool_ids[validation_data.sample_indices]
    assert np.all(split[actual_train] == 0) and np.all(split[actual_validation] == 1)
    assert not np.isin(pool_ids, np.flatnonzero(split == 2)).any()
    assert np.load(str(root/'training_input/data1.npy')).shape[1] == opt.num_vars
    assert sorted(p.name for p in (root/'training_input').iterdir()) == ['DAG1.npy', 'data1.npy', 'sample_indices.npy']
    np.save(str(root/'normalization_mean.npy'),
            train_data.mean.cpu().numpy() if opt.normalize_data else np.zeros((1, opt.num_vars)))
    np.save(str(root/'normalization_std.npy'),
            train_data.std.cpu().numpy() if opt.normalize_data else np.ones((1, opt.num_vars)))
    if opt.normalize_data and (not torch.isfinite(train_data.std).all().item() or
                               (train_data.std <= 0).any().item()):
        raise ValueError('Invalid normalization statistics')

    def callback(stage, step, metrics, throttle=None):
        safe = {key: float(value) for key, value in metrics.items()}
        # Ground-truth graph metrics compare direct observed edges only. They
        # cannot score latent confounding or select hyperparameters here.
        with (root/'training_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'stage': stage, 'step': step, 'metrics': safe})+'\n')
        if stage != 'train' or step % 1000 == 0:
            print(stage, step, 'h=', safe.get('constraint_violation'),
                  'nll-val=', safe.get('nll-val'), flush=True)

    start = time.time()
    previous_status = (json.loads((root/'training_status.json').read_text())
                       if (root/'training_status.json').exists() else {})
    already_complete = (root/'train/DAG.npy').exists()
    trained = train(model, train_data.adjacency.cpu().numpy(), train_data,
                    validation_data, opt, callback, None)
    converged = trained is not None and (not already_complete or previous_status.get('converged', True))
    if trained is None:
        trained = model
        # Save an explicitly marked terminal model for diagnostics/resume. Do
        # not pass it off as a converged augmented-Lagrangian solution.
        terminal = root/'unconverged'
        terminal.mkdir(exist_ok=True)
        dump(terminal/'model.pkl', model)
        dump(terminal/'opt.pkl', opt)
        (root/'training_status.json').write_text(json.dumps({
            'converged': False, 'iteration_limit': opt.num_train_iter,
            'final_h': float(compute_constraint(model, model.get_w_adj()).item())}, indent=2))
        if settings['require_convergence']:
            raise RuntimeError('GraN-DAG hit iteration limit; unconverged checkpoint saved. '
                               'Use a NEW run/config with more iterations; no converged results reported.')
        (root/'train').mkdir(exist_ok=True)
        dump(root/'train/model.pkl', model)
        dump(root/'train/opt.pkl', opt)
        np.save(str(root/'train/DAG.npy'), model.adjacency.detach().cpu().numpy())
    dag_model = to_dag(copy.deepcopy(trained), train_data, validation_data, opt, callback, None)
    pruned = cam_pruning(copy.deepcopy(dag_model), train_data, validation_data, opt,
                         callback, None, cutoff=settings['cam_pruning_cutoff'])
    pruning_stage = 'cam-pruning/cutoff_%.6f' % settings['cam_pruning_cutoff']

    # Fitting and graph selection are now complete. Only frozen inference sees
    # final-test X. The ground-truth latent labels are still never loaded here.
    raw = np.load(str(root/'data/observed_X.npy'))
    data = torch.as_tensor(raw).type(torch.Tensor)
    if opt.normalize_data:
        data = (data-train_data.mean)/train_data.std
    mean = train_data.mean.cpu().numpy() if opt.normalize_data else np.zeros((1, opt.num_vars))
    std = train_data.std.cpu().numpy() if opt.normalize_data else np.ones((1, opt.num_vars))
    stages = {'train': trained, 'to-dag': dag_model, pruning_stage: pruned,
              'random_full': initial}
    for stage, mask in [('random_train_mask', trained.adjacency),
                        ('random_cam_mask', pruned.adjacency)]:
        random_model = copy.deepcopy(initial)
        random_model.adjacency = mask.detach().clone()
        assert all(torch.equal(a, b) for a, b in zip(initial.parameters(), random_model.parameters()))
        directory = root/stage
        directory.mkdir(exist_ok=True)
        dump(directory/'model.pkl', random_model)
        dump(directory/'opt.pkl', opt)
        stages[stage] = random_model
    stage_manifest = {}
    for stage, stage_model in stages.items():
        checkpoint = root/stage/'model.pkl'
        directory = root/stage/'hidden'
        save_hidden_activations(
            stage_model, data, str(directory), batch_size=256,
            sample_indices=np.arange(len(raw)), sample_split=split,
            mean=mean, std=std, overwrite=(directory/'metadata.json').exists(),
            metadata={'stage': stage, 'checkpoint': str(checkpoint),
                      'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      'source_data': str(root/'data/observed_X.npy'),
                      'normalized': opt.normalize_data,
                      'preprocessing_fit_split': 'train only; original DataManagerFile',
                      'graph_selection_splits': ['train', 'validation'],
                      'row_order': 'original observed_X/true_Z sample order',
                      'split': 'all original rows; final-test inference only after fitting'})
        stage_manifest[stage] = {'hidden_dir': str(directory),
                                 'edges': int(stage_model.adjacency.sum().item())}
    history = root/'train/aug-lagrangians.pkl'
    if history.exists():
        with history.open('rb') as stream:
            iterations = len(pickle.load(stream))
    else:
        iterations = opt.num_train_iter
    status = {
        'converged': converged, 'iterations': iterations, 'resumed_training_cache': already_complete,
        'worker_elapsed_seconds': time.time()-start,
        'original_worker_elapsed_seconds': previous_status.get(
            'original_worker_elapsed_seconds', previous_status.get('worker_elapsed_seconds', time.time()-start)),
        'final_h': float(compute_constraint(trained, trained.get_w_adj()).item()),
        'n_train': len(actual_train), 'n_validation': len(actual_validation),
        'n_final_test': int((split == 2).sum()), 'training_input_columns': opt.num_vars,
        'latent_labels_used_by_grandag': False, 'final_test_used_by_grandag': False,
        'cam_pruning_input_rows': len(pool_ids),
        'cam_pruning_retrains_nn': False,
        'stages': stage_manifest,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'torch': torch.__version__},
        'source_sha256': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for path in ['gran_dag/train.py', 'gran_dag/models/base_model.py',
                                       'gran_dag/hidden.py', 'latent_experiments/train.py']},
    }
    (root/'training_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('Training/exports complete:', json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    train_experiment(parser.parse_args().run_dir)
