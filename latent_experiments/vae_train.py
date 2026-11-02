"""One training attempt per A/B; label-free inference is finalized before Z evaluation."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from latent_unsupervised import mapped
from .additional import array_digest, digest
from .evaluate import write_csv
from .unsupervised import file_inventory, forbid_latent_reads, save_json
from .vae import (epoch_orders, make_model, marginal_log_prob,
                  quadrature_terms, stochastic_terms)
from .vae_data import load_inputs


def state_digest(state):
    sha = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        sha.update(name.encode())
        sha.update(array_digest(tensor.detach().cpu().numpy()).encode())
    return sha.hexdigest()


def validate_config(config):
    fixed = dict(seed=201, hidden_layer=1, hidden_stage='train', latent_dim=1, width=16,
                 negative_slope=0.01, dtype='float64', optimizer='Adam', learning_rate=0.001,
                 batch_size=64, max_epochs=500, patience=30, min_delta=0.0,
                 kl_weight=1.0, train_latent_samples=1, quadrature_points=64,
                 quadrature_check_points=[32, 96])
    for key, value in fixed.items():
        if config[key] != value:
            raise ValueError('This single-comparison protocol fixes '+key)


def train_model(name, features, target, labels, config, output, orders):
    folder = output/name
    folder.mkdir()
    save_json(folder/'training_started.json', dict(attempt=1, time=time.time(), seed=config['seed']))
    model = make_model(features.shape[1], config)
    initial = copy.deepcopy(model.state_dict())
    torch.save(initial, str(folder/'initial_state.pt'))
    decoder_hash = state_digest({k: v for k, v in initial.items()
                                if k.startswith('decoder.') or k == 'raw_observation_std'})
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # Independent of encoder initialization: common training epsilon sequence.
    torch.manual_seed(config['seed'])
    train_ids, validation_ids = np.where(labels == 0)[0], np.where(labels == 1)[0]
    features = torch.as_tensor(features, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    started = time.monotonic()
    history, best, bad_epochs, best_epoch = [], -float('inf'), 0, None
    for epoch in range(config['max_epochs']):
        model.train()
        totals = np.zeros(3)
        for start in range(0, len(train_ids), config['batch_size']):
            idx = train_ids[orders[epoch, start:start+config['batch_size']]]
            optimizer.zero_grad()
            terms = stochastic_terms(model, features[idx], target[idx])
            loss = -terms[0].mean()
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError('Nonfinite training ELBO; no retry permitted')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
                   for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient; no retry permitted')
            optimizer.step()
            totals += [float(t.detach().sum().item()) for t in terms]
        model.eval()
        with torch.no_grad():
            val = quadrature_terms(model, features[validation_ids], target[validation_ids],
                                   config['quadrature_points'])
            train_mu, train_std = model.encode(features[train_ids])
            noise_variance = model.observation_std().numpy() ** 2
        score = float(val['elbo'].mean().item())
        if not np.isfinite(score) or not np.isfinite(noise_variance).all():
            raise FloatingPointError('Nonfinite validation score/noise variance')
        record = dict(epoch=epoch+1, train_mc_elbo=totals[0]/600,
                      train_mc_reconstruction=totals[1]/600, train_mc_kl=totals[2]/600,
                      validation_elbo=score,
                      validation_reconstruction=float(val['reconstruction'].mean().item()),
                      validation_kl=float(val['kl'].mean().item()),
                      train_mu_variance=float(np.var(train_mu.numpy())),
                      train_posterior_std_min=float(train_std.min().item()),
                      train_posterior_std_max=float(train_std.max().item()),
                      batch_order_sha256=array_digest(orders[epoch]), elapsed_seconds=time.monotonic()-started)
        record.update({'noise_variance_X'+str(j+1): float(v) for j, v in enumerate(noise_variance)})
        history.append(record)
        if score > best + config['min_delta']:
            best, best_epoch, bad_epochs = score, epoch+1, 0
            torch.save(dict(state_dict=model.state_dict(), input_dim=model.input_dim, epoch=best_epoch,
                            validation_elbo=best), str(folder/'best.pt'))
        else:
            bad_epochs += 1
        if (epoch+1) % 25 == 0 or bad_epochs >= config['patience']:
            print('{} epoch={} val_ELBO={:.6f} best_epoch={} patience={}'.format(
                name, epoch+1, score, best_epoch, bad_epochs), flush=True)
        if bad_epochs >= config['patience']:
            break
    elapsed = time.monotonic()-started
    write_csv(folder/'history.csv', history)
    torch.save(dict(state_dict=model.state_dict(), epoch=len(history)), str(folder/'last.pt'))
    checkpoint = torch.load(str(folder/'best.pt'), map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    status = dict(model=name, attempts=1, input_dimension=model.input_dim, best_epoch=best_epoch,
                  epochs_completed=len(history), best_validation_elbo=best,
                  stop_reason='patience' if bad_epochs >= config['patience'] else 'max_epochs',
                  optimization_convergence_guaranteed=False, elapsed_seconds=elapsed,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  encoder_parameters=sum(p.numel() for p in model.encoder.parameters())+
                                     sum(p.numel() for p in model.posterior_head.parameters()),
                  decoder_parameters=sum(p.numel() for p in model.decoder.parameters())+3,
                  initial_decoder_sha256=decoder_hash, best_checkpoint_sha256=digest(folder/'best.pt'),
                  batch_orders_used_sha256=array_digest(orders[:len(history)]),
                  training_noise_seed=config['seed'], nonfinite_detected=False)
    save_json(folder/'training_status.json', status)
    return model, status


def summarize_predictions(name, model, data, config, output, status):
    """Called only AFTER both training runs have fixed their checkpoints. No Z."""
    features = torch.as_tensor(data['features_'+name], dtype=torch.float64)
    target = torch.as_tensor(data['target'], dtype=torch.float64)
    points = config['quadrature_points']
    with torch.no_grad():
        terms = quadrature_terms(model, features, target, points)
        predictions = {key: value.numpy() for key, value in terms.items()}
        for order in config['quadrature_check_points'] + [points]:
            predictions['log_marginal_'+str(order)] = marginal_log_prob(model, target, order).numpy()
        # Accuracy diagnostics only. These orders never alter selection or the primary result.
        for order in config['quadrature_check_points']:
            predictions['elbo_check_'+str(order)] = quadrature_terms(model, features, target, order)['elbo'].numpy()
        noise = model.observation_std().numpy() ** 2
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise FloatingPointError('Nonfinite frozen predictions')
    predictions['x_mean_raw'] = predictions['x_mean'] * data['sx']['scale'] + data['sx']['mean']
    predictions['sample_ids'], predictions['split'] = data['ids'], data['labels']
    np.savez(str(output/name/'predictions.npz'), **predictions)
    factor_rows = [dict(sample_id=int(i), split=int(k), mu=float(m), posterior_std=float(s))
                   for i, k, m, s in zip(data['ids'], data['labels'], predictions['mu'][:, 0], predictions['std'][:, 0])]
    write_csv(output/name/'latent_posterior.csv', factor_rows)
    jacobian = float(np.log(data['sx']['scale']).sum())
    rows = []
    for split, k in [('train', 0), ('validation', 1), ('test', 2)]:
        idx = data['labels'] == k
        mean_kl = float(predictions['kl'][idx].mean())
        variance = float(predictions['mu'][idx].var())
        row = dict(model=name, split=split, n_samples=int(idx.sum()),
                   elbo_standardized=float(predictions['elbo'][idx].mean()),
                   elbo_raw=float(predictions['elbo'][idx].mean()-jacobian),
                   reconstruction_log_prob_standardized=float(predictions['reconstruction'][idx].mean()),
                   kl=mean_kl, mu_variance=variance,
                   posterior_variance_mean=float((predictions['std'][idx] ** 2).mean()),
                   x_mse_raw=float(np.mean((predictions['x_mean_raw'][idx]-data['x'][idx]) ** 2)),
                   x_mse_standardized=float(np.mean((predictions['x_mean'][idx]-data['target'][idx]) ** 2)),
                   log_marginal_standardized=float(predictions['log_marginal_'+str(points)][idx].mean()),
                   log_marginal_raw=float(predictions['log_marginal_'+str(points)][idx].mean()-jacobian),
                   quadrature_points=points,
                   low_kl=mean_kl < config['collapse_mean_kl_threshold'],
                   low_mu_variance=variance < config['collapse_mu_variance_threshold'],
                   posterior_collapse_flag=(mean_kl < config['collapse_mean_kl_threshold'] and
                                            variance < config['collapse_mu_variance_threshold']),
                   best_epoch=status['best_epoch'], epochs_completed=status['epochs_completed'],
                   total_parameters=status['total_parameters'], encoder_parameters=status['encoder_parameters'],
                   decoder_parameters=status['decoder_parameters'], training_seconds=status['elapsed_seconds'])
        for j in range(3):
            row['noise_variance_standardized_X'+str(j+1)] = float(noise[j])
            row['noise_variance_raw_X'+str(j+1)] = float(noise[j]*data['sx']['scale'][j]**2)
            row['x_mse_raw_X'+str(j+1)] = float(np.mean((predictions['x_mean_raw'][idx,j]-data['x'][idx,j])**2))
        for order in config['quadrature_check_points']:
            delta = predictions['log_marginal_'+str(points)][idx]-predictions['log_marginal_'+str(order)][idx]
            row['log_marginal_raw_'+str(order)] = float(predictions['log_marginal_'+str(order)][idx].mean()-jacobian)
            row['quadrature_mean_absolute_delta_'+str(order)] = float(np.abs(delta).mean())
            row['quadrature_max_absolute_delta_'+str(order)] = float(np.abs(delta).max())
            row['elbo_check_mean_'+str(order)] = float(predictions['elbo_check_'+str(order)][idx].mean())
        row['quadrature_accuracy_warning'] = row['quadrature_mean_absolute_delta_96'] > config['quadrature_warning_nats']
        rows.append(row)
    return rows


def run_training(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    validate_config(config)
    root, output = mapped(config['source_dir']).resolve(), mapped(config['output_dir']).resolve()
    if output.exists():
        raise FileExistsError('Refusing a second training attempt or overwrite: '+str(output))
    before = file_inventory(root)
    # The same integration tests are mandatory before either actual VAE fit.
    test = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                           '-p', 'test_latent_vae.py', '-v'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    print(test.stdout, flush=True)
    if test.returncode:
        raise RuntimeError('Preflight tests failed; no VAE training started')
    output.mkdir()
    (output/'preflight_tests.txt').write_text(test.stdout)
    save_json(output/'config.json', config)
    save_json(output/'source_inventory_before.json', before)
    try:
        torch.set_num_threads(1)
        np.random.seed(config['seed'])
        with forbid_latent_reads(root) as audit:
            data = load_inputs(root)
            save_json(output/'source_verification.json', data['verification'])
            prep = {prefix+'_'+key: value for prefix in ('sx', 'sh') for key, value in data[prefix].items()}
            np.savez(str(output/'preprocessing.npz'), **prep)
            np.save(str(output/'sample_indices.npy'), data['ids'])
            np.save(str(output/'sample_split.npy'), data['labels'])
            orders = epoch_orders(config['seed'], config['max_epochs'], 600)
            np.save(str(output/'batch_orders.npy'), orders)
            models, statuses = {}, {}
            for name in ('A', 'B'):
                models[name], statuses[name] = train_model(name, data['features_'+name], data['target'],
                                                          data['labels'], config, output, orders)
            if statuses['A']['initial_decoder_sha256'] != statuses['B']['initial_decoder_sha256']:
                raise AssertionError('Decoder initialization mismatch')
            save_json(output/'checkpoints_finalized.json', statuses)
            rows = []
            for name in ('A', 'B'):
                rows.extend(summarize_predictions(name, models[name], data, config, output, statuses[name]))
            write_csv(output/'label_free_metrics.csv', rows)
        save_json(output/'latent_read_audit.json', audit)
        if audit['blocked_attempts']:
            raise AssertionError('Unexpected attempt to read latent labels')
        if before != file_inventory(root):
            raise AssertionError('Original experiment files changed')
        save_json(output/'metadata.json', dict(source_dir=str(root), config=config,
            model_statuses=statuses, verification=data['verification'],
            input_feature_order='MLP X1 units 1..10, MLP X2 units 1..10, MLP X3 units 1..10',
            preprocessing_fit_rows=600, scaler_std_ddof=0, posterior_mu_variance_ddof=0,
            source_hidden_dtype='float64', prior='N(0,1)', latent_dimension=1,
            X_reconstruction='64-point q expectation of decoder mean; raw and standardized MSE',
            validation_selection='64-point q quadrature ELBO, max, strict improvement',
            marginal_likelihood='prior Gauss-Hermite 64 points; 32/96 diagnostics fixed before training',
            log_density_units='nats/sample, sum over three X variables',
            source_checkpoint_not_in_optimizer=True, same_initial_decoder_verified=True,
            batch_orders_sha256=array_digest(orders), old_results_read_during_training=False))
        artifacts = [output/'config.json', output/'preprocessing.npz', output/'sample_indices.npy',
                     output/'sample_split.npy', output/'label_free_metrics.csv', output/'checkpoints_finalized.json',
                     output/'metadata.json']
        artifacts += [output/name/file for name in ('A', 'B') for file in ('best.pt', 'predictions.npz')]
        save_json(output/'training_complete.json', dict(status='complete', new_vae_training_attempts=2,
            grandag_training_attempts=0, Z_read=False, both_checkpoints_fixed_before_test_inference=True,
            existing_source_files_unchanged=True, artifacts={str(p.relative_to(output)): digest(p) for p in artifacts},
            versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
            code_sha256={str(p): digest(p) for p in list(Path('latent_experiments').glob('vae*.py'))+
                         [Path('latent_vae.py'), Path('tests/test_latent_vae.py')]}, completed_at=time.time()))
    except Exception as exc:
        save_json(output/'failure.json', dict(error=repr(exc), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run_training(parser.parse_args().config)
