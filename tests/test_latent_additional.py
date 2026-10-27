"""Additional analyses: controlled weights/masks, reuse, split safety and deltas."""
import copy
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gran_dag.hidden import save_hidden_activations
from gran_dag.models.learnables import LearnableModel_NonLinGaussANM
from latent_experiments.additional import (
    CONDITIONS, comparison_rows, controlled_features, digest, fresh_output,
    parameter_digest, score_features, snapshot)
from latent_experiments.config import load_config
from latent_experiments.evaluate import evaluate_probe, fit_probe, write_csv


class AdditionalAnalysisTests(unittest.TestCase):
    def test_new_output_guard_and_source_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'old.txt').write_text('preserve')
            original = snapshot(root)
            output = fresh_output(root)
            output.mkdir()
            (output/'result.txt').write_text('new')
            self.assertEqual(original, snapshot(root, excluded=output))
            with self.assertRaises(FileExistsError):
                fresh_output(root)
            with self.assertRaises(ValueError):
                fresh_output(root, root/'data'/'nested')
            with self.assertRaises(ValueError):
                fresh_output(root, root.parent/'elsewhere')

    def test_controlled_features_reuse_with_equal_cam_and_dag(self):
        config = load_config('configs/latent_linear.json')
        config['training']['normalize_data'] = False
        rng = np.random.RandomState(18)
        x = rng.normal(size=(30, 3))
        ids, labels = np.arange(30), np.repeat([0, 1, 2], [18, 6, 6])
        initial = LearnableModel_NonLinGaussANM(3, 2, 4).double()
        learned = copy.deepcopy(initial)
        with torch.no_grad():
            learned.biases[0].add_(.3)
        train_mask = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        dag_mask = train_mask.copy()
        dag_mask[2, 0] = 0
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/'additional'
            output.mkdir()
            np.save(str(root/'normalization_mean.npy'), np.zeros((1, 3)))
            np.save(str(root/'normalization_std.npy'), np.ones((1, 3)))
            paths = [('random_full', initial, initial.adjacency.numpy()),
                     ('random_train_mask', initial, train_mask),
                     ('train', learned, train_mask), ('to-dag', learned, dag_mask),
                     ('random_cam_mask', initial, dag_mask),
                     ('cam-pruning/cutoff_0.001000', learned, dag_mask)]
            for stage, model, mask in paths:
                model = copy.deepcopy(model)
                model.adjacency = torch.as_tensor(mask, dtype=torch.float64)
                directory = root/stage
                directory.mkdir(parents=True)
                with (directory/'model.pkl').open('wb') as stream:
                    pickle.dump(model, stream)
                save_hidden_activations(model, torch.as_tensor(x), str(directory/'hidden'),
                                        sample_indices=ids, sample_split=labels,
                                        metadata={'checkpoint_sha256': digest(directory/'model.pkl')})
            before = snapshot(root, excluded=output)
            conditions, hidden, verified = controlled_features(root, output, config, x, ids, labels)
            self.assertTrue(verified['learned_stage_parameters_equal_train'])
            self.assertTrue(verified['all_random_parameters_equal_initial'])
            self.assertTrue(verified['dag_equals_cam_mask'])
            self.assertTrue(verified['learned_parameters_differ_from_initial'])
            self.assertEqual(next(c for c in conditions if c['condition'] == 'D')['ridge_reuse_stage'],
                             'random_cam_mask')
            for code in ('D', 'E', 'F', 'G'):
                for h in hidden[code]:
                    np.testing.assert_array_equal(h[:, 0, :], np.repeat(h[:1, 0, :], len(x), axis=0))
            for a, b in [('D', 'F'), ('E', 'G')]:
                for ha, hb in zip(hidden[a], hidden[b]):
                    np.testing.assert_array_equal(ha, hb)
            self.assertEqual(before, snapshot(root, excluded=output))
            # Corrupted cached activations must never be silently reused.
            path = root/'train/hidden/hidden_layer_001.npy'
            broken = np.load(str(path))
            broken[0, 0, 0] += 1
            np.save(str(path), broken)
            with self.assertRaises(AssertionError):
                controlled_features(root, output, config, x, ids, labels)

    def test_parameter_hash_ignores_only_mask(self):
        model = LearnableModel_NonLinGaussANM(3, 2, 4)
        changed = copy.deepcopy(model)
        changed.adjacency.zero_()
        self.assertEqual(parameter_digest(model), parameter_digest(changed))
        with torch.no_grad():
            changed.weights[0][0, 0, 0].add_(1)
        self.assertNotEqual(parameter_digest(model), parameter_digest(changed))

    def test_train_only_probe_arguments_and_existing_ridge_reuse(self):
        config = load_config('configs/latent_linear.json')
        config['probe']['cv_folds'] = 2
        config['probe']['ridge_alphas'] = [.01, 1.]
        config['probe']['nonlinear'].update(n_estimators=3, max_depth=[3], min_samples_leaf=[2])
        rng = np.random.RandomState(31)
        x = rng.normal(size=(40, 2))
        z = x[:, :1]*.7 + rng.normal(size=(40, 1))*.1
        labels = np.repeat([0, 1, 2], [24, 8, 8])
        ids = np.arange(40)
        h = np.stack([x, x**2], axis=2)
        features = {'B': [h, h*.5], 'C': [h, h*.5]}
        conditions = [dict(condition=code, weights=weights, mask='train',
                           ridge_reuse_stage='random_train_mask', parent_counts=[1, 1])
                      for code, weights in [('B', 'initial'), ('C', 'learned')]]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'analysis/probes').mkdir(parents=True)
            output = root/'additional'
            (output/'probes').mkdir(parents=True)
            old, params, cv = fit_probe(x[:24], z[:24, 0], config['probe'])
            row = dict(stage='X', representation='X_ridge', layer=0, scope='all', latent='Z1',
                       cv_mse=cv, best_params=json.dumps(params))
            row.update(evaluate_probe(old, x, z[:, 0], labels))
            write_csv(root/'analysis/metrics.csv', [row])
            with (root/'analysis/probes/X_X_ridge_L0_all_Z1.pkl').open('wb') as stream:
                pickle.dump(old, stream)
            train_inputs = [x[:24], h[:24].reshape(24, -1), (h*.5)[:24].reshape(24, -1)]
            train_inputs += [values[:24, j, :] for values in (h, h*.5) for j in range(2)]

            def guarded_fit(values, target, settings, nonlinear=False):
                self.assertEqual(len(values), 24)
                np.testing.assert_array_equal(target, z[:24, 0])
                self.assertTrue(any(np.array_equal(values, candidate) for candidate in train_inputs))
                return fit_probe(values, target, settings, nonlinear)

            with patch('latent_experiments.additional.fit_probe', side_effect=guarded_fit) as mocked:
                metrics, predictions, status = score_features(
                    root, output, config, conditions, features, x, z, ids, labels)
            self.assertEqual(status['reused_existing_probes'], 1)
            self.assertEqual(status['new_probe_fits'], mocked.call_count)
            self.assertEqual(len(metrics), 26)
            self.assertEqual(len(predictions), 26*8)
            raw = next(r for r in metrics if r['condition'] == 'X' and r['decoder'] == 'ridge')
            self.assertEqual(raw['probe_source'], 'existing_probe_verified')
            self.assertAlmostEqual(raw['test_r2'], row['test_r2'])

    def test_paired_comparison_signs_and_all_controls(self):
        metrics = []
        for i, (code, _, _, _) in enumerate(CONDITIONS):
            metrics.append(dict(condition=code, decoder='ridge', layer=1, scope='X1',
                                latent='Z1', test_r2=i*.1, test_mse=1-i*.05))
        rows = comparison_rows(metrics)
        self.assertEqual(len(rows), 8)
        learning = next(r for r in rows if r['comparison'] == 'learning_train_mask')
        self.assertAlmostEqual(learning['delta_r2'], .1)
        self.assertAlmostEqual(learning['delta_mse'], -.05)
        dag = next(r for r in rows if r['comparison'] == 'dag_mask_learned_weights')
        self.assertAlmostEqual(dag['delta_r2'], .2)
        self.assertAlmostEqual(dag['delta_mse'], -.1)


if __name__ == '__main__':
    unittest.main()
