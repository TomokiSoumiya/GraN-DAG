"""Synthetic equations, leakage boundaries, probes and experiment safety."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from latent_experiments.config import load_config
from latent_experiments.data import generate_arrays, save_dataset, split_indices, topological_order
from latent_experiments.evaluate import fit_probe, evaluate_probe, pca_on_train, gaussian_reference
from latent_experiments.runner import run


class LatentExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('configs/latent_linear.json')

    def test_initial_equations_and_reproducibility(self):
        a = generate_arrays(self.config['data'])
        b = generate_arrays(self.config['data'])
        self.assertEqual(a['observed_X'].shape, (1000, 3))
        self.assertEqual(a['true_Z'].shape, (1000, 1))
        np.testing.assert_array_equal(a['observed_X'], b['observed_X'])
        np.testing.assert_allclose(a['observed_X'], a['true_Z'] @ np.array([[2., 3., 1.]]) + a['noise'])
        self.assertFalse(a['observed_adjacency'].any())
        self.assertTrue(a['latent_adjacency'].all())
        self.assertLess(abs(a['noise'].std()-1.), .06)

    def test_configured_generators_b_to_e_and_distributions(self):
        for file in ['latent_nonlinear', 'latent_two_factors', 'latent_observed_edges', 'latent_ten_observed']:
            config = load_config('configs/'+file+'.json')
            config['data']['n_samples'] = 50
            data = generate_arrays(config['data'])
            self.assertEqual(data['observed_X'].shape, (50, config['data']['n_observed']))
            self.assertEqual(data['true_Z'].shape, (50, config['data']['n_latent']))
            self.assertTrue(np.isfinite(data['observed_X']).all())
            topological_order(data['observed_coefficients'])
        for distribution in [{'type': 'uniform', 'low': -2., 'high': 2.}, {'type': 'laplace', 'loc': 0., 'scale': 1.}]:
            config = copy.deepcopy(self.config['data'])
            config['latent_distribution'] = distribution
            self.assertTrue(np.isfinite(generate_arrays(config)['true_Z']).all())
        bad = copy.deepcopy(self.config['data'])
        bad['observed_coefficients'] = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
        with self.assertRaises(ValueError):
            generate_arrays(bad)

    def test_observed_edges_equation(self):
        config = load_config('configs/latent_observed_edges.json')
        a = generate_arrays(config['data'])
        np.testing.assert_allclose(a['observed_X'][:, 1], 3*a['true_Z'][:, 0] + .5*a['observed_X'][:, 0] + a['noise'][:, 1])
        np.testing.assert_allclose(a['observed_X'][:, 2], a['true_Z'][:, 0] + .5*a['observed_X'][:, 1] + a['noise'][:, 2])

    def test_split_exactly_matches_original_loader_and_excludes_final_test(self):
        import torch
        from gran_dag.data import DataManagerFile
        old = torch.tensor(0.).type()
        torch.set_default_tensor_type('torch.DoubleTensor')
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                split = save_dataset(self.config, root/'data')
                train = DataManagerFile(str(root/'training_input'), '1', 600, 200, normalize=True)
                val = DataManagerFile(str(root/'training_input'), '1', 600, 200, train=False,
                                       normalize=True, mean=train.mean, std=train.std)
                np.testing.assert_array_equal(split['pool'][train.sample_indices], split['train'])
                np.testing.assert_array_equal(split['pool'][val.sample_indices], split['validation'])
                self.assertFalse(np.isin(split['test'], split['pool']).any())
                self.assertEqual(set((root/'training_input').iterdir()),
                                 {root/'training_input'/f for f in ['data1.npy', 'DAG1.npy', 'sample_indices.npy']})
                raw = np.load(str(root/'data/observed_X.npy'))
                np.testing.assert_allclose(train.mean.numpy(), raw[split['train']].mean(0, keepdims=True))
                np.testing.assert_allclose(train.std.numpy(), raw[split['train']].std(0, ddof=1, keepdims=True))
                reference = gaussian_reference(self.config, root)
                self.assertAlmostEqual(reference['population_bayes_mse_per_z'][0], 1/15)
        finally:
            torch.set_default_tensor_type(old)

    def test_probe_preprocessing_fit_and_metric_values(self):
        rng = np.random.RandomState(8)
        x = rng.normal(size=(150, 3))
        z = x @ np.array([2., -1., 3.])
        labels = np.repeat([0, 1, 2], [90, 30, 30])
        model, params, _ = fit_probe(x[:90], z[:90], self.config['probe'])
        np.testing.assert_allclose(model.named_steps['scale'].mean_, x[:90].mean(0))
        result = evaluate_probe(model, x, z, labels)
        pred = model.predict(x[labels == 2])
        self.assertAlmostEqual(result['test_mse'], float(np.mean((pred-z[labels == 2])**2)))
        self.assertAlmostEqual(result['test_r2'], 1-float(np.sum((pred-z[labels == 2])**2)/np.sum((z[labels == 2]-z[labels == 2].mean())**2)))
        self.assertGreater(result['test_r2'], .999)
        changed = x.copy()
        changed[labels == 2] += 1000
        again, params2, _ = fit_probe(changed[:90], z[:90], self.config['probe'])
        self.assertEqual(params, params2)
        np.testing.assert_array_equal(model.named_steps['ridge'].coef_, again.named_steps['ridge'].coef_)
        const, _, _ = fit_probe(np.ones((90, 10)), z[:90], self.config['probe'])
        np.testing.assert_allclose(const.predict(np.ones((30, 10))), z[:90].mean())
        coords, _, dims = pca_on_train(np.ones((150, 10)), labels)
        self.assertFalse(coords.any())
        self.assertEqual(dims, 0)
        p1, _, _ = pca_on_train(x, labels)
        p2, _, _ = pca_on_train(changed, labels)
        np.testing.assert_array_equal(p1[labels == 0], p2[labels == 0])

    def test_run_safety_and_config_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            run(self.config, root, stage='generate')
            with self.assertRaises(FileExistsError):
                run(self.config, root, stage='generate')
            run(self.config, root, stage='generate', resume=True)
            changed = copy.deepcopy(self.config)
            changed['data']['seed'] = 91
            with self.assertRaises(ValueError):
                run(changed, root, stage='generate', resume=True)

    def test_multi_latent_evaluation_with_frozen_dummy_networks_and_constant_cam(self):
        # An evaluation integration test, NOT a trained B--E experiment.
        import torch
        from gran_dag.hidden import save_hidden_activations
        from gran_dag.models.learnables import LearnableModel_NonLinGaussANM
        from latent_experiments.evaluate import evaluate
        config = load_config('configs/latent_two_factors.json')
        config['data']['n_samples'] = 80
        config['training']['hid_dim'] = 3
        config['probe']['cv_folds'] = 2
        config['probe']['ridge_alphas'] = [1.]
        config['probe']['nonlinear'].update(n_estimators=3, max_depth=[2], min_samples_leaf=[2])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'run'
            run(config, root, stage='generate')
            x = np.load(str(root/'data/observed_X.npy'))
            labels = np.load(str(root/'data/sample_split.npy'))
            model = LearnableModel_NonLinGaussANM(5, 2, 3, norm_prod='paths')
            original_mask = model.adjacency.clone()
            paths = ['train', 'to-dag', 'cam-pruning/cutoff_0.001000',
                     'random_full', 'random_train_mask', 'random_cam_mask']
            for stage in paths:
                model.adjacency = (torch.zeros_like(original_mask) if stage in
                                   ('cam-pruning/cutoff_0.001000', 'random_cam_mask', 'to-dag') else original_mask)
                save_hidden_activations(model, torch.as_tensor(x).type_as(next(model.parameters())),
                                        str(root/stage/'hidden'), sample_split=labels)
            status = dict(n_train=48, n_validation=16, n_final_test=16, converged=False,
                          iterations=0, final_h=0., worker_elapsed_seconds=0.)
            (root/'training_status.json').write_text(json.dumps(status))
            metrics = evaluate(root, config)
            self.assertEqual({r['latent'] for r in metrics}, {'Z1', 'Z2'})
            self.assertTrue(all(np.isfinite(r['test_r2']) and np.isfinite(r['test_mse']) for r in metrics))
            roots = [r for r in metrics if r['stage'] == 'cam' and r['representation'] == 'hidden_ridge']
            self.assertTrue(all(r['varying_features'] == 0 for r in roots))
            self.assertTrue((root/'analysis/metrics.csv').is_file())
            self.assertTrue((root/'analysis/figures/probe_scores_Z2.png').is_file())
            self.assertTrue((root/'analysis/figures/pca_Z2.png').is_file())


if __name__ == '__main__':
    unittest.main()
