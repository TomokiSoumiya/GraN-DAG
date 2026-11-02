"""Preflight only: analytic identities, leakage checks and saved-source verification.

No optimizer steps or experiment-data generation are performed by these tests.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from latent_unsupervised import mapped
from latent_experiments.unsupervised import forbid_latent_reads
from latent_experiments.vae import (epoch_orders, gaussian_kl, inverse_softplus,
    make_model, marginal_log_prob, quadrature_rule, quadrature_terms, stochastic_terms)
from latent_experiments.vae_data import fit_scaler, load_inputs, transform_scaler, validate_arrays


CONFIG = json.loads(Path('configs/latent_vae_seed201.json').read_text())


class VAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_decoder_and_batch_order_despite_different_encoder_dimensions(self):
        a, b = make_model(3, CONFIG), make_model(30, CONFIG)
        for x, y in zip(a.decoder.parameters(), b.decoder.parameters()):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(a.raw_observation_std, b.raw_observation_std))
        self.assertNotEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        orders = epoch_orders(201, 5, 600)
        np.testing.assert_array_equal(orders, epoch_orders(201, 5, 600))
        for row in orders:
            np.testing.assert_array_equal(np.sort(row), np.arange(600))

    def test_kl_matches_torch_distribution(self):
        mu = torch.tensor([[-2.], [0.], [1.]], dtype=torch.float64)
        std = torch.tensor([[.2], [1.], [3.]], dtype=torch.float64)
        q = torch.distributions.Normal(mu, std)
        prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        actual = torch.distributions.kl_divergence(q, prior).sum(-1)
        np.testing.assert_allclose(gaussian_kl(mu, std).numpy(), actual.numpy(), atol=1e-12)

    def test_constant_decoder_elbo_and_marginal_are_analytic(self):
        model = make_model(3, CONFIG)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.raw_observation_std.fill_(inverse_softplus(1.-model.observation_floor))
            model.posterior_head.bias[1] = inverse_softplus(1.-model.posterior_floor)
        x = torch.zeros(7, 3, dtype=torch.float64)
        with torch.no_grad():
            result = quadrature_terms(model, x, x)
            expected = -1.5*np.log(2*np.pi)
            np.testing.assert_allclose(result['elbo'].numpy(), expected, atol=1e-12)
            np.testing.assert_allclose(result['kl'].numpy(), 0., atol=1e-12)
            self.assertEqual(tuple(result['mu'].shape), (7, 1))
            self.assertEqual(tuple(result['x_mean'].shape), (7, 3))
            for order in (32, 64, 96):
                np.testing.assert_allclose(marginal_log_prob(model, x, order).numpy(), expected, atol=1e-12)

    def test_linear_gaussian_marginal_matches_exact_covariance(self):
        model = make_model(3, CONFIG)
        model.decoder = torch.nn.Linear(1, 3).double()
        loading = np.array([.4, .6, -.3])
        with torch.no_grad():
            model.decoder.weight.copy_(torch.as_tensor(loading[:, None]))
            model.decoder.bias.zero_()
            model.raw_observation_std.fill_(inverse_softplus(1.-model.observation_floor))
        x = np.array([[0.,0.,0.], [.3,-.7,.8], [-1.,.4,1.1]])
        covariance = np.eye(3) + np.outer(loading, loading)
        expected = -0.5*(3*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]+
                         np.einsum('bi,ij,bj->b', x, np.linalg.inv(covariance), x))
        with torch.no_grad():
            actual = marginal_log_prob(model, torch.as_tensor(x), 64).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_train_only_scaling_and_constant_columns(self):
        values = np.array([[1.,7.,2.], [3.,7.,6.], [1000.,-8.,-20.]])
        scaler = fit_scaler(values[:2], True)
        np.testing.assert_array_equal(scaler['keep'], [True,False,True])
        np.testing.assert_allclose(transform_scaler(values[:2], scaler).mean(0), 0., atol=1e-12)
        np.testing.assert_allclose(transform_scaler(values[:2], scaler).std(0), 1., atol=1e-12)
        transformed = transform_scaler(values, scaler)
        self.assertGreater(abs(transformed[2,0]), 100.)
        self.assertEqual(scaler['n_fit'], 2)
        with self.assertRaises(ValueError):
            fit_scaler(np.ones((4,3)), True)
        with self.assertRaises(ValueError):
            transform_scaler(np.full((3,3), np.nan), scaler)

    def test_shape_split_and_row_mismatch_fail(self):
        x, h, ids = np.zeros((1000,3)), np.zeros((1000,3,10)), np.arange(1000)
        labels = np.repeat([0,1,2], [600,200,200])
        validate_arrays(x,h,ids,labels,ids,labels)
        with self.assertRaises(AssertionError):
            validate_arrays(x,h,ids,labels,ids[::-1],labels)
        with self.assertRaises(AssertionError):
            validate_arrays(x,h,ids,labels,ids,labels[::-1])
        with self.assertRaises(ValueError):
            validate_arrays(x,h[:999],ids,labels,ids,labels)
        with self.assertRaises(ValueError):
            validate_arrays(x,h,ids,np.zeros(1000),ids,np.zeros(1000))

    def test_positive_scales_and_finite_gradients_without_optimizer_step(self):
        model = make_model(30, CONFIG)
        features = torch.ones(5,30,dtype=torch.float64)
        target = torch.ones(5,3,dtype=torch.float64)
        elbo, reconstruction, kl = stochastic_terms(model, features, target)
        self.assertEqual(tuple(elbo.shape), (5,))
        np.testing.assert_allclose((reconstruction-kl).detach().numpy(), elbo.detach().numpy())
        (-elbo.mean()).backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all().item()) for p in model.parameters()))
        with torch.no_grad():
            model.raw_observation_std.fill_(-1000.)
            model.posterior_head.weight.zero_()
            model.posterior_head.bias[1] = -1000.
            self.assertTrue(bool((model.observation_std() > 0).all().item()))
            self.assertTrue(bool((model.encode(features)[1] > 0).all().item()))

    def test_true_z_read_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with forbid_latent_reads(root) as audit:
                with self.assertRaises(PermissionError):
                    np.load(str(root/'data/true_Z.npy'))
            self.assertEqual(len(audit['blocked_attempts']), 1)

    def test_quadrature_standard_normal_moments(self):
        nodes, weights = quadrature_rule(64)
        self.assertAlmostEqual(float(weights.sum()), 1., places=12)
        self.assertAlmostEqual(float((nodes*weights).sum()), 0., places=12)
        self.assertAlmostEqual(float((nodes**2*weights).sum()), 1., places=12)

    def test_existing_source_is_aligned_frozen_and_reproducible(self):
        root = mapped(CONFIG['source_dir'])
        with forbid_latent_reads(root) as audit:
            data = load_inputs(root)
        self.assertEqual(audit['blocked_attempts'], [])
        self.assertEqual(data['features_B'].shape, (1000,30))
        self.assertEqual(data['verification']['hidden_constant'], 0)
        self.assertTrue(data['verification']['frozen_parameters_unchanged'])
        self.assertTrue(data['verification']['all_source_parameters_require_grad_false'])
        self.assertTrue(data['verification']['original_training_excludes_final_test'])


if __name__ == '__main__':
    unittest.main()
