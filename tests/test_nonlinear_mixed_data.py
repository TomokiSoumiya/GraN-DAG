"""Unit checks only: no NN training or experimental runs."""
import copy
import unittest
from pathlib import Path

import numpy as np

from latent_experiments.config import load_config
from latent_experiments.data import generate_arrays, split_indices


class MixedNonlinearDataTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path('configs/latent_nonlinear_mixed_seed201.json'))

    def test_exact_requested_equations_and_splits(self):
        arrays = generate_arrays(self.config['data'])
        rng = np.random.RandomState(201)
        z = rng.normal(size=(1000, 1))
        noise = rng.normal(size=(1000, 3))
        expected = np.column_stack((2*z[:, 0], z[:, 0]**2, np.sin(z[:, 0])))+noise
        np.testing.assert_array_equal(arrays['true_Z'], z)
        np.testing.assert_array_equal(arrays['noise'], noise)
        np.testing.assert_array_equal(arrays['observed_X'], expected)
        np.testing.assert_array_equal(arrays['sample_indices'], np.arange(1000))
        self.assertFalse(arrays['observed_adjacency'].any())
        np.testing.assert_array_equal(arrays['latent_adjacency'], np.ones((1, 3)))
        splits = split_indices(1000, self.config['split'], 201)
        self.assertEqual([len(splits[key]) for key in ('train', 'validation', 'test')], [600, 200, 200])
        np.testing.assert_array_equal(np.sort(np.concatenate([splits[key] for key in
                                      ('train', 'validation', 'test')])), np.arange(1000))
        self.assertFalse(np.isin(splits['pool'], splits['test']).any())

    def test_shared_function_backward_compatibility(self):
        config = copy.deepcopy(self.config['data'])
        config.pop('latent_functions')
        config['latent_function'] = 'tanh'
        arrays = generate_arrays(config)
        rng = np.random.RandomState(201)
        z = rng.normal(size=(1000, 1))
        expected = np.tanh(z) @ np.array(config['latent_coefficients'])+rng.normal(size=(1000, 3))
        np.testing.assert_array_equal(arrays['observed_X'], expected)

    def test_invalid_target_functions_rejected(self):
        for names in (['identity'], 'sin', ['identity', 'missing', 'sin']):
            config = copy.deepcopy(self.config['data'])
            config['latent_functions'] = names
            with self.assertRaises(ValueError):
                generate_arrays(config)


if __name__ == '__main__':
    unittest.main()
