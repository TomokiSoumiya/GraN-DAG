"""Generator-only unit tests, with no experimental runs or NN training."""
import copy
import unittest

import numpy as np

from latent_experiments.config import load_config
from latent_experiments.data import generate_arrays, split_indices


class NormalizedSignalTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('configs/latent_nonlinear_normalized_seed201.json')
        self.train = split_indices(1000,self.config['split'],201)['train']

    def test_exact_formula_and_training_only_statistics(self):
        arrays = generate_arrays(self.config['data'],self.train)
        rng = np.random.RandomState(201)
        z = rng.normal(size=(1000,1))[:,0]
        raw = np.column_stack([np.tanh(z),np.sin(z),z**3])
        noise = rng.normal(size=(1000,3))
        np.testing.assert_array_equal(arrays['raw_signal'],raw)
        np.testing.assert_array_equal(arrays['noise'],noise)
        np.testing.assert_array_equal(arrays['signal_mean'],raw[self.train].mean(0))
        np.testing.assert_array_equal(arrays['signal_std'],raw[self.train].std(0))
        expected=(raw-raw[self.train].mean(0))/raw[self.train].std(0)
        np.testing.assert_array_equal(arrays['observed_X'],expected+noise)
        np.testing.assert_allclose(arrays['normalized_signal'][self.train].mean(0),np.zeros(3),atol=1e-14)
        np.testing.assert_allclose(arrays['normalized_signal'][self.train].std(0),np.ones(3),atol=1e-14)

    def test_invalid_training_ids_and_missing_ids_fail(self):
        for ids in (None,[],[0,0],[0,1000],[0.5,1.5]):
            with self.assertRaises(ValueError):
                generate_arrays(self.config['data'],ids)

    def test_legacy_path_unchanged(self):
        config = copy.deepcopy(self.config['data'])
        config.pop('signal_normalization')
        arrays = generate_arrays(config)
        rng=np.random.RandomState(201)
        z=rng.normal(size=(1000,1))[:,0]
        np.testing.assert_array_equal(arrays['observed_X'],
            np.column_stack([np.tanh(z),np.sin(z),z**3])+rng.normal(size=(1000,3)))
        self.assertNotIn('signal_mean',arrays)


if __name__=='__main__':
    unittest.main()
