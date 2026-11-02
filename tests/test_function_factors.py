"""Label-free small unit tests; no GraN-DAG training/experimental runs."""
import inspect
import json
import pickle
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from latent_experiments.function_factors import apply_model, fit_kernel, observed_diagnostics


class FunctionFactorTests(unittest.TestCase):
    def setUp(self):
        self.settings=json.loads(Path('configs/latent_factors_normalized_seed201.json').read_text())
        self.x=np.random.RandomState(201).normal(size=(30,3))

    def test_kernel_gamma_and_scaler_fit_on_training_only(self):
        self.assertEqual(list(inspect.signature(fit_kernel).parameters),['features_train','x_train','settings'])
        model=fit_kernel(self.x[:20],self.x[:20],self.settings)
        np.testing.assert_allclose(model['scaler'].mean_,self.x[:20].mean(0))
        scaled=(self.x[:20]-self.x[:20].mean(0))/self.x[:20].std(0)
        expected=1/(2*np.median(pdist(scaled,metric='sqeuclidean')))
        self.assertAlmostEqual(model['gamma'],expected,places=14)
        self.assertEqual(model['n_positive_training_pairs'],190)
        factors,xhat=apply_model(model,self.x)
        self.assertEqual(factors.shape,(30,1));self.assertEqual(xhat.shape,(30,3))
        changed=self.x.copy();changed[20:]+=1000
        altered,_=apply_model(model,changed)
        np.testing.assert_array_equal(factors[:20],altered[:20])
        restored=pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(factors,apply_model(restored,self.x)[0])
        design=np.column_stack([np.ones(20),factors[:20]])
        np.testing.assert_allclose(model['x_decoder'],np.linalg.lstsq(design,self.x[:20],rcond=None)[0])

    def test_observed_diagnostic_detects_affine_factor_without_labels(self):
        labels=np.r_[np.zeros(20,dtype=int),np.ones(5,dtype=int),np.full(5,2,dtype=int)]
        rows=observed_diagnostics('dummy',3*self.x[:,0]+2,self.x,labels)
        self.assertEqual(len(rows),3)
        self.assertLess(rows[0]['affine_test_RMSE_over_train_factor_std'],1e-12)
        self.assertAlmostEqual(rows[0]['affine_slope_train'],3)
        self.assertTrue(all(row['Z_used'] is False for row in rows))

    def test_kernel_constant_or_invalid_training_inputs_rejected(self):
        for x in (np.ones((20,3)),np.full((20,3),np.nan)):
            with self.assertRaises(ValueError):fit_kernel(x,x,self.settings)


if __name__=='__main__':
    unittest.main()
