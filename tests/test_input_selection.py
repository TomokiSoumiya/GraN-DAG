"""Small unit checks only: no GraN-DAG training or experiment runs."""
import inspect
import unittest

import numpy as np

from latent_experiments.input_selection import CANDIDATES, apply_candidate, fit_single_observed


class InputSelectionTests(unittest.TestCase):
    def test_single_column_uses_train_only_and_replays(self):
        self.assertEqual(list(inspect.signature(fit_single_observed).parameters),
                         ['features_train','x_train'])
        values = np.random.RandomState(201).normal(size=(30,3))
        model = fit_single_observed(values[:20,:1],values[:20])
        np.testing.assert_allclose(model['scaler'].mean_,values[:20,:1].mean(0))
        np.testing.assert_allclose(model['scaler'].var_,values[:20,:1].var(0))
        factors, xhat = apply_candidate(model,values[:,:1])
        self.assertEqual(factors.shape,(30,1))
        self.assertEqual(xhat.shape,(30,3))
        np.testing.assert_allclose(factors,(values[:,:1]-values[:20,:1].mean(0))/values[:20,:1].std(0))
        changed = values[:,:1].copy()
        changed[20:] += 1000
        changed_factors, _ = apply_candidate(model,changed)
        np.testing.assert_array_equal(factors[:20],changed_factors[:20])
        design = np.column_stack([np.ones(20),factors[:20]])
        np.testing.assert_allclose(model['x_decoder'],np.linalg.lstsq(design,values[:20],rcond=None)[0])

    def test_constant_single_column_and_invalid_inputs(self):
        values = np.ones((30,1))*3
        x = np.random.RandomState(201).normal(size=(30,3))
        model = fit_single_observed(values[:20],x[:20])
        factors, xhat = apply_candidate(model,values)
        np.testing.assert_array_equal(factors,np.zeros((30,1)))
        np.testing.assert_allclose(xhat,np.repeat(x[:20].mean(0,keepdims=True),30,axis=0))
        self.assertTrue(model['constant_column'])
        for invalid in (np.ones((20,2)),np.full((20,1),np.nan)):
            with self.assertRaises(ValueError):
                fit_single_observed(invalid,x[:20])

    def test_fixed_candidates_no_singleton_fa(self):
        self.assertEqual(len(CANDIDATES),10)
        self.assertEqual(sum(reuse is not None for _,_,_,reuse in CANDIDATES),6)
        self.assertEqual([(c,m) for c,m,_,_ in CANDIDATES if c in ('X1','X3')],
                         [('X1','standardized_observed'),('X3','standardized_observed')])
        self.assertEqual([columns for c,m,columns,_ in CANDIDATES if c=='X13'],[[0,2],[0,2]])


if __name__=='__main__':
    unittest.main()
