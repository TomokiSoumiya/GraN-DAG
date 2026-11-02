"""Label-free checks for block PCA and the matched linear-duplication control."""
import inspect
import json
import pickle
from pathlib import Path
import unittest

import numpy as np

from latent_experiments.feature_controls import (feature_diagnostics, fit_blocks, linear_features,
                                                transform_blocks)
from latent_experiments.unsupervised import fit_extractor, transform_extractor


class FeatureControlTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('configs/latent_feature_controls_seed201.json').read_text())
        self.x=np.random.RandomState(201).normal(size=(40,3))
        self.mask=np.array([[0,0,1],[1,0,0],[0,1,0]])

    def test_linear_allocation_coefficients_and_exact_pca_equivalence(self):
        features,coef,parents=linear_features(self.x,self.mask,201)
        np.testing.assert_array_equal(parents,[1,2,0])
        np.testing.assert_array_equal(coef,np.random.RandomState(201).normal(size=(3,10)))
        for j in range(3):
            np.testing.assert_array_equal(features[:,j,:],self.x[:,parents[j],None]*coef[j])
        a=fit_extractor(self.x[:25],self.x[:25],'pca',self.cfg)
        e=fit_extractor(features[:25].reshape(25,30),self.x[:25],'pca',self.cfg)
        fa,_,xa=transform_extractor(a,self.x)
        fe,_,xe=transform_extractor(e,features.reshape(40,30))
        self.assertAlmostEqual(abs(np.corrcoef(fa[:,0],fe[:,0])[0,1]),1.,places=12)
        np.testing.assert_allclose(xa,xe,atol=1e-12,rtol=1e-12)

    def test_block_summary_fit_only_training_and_saved_transform(self):
        h,_,_=linear_features(self.x,self.mask,201)
        model=fit_blocks(h[:25],self.x[:25],self.cfg)
        f,xhat,scores=transform_blocks(model,h)
        self.assertEqual(f.shape,(40,1));self.assertEqual(xhat.shape,(40,3));self.assertEqual(scores.shape,(40,3))
        self.assertEqual(model['final']['n_train'],25)
        np.testing.assert_allclose(model['final']['scaler'].mean_,scores[:25].mean(0),atol=1e-12)
        np.testing.assert_allclose(model['final']['scaler'].scale_,scores[:25].std(0),atol=1e-12)
        restored=pickle.loads(pickle.dumps(model))
        # Pickle can change array memory layout and BLAS rounding; the saved
        # transformation must reproduce values to float64 accuracy, not bits.
        np.testing.assert_allclose(f,transform_blocks(restored,h)[0],rtol=1e-12,atol=1e-12)
        changed=h.copy();changed[25:]+=1000
        np.testing.assert_array_equal(f[:25],transform_blocks(model,changed)[0][:25])
        self.assertEqual(list(inspect.signature(fit_blocks).parameters),['hidden_train','x_train','settings'])
        self.assertEqual(list(inspect.signature(linear_features).parameters),['x','mask','seed'])

    def test_zero_variance_blocks_are_not_invented(self):
        h,_,_=linear_features(self.x,self.mask,201)
        h[:,1,:]=7
        model=fit_blocks(h[:25],self.x[:25],self.cfg)
        f,_,scores=transform_blocks(model,h)
        self.assertFalse(model['blocks'][1]['informative'])
        self.assertFalse(model['final']['keep'][1])
        np.testing.assert_array_equal(scores[:,1],0)
        self.assertTrue(np.isfinite(f).all())
        with self.assertRaises(ValueError):fit_blocks(np.ones((25,3,10)),self.x[:25],self.cfg)

    def test_rank_and_redundancy_detect_linear_replicates(self):
        h,_,_=linear_features(self.x,self.mask,201)
        rows,matrix=feature_diagnostics(h.reshape(40,30),np.repeat(np.arange(3),10),'E','train',self.cfg)
        self.assertEqual(matrix.shape,(30,30))
        for r in rows:
            if r['group_left']==r['group_right']:
                self.assertEqual(r['numerical_rank'],1)
                self.assertAlmostEqual(r['participation_rank'],1.,places=12)
                self.assertAlmostEqual(r['mean_absolute_r'],1.,places=12)

    def test_invalid_mask_or_nonfinite_input_fails(self):
        for mask in (np.eye(3),np.ones((3,3)),np.zeros((3,3))):
            with self.assertRaises(ValueError):linear_features(self.x,mask,201)
        with self.assertRaises(ValueError):linear_features(np.full((40,3),np.nan),self.mask,201)


if __name__=='__main__':unittest.main()
