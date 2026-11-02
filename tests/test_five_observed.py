"""Preflight on small fixtures: no GraN-DAG optimizer or experimental fit is run."""
import copy
import inspect
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from gran_dag.data import DataManagerFile
from gran_dag.models.learnables import LearnableModel_NonLinGaussANM
from latent_experiments.config import load_config
from latent_experiments import data
from latent_experiments.five_observed import (extra_signal_functions,fit_mlp_summary,
    apply_mlp_summary,mask_summary,validate_config)
from latent_experiments.unsupervised import fit_extractor,transform_extractor,forbid_latent_reads


class FiveObservedTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config('configs/latent_five_observed_seed201.json')
        self.small=copy.deepcopy(self.cfg);self.small['data']['n_samples']=60
        self.splits=data.split_indices(60,self.cfg['split'],201)

    def test_exact_five_functions_training_normalization_and_registry_restoration(self):
        original=dict(data.FUNCTIONS)
        with extra_signal_functions():
            arrays=data.generate_arrays(self.small['data'],self.splits['train'])
        self.assertEqual(data.FUNCTIONS,original)
        z=arrays['true_Z'][:,0]
        expected=np.column_stack([np.tanh(z),np.sin(z),z**3,np.tanh(2*z),np.sin(2*z)])
        np.testing.assert_array_equal(arrays['raw_signal'],expected)
        idx=self.splits['train'];mean=expected[idx].mean(0);std=expected[idx].std(0)
        np.testing.assert_array_equal(arrays['signal_mean'],mean)
        np.testing.assert_array_equal(arrays['signal_std'],std)
        np.testing.assert_array_equal(arrays['observed_X'],(expected-mean)/std+arrays['noise'])
        np.testing.assert_allclose(arrays['normalized_signal'][idx].mean(0),0,atol=1e-14)
        np.testing.assert_allclose(arrays['normalized_signal'][idx].std(0),1,atol=1e-14)
        self.assertEqual(arrays['observed_X'].shape,(60,5))
        np.testing.assert_array_equal(arrays['observed_adjacency'],np.zeros((5,5)))
        np.testing.assert_array_equal(arrays['latent_adjacency'],np.ones((1,5)))

    def test_production_split_is_exact_disjoint_and_matches_original_loader(self):
        split=data.split_indices(1000,self.cfg['split'],201)
        self.assertEqual([len(split[k]) for k in ['train','validation','test']],[600,200,200])
        self.assertEqual(len(set(split['train'])&set(split['validation'])),0)
        self.assertEqual(len(set(split['pool'])&set(split['test'])),0)
        old=torch.Tensor().type()
        torch.set_default_tensor_type('torch.DoubleTensor')
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                with extra_signal_functions(): saved=data.save_dataset(self.small,root/'data')
                train=DataManagerFile(str(root/'training_input'),'1',36,12,normalize=True,random_seed=201)
                pool=np.load(str(root/'training_input/sample_indices.npy'))
                np.testing.assert_array_equal(pool[train.sample_indices],saved['train'])
                self.assertEqual(train.dataset.shape[1],5)
                self.assertFalse((root/'training_input/true_Z.npy').exists())
        finally: torch.set_default_tensor_type(old)

    def test_hidden_shape_and_zero_parent_constant_block_without_optimization(self):
        torch.manual_seed(201)
        model=LearnableModel_NonLinGaussANM(5,2,10,norm_prod='paths')
        with torch.no_grad():
            model.adjacency[:,0]=0
            x=torch.as_tensor(np.random.RandomState(201).normal(size=(20,5))).type_as(next(model.parameters()))
            h=model.get_hidden_activations(x)
        self.assertEqual(tuple(h[0].shape),(20,5,10));self.assertEqual(tuple(h[1].shape),(20,5,10))
        np.testing.assert_array_equal(h[0][:,0,:].numpy(),np.repeat(h[0][0:1,0,:].numpy(),20,axis=0))

    def test_float64_weighted_paths_for_saved_model_diagnostics(self):
        old=torch.Tensor().type()
        torch.set_default_tensor_type('torch.DoubleTensor')
        try:
            model=LearnableModel_NonLinGaussANM(5,2,10,norm_prod='paths')
            for p in model.parameters():p.requires_grad=False
            with torch.no_grad():weighted=model.get_w_adj()
            self.assertEqual(tuple(weighted.shape),(5,5))
            self.assertEqual(weighted.dtype,torch.float64)
            self.assertTrue(bool(torch.isfinite(weighted).all().item()))
        finally:torch.set_default_tensor_type(old)

    def test_pca_and_fa_fit_only_supplied_training_rows(self):
        x=np.random.RandomState(201).normal(size=(40,5))
        for method in ['pca','factor_analysis']:
            bundle=fit_extractor(x[:30],x[:30],method,self.cfg['analysis'])
            np.testing.assert_allclose(bundle['scaler'].mean_,x[:30].mean(0))
            f,_,r=transform_extractor(bundle,x)
            self.assertEqual(f.shape,(40,1));self.assertEqual(r.shape,(40,5))
            changed=x.copy();changed[30:]+=1000
            np.testing.assert_allclose(f[:30],transform_extractor(bundle,changed)[0][:30],atol=1e-12)
            self.assertEqual(bundle['n_train'],30)

    def test_mlp_summary_removes_constant_blocks_and_is_train_only(self):
        rng=np.random.RandomState(201)
        h=rng.normal(size=(40,5,10));h[:,2,:]=7
        x=rng.normal(size=(40,5))
        bundle=fit_mlp_summary(h[:30],x[:30],self.cfg['analysis'])
        f,r,s=apply_mlp_summary(bundle,h)
        self.assertEqual(f.shape,(40,1));self.assertEqual(r.shape,(40,5));self.assertEqual(s.shape,(40,5))
        self.assertFalse(bundle['blocks'][2]['informative']);self.assertFalse(bundle['final']['keep'][2])
        np.testing.assert_array_equal(s[:,2],0)
        np.testing.assert_allclose(bundle['final']['scaler'].mean_,s[:30,bundle['final']['keep']].mean(0),atol=1e-12)
        changed=h.copy();changed[30:]+=1000
        np.testing.assert_allclose(f[:30],apply_mlp_summary(bundle,changed)[0][:30],atol=1e-12)
        with self.assertRaises(ValueError):fit_mlp_summary(np.ones((30,5,10)),x[:30],self.cfg['analysis'])
        self.assertEqual(list(inspect.signature(fit_mlp_summary).parameters),['hidden_train','x_train','settings'])

    def test_mask_orientation_and_multiple_parent_counts(self):
        a=np.zeros((5,5));a[0,2]=a[1,2]=a[3,1]=1
        rows=mask_summary(a,'train')
        self.assertEqual(rows[2]['parents'],'X1,X2');self.assertEqual(rows[2]['parent_count'],2)
        self.assertTrue(rows[2]['multiple_inputs']);self.assertEqual(rows[2]['hidden_mlp_axis_index'],2)
        self.assertEqual(sum(r['multiple_inputs'] for r in rows),1)
        with self.assertRaises(ValueError):mask_summary(np.eye(5),'bad')

    def test_latent_file_read_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            with forbid_latent_reads(Path(folder)) as audit:
                with self.assertRaises(PermissionError):np.load(str(Path(folder)/'data/true_Z.npy'))
                # The existing guard also blocks legacy supervised analysis/;
                # new unsupervised outputs must use the permitted directory.
                output=Path(folder)/'unsupervised_factors'
                output.mkdir()
                np.save(str(output/'allowed.npy'),np.zeros((2,1)))
                np.testing.assert_array_equal(np.load(str(output/'allowed.npy')),np.zeros((2,1)))
            self.assertEqual(len(audit['blocked_attempts']),1)

    def test_fixed_production_configuration(self):
        validate_config(self.cfg)
        bad=copy.deepcopy(self.cfg);bad['training']['seed']=202
        with self.assertRaises(ValueError):validate_config(bad)


if __name__=='__main__':unittest.main()
