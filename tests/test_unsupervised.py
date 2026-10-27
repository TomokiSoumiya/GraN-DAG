"""Unsupervised-only fitting, split/row safety, constant features and frozen replay."""
import copy
import inspect
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from latent_experiments.config import load_config
from latent_experiments.data import save_dataset
from latent_experiments.unsupervised import (
    METHODS, digest, evaluate_experiment, extract_experiment, factor_correlations,
    fit_extractor, forbid_latent_reads, load_features, read_csv, transform_extractor)


class UnsupervisedTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads(Path('configs/latent_unsupervised.json').read_text())
        self.settings['factor_analysis'].update(max_iter=300, tol=.001)

    def test_fit_api_has_no_z_and_uses_training_only(self):
        self.assertEqual(list(inspect.signature(fit_extractor).parameters),
                         ['features_train','x_train','method','settings'])
        rng = np.random.RandomState(8)
        x = rng.normal(size=(80,3))
        values = np.column_stack([x, x[:,0]**2, np.ones(len(x))])
        changed = values.copy()
        changed[50:,:] += 10000
        for method in METHODS:
            with patch('latent_experiments.evaluate.fit_probe', side_effect=AssertionError('No Z regression')):
                a = fit_extractor(values[:50],x[:50],method,self.settings)
                b = fit_extractor(changed[:50],x[:50],method,self.settings)
            self.assertFalse(a['keep'][-1])
            np.testing.assert_allclose(a['scaler'].mean_,values[:50,:4].mean(0),rtol=1e-14,atol=1e-14)
            np.testing.assert_array_equal(a['extractor'].components_,b['extractor'].components_)
            fa,ha,xa = transform_extractor(a,values)
            fb,_,_ = transform_extractor(b,changed)
            self.assertEqual(fa.shape,(80,1))
            self.assertEqual(ha.shape,values.shape)
            self.assertEqual(xa.shape,x.shape)
            np.testing.assert_array_equal(fa[:50],fb[:50])
            restored = pickle.loads(pickle.dumps(a))
            for old,new in zip(transform_extractor(a,values),transform_extractor(restored,values)):
                np.testing.assert_array_equal(old,new)
            design = np.column_stack([np.ones(50),fa[:50]])
            np.testing.assert_allclose(a['x_decoder'],np.linalg.lstsq(design,x[:50],rcond=None)[0])

    def test_constant_and_nearly_constant_columns(self):
        rng = np.random.RandomState(7)
        x = rng.normal(size=(30,3))
        for method in METHODS:
            values = np.ones((30,5))*3
            bundle = fit_extractor(values[:20],x[:20],method,self.settings)
            factors,features,xhat = transform_extractor(bundle,values)
            np.testing.assert_array_equal(factors,np.zeros((30,1)))
            np.testing.assert_array_equal(features,values)
            np.testing.assert_allclose(xhat,np.repeat(x[:20].mean(0,keepdims=True),30,axis=0))
            self.assertFalse(factor_correlations(factors[:,0],x[:,0])['correlation_defined'])
            self.assertIsNone(factor_correlations(factors[:,0],x[:,0])['pearson_r'])
            nearly = np.column_stack([x[:,0],3+np.arange(30)*1e-15])
            fitted = fit_extractor(nearly[:20],x[:20],method,self.settings)
            np.testing.assert_array_equal(fitted['keep'],[True,False])

    def test_read_guard_blocks_z_and_old_supervised_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'data').mkdir()
            (root/'analysis').mkdir()
            np.save(str(root/'data/true_Z.npy'),np.arange(10))
            np.save(str(root/'data/observed_X.npy'),np.ones((10,3)))
            (root/'analysis/test_predictions.csv').write_text('Z values must not be read')
            with forbid_latent_reads(root) as audit:
                np.testing.assert_array_equal(np.load(str(root/'data/observed_X.npy')),np.ones((10,3)))
                for operation in [lambda:np.load(str(root/'data/true_Z.npy')),
                                  lambda:(root/'data/true_Z.npy').read_bytes(),
                                  lambda:io.open(str(root/'data/true_Z.npy'),'rb'),
                                  lambda:(root/'analysis/test_predictions.csv').read_text()]:
                    with self.assertRaises(PermissionError):
                        operation()
            self.assertEqual(len(audit['blocked_attempts']),4)
            np.testing.assert_array_equal(np.load(str(root/'data/true_Z.npy')),np.arange(10))

    def fixture(self,root):
        config = load_config('configs/latent_linear.json')
        config['data']['n_samples'] = 40
        (root/'config.json').write_text(json.dumps(config))
        save_dataset(config,root/'data')
        x = np.load(str(root/'data/observed_X.npy'))
        ids = np.load(str(root/'data/sample_indices.npy'))
        labels = np.load(str(root/'data/sample_split.npy'))
        np.save(str(root/'normalization_mean.npy'),x[labels==0].mean(0,keepdims=True))
        np.save(str(root/'normalization_std.npy'),x[labels==0].std(0,ddof=1,keepdims=True))
        (root/'data_manifest.json').write_text(json.dumps(dict(data_files={
            'true_Z.npy':digest(root/'data/true_Z.npy')})))
        (root/'fixture_features').mkdir()
        features = [('X',x,dict(condition='X',feature_kind='observed_X',weights='none',mask='none',
                                layer=0,source='data/observed_X.npy',columns=['X1','X2','X3']))]
        for code in ('C','E','B','D'):
            for layer in (1,2):
                h = np.column_stack([x,np.tanh(x),np.ones((len(x),2))])
                if code in ('E','D'):
                    h[:,2] = 0
                name = code+'_L'+str(layer)
                source = 'fixture_features/'+name+'.npy'
                np.save(str(root/source),h)
                features.append((name,h,dict(condition=code,feature_kind='hidden',weights='fixture',
                     mask='fixture',layer=layer,source=source,columns=['u'+str(j) for j in range(h.shape[1])])) )
        return features,x,ids,labels,dict(fixture_only=True)

    def test_row_and_split_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            ids = np.load(str(root/'data/sample_indices.npy'))
            swapped = ids.copy()
            swapped[[0,1]] = swapped[[1,0]]
            np.save(str(root/'data/sample_indices.npy'),swapped)
            with self.assertRaises(AssertionError):
                load_features(root,root/'out',self.settings)
            np.save(str(root/'data/sample_indices.npy'),ids)
            labels = np.load(str(root/'data/sample_split.npy'))
            train,test = np.flatnonzero(labels==0)[0],np.flatnonzero(labels==2)[0]
            labels[train],labels[test] = labels[test],labels[train]
            np.save(str(root/'data/sample_split.npy'),labels)
            with self.assertRaises(AssertionError):
                load_features(root,root/'out',self.settings)

    def test_end_to_end_label_change_cannot_change_extraction_or_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self.fixture(root)
            original_z = np.load(str(root/'data/true_Z.npy'))
            output = root/'unsupervised'
            with patch('latent_experiments.unsupervised.load_features',return_value=fixture), \
                 patch('latent_experiments.unsupervised.factor_correlations',side_effect=AssertionError('No Z evaluation yet')), \
                 patch('latent_experiments.additional.fit_probe',side_effect=AssertionError('No supervised fit')):
                extract_experiment(root,output,self.settings)
                np.save(str(root/'data/true_Z.npy'),original_z*1000+999)
                repeated = root/'repeat'
                extract_experiment(root,repeated,self.settings)
                # Extraction must also work when no ground-truth Z file exists.
                (root/'data/true_Z.npy').unlink()
                no_z = root/'without_z'
                extract_experiment(root,no_z,self.settings)
            # Factors/models do not depend on even drastically changed latent labels.
            for path in (output/'factors').glob('*.npy'):
                self.assertEqual(digest(path),digest(repeated/'factors'/path.name))
                self.assertEqual(digest(path),digest(no_z/'factors'/path.name))
            self.assertEqual(len(read_csv(output/'extracted_factors.csv')),18*40)
            saved = json.loads((output/'extraction_status.json').read_text())
            self.assertEqual(saved['z_read_attempts'],[])
            self.assertEqual(saved['extractor_count'],18)
            np.save(str(root/'data/true_Z.npy'),original_z)
            before = {p.name:digest(p) for p in (output/'factors').glob('*.npy')}
            with patch('latent_experiments.unsupervised.fit_extractor',side_effect=AssertionError('Evaluation must not fit')):
                metrics = evaluate_experiment(root,output)
            self.assertEqual(len(metrics),54)
            self.assertEqual(len(list((output/'figures').glob('*.png'))),6)
            self.assertEqual(before,{p.name:digest(p) for p in (output/'factors').glob('*.npy')})
            row = next(r for r in metrics if r['split']=='test' and r['condition']=='X' and r['method']=='pca')
            factors = np.load(str(output/'factors/X_pca.npy'))[:,0]
            test = fixture[3]==2
            self.assertAlmostEqual(row['pearson_r'],np.corrcoef(factors[test],original_z[test,0])[0,1])
            with self.assertRaises(FileExistsError):
                extract_experiment(root,output,self.settings)
            with self.assertRaises(FileExistsError):
                evaluate_experiment(root,output)

    def test_frozen_artifact_tampering_rejected_before_z_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = self.fixture(root)
            output = root/'unsupervised'
            with patch('latent_experiments.unsupervised.load_features',return_value=fixture):
                extract_experiment(root,output,self.settings)
            path = output/'factors/X_pca.npy'
            values = np.load(str(path))
            values[0,0] += 1
            np.save(str(path),values)
            with self.assertRaisesRegex(ValueError,'Frozen extraction changed'):
                evaluate_experiment(root,output)


if __name__ == '__main__':
    unittest.main()
