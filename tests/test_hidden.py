"""Regression tests runnable with Python 3.7/PyTorch in the original container."""
import argparse
import json
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from gran_dag.data import DataManagerFile
from gran_dag.hidden import (extract_from_checkpoint, export_hidden_from_managers,
                             save_hidden_activations)
from gran_dag.models.learnables import LearnableModel_NonLinGaussANM, LearnableModel_NonLinGauss


class HiddenTests(unittest.TestCase):
    def setUp(self):
        self.previous_type = torch.tensor(0.).type()
        torch.set_default_tensor_type('torch.DoubleTensor')
        torch.manual_seed(17)
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.raw = np.random.RandomState(7).normal(size=(20, 3)) * [2., 3., 4.] + [1., 5., 10.]
        np.save(os.path.join(self.root, 'data1.npy'), self.raw)
        np.save(os.path.join(self.root, 'DAG1.npy'), np.zeros((3, 3)))

    def tearDown(self):
        self.temp.cleanup()
        torch.set_default_tensor_type(self.previous_type)

    def model(self, kind=LearnableModel_NonLinGaussANM, nonlin='leaky-relu', layers=2):
        model = kind(3, layers, 4, nonlin=nonlin, norm_prod='paths')
        model.adjacency[0, 1] = 0
        return model

    def managers(self, normalize):
        train = DataManagerFile(self.root, '1', 0.6, 5, normalize=normalize, random_seed=9)
        test = DataManagerFile(self.root, '1', 0.6, 5, train=False,
                               normalize=normalize, mean=train.mean, std=train.std, random_seed=9)
        return train, test

    def test_all_mlps_match_independent_linear_layers_and_gradients(self):
        # Independent per-variable computation checks graph orientation, each
        # layer and activation, and the unchanged likelihood forward return.
        for kind in (LearnableModel_NonLinGaussANM, LearnableModel_NonLinGauss):
            for nonlin in ('leaky-relu', 'sigmoid'):
                model = self.model(kind, nonlin)
                x = torch.as_tensor(self.raw[:6]).requires_grad_(True)
                weights, biases = model.get_parameters(mode='wb')
                outputs = model.forward_given_params(x, weights, biases)
                density, hidden = model.forward_given_params(x, weights, biases, return_hidden=True)
                reference_outputs = []
                for j in range(3):
                    value = x * model.adjacency[:, j]
                    for k in range(model.num_layers + 1):
                        value = F.linear(value, weights[k][j], biases[k][j])
                        if k < model.num_layers:
                            value = F.leaky_relu(value) if nonlin == 'leaky-relu' else torch.sigmoid(value)
                            np.testing.assert_allclose(hidden[k][:, j].detach().numpy(),
                                                       value.detach().numpy(), rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(outputs[j].detach().numpy(), value.detach().numpy(), atol=1e-12)
                    self.assertTrue(torch.equal(outputs[j], density[j]))
                    reference_outputs.append(value)
                loss = sum(t.sum() for t in outputs)
                captured_loss = sum(t.sum() for t in density)
                targets = [x] + list(model.parameters())[:2]
                original_grads = torch.autograd.grad(loss, targets, retain_graph=True)
                captured_grads = torch.autograd.grad(captured_loss, targets)
                reference_grads = torch.autograd.grad(sum(t.sum() for t in reference_outputs), targets)
                for a, b, reference in zip(original_grads, captured_grads, reference_grads):
                    self.assertTrue(torch.equal(a, b))
                    np.testing.assert_allclose(a.detach().numpy(), reference.detach().numpy(),
                                               atol=1e-12, rtol=1e-12)

    def test_export_row_mapping_batches_state_and_overwrite_guard(self):
        model = self.model()
        train, test = self.managers(False)
        model.train()
        ratio = model.zero_weights_ratio
        params = [p.detach().clone() for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        for batch_size in (1, 7, 100):
            output = os.path.join(self.root, 'batch{}'.format(batch_size))
            info = export_hidden_from_managers(model, train, test, output, batch_size=batch_size)
            ids = np.load(os.path.join(output, 'sample_indices.npy'))
            expected_ids = np.sort(np.r_[train.sample_indices, test.sample_indices])
            np.testing.assert_array_equal(ids, expected_ids)
            labels = np.load(os.path.join(output, 'sample_split.npy'))
            np.testing.assert_array_equal(ids[labels == 0], np.sort(train.sample_indices))
            expected = model.get_hidden_activations(torch.as_tensor(self.raw[ids]))
            for k in range(2):
                actual = np.load(os.path.join(output, 'hidden_layer_{:03d}.npy'.format(k+1)))
                self.assertEqual(actual.shape, (17, 3, 4))
                np.testing.assert_allclose(actual, expected[k].detach().numpy(), atol=1e-12)
            self.assertEqual(info['variables'][2], {'axis_index': 2, 'input_column': 2, 'name': 'X3'})
            with self.assertRaises(FileExistsError):
                export_hidden_from_managers(model, train, test, output)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        # Direct forward changes only the existing diagnostic ratio; reset it
        # before checking that file export restores the value it was given.
        model.zero_weights_ratio = ratio
        save_hidden_activations(model, train.dataset, os.path.join(self.root, 'state'))
        self.assertEqual(model.zero_weights_ratio, ratio)
        for before, after in zip(params, model.parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_saved_checkpoint_preprocessing_and_new_input(self):
        for normalize in (False, True):
            model = self.model()
            stage = os.path.join(self.root, 'stage{}'.format(normalize))
            os.makedirs(stage)
            model_path = os.path.join(stage, 'model.pkl')
            opt = argparse.Namespace(data_path='/old/container/path', i_dataset='1', train_samples=0.6,
                                     test_samples=5, normalize_data=normalize, random_seed=9, float=False)
            for name, obj in [('model.pkl', model), ('opt.pkl', opt)]:
                with open(os.path.join(stage, name), 'wb') as f:
                    pickle.dump(obj, f)
            train, test = self.managers(normalize)
            output = os.path.join(stage, 'hidden')
            info = extract_from_checkpoint(model_path, self.root, output, batch_size=5,
                                            variable_names=['A', 'B', 'C'])
            ids = np.load(os.path.join(output, 'sample_indices.npy'))
            mean = np.load(os.path.join(output, 'mean.npy'))
            std = np.load(os.path.join(output, 'std.npy'))
            expected = torch.as_tensor(self.raw[ids])
            if normalize:
                np.testing.assert_allclose(mean, self.raw[train.sample_indices].mean(0, keepdims=True))
                np.testing.assert_allclose(std, self.raw[train.sample_indices].std(0, ddof=1, keepdims=True))
                expected = (expected - train.mean) / train.std
            np.testing.assert_allclose(np.load(os.path.join(output, 'hidden_layer_001.npy')),
                                       model.get_hidden_activations(expected)[0].detach().numpy(), atol=1e-12)
            self.assertEqual(info['variables'][0]['name'], 'A')
            self.assertEqual(len(info['checkpoint_sha256']), 64)
            for split, manager in [('train', train), ('test', test)]:
                split_dir = os.path.join(stage, split)
                extract_from_checkpoint(model_path, self.root, split_dir, split=split)
                np.testing.assert_array_equal(np.load(os.path.join(split_dir, 'sample_indices.npy')),
                                              np.sort(manager.sample_indices))
            new_path = os.path.join(self.root, 'new.npy')
            # The new data has a different distribution. Normalization must
            # still use training statistics, never statistics of this new file.
            new = self.raw[:4] + 20
            np.save(new_path, new)
            new_dir = os.path.join(stage, 'new')
            extract_from_checkpoint(model_path, self.root, new_dir, input_data=new_path)
            expected = torch.as_tensor(new)
            if normalize:
                expected = (expected - train.mean) / train.std
            np.testing.assert_allclose(np.load(os.path.join(new_dir, 'hidden_layer_002.npy')),
                                       model.get_hidden_activations(expected)[1].detach().numpy(), atol=1e-12)
            self.assertTrue((np.load(os.path.join(new_dir, 'sample_split.npy')) == -1).all())

    def test_float_model_no_checkpoint_and_bad_inputs(self):
        # Export also accepts a model that exists only in memory.
        model = self.model(layers=3).float()
        model.adjacency = model.adjacency.float()
        data = torch.as_tensor(self.raw).float()
        output = os.path.join(self.root, 'float')
        info = save_hidden_activations(model, data, output, batch_size=3)
        self.assertEqual(len(info['layers']), 3)
        self.assertEqual(np.load(os.path.join(output, 'hidden_layer_003.npy')).dtype, np.float32)
        for bad_data in (torch.ones(2, 4), torch.ones(0, 3), torch.full((2, 3), float('nan'))):
            with self.assertRaises(ValueError):
                save_hidden_activations(model, bad_data, os.path.join(self.root, 'bad'))
        with self.assertRaises(ValueError):
            save_hidden_activations(model, data, output, batch_size=0)

    def test_training_flow_exports_in_memory_when_no_checkpoint_is_saved(self):
        from gran_dag.main import main
        opt = argparse.Namespace(random_seed=9, lr=0.001, lr_reinit=None, gpu=False,
                                 float=False, exp_path=os.path.join(self.root, 'flow'),
                                 model='NonLinGaussANM', num_vars=3, num_layers=2,
                                 hid_dim=4, nonlin='leaky-relu', norm_prod='paths',
                                 square_prod=False, data_path=self.root, i_dataset='1',
                                 train_samples=0.6, test_samples=5, normalize_data=True,
                                 pns=False, train=True, to_dag=False, cam_pruning=False,
                                 retrain=False, export_hidden=True, hidden_split='all',
                                 hidden_batch_size=3, hidden_overwrite=False)
        with patch('gran_dag.main.train', return_value=None) as train_call:
            main(opt)
        train_call.assert_called_once()
        output = os.path.join(opt.exp_path, 'train', 'hidden')
        self.assertTrue(os.path.isfile(os.path.join(output, 'hidden_layer_001.npy')))
        with open(os.path.join(output, 'metadata.json')) as stream:
            self.assertFalse(json.load(stream)['checkpoint_exists'])


if __name__ == '__main__':
    unittest.main()
