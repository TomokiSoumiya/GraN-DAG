# coding=utf-8
"""
GraN-DAG

Copyright © 2019 Authors of Gradient-Based Neural DAG Learning

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import argparse
import copy
import cdt

import numpy as np
import torch

from .models.learnables import LearnableModel_NonLinGauss, LearnableModel_NonLinGaussANM
from .train import pns, train, to_dag, cam_pruning, retrain
from .data import DataManagerFile
from .utils.save import load, dump
from .utils.metrics import edge_errors

def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)

def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))

def main(opt, metrics_callback=None, plotting_callback=None):
    """
    :param opt: a Bunch-like object containing hyperparameter values
    :param metrics_callback: a function of the form f(step, metrics_dict) used to log metric values during training

    """
    # Control as much randomness as possible
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)

    # Initialize metric logger if needed
    if metrics_callback is None:
        metrics_callback = _print_metrics

    # adjust some default hparams
    if opt.lr_reinit is None: opt.lr_reinit = opt.lr

    # Use GPU
    if opt.gpu:
        if opt.float:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        if opt.float:
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')

    # create experiment path
    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)

    # create learning model and ground truth model
    if opt.model == "NonLinGauss":
        model = LearnableModel_NonLinGauss(opt.num_vars, opt.num_layers, opt.hid_dim, nonlin=opt.nonlin,
                                           norm_prod=opt.norm_prod, square_prod=opt.square_prod)
    elif opt.model == "NonLinGaussANM":
        model = LearnableModel_NonLinGaussANM(opt.num_vars, opt.num_layers, opt.hid_dim, nonlin=opt.nonlin,
                                              norm_prod=opt.norm_prod,
                                              square_prod=opt.square_prod)
    else:
        raise ValueError("opt.model has to be in {NonLinGauss, NonLinGaussANM}")

    # create DataManager for training
    train_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data, random_seed=opt.random_seed)
    test_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=False,
                                normalize=opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                random_seed=opt.random_seed)

    def export_stage_hidden(stage_model, stage, checkpoint_name='model.pkl'):
        if not getattr(opt, 'export_hidden', False):
            return
        from .hidden import export_hidden_from_managers
        checkpoint = os.path.join(opt.exp_path, stage, checkpoint_name)
        info = export_hidden_from_managers(
            stage_model, train_data, test_data,
            os.path.join(opt.exp_path, stage, 'hidden'),
            split=opt.hidden_split, batch_size=opt.hidden_batch_size,
            metadata={'stage': stage, 'checkpoint': checkpoint,
                      'checkpoint_exists': os.path.isfile(checkpoint),
                      'source_data': train_data.data_path,
                      'random_seed': opt.random_seed,
                      'note': 'Current adjacency mask; CAM pruning does not retrain NN weights.'},
            overwrite=opt.hidden_overwrite)
        print('Hidden layers exported for {}: {}'.format(
            stage, [layer['shape'] for layer in info['layers']]))

    # save gt adjacency
    dump(train_data.adjacency.detach().cpu().numpy(), opt.exp_path, 'gt-adjacency')

    # apply preliminary neighborhood selection
    if opt.pns:
        if opt.num_neighbors is None:
            num_neighbors = opt.num_vars
        else:
            num_neighbors = opt.num_neighbors
        print("Making pns folder")
        pns(model, train_data, test_data, num_neighbors, opt.pns_thresh, opt.exp_path, metrics_callback,
            plotting_callback)

    # train until constraint is sufficiently close to being satisfied
    if opt.train:
        if file_exists(opt.exp_path, "pns"):
            print("Training with pns folder")
            model = load(os.path.join(opt.exp_path, "pns"), "model.pkl")
        else:
            print("Training from scratch")
        trained = train(model, train_data.adjacency.detach().cpu().numpy(), train_data, test_data, opt, metrics_callback,
                        plotting_callback)
        # If training hit its iteration limit without saving a checkpoint,
        # extraction still works on the in-memory model at that point.
        export_stage_hidden(trained if trained is not None else model, 'train')

    # remove edges until we have a DAG
    if opt.to_dag:
        # load model
        assert file_exists(opt.exp_path, "train"), \
            "The /train folder is required to run --to_dag. Add --train to the command line"
        model = load(os.path.join(opt.exp_path, "train"), "model.pkl")

        # run
        dag_model = to_dag(model, train_data, test_data, opt, metrics_callback, plotting_callback)
        export_stage_hidden(dag_model, 'to-dag')

    # do further pruning of the DAG
    if opt.cam_pruning:
        old_adj = np.eye(opt.num_vars)  # useful to not train the same graph many times...
        # if opt.cam_pruning is iterable, extract the different cutoffs, otherwise use only the cutoff specified
        try:
            cam_pruning_cutoff = [float(i) for i in opt.cam_pruning_cutoff]
        except:
            cam_pruning_cutoff = [float(opt.cam_pruning_cutoff)]
        for cutoff in cam_pruning_cutoff:
            # load model
            assert file_exists(opt.exp_path, "to-dag"), \
                "The /to-dag folder is required to run --cam-pruning. Add --to-dag to the command line"
            model = load(os.path.join(opt.exp_path, "to-dag"), "model.pkl")

            # run
            model = cam_pruning(model, train_data, test_data, opt, cutoff=cutoff,
                                metrics_callback=metrics_callback, plotting_callback=plotting_callback)
            export_stage_hidden(model, 'cam-pruning/cutoff_%.6f' % cutoff)
            current_adj = model.adjacency.detach().cpu().numpy()

            # retrain model from scratch using the learned graph (and make sure not learning on the same graph twice)
            # If we decide not to retrain, the metrics can be obtained by looking at the previous cutoff value
            if opt.retrain and (current_adj != old_adj).any():
                path_cam = "cam-pruning/cutoff_%.6f" % cutoff
                print("Retraining NNs with estimated dag from {}".format(path_cam))
                model.reset_params()
                retrain(model, train_data, test_data, path_cam, opt, metrics_callback, plotting_callback)
                if getattr(opt, 'export_hidden', False):
                    stage = 'retrain_' + path_cam
                    best = os.path.join(opt.exp_path, stage, 'best-model.pkl')
                    export_stage_hidden(load(os.path.dirname(best), 'best-model.pkl')
                                        if os.path.isfile(best) else model, stage, 'best-model.pkl')
                old_adj = current_adj

    # if no cam pruning, run it on to-dag exclusively
    elif opt.retrain:
        assert file_exists(opt.exp_path, "to-dag"), \
            "The /to-dag folder is required to run --retrain. Add --to-dag to the command line"
        model = load(os.path.join(opt.exp_path, "to-dag"), "model.pkl")

        print("Retraining NNs with estimated dag from /to-dag")
        model.reset_params()
        retrain(model, train_data, test_data, "to-dag", opt, metrics_callback, plotting_callback)
        if getattr(opt, 'export_hidden', False):
            stage = 'retrain_to-dag'
            best = os.path.join(opt.exp_path, stage, 'best-model.pkl')
            export_stage_hidden(load(os.path.dirname(best), 'best-model.pkl')
                                if os.path.isfile(best) else model, stage, 'best-model.pkl')
