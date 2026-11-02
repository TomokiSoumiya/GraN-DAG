"""Read-only source verification and train-only VAE preprocessing. No Z input."""
import json
from pathlib import Path

import numpy as np

from analyze_hidden import varying_columns
from gran_dag.hidden import load_checkpoint_cpu
from .additional import array_digest, digest, parameter_digest, verify_hidden


def validate_arrays(x, h, ids, labels, hidden_ids, hidden_labels):
    if x.shape != (1000, 3) or h.shape != (1000, 3, 10):
        raise ValueError('Expected X=(1000,3), H=(1000,3,10)')
    if not np.isfinite(x).all() or not np.isfinite(h).all():
        raise ValueError('Nonfinite observed features')
    np.testing.assert_array_equal(ids, np.arange(1000))
    np.testing.assert_array_equal(hidden_ids, ids)
    np.testing.assert_array_equal(hidden_labels, labels)
    if labels.shape != (1000,) or [int((labels == k).sum()) for k in range(3)] != [600, 200, 200]:
        raise ValueError('Expected original 600/200/200 split')


def fit_scaler(train, drop_constants):
    train = np.asarray(train, dtype=np.float64)
    if train.ndim != 2 or len(train) == 0 or not np.isfinite(train).all():
        raise ValueError('Finite training matrix required')
    varying = varying_columns(train)
    if drop_constants and not varying.any():
        raise ValueError('All hidden columns are constant')
    keep = varying if drop_constants else np.ones(train.shape[1], dtype=bool)
    mean, scale = train.mean(0), train.std(0, ddof=0)
    scale[~varying] = 1.0
    return dict(mean=mean, scale=scale, keep=keep, constant=~varying, n_fit=len(train))


def transform_scaler(values, scaler):
    if values.ndim != 2 or values.shape[1] != len(scaler['keep']) or not np.isfinite(values).all():
        raise ValueError('Invalid transform input')
    keep = scaler['keep']
    return (values[:, keep] - scaler['mean'][keep]) / scaler['scale'][keep]


def load_inputs(root):
    root = Path(root)
    config = json.loads((root/'config.json').read_text())
    d = config['data']
    if (d['seed'] != 201 or d['noise_std'] != 1 or d['latent_functions'] != ['tanh', 'sin', 'cubic']
            or config['split']['seed'] != 201 or config['training']['seed'] != 201):
        raise ValueError('Only the existing normalized seed201 condition is authorized')
    if d['signal_normalization'] != {'fit_split': 'train', 'ddof': 0}:
        raise ValueError('Wrong original signal normalization')
    data, hidden = root/'data', root/'train/hidden'
    x = np.load(str(data/'observed_X.npy'))
    ids = np.load(str(data/'sample_indices.npy'))
    labels = np.load(str(data/'sample_split.npy'))
    h = np.load(str(hidden/'hidden_layer_001.npy'))
    validate_arrays(x, h, ids, labels, np.load(str(hidden/'sample_indices.npy')),
                    np.load(str(hidden/'sample_split.npy')))
    for name, k in [('train', 0), ('validation', 1), ('test', 2)]:
        np.testing.assert_array_equal(np.sort(np.load(str(data/(name+'_indices.npy')))), ids[labels == k])
    pool = np.load(str(root/'training_input/sample_indices.npy'))
    np.testing.assert_array_equal(np.sort(pool), ids[labels != 2])
    np.testing.assert_array_equal(np.load(str(root/'training_input/data1.npy')), x[pool])
    mean = np.load(str(root/'normalization_mean.npy'))
    std = np.load(str(root/'normalization_std.npy'))
    np.testing.assert_allclose(mean, x[labels == 0].mean(0, keepdims=True), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(std, x[labels == 0].std(0, ddof=1, keepdims=True), atol=1e-12, rtol=1e-12)
    model = load_checkpoint_cpu(str(root/'train/model.pkl'))
    before, mask_before = parameter_digest(model), array_digest(model.adjacency.numpy())
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    if (model.num_vars, model.num_layers, model.hid_dim, model.nonlin) != (3, 2, 10, 'leaky-relu'):
        raise ValueError('Unexpected source architecture')
    # Existing verifier reproduces both saved layers, using original float64
    # normalization; no optimizer, dataset generation or file export is called.
    actual = verify_hidden(hidden, model, x, ids, labels, mean, std, root/'train/model.pkl')
    np.testing.assert_array_equal(h, actual[0])
    if before != parameter_digest(model) or mask_before != array_digest(model.adjacency.numpy()):
        raise ValueError('Frozen source weights or mask changed')
    flat = h.reshape(1000, 30)
    sx, sh = fit_scaler(x[labels == 0], False), fit_scaler(flat[labels == 0], True)
    verification = dict(checkpoint=str(root/'train/model.pkl'), checkpoint_sha256=digest(root/'train/model.pkl'),
        hidden_sha256=digest(hidden/'hidden_layer_001.npy'), source_parameters_sha256=before,
        mask_sha256=mask_before, mask=model.adjacency.numpy().tolist(),
        mask_orientation='source row -> target column', frozen_parameters_unchanged=True,
        all_source_parameters_require_grad_false=all(not p.requires_grad for p in model.parameters()),
        source_not_part_of_vae_optimizer=True, source_forward_reproduced_saved_hidden=True,
        sample_ids_and_splits_verified=True, original_training_excludes_final_test=True,
        n_train=600, n_validation=200, n_test=200, hidden_shape=list(h.shape),
        hidden_kept=int(sh['keep'].sum()), hidden_constant=int(sh['constant'].sum()),
        original_signal_normalization=json.loads((data/'config.json').read_text())['resolved_signal_normalization'])
    return dict(x=x, h=flat, ids=ids, labels=labels, sx=sx, sh=sh,
                target=transform_scaler(x, sx), features_A=transform_scaler(x, sx),
                features_B=transform_scaler(flat, sh), verification=verification)
