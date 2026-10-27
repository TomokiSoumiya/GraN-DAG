"""Batch export of post-activation MLP hidden layers; no optimization is run."""
import hashlib
import io
import json
import os
import pickle

import numpy as np
import torch

from .data import DataManagerFile


class _CPUUnpickler(pickle.Unpickler):
    """Read this repository's model pickles even if tensors were saved on CUDA."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu")
        return super().find_class(module, name)


def load_checkpoint_cpu(path):
    """Load a trusted model.pkl (or best-model.pkl) without a GPU dependency."""
    with open(path, "rb") as stream:
        model = _CPUUnpickler(stream).load()
    model.cpu()
    # adjacency is a plain Tensor in the original model, not a registered buffer.
    model.adjacency = model.adjacency.cpu()
    return model


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_hidden_activations(model, data, output_dir, batch_size=256,
                            sample_indices=None, sample_split=None,
                            variable_names=None, mean=None, std=None,
                            metadata=None, overwrite=False):
    """Export a preprocessed (N, D) tensor using the model's current graph mask.

    hidden_layer_001.npy, ... contain (N, D, H) post-activation values.
    Axis 1 is the MLP predicting the corresponding original data column.
    sample_indices.npy links rows to the input .npy (zero-based).
    sample_split.npy uses 0=train, 1=validation, 2=final test,
    -1=new/unassigned input (older exports label 1 as test/validation).
    Files are written by batch using NumPy memmaps to limit activation memory.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if data.dim() != 2 or data.size(1) != model.num_vars or data.size(0) == 0:
        raise ValueError("data must have nonempty shape (samples, model.num_vars)")
    if model.num_layers < 1:
        raise ValueError("model has no hidden layers")
    if not bool(torch.isfinite(data).all().item()):
        raise ValueError("preprocessed data contains NaN or infinity")
    count = data.size(0)
    if sample_indices is None:
        sample_indices = np.arange(count, dtype=np.int64)
    if sample_split is None:
        sample_split = np.full(count, -1, dtype=np.int8)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    sample_split = np.asarray(sample_split, dtype=np.int8)
    if sample_indices.shape != (count,) or sample_split.shape != (count,):
        raise ValueError("sample indices/split must have shape (samples,)")
    if variable_names is None:
        variable_names = ["X{}".format(i + 1) for i in range(model.num_vars)]
    if (len(variable_names) != model.num_vars or
            not all(isinstance(name, str) for name in variable_names) or
            len(set(variable_names)) != len(variable_names)):
        raise ValueError("variable_names must contain one unique string per data column")
    if mean is None:
        mean = np.zeros((1, model.num_vars))
    if std is None:
        std = np.ones((1, model.num_vars))

    filenames = ["hidden_layer_{:03d}.npy".format(i + 1)
                 for i in range(model.num_layers)]
    filenames += ["sample_indices.npy", "sample_split.npy", "mean.npy", "std.npy",
                  "adjacency.npy", "metadata.json"]
    if not overwrite:
        existing = [name for name in filenames if os.path.exists(os.path.join(output_dir, name))]
        if existing:
            raise FileExistsError("Export already exists in {} (use --overwrite): {}".format(
                output_dir, ", ".join(existing)))
    os.makedirs(output_dir, exist_ok=True)

    weights = next(model.parameters())
    was_training = model.training
    zero_weights_ratio = model.zero_weights_ratio
    arrays = []
    layers = []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, count, batch_size):
                batch = data[start:start + batch_size].to(
                    device=weights.device, dtype=weights.dtype)
                hidden = model.get_hidden_activations(batch)
                for layer_index, tensor in enumerate(hidden):
                    values = tensor.detach().cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite hidden values in layer {}".format(layer_index + 1))
                    if start == 0:
                        filename = filenames[layer_index]
                        shape = (count, model.num_vars, values.shape[2])
                        arrays.append(np.lib.format.open_memmap(
                            os.path.join(output_dir, filename), mode="w+",
                            dtype=values.dtype, shape=shape))
                        layers.append({"layer": layer_index + 1, "file": filename,
                                       "shape": list(shape), "dtype": str(values.dtype)})
                    arrays[layer_index][start:start + len(values)] = values
        for array in arrays:
            array.flush()
    finally:
        model.train(was_training)
        model.zero_weights_ratio = zero_weights_ratio

    np.save(os.path.join(output_dir, "sample_indices.npy"), sample_indices)
    np.save(os.path.join(output_dir, "sample_split.npy"), sample_split)
    np.save(os.path.join(output_dir, "mean.npy"), mean)
    np.save(os.path.join(output_dir, "std.npy"), std)
    np.save(os.path.join(output_dir, "adjacency.npy"), model.adjacency.detach().cpu().numpy())
    info = dict(metadata or {})
    info.update({
        "format_version": 1,
        "activation_position": "after activation function; before next affine layer",
        "activation_function": model.nonlin,
        "axes": ["sample", "target_variable_mlp", "hidden_unit"],
        "layer_order": "1-based, from input to output; density output layer excluded",
        "variables": [{"axis_index": i, "input_column": i, "name": name}
                      for i, name in enumerate(variable_names)],
        "sample_indices": "sample_indices.npy; zero-based row IDs in source data",
        "sample_split": {"file": "sample_split.npy", "codes": {
            "0": "train", "1": "test/validation", "2": "final test", "-1": "new input"}},
        "normalization_formula": "(raw_data - mean.npy) / std.npy",
        "adjacency_file": "adjacency.npy; source row -> target column",
        "layers": layers,
    })
    with open(os.path.join(output_dir, "metadata.json"), "w") as stream:
        json.dump(info, stream, indent=2, ensure_ascii=False)
    return info


def export_hidden_from_managers(model, train_data, test_data, output_dir,
                                split="all", batch_size=256, variable_names=None,
                                metadata=None, overwrite=False):
    """Use exactly the tensors/normalization already prepared for training."""
    if split not in ("all", "train", "test"):
        raise ValueError("split must be all, train, or test")
    managers = [(train_data, 0)] if split == "train" else [(test_data, 1)]
    if split == "all":
        managers = [(train_data, 0), (test_data, 1)]
    tensors = [manager.dataset for manager, _ in managers]
    indices = np.concatenate([manager.sample_indices for manager, _ in managers])
    labels = np.concatenate([np.full(manager.num_samples, label, dtype=np.int8)
                             for manager, label in managers])
    # Restore original source row order to align with later latent ground truth.
    order = np.argsort(indices)
    data = torch.cat(tensors, dim=0)
    data = data[torch.as_tensor(order, device=data.device).long()]
    mean = None if train_data.mean is None else train_data.mean.detach().cpu().numpy()
    std = None if train_data.std is None else train_data.std.detach().cpu().numpy()
    info = dict(metadata or {})
    info.update({"split": split, "row_order": "ascending original row index",
                 "normalized": train_data.mean is not None})
    return save_hidden_activations(
        model, data, output_dir, batch_size, indices[order], labels[order],
        variable_names, mean, std, info, overwrite)


def extract_from_checkpoint(model_path, data_path=None, output_dir=None,
                            split="all", batch_size=256, input_data=None,
                            variable_names=None, overwrite=False):
    """Load a checkpoint and its opt.pkl; reproduce original preprocessing.

    data_path overrides the original training directory (e.g. /dataset in a
    container). It must contain the unchanged training data and DAG so that
    the original shuffled split and train-only normalization can be rebuilt.
    Optional input_data is a NEW (N, D) observed-only .npy; it uses those same
    training statistics, keeps source row order, and needs no ground-truth DAG.
    """
    model_path = os.path.abspath(model_path)
    with open(os.path.join(os.path.dirname(model_path), "opt.pkl"), "rb") as stream:
        opt = pickle.load(stream)
    model = load_checkpoint_cpu(model_path)
    data_path = data_path or opt.data_path
    output_dir = output_dir or os.path.join(os.path.dirname(model_path), "hidden")
    if input_data is not None and split != "all":
        raise ValueError("--input-data must be used with --split all")

    # DataManagerFile uses the global Tensor type, exactly as gran_dag.main.
    # Rebuild with the ORIGINAL training precision even when exporting on CPU.
    previous_type = torch.tensor(0.).type()
    try:
        torch.set_default_tensor_type("torch.FloatTensor" if opt.float else "torch.DoubleTensor")
        train_data = DataManagerFile(data_path, opt.i_dataset, opt.train_samples,
                                     opt.test_samples, train=True,
                                     normalize=opt.normalize_data, random_seed=opt.random_seed)
        test_data = DataManagerFile(data_path, opt.i_dataset, opt.train_samples,
                                    opt.test_samples, train=False,
                                    normalize=opt.normalize_data, mean=train_data.mean,
                                    std=train_data.std, random_seed=opt.random_seed)
        if train_data.dataset.size(1) != model.num_vars:
            raise ValueError("training data columns do not match checkpoint")
        if opt.normalize_data and (not bool(torch.isfinite(train_data.std).all().item()) or
                                   bool((train_data.std <= 0).any().item())):
            raise ValueError("training normalization has zero or invalid standard deviation")
        info = {"checkpoint": model_path, "checkpoint_sha256": _file_sha256(model_path),
                "training_data": os.path.abspath(train_data.data_path),
                "training_data_sha256": _file_sha256(train_data.data_path),
                "dataset_index": str(opt.i_dataset), "random_seed": opt.random_seed,
                "train_samples_option": opt.train_samples,
                "test_samples_option": opt.test_samples,
                "preprocessing_source": "original opt.pkl and unchanged training data"}
        if input_data is None:
            info["source_data"] = info["training_data"]
            return export_hidden_from_managers(
                model, train_data, test_data, output_dir, split, batch_size,
                variable_names, info, overwrite)
        raw = np.load(input_data, allow_pickle=False)
        if raw.ndim != 2 or raw.shape[1] != model.num_vars:
            raise ValueError("input_data must have shape (samples, model.num_vars)")
        data = torch.as_tensor(raw).type(torch.Tensor)
        mean, std = None, None
        if opt.normalize_data:
            data = (data - train_data.mean) / train_data.std
            mean = train_data.mean.cpu().numpy()
            std = train_data.std.cpu().numpy()
        info.update({"source_data": os.path.abspath(input_data),
                     "source_data_sha256": _file_sha256(input_data),
                     "normalized": opt.normalize_data, "split": "new input",
                     "row_order": "original input row order"})
        return save_hidden_activations(
            model, data, output_dir, batch_size, variable_names=variable_names,
            mean=mean, std=std, metadata=info, overwrite=overwrite)
    finally:
        torch.set_default_tensor_type(previous_type)
