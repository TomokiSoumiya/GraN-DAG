# GraN-DAG hidden activation export

## What is captured

`gran_dag/models/base_model.py:BaseModel.forward_given_params` implements all
per-variable MLPs together with `torch.einsum`. The first layer applies the
model's current `adjacency` mask. After every hidden affine transform and
Leaky ReLU/sigmoid, the internal tensor already has shape `(N, D, H)`:

- `N`: number of exported samples;
- `D`: number of observed variables / target MLPs;
- `H`: hidden units in that layer.

`model.get_hidden_activations(preprocessed_tensor)` returns these tensors in
input-to-output layer order. Layer 1 is the first hidden layer, not the input.
The density-parameter output layer is excluded. The existing likelihood forward
return is unchanged unless `return_hidden=True` is explicitly requested.

For example, `hidden_layer_001.npy[:, 2, :]` contains the first hidden layer of
the MLP predicting the third observed variable (input column 2, default name
`X3`). These are post-activation values, not individual NN weights or estimates
of latent variables.

## Extract an existing trained model

With Python/PyTorch already installed in the original environment:

```bash
python extract_hidden.py \
  --model-path experiments/p10-run1/train/model.pkl \
  --data-path data/unpacked/data_p10_e10_n1000_GP \
  --batch-size 128
```

On the prepared host, use the existing container dependencies:

```bash
cd /work/somiya/research/GraN-DAG
bash run_in_container.sh python extract_hidden.py \
  --model-path /final_log/p10-run1/train/model.pkl \
  --data-path /dataset/unpacked/data_p10_e10_n1000_GP \
  --batch-size 128
```

The helper maps `data/` to `/dataset`, the entire `experiments/` directory to
`/final_log`, and repository code to `/code`. These mounts differ from the
single-experiment mounts in `start_example.sh`. On this host, container commands
must run outside the coding agent's filesystem sandbox (a normal host terminal
works); `bwrap` is blocked inside that sandbox.

Default output is `<checkpoint directory>/hidden/`. Use `--output-dir` to change
it, `--split train` or `--split test` to restrict rows, and `--overwrite` to
explicitly replace an earlier export. By default, existing export files are
protected. Use only trusted checkpoint/option pickle files.

Any saved stage model can be used, including `to-dag/model.pkl`,
`cam-pruning/cutoff_0.001000/model.pkl`, and `best-model.pkl` from retraining,
provided its `opt.pkl` is alongside it. The standalone tool loads tensors on
CPU, including the original model's unregistered adjacency tensor.

The checkpoint's graph mask is preserved. The training checkpoint and the CAM
checkpoint may yield different activations despite having the same NN weights:
CAM pruning changes which input connections are enabled, without retraining
those weights. For representation studies, record the stage and choose the
checkpoint intentionally. CAM-pruned activations are not activations of a model
retrained on the pruned graph unless that extra training was actually run.

## Saved files and row alignment

For the current 1000-row, 10-variable, 2-hidden-layer, 10-unit model:

| File | Shape | Meaning |
| --- | --- | --- |
| `hidden_layer_001.npy` | `(1000, 10, 10)` | First post-activation hidden layer |
| `hidden_layer_002.npy` | `(1000, 10, 10)` | Second post-activation hidden layer |
| `sample_indices.npy` | `(1000,)` | Zero-based original input row IDs |
| `sample_split.npy` | `(1000,)` | 0=train, 1=test/validation; -1=new input |
| `mean.npy`, `std.npy` | `(1, 10)` | Training normalization statistics, or zeros/ones if disabled |
| `adjacency.npy` | `(10, 10)` | The exact graph mask used for extraction |
| `metadata.json` | — | Layer files/shapes, variable names/indices, preprocessing, checkpoint/source |

All selected rows are exported in ascending original row order, rather than
random minibatch order. `--split all` includes the rows selected by the original
train/test settings, not unused rows if the training run limited sample counts.
For a selected split, or limited sample counts, `N` is smaller than 1000.

```python
import json
import numpy as np

folder = "experiments/p10-run1/train/hidden"
hidden = np.load(folder + "/hidden_layer_001.npy", mmap_mode="r")
row_ids = np.load(folder + "/sample_indices.npy")
with open(folder + "/metadata.json") as stream:
    info = json.load(stream)
print(hidden.shape, info["variables"])
third_variable_hidden = hidden[:, 2, :]
# If latent_truth.npy later contains one row per ORIGINAL data sample:
# aligned_latent_truth = np.load("latent_truth.npy")[row_ids]
```

Layer activations are written in batches to NumPy memmaps. The exporter does
not accumulate all layers for the entire dataset in RAM. The prepared input
data and source indices still reside in RAM. Exports preserve model precision
(`float64` in the current checkpoints), use eval/no-grad mode, and restore the
prior model mode and diagnostic counter. They do not update model parameters,
the graph mask, optimizer state, or random sampling state.

Optional `--variable-names names.json` accepts a JSON list such as
`["temperature", "pressure", "output"]`, in the exact original column order.
Default names are `X1, X2, ...`; `metadata.json` also stores zero-based indices.

To open the results in a spreadsheet or text editor, convert them to CSV:

```bash
python3 hidden_to_csv.py experiments/p10-run1/train/hidden
```

This creates `hidden_layer_001.csv`, etc., beside the NumPy files. Each row is
one sample; columns are `sample_index`, `split`, followed by
`X1_unit001, ..., X1_unit010, X2_unit001, ...`. Thus `(1000, 10, 10)` becomes
1000 data rows and 102 columns, retaining original row IDs and train/validation
labels. The original `.npy` files are retained. Use `--overwrite` to replace an
existing CSV export.

## Preprocessing

The standalone extractor reads the saved `opt.pkl` and uses the original
`DataManagerFile` with the original seed, sample counts, precision, and
normalization flag. It reconstructs the training split and computes any
normalization from that training split only. Test/validation and new data use
those training statistics. The original PyTorch sample standard deviation
(`torch.std`, equivalent to `ddof=1`) is retained.

Old checkpoints do not store normalization statistics or original data inside
the model pickle. Therefore keep the unchanged original `dataN.npy` and
`DAGN.npy`, and pass `--data-path` when paths recorded in `opt.pkl` are no longer
valid. Do not point `--data-path` at unrelated new data: normalization would then
be reconstructed from the wrong training set. Exported `mean.npy`, `std.npy`,
sample IDs, and source/checkpoint SHA-256 hashes make the extraction traceable.

## Export as part of training

Add `--export-hidden` to an existing `main.py` training command:

```bash
python main.py --exp-path /new-experiment --data-path /training-data \
  --i-dataset 1 --model NonLinGaussANM --num-vars 10 \
  --train --to-dag --export-hidden --hidden-batch-size 128
```

When using `start_example.sh`, add `--export-hidden` to its `GRANDAG_COMMAND`.
Each requested stage exports to its own `<stage>/hidden/`. In this path the
already prepared data tensors are used directly, so preprocessing is identical
to the training run. `--hidden-split` and `--hidden-overwrite` control split and
replacement. If the iteration limit is reached before the training code saves
a model, the in-memory model is still exported for the train-only flow;
metadata records `checkpoint_exists=false`. Subsequent DAG/pruning stages still
require their original saved-model inputs. Retraining exports use
`best-model.pkl` when available; otherwise the current in-memory model is used.

The flag is opt-in. Existing training commands, losses, acyclicity constraints,
and optimization steps retain their original behavior.

## Future data with latent variables

For future training, put **only observed columns** into `dataN.npy`, shaped
`(samples, observed_variables)`, with `--num-vars` equal to the observed column
count. Keep latent ground truth separately, aligned by original sample row;
never include it in the NN input or training normalization. After training,
the same exporter works without modifications. `DAGN.npy` is currently required
by the original training/evaluation loader; its definition for a hidden-variable
experiment must be chosen for that study. Do not treat a sliced full DAG as
automatically representing all dependencies induced by latent variables.

To apply an existing NN to a new observed-only matrix with the **same variable
meaning and column order**, use:

```bash
bash run_in_container.sh python extract_hidden.py \
  --model-path /final_log/p10-run1/train/model.pkl \
  --data-path /dataset/unpacked/data_p10_e10_n1000_GP \
  --input-data /dataset/new_observed.npy \
  --output-dir /final_log/new_observed_hidden
```

`--input-data` needs no graph file of its own and keeps all new rows in their
original order. It uses statistics from the original training data supplied by
`--data-path`. A model trained on the existing dataset is not thereby trained
for a different causal system. No latent variable generation, recovery,
correlation analysis, or causal identification is performed by this feature.

## Verification

```bash
bash run_in_container.sh python -m unittest discover -s tests -v
```

Tests compare every MLP/layer against independent `F.linear` evaluation with
the graph mask, verify unchanged forward outputs and gradients, batches, float
precision, row and split alignment, normalization (including new input), saved
checkpoint reloads, state restoration, overwrite protection, and invalid input.
The no-checkpoint training-flow test mocks the training call to isolate the
export hook; it is not evidence of a new converged training run.

Verified in the prepared Python 3.7.3/PyTorch CPU container on 2026-10-27:

- All five unittest cases passed, including independent per-variable forward
  and gradient comparisons.
- Real `p10-run1/train/model.pkl` and
  `p10-run1/cam-pruning/cutoff_0.001000/model.pkl` both exported two
  `(1000, 10, 10)` float64 layers into their respective `hidden/` directories.
- Every exported row, target MLP, and layer matched independent per-variable
  `F.linear` evaluation (`rtol=atol=1e-11`). Source row IDs were `0..999`, with
  the original 800 training and 200 validation labels.
- Model file SHA-256 hashes matched the values recorded during extraction.
- The actual `main.py --train --export-hidden` command was tested with a copy
  of the existing completed training cache in a temporary directory. It loaded
  that cache without optimization and produced the same arrays as the standalone
  command. No new full training run was performed.
- GPU execution and latent-variable recovery were not tested.
