# Gradient-Based Neural DAG Learning

This code was written by the authors of the ICLR 2020 submission: "Gradient-Based Neural DAG Learning".
Our implementation is in PyTorch but some functions rely on the
[Causal Discovery Toolbox](https://diviyan-kalainathan.github.io/CausalDiscoveryToolbox/html/index.html) which relies
partly on the R programming language.

## Run the code
To use our implementation of GraN-DAG, simply install Singularity (instructions: [https://www.sylabs.io/guides/3.0/user-guide/installation.html](https://www.sylabs.io/guides/3.0/user-guide/installation.html))
and run the code in our container (download it here: [https://zenodo.org/record/7700409](https://zenodo.org/record/7700409)). Use start_example.sh (update the paths) to launch the differents methods (GraN-DAG, DAG-GNN, NOTEARS, CAM).

## Export hidden layer activations

Hidden activations can be exported from existing model checkpoints or after a
training stage. This only records representations; it does not estimate latent
variables or change the optimizer, likelihood, or DAG constraint.

See [docs/hidden_activations.md](docs/hidden_activations.md) for commands, array
axes, preprocessing, source row mapping, and future observed-data inputs.

## Unsupervised latent-factor extraction (current workflow)

Use frozen GraN-DAG hidden layers or raw X to extract one PCA/Factor Analysis
factor without latent labels. Fit on training rows only; open true Z only in a
separate final-evaluation process. This does not identify a causal latent cause.

```bash
bash run_in_container.sh python latent_unsupervised.py
```

See [docs/latent_unsupervised.md](docs/latent_unsupervised.md) for extraction-only
operation when Z is unknown, matched random-network masks, results, and safe
cleanup of historical supervised artifacts. Existing outputs are never overwritten.

For the matched-noise robustness study (four noise levels × five seeds), see
[docs/latent_noise_study.md](docs/latent_noise_study.md). Its entry point is
`latent_noise_study.py`; all runs finish label-free extraction before any Z evaluation.

## Historical synthetic experiments (supervised Z readout)

The following older workflows train regressors on known Z for validation. They
are retained for reproducibility and are not run by the unsupervised workflow.

Generate X/Z separately, train GraN-DAG on X only, and compare latent-factor
readout from X, learned hidden layers, mask-matched random networks, and an
X-only nonlinear regressor using an independent final test split:

```bash
bash run_in_container.sh python latent_experiment.py --config configs/latent_linear.json
```

See [docs/latent_experiments.md](docs/latent_experiments.md) for configuration,
leakage prevention, output files, and the limits of this experiment.

For additional fixed-weight/fixed-mask comparisons of an existing run, including
nonlinear hidden-layer probes, see
[docs/latent_additional_analysis.md](docs/latent_additional_analysis.md).
`latent_additional_analysis.py` writes to a new directory and never retrains
GraN-DAG or overwrites the original experiment.

For the nonlinear `tanh` experiment and a separately reported five-fresh-seed
study, see [docs/latent_nonlinear_study.md](docs/latent_nonlinear_study.md).
