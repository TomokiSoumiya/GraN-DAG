#!/usr/bin/env python3
"""Generate, train, and evaluate configurable GraN-DAG latent experiments."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, help='Override output_dir without changing experiment conditions')
    parser.add_argument('--stage', choices=['all', 'generate', 'train', 'evaluate'], default='all')
    parser.add_argument('--resume', action='store_true', help='Reuse only a run with an identical saved config')
    args = parser.parse_args()
    from latent_experiments.config import load_config
    config = load_config(args.config)
    # Set before importing NumPy/PyTorch to avoid CPU oversubscription on this host.
    threads = str(config['training'].get('torch_num_threads', 1))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = threads
    from latent_experiments.runner import run
    output = args.output_dir or Path(config['output_dir'])
    if (not output.is_absolute() and output.parts[0] == 'experiments' and
            Path.cwd() == Path('/code') and Path('/final_log').is_dir()):
        output = Path('/final_log').joinpath(*output.parts[1:])
    run(config, output, args.stage, args.resume)


if __name__ == '__main__':
    main()
