#!/usr/bin/env python3
"""Unsupervised extraction first; open true Z only in a separate evaluation process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def mapped(path):
    path = Path(path)
    if (not path.is_absolute() and path.parts[0] == 'experiments' and
            Path.cwd() == Path('/code') and Path('/final_log').is_dir()):
        return Path('/final_log').joinpath(*path.parts[1:])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/latent_unsupervised.json'))
    parser.add_argument('--output-dir', type=Path, help='A NEW direct child of the source run')
    parser.add_argument('--stage', choices=['all', 'extract', 'evaluate'], default='all')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    root = mapped(settings['run_dir']).resolve()
    output = mapped(args.output_dir or settings['output_dir']).resolve()
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    stages = ['extract', 'evaluate'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        command = [sys.executable, '-m', 'latent_experiments.unsupervised', stage,
                   '--run-dir', str(root), '--output-dir', str(output), '--config', str(args.config.resolve())]
        subprocess.check_call(command, env=env)


if __name__ == '__main__':
    main()
