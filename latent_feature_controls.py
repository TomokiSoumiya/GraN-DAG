#!/usr/bin/env python3
"""Extract D/E once, reuse A/B/C, then evaluate in a separate process."""
import argparse
import os
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/latent_feature_controls_seed201.json')
    parser.add_argument('--stage', choices=['all', 'extract', 'evaluate'], default='all')
    args = parser.parse_args()
    for stage in (['extract','evaluate'] if args.stage == 'all' else [args.stage]):
        subprocess.check_call([sys.executable, '-B', '-m', 'latent_experiments.feature_controls',
                               stage, '--config', args.config])


if __name__ == '__main__':
    main()
