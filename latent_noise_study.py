#!/usr/bin/env python3
"""Matched-noise, five-seed unsupervised factor study; train/extract before Z evaluation."""
import argparse
import json
import os
from pathlib import Path


def mapped(path):
    path = Path(path)
    if (not path.is_absolute() and path.parts[0]=='experiments' and
            Path.cwd()==Path('/code') and Path('/final_log').is_dir()):
        return Path('/final_log').joinpath(*path.parts[1:])
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/latent_noise_study.json'))
    parser.add_argument('--stage',choices=['all','prepare','fit','evaluate','summarize'],default='all')
    args=parser.parse_args()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    from latent_experiments import noise_study
    spec=json.loads(args.config.read_text())
    root=mapped(spec['output_dir']).resolve()
    stages=['prepare','fit','evaluate','summarize'] if args.stage=='all' else [args.stage]
    for stage in stages:
        getattr(noise_study,stage)(spec,root)


if __name__=='__main__':
    main()
