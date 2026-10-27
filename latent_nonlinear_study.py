#!/usr/bin/env python3
"""Prepare/run/aggregate a preregistered fresh-seed nonlinear GraN-DAG study."""
import argparse
import os
from pathlib import Path


def mapped(path):
    path = Path(path)
    if (not path.is_absolute() and path.parts[0] == 'experiments' and
            Path.cwd() == Path('/code') and Path('/final_log').is_dir()):
        path = Path('/final_log').joinpath(*path.parts[1:])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/latent_nonlinear_study.json'))
    parser.add_argument('--stage', choices=['prepare', 'run', 'aggregate', 'all'], default='all')
    parser.add_argument('--summary-name', default='summary', help='New direct child for aggregate output')
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import json
    from latent_experiments.nonlinear_study import prepare, run_seeds, aggregate
    spec = json.loads(args.config.read_text())
    root, pilot = mapped(spec['output_dir']), mapped(spec['pilot_dir'])
    if args.stage in ('prepare', 'all'):
        prepare(spec, root, pilot)
    if args.stage in ('run', 'all'):
        run_seeds(spec, root, pilot)
    if args.stage in ('aggregate', 'all'):
        aggregate(spec, root, pilot, args.summary_name)


if __name__ == '__main__':
    main()
