#!/usr/bin/env python3
"""Read-only reuse of a latent experiment; write controlled probes to a NEW folder."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('experiments/latent_linear'))
    parser.add_argument('--output-dir', type=Path,
                        help='New direct child of run-dir; default: additional_analysis')
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    root = args.run_dir
    if (not root.is_absolute() and root.parts[0] == 'experiments' and
            Path.cwd() == Path('/code') and Path('/final_log').is_dir()):
        root = Path('/final_log').joinpath(*root.parts[1:])
    output = args.output_dir
    if output is not None and (not output.is_absolute() and output.parts[0] == 'experiments' and
                              Path.cwd() == Path('/code') and Path('/final_log').is_dir()):
        output = Path('/final_log').joinpath(*output.parts[1:])
    from latent_experiments.additional import analyze
    analyze(root, output)


if __name__ == '__main__':
    main()
