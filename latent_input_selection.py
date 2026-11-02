#!/usr/bin/env python3
"""Frozen seed201 input-selection analysis; never trains GraN-DAG."""
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

if __name__ == '__main__':
    from latent_experiments.input_selection import main
    main()
