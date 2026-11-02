#!/usr/bin/env python3
"""Reproduce the frozen PCA score, validate it, then fit one Z_hat graph."""
import os

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

if __name__=='__main__':
    from latent_experiments.zhat_graph import main
    main()
