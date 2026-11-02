#!/usr/bin/env python3
"""One normalized nonlinear seed201 run: frozen factor extraction/evaluation."""
import os

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

if __name__=='__main__':
    from latent_experiments.function_factors import main
    main()
