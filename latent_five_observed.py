#!/usr/bin/env python3
"""Test, generate one five-variable dataset, fit GraN-DAG once, freeze factors, evaluate."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'


def main():
    from latent_experiments.config import load_config
    from latent_experiments.five_observed import generate,validate_config
    from latent_unsupervised import mapped
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/latent_five_observed_seed201.json')
    parser.add_argument('--stage',choices=['all','generate','train','extract','evaluate'],default='all')
    args=parser.parse_args();config=load_config(args.config);validate_config(config)
    root=mapped(config['output_dir']).resolve()
    stages=['generate','train','extract','evaluate'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='generate':
            if root.exists(): raise FileExistsError('Experiment already exists: '+str(root))
            tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_five_observed.py','-v'],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
            print(tests.stdout,flush=True)
            if tests.returncode: raise RuntimeError('Preflight failed; experiment not started')
            # Generation is isolated from the subsequent training subprocess.
            generate(config,root,tests.stdout)
        else:
            cmd=[sys.executable,'-B','-m','latent_experiments.five_observed',stage,'--config',args.config]
            if stage=='train':
                print('One GraN-DAG fit; progress log:',root/'training.log',flush=True)
                with (root/'training.log').open('x') as log:
                    subprocess.check_call(cmd,stdout=log,stderr=subprocess.STDOUT)
            else: subprocess.check_call(cmd)


if __name__=='__main__': main()
