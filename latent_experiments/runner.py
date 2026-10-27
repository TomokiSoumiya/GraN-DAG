"""Stage orchestration with config fingerprints and immutable run identities."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .data import save_dataset


def canonical(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def run(config, root, stage='all', resume=False):
    root = Path(root).resolve()
    fingerprint = hashlib.sha256(canonical(config).encode()).hexdigest()
    if root.exists():
        if not resume:
            raise FileExistsError('Run directory already exists: {}. Choose a new output or --resume.'.format(root))
        saved = json.loads((root/'config.json').read_text())
        if canonical(saved) != canonical(config):
            raise ValueError('Configuration differs from saved run. Choose a new output directory.')
    else:
        root.mkdir(parents=True)
        (root/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (root/'config.sha256').write_text(fingerprint+'\n')
    if stage in ('all', 'generate'):
        if not (root/'data').exists():
            split = save_dataset(config, root/'data')
            settings = dict(config['training'])
            settings.update(n_train=len(split['train']), n_validation=len(split['validation']),
                            n_observed=config['data']['n_observed'])
            (root/'training_options.json').write_text(json.dumps(settings, indent=2))
            manifest = {'config_sha256': fingerprint, 'data_files': {}}
            for file in sorted((root/'data').glob('*.npy')):
                manifest['data_files'][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
            (root/'data_manifest.json').write_text(json.dumps(manifest, indent=2))
            print('Generated data and disjoint train/validation/final-test split:', root, flush=True)
        else:
            print('Reusing generated data:', root, flush=True)
    if stage in ('all', 'train'):
        if not (root/'training_options.json').exists():
            raise FileNotFoundError('Generate this run before training')
        print('Starting X-only original GraN-DAG training. Progress:', root/'training.log', flush=True)
        with (root/'training.log').open('a') as log:
            subprocess.check_call([sys.executable, '-m', 'latent_experiments.train',
                                   '--run-dir', str(root)], stdout=log, stderr=subprocess.STDOUT)
    if stage in ('all', 'evaluate'):
        from .evaluate import evaluate
        evaluate(root, config)
    return root
