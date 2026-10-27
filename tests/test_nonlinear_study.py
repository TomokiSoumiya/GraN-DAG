"""Frozen seed plan, sample SD, no result overwrite, and existing CLI reuse."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from latent_experiments.config import load_config
from latent_experiments.additional import comparison_rows
from latent_experiments.evaluate import write_csv
from latent_experiments.nonlinear_study import (
    aggregate, prepare, resolved_seed_config, run_seeds, summarize, validate_plan)


class NonlinearStudyTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(Path('configs/latent_nonlinear_study.json').read_text())

    def test_exact_base_and_fresh_seed_changes(self):
        base = load_config(self.spec['base_config'])
        self.assertEqual(base['data']['latent_function'], 'tanh')
        self.assertEqual(base['data']['noise_std'], .5)
        for seed in self.spec['seeds']:
            config = resolved_seed_config(self.spec, seed)
            for key in ('data', 'split', 'training'):
                self.assertEqual(config[key]['seed'], seed)
                config[key]['seed'] = base[key]['seed']
                self.assertEqual(config[key], base[key])
            self.assertEqual(config['probe']['seed'], seed+1)
            config['probe']['seed'] = base['probe']['seed']
            self.assertEqual(config['probe'], base['probe'])

    def test_mean_sample_sd_and_paired_delta(self):
        rows = [dict(seed=seed, condition='C', score=value, delta=-value)
                for seed, value in zip(self.spec['seeds'], [1., 2., 3., 4., 5.])]
        result = summarize(rows, ['condition'], ['score', 'delta'])[0]
        self.assertEqual(result['n_seeds'], 5)
        self.assertEqual(result['score_mean'], 3.)
        self.assertAlmostEqual(result['score_std'], np.sqrt(2.5))
        self.assertEqual(result['delta_mean'], -3.)
        self.assertAlmostEqual(result['delta_std'], result['score_std'])
        with self.assertRaises(ValueError):
            summarize(rows+[rows[0]], ['condition'], ['score'])
        bad = copy.deepcopy(rows)
        bad[0]['score'] = np.nan
        with self.assertRaises(ValueError):
            summarize(bad, ['condition'], ['score'])

    def fixture(self, folder):
        root, pilot = Path(folder)/'study', Path(folder)/'pilot'
        pilot.mkdir()
        (pilot/'config.json').write_text(json.dumps(load_config(self.spec['base_config'])))
        prepare(self.spec, root, pilot)
        return root, pilot

    def test_plan_immutability_and_output_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root, pilot = self.fixture(folder)
            plan = validate_plan(self.spec, root)
            self.assertFalse(plan['pilot_in_summary'])
            self.assertEqual(plan['std_ddof'], 1)
            with self.assertRaises(FileExistsError):
                prepare(self.spec, root, pilot)
            changed = copy.deepcopy(self.spec)
            changed['seeds'][0] = 999
            with self.assertRaises(ValueError):
                validate_plan(changed, root)
            path = root/'configs/seed_101.json'
            path.write_text(path.read_text()+'\n')
            with self.assertRaises(ValueError):
                validate_plan(self.spec, root)

    def test_reuses_existing_cli_and_requires_completed_pilot(self):
        with tempfile.TemporaryDirectory() as folder:
            root, pilot = self.fixture(folder)
            (pilot/'additional_analysis').mkdir()
            (pilot/'training_status.json').write_text(json.dumps(dict(converged=False)))
            (pilot/'additional_analysis/status.json').write_text(json.dumps(dict(completed=True)))
            with self.assertRaises(RuntimeError):
                run_seeds(self.spec, root, pilot)
            (pilot/'training_status.json').write_text(json.dumps(dict(converged=True)))
            with patch('latent_experiments.nonlinear_study.subprocess.check_call') as called:
                run_seeds(self.spec, root, pilot)
            self.assertEqual(called.call_count, 10)
            scripts = [row[0][0][1] for row in called.call_args_list]
            self.assertEqual(scripts.count('latent_experiment.py'), 5)
            self.assertEqual(scripts.count('latent_additional_analysis.py'), 5)
            jobs = json.loads((root/'jobs.json').read_text())
            self.assertEqual([r['seed'] for r in jobs], self.spec['seeds'])
            self.assertTrue(all(r['completed'] for r in jobs))
            with self.assertRaises(FileExistsError):
                run_seeds(self.spec, root, pilot)

    def test_aggregation_tables_and_plots_with_mock_completed_runs(self):
        # Synthetic metric fixtures, NOT a claim of any completed real seed.
        def metrics(offset):
            rows = []
            keys = [('X', '0', 'all')]
            keys += [(c, l, s) for c in 'ABCDEFG' for l in ('1','2')
                     for s in ('X1','X2','X3','concat')]
            for code, layer, scope in keys:
                for decoder in ('ridge','extra_trees'):
                    score = .6+offset+int(layer)*.01
                    rows.append(dict(condition=code, decoder=decoder, layer=layer, scope=scope,
                                     latent='Z1', test_r2=score, test_mse=1-score,
                                     train_r2=score+.02, validation_r2=score+.01))
            return rows

        with tempfile.TemporaryDirectory() as folder:
            root, pilot = self.fixture(folder)
            (root/'jobs.json').write_text(json.dumps([
                dict(seed=s, completed=True) for s in self.spec['seeds']]))
            linear = Path(folder)/'latent_linear/additional_analysis'
            linear.mkdir(parents=True)
            write_csv(linear/'metrics.csv', metrics(0))
            results = {}
            for i, directory in enumerate([pilot]+[root/'runs'/('seed_'+str(s)) for s in self.spec['seeds']]):
                directory.mkdir(exist_ok=True)
                (directory/'config.json').write_text(json.dumps(load_config(self.spec['base_config'])))
                (directory/'additional_analysis').mkdir()
                values = metrics(i*.01)
                write_csv(directory/'additional_analysis/paired_comparisons.csv', comparison_rows(values))
                for stage in ('train', 'to-dag', 'cam-pruning/cutoff_0.001000'):
                    path = directory/stage/'hidden'
                    path.mkdir(parents=True)
                    np.save(str(path/'adjacency.npy'), np.array([[0,1,0],[0,0,1],[0,0,0]]))
                results[directory] = (
                    dict(converged=True, iterations=100, original_worker_elapsed_seconds=1),
                    values, dict(dag_equals_cam_mask=True, learned_stage_parameters_equal_train=True))
            with patch('latent_experiments.nonlinear_study.verify_completed_run',
                       side_effect=lambda directory: results[directory]):
                result = aggregate(self.spec, root, pilot)
            self.assertEqual(len(result), 114)
            self.assertTrue(all(r['n_seeds'] == 5 for r in result))
            self.assertEqual(len(list((root/'summary/figures').glob('*.png'))), 8)
            status = json.loads((root/'summary/status.json').read_text())
            self.assertFalse(status['pilot_in_mean_std'])
            self.assertTrue(status['completed'])
            with self.assertRaises(FileExistsError):
                aggregate(self.spec, root, pilot)


if __name__ == '__main__':
    unittest.main()
