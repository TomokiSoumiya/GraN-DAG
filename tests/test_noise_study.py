"""Matched-noise identities, paired SD, global Z barrier and safe model reuse."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from latent_experiments.config import load_config
from latent_experiments.data import generate_arrays,save_dataset,split_indices
from latent_experiments.noise_study import (check_global_barrier,copy_existing_if_exact,
    expected_jobs,make_figures,make_report,noise_paired_rows,paired_rows,run_config,
    sigma_name,summarize_rows,evaluate)
from latent_experiments.unsupervised import digest,file_inventory,forbid_latent_reads,save_json


class NoiseStudyTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads(Path('configs/latent_noise_study.json').read_text())

    def synthetic_metrics(self):
        rows=[]
        conditions=['X','C_L1','C_L2','E_L1','E_L2','B_L1','B_L2','D_L1','D_L2']
        for sigma in self.spec['sigmas']:
            for i,seed in enumerate(self.spec['seeds']):
                for j,condition in enumerate(conditions):
                    for method in ('pca','factor_analysis'):
                        for split in ('train','validation','test'):
                            score=.9-.05*sigma+.001*j+.002*i
                            rows.append(dict(sigma=sigma,seed=seed,condition=condition,method=method,
                                split=split,latent='Z1',pearson_r=-score,absolute_pearson_r=score,
                                spearman_r=-score,x_reconstruction_raw_mse=1-score,
                                x_reconstruction_standardized_mse=1-score,factor_variance=1.))
        return rows

    def test_all_four_sigmas_share_base_randomness_splits_and_training_settings(self):
        for seed in self.spec['seeds']:
            reference=None
            for sigma in self.spec['sigmas']:
                config=run_config(self.spec,sigma,seed)
                data=generate_arrays(config['data'])
                splits=split_indices(1000,config['split'],config['training']['seed'])
                np.testing.assert_allclose(data['observed_X'],data['true_Z']@np.array([[2.,3.,1.]])+data['noise'])
                actual=(data['true_Z'],data['noise']/sigma,splits['labels'])
                if reference is None:reference=actual
                else:
                    for a,b in zip(actual,reference):np.testing.assert_array_equal(a,b)
                self.assertEqual([int((splits['labels']==c).sum()) for c in (0,1,2)],[600,200,200])
                expected=load_config('configs/latent_linear.json')['training']
                expected['seed']=seed
                self.assertEqual(config['training'],expected)
        self.assertEqual(len(expected_jobs(self.spec,Path('/unused'))),20)

    def test_full_cohort_sd_and_undefined_are_not_silently_excluded(self):
        rows=[dict(seed=s,group='g',score=float(i)) for i,s in enumerate(self.spec['seeds'])]
        r=summarize_rows(rows,['group'],['score'])[0]
        self.assertEqual(r['score_mean'],2.)
        self.assertAlmostEqual(r['score_std'],np.sqrt(2.5))
        rows[0]['score']=None
        r=summarize_rows(rows,['group'],['score'])[0]
        self.assertEqual(r['score_n_defined'],4)
        self.assertIsNone(r['score_mean']);self.assertIsNone(r['score_std'])
        with self.assertRaises(ValueError):summarize_rows(rows[:-1],['group'],['score'])
        with self.assertRaises(ValueError):summarize_rows(rows[:-1]+[rows[1]],['group'],['score'])

    def test_paired_differences_within_seed_and_across_noise(self):
        metrics=self.synthetic_metrics()
        within=paired_rows(metrics)
        across=noise_paired_rows(metrics,self.spec['sigmas'])
        self.assertEqual(len(metrics),1080);self.assertEqual(len(within),800);self.assertEqual(len(across),540)
        r=next(r for r in within if r['before']=='B_L1' and r['after']=='C_L1')
        self.assertAlmostEqual(r['delta_absolute_pearson_r'],-.004)
        self.assertAlmostEqual(r['delta_x_reconstruction_raw_mse'],.004)
        r=next(r for r in across if r['before_sigma']==.5 and r['after_sigma']==4.)
        self.assertAlmostEqual(r['delta_absolute_pearson_r'],-.175)
        summarized=summarize_rows(within,['sigma','comparison','before','after','method'],['delta_absolute_pearson_r'])
        self.assertEqual(len(summarized),160)
        self.assertTrue(all(abs(r['delta_absolute_pearson_r_std'])<1e-15 for r in summarized))

    def test_global_barrier_requires_all_runs_and_unmodified_artifacts_without_reading_z(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            jobs=expected_jobs(self.spec,root); plan=dict(jobs=jobs)
            records=[]
            for job in jobs:
                directory=root/job['directory']/'unsupervised_factors'
                directory.mkdir(parents=True)
                (directory/'factor.txt').write_text('frozen')
                save_json(directory/'extraction_status.json',dict(completed=True))
                save_json(directory/'extraction_manifest.json',{'factor.txt':digest(directory/'factor.txt')})
                records.append(dict(directory=job['directory'],
                    manifest_sha256=digest(directory/'extraction_manifest.json'),
                    status_sha256=digest(directory/'extraction_status.json')))
            completed=[dict(j,completed=True) for j in jobs]
            save_json(root/'fit_jobs.json',completed[:-1])
            save_json(root/'global_extraction_freeze.json',dict(completed=True,all_run_count=20,extractors_frozen=records))
            with patch('latent_experiments.noise_study.validate_plan',return_value=plan), \
                 patch('latent_experiments.noise_study.np.load',side_effect=AssertionError('No Z read at barrier')):
                with self.assertRaises(RuntimeError):check_global_barrier(self.spec,root)
                save_json(root/'fit_jobs.json',completed)
                check_global_barrier(self.spec,root)
                directory=root/records[0]['directory']/'unsupervised_factors'
                (directory/'factor.txt').write_text('tampered')
                with self.assertRaises(ValueError):check_global_barrier(self.spec,root)

    def test_evaluation_never_starts_when_global_barrier_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('latent_experiments.noise_study.check_global_barrier',side_effect=RuntimeError('unfinished')), \
                 patch('latent_experiments.noise_study.verify_pairing') as pairing, \
                 patch('latent_experiments.noise_study.subprocess.check_call') as command:
                with self.assertRaises(RuntimeError):evaluate(self.spec,Path(folder))
                pairing.assert_not_called();command.assert_not_called()

    def test_reuse_checks_exact_x_split_config_and_preserves_original_without_z(self):
        with tempfile.TemporaryDirectory() as folder:
            source,target=Path(folder)/'source',Path(folder)/'target'
            source.mkdir();target.mkdir()
            config=run_config(self.spec,1.,42)
            for directory in (source,target):
                save_json(directory/'config.json',config)
                save_dataset(config,directory/'data')
            for stage in ('train','to-dag','cam-pruning','random_full','random_train_mask','random_cam_mask'):
                (source/stage).mkdir()
                # Opaque copy fixture only: no fake NN fit/checkpoint loading.
                (source/stage/'model.pkl').write_bytes(b'checkpoint-copy-fixture')
            save_json(source/'training_status.json',dict(converged=True,
                latent_labels_used_by_grandag=False,final_test_used_by_grandag=False))
            for name in ('normalization_mean.npy','normalization_std.npy'):
                np.save(str(source/name),np.ones((1,3)))
            spec=copy.deepcopy(self.spec);spec['reuse']['source']=str(source)
            before=file_inventory(source)
            with forbid_latent_reads(source):
                self.assertTrue(copy_existing_if_exact(spec,dict(sigma=1.,seed=42),target))
            self.assertEqual(before,file_inventory(source))
            self.assertEqual(digest(source/'train/model.pkl'),digest(target/'train/model.pkl'))
            self.assertEqual(json.loads((target/'model_origin.json').read_text())['origin'],'copied_existing_exact_match')
            config['training']['lr']=.5;save_json(target/'config.json',config)
            with self.assertRaises(ValueError):copy_existing_if_exact(spec,dict(sigma=1.,seed=42),target)

    def test_summary_figures_and_report_with_synthetic_metrics_not_real_runs(self):
        metrics=self.synthetic_metrics()
        summary=summarize_rows(metrics,['sigma','condition','method','split','latent'],
            ['absolute_pearson_r','x_reconstruction_raw_mse'])
        pairs=summarize_rows(paired_rows(metrics),['sigma','comparison','before','after','method'],
            ['delta_absolute_pearson_r','delta_x_reconstruction_raw_mse'])
        tiny_png=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jXioAAAAASUVORK5CYII=')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for sigma in self.spec['sigmas']:
                path=root/'runs'/sigma_name(sigma)/('seed_'+str(self.spec['representative_seed']))/'unsupervised_factors/figures'
                path.mkdir(parents=True)
                for method in ('pca','factor_analysis'):(path/('scatter_'+method+'_Z1.png')).write_bytes(tiny_png)
            make_figures(self.spec,root,summary,pairs)
            make_report(self.spec,root,summary,pairs,[],[])
            self.assertTrue((root/'report.md').exists())
            self.assertEqual(len(list((root/'figures').rglob('*.png'))),14)


if __name__=='__main__':unittest.main()
