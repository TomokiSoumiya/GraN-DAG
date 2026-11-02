"""Five fixed PCA controls. Only D/E fit new extractors; no NN optimizers exist here."""
import argparse
import itertools
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analyze_hidden import correlations, varying_columns
from gran_dag.hidden import load_checkpoint_cpu
from latent_unsupervised import mapped
from .additional import parameter_digest, verify_hidden
from .unsupervised import (SPLITS, digest, factor_correlations, file_inventory, fit_extractor,
                           forbid_latent_reads, read_csv, save_json, transform_extractor, write_csv)


NAMES = dict(A='元X＋PCA', B='未学習NN H1＋PCA', C='学習済みNN H1＋PCA',
             D='MLP別PCA→標準化→PCA', E='マスク対応の線形複製＋PCA')


def fit_blocks(hidden_train, x_train, settings):
    """Training rows only. No labels or evaluation data can enter this API."""
    if (hidden_train.ndim != 3 or hidden_train.shape[1:] != (3,10)
            or x_train.shape != (len(hidden_train),3) or not np.isfinite(hidden_train).all()):
        raise ValueError('Aligned finite training-only H=(N,3,10), X=(N,3) required')
    blocks, scores, details = [], np.zeros((len(hidden_train),3)), []
    for j in range(3):
        values = hidden_train[:,j,:]
        keep = varying_columns(values)
        block = dict(keep=keep, scaler=None, pca=None, informative=False)
        ratio = None
        if keep.any():
            block['scaler'] = StandardScaler().fit(values[:,keep])
            scaled = block['scaler'].transform(values[:,keep])
            block['pca'] = PCA(n_components=1,svd_solver='full').fit(scaled)
            candidate = block['pca'].transform(scaled)
            if varying_columns(candidate).any():
                scores[:,j] = candidate[:,0]
                block['informative'] = True
                ratio = float(block['pca'].explained_variance_ratio_[0])
        blocks.append(block)
        details.append(dict(mlp='X'+str(j+1), n_train=len(hidden_train), input_units=10,
            nonconstant_units=int(keep.sum()), informative=block['informative'],
            pc1_explained_variance_ratio=ratio, block_score_variance=float(scores[:,j].var(ddof=1))))
    if not any(b['informative'] for b in blocks):
        raise ValueError('No informative MLP block; no factor is invented')
    final = fit_extractor(scores,x_train,'pca',settings)
    return dict(blocks=blocks, final=final, details=details, n_train=len(hidden_train))


def transform_blocks(bundle, hidden):
    if hidden.ndim != 3 or hidden.shape[1:] != (3,10) or not np.isfinite(hidden).all():
        raise ValueError('Finite H=(N,3,10) required')
    scores = np.zeros((len(hidden),3))
    for j, block in enumerate(bundle['blocks']):
        if block['informative']:
            scores[:,j] = block['pca'].transform(block['scaler'].transform(hidden[:,j,block['keep']]))[:,0]
    factors, _, xhat = transform_extractor(bundle['final'],scores)
    return factors, xhat, scores


def linear_features(x, mask, seed):
    """Random linear features depend only on fixed seed, mask and observed X."""
    if (x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all()
            or mask.shape != (3,3) or not np.isin(mask,[0,1]).all()
            or np.any(np.diag(mask)) or not np.all(mask.sum(0)==1)):
        raise ValueError('Exactly one binary off-diagonal parent per MLP required')
    coefficients = np.random.RandomState(seed).normal(size=(3,10))
    if np.any(coefficients==0):
        raise ValueError('Zero random coefficient; protocol does not reroll')
    parents = mask.argmax(0)
    return x[:,parents,None]*coefficients[None,:,:], coefficients, parents


def feature_diagnostics(values, group_ids, condition, split, settings):
    """Descriptive statistics only; rank/SVD never selects any extracted factor."""
    keep = varying_columns(values)
    corr = correlations(values,values)
    rows = []
    for left in range(3):
        for right in range(left,3):
            li = np.where((group_ids==left)&keep)[0]
            ri = np.where((group_ids==right)&keep)[0]
            pair = corr[np.ix_(li,ri)]
            pair = pair[np.triu_indices(len(li),1)] if left==right else pair.ravel()
            pair = np.abs(pair[np.isfinite(pair)])
            row = dict(condition=condition, split=split, group_left=left+1, group_right=right+1,
                group_kind='observed X' if condition=='A' else 'target MLP', n_feature_pairs=len(pair),
                mean_absolute_r=float(pair.mean()) if len(pair) else None,
                max_absolute_r=float(pair.max()) if len(pair) else None,
                fraction_absolute_r_ge_095=float((pair>=settings['high_absolute_correlation_threshold']).mean()) if len(pair) else None,
                n_nonconstant_left=len(li), n_nonconstant_right=len(ri), numerical_rank=None,
                participation_rank=None, pc1_variance_fraction=None)
            if left==right and len(li):
                block = values[:,li]
                standardized = (block-block.mean(0))/block.std(0)
                singular = np.linalg.svd(standardized,compute_uv=False)
                energy = singular**2
                row.update(numerical_rank=int((singular>settings['rank_relative_tolerance']*singular[0]).sum()),
                    participation_rank=float(energy.sum()**2/(energy**2).sum()),
                    pc1_variance_fraction=float(energy[0]/energy.sum()))
            rows.append(row)
    return rows, corr


def verify_reused(root, old, entry, features, x, labels, manifest):
    for key in ('model_file','factor_file'):
        if digest(old/entry[key]) != manifest[entry[key]]:
            raise ValueError('Saved extractor/factors changed: '+entry[key])
    with (old/entry['model_file']).open('rb') as stream:
        bundle = pickle.load(stream)
    train = labels==0
    if bundle['n_train'] != 600 or bundle['method'] != 'pca':
        raise ValueError('Expected saved train600 PCA')
    np.testing.assert_array_equal(bundle['keep'],varying_columns(features[train]))
    np.testing.assert_allclose(bundle['scaler'].mean_,features[train][:,bundle['keep']].mean(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(bundle['scaler'].scale_,features[train][:,bundle['keep']].std(0),rtol=1e-12,atol=1e-12)
    if int(bundle['scaler'].n_samples_seen_) != 600 or bundle['extractor'].n_samples_ != 600:
        raise ValueError('Saved preprocessing/PCA not fitted on 600 rows')
    factor = np.load(str(old/entry['factor_file']))
    replay, _, xhat = transform_extractor(bundle,features)
    np.testing.assert_allclose(factor,replay,rtol=1e-12,atol=1e-12)
    reconstruction_file = 'reconstructions/'+entry['key']+'_X.npy'
    if digest(old/reconstruction_file) != manifest[reconstruction_file]:
        raise ValueError('Saved OLS reconstructions changed')
    np.testing.assert_allclose(xhat,np.load(str(old/reconstruction_file)),rtol=1e-12,atol=1e-12)
    design = np.column_stack([np.ones(600),factor[train]])
    np.testing.assert_allclose(design.T@(x[train]-xhat[train]),0,rtol=0,atol=1e-8)
    return bundle, factor, xhat


def extract(settings):
    root, output = mapped(settings['source_dir']).resolve(), mapped(settings['output_dir']).resolve()
    if output.exists():
        raise FileExistsError('Existing analysis must not be overwritten: '+str(output))
    if settings['conditions'] != list('ABCDE') or settings['linear_feature_seed'] != 201 or settings['n_factors'] != 1:
        raise ValueError('Only the five predeclared controls at seed201 are authorized')
    before = file_inventory(root)
    with forbid_latent_reads(root) as audit:
        config = json.loads((root/'config.json').read_text())
        if (config['data']['latent_functions'] != ['tanh','sin','cubic'] or config['data']['noise_std'] != 1
                or any(config[k]['seed'] != 201 for k in ('data','split','training'))):
            raise ValueError('Wrong source dataset')
        x = np.load(str(root/'data/observed_X.npy'))
        ids = np.load(str(root/'data/sample_indices.npy'))
        labels = np.load(str(root/'data/sample_split.npy'))
        if x.shape != (1000,3) or not np.isfinite(x).all():
            raise ValueError('Wrong X dimensions or nonfinite values')
        np.testing.assert_array_equal(ids,np.arange(1000))
        np.testing.assert_array_equal(np.bincount(labels),[600,200,200])
        for name,code in SPLITS:
            np.testing.assert_array_equal(np.sort(np.load(str(root/'data'/(name+'_indices.npy')))),ids[labels==code])
        old = root/'unsupervised_factors'
        meta = json.loads((old/'metadata.json').read_text())
        status = json.loads((old/'extraction_status.json').read_text())
        manifest = json.loads((old/'extraction_manifest.json').read_text())
        if meta['z_used_in_fit'] or meta['z_used_for_selection'] or not status['all_factors_saved_before_Z'] or status['z_read_attempts']:
            raise ValueError('Reused factors lack clean label-free provenance')
        mean, std = [np.load(str(root/name)) for name in ('normalization_mean.npy','normalization_std.npy')]
        models, hidden, verification = {}, {}, {}
        for code,stage in [('B','random_train_mask'),('C','train')]:
            model = load_checkpoint_cpu(str(root/stage/'model.pkl'))
            original = parameter_digest(model)
            for p in model.parameters(): p.requires_grad = False
            layers = verify_hidden(root/stage/'hidden',model,x,ids,labels,mean,std,root/stage/'model.pkl')
            if original != parameter_digest(model): raise ValueError('Source weights changed')
            models[code], hidden[code] = model, layers[0]
            verification[code] = dict(checkpoint=str(root/stage/'model.pkl'),parameters_sha256=original,
                                     hidden_reproduced=True, checkpoint_sha256=digest(root/stage/'model.pkl'))
        np.testing.assert_array_equal(models['B'].adjacency.numpy(),models['C'].adjacency.numpy())
        initial = load_checkpoint_cpu(str(root/'random_full/model.pkl'))
        if parameter_digest(initial) != parameter_digest(models['B']):
            raise ValueError('Random-mask NN differs from the saved initial weights')
        mask = models['C'].adjacency.numpy()
        e, coefficients, parents = linear_features(x,mask,settings['linear_feature_seed'])
        output.mkdir()
        for directory in ('models','factors','reconstructions','features','figures'):
            (output/directory).mkdir()
        save_json(output/'config.json',settings)
        save_json(output/'source_inventory_before.json',before)
        np.save(str(output/'sample_indices.npy'),ids)
        np.save(str(output/'sample_split.npy'),labels)
        coefficients_rows = [dict(target_mlp='X'+str(j+1), source_X='X'+str(parents[j]+1), unit=k+1,
                                 coefficient=float(coefficients[j,k]),intercept=0)
                             for j in range(3) for k in range(10)]
        write_csv(output/'linear_coefficients.csv',coefficients_rows)
        np.save(str(output/'features/E_linear_features.npy'),e)
        bundles, factors, recon, features = {}, {}, {}, {}
        provenance = []
        for code in 'ABC':
            entry = next(c for c in meta['candidates'] if c['key']==settings['reused_keys'][code])
            values = x if code=='A' else hidden[code].reshape(1000,30)
            bundles[code], factors[code], recon[code] = verify_reused(root,old,entry,values,x,labels,manifest)
            features[code] = values
            provenance.append(dict(condition=code,action='reused, no fit',source_key=entry['key'],
                source_model=entry['model_file'],source_factor=entry['factor_file'],
                model_sha256=digest(old/entry['model_file']),factor_sha256=digest(old/entry['factor_file'])))
        d = fit_blocks(hidden['C'][labels==0],x[labels==0],settings)
        factors['D'], recon['D'], features['D'] = transform_blocks(d,hidden['C'])
        bundles['D'] = d
        features['E'] = e.reshape(1000,30)
        bundles['E'] = fit_extractor(features['E'][labels==0],x[labels==0],'pca',settings)
        factors['E'], _, recon['E'] = transform_extractor(bundles['E'],features['E'])
        np.save(str(output/'features/D_block_scores.npy'),features['D'])
        write_csv(output/'D_block_pca.csv',d['details'])
        metric_rows, factor_rows, redundancy, loadings, pairs = [], [], [], [], []
        for code in 'ABCDE':
            if factors[code].shape != (1000,1) or not np.isfinite(factors[code]).all():
                raise ValueError('Invalid extracted factor')
            np.save(str(output/'factors'/(code+'.npy')),factors[code])
            np.save(str(output/'reconstructions'/(code+'_X.npy')),recon[code])
            # Copying already fitted A/B/C bundles does not refit them.
            with (output/'models'/(code+'.pkl')).open('wb') as stream: pickle.dump(bundles[code],stream)
            final = bundles[code]['final'] if code=='D' else bundles[code]
            groups = np.arange(3) if code in ('A','D') else np.repeat(np.arange(3),10)
            component = final['extractor'].components_[0]
            kept_groups = groups[final['keep']]
            for group in range(3):
                loadings.append(dict(condition=code,group=group+1,
                    source_X='X'+str(group+1 if code=='A' else parents[group]+1),
                    squared_loading_share=float((component[kept_groups==group]**2).sum()),
                    input_columns=int((groups==group).sum())))
            for i in range(1000):
                factor_rows.append(dict(condition=code,sample_id=int(ids[i]),split=int(labels[i]),factor=float(factors[code][i,0])))
            for name,k in SPLITS:
                chosen = labels==k
                row = dict(condition=code,description=NAMES[code],split=name,n_samples=int(chosen.sum()),
                    factor_variance=float(factors[code][chosen,0].var(ddof=1)),
                    x_reconstruction_raw_mse=float(np.mean((x[chosen]-recon[code][chosen])**2)),
                    input_dimension=features[code].shape[1],used_dimension=int(final['keep'].sum()))
                for j in range(3):
                    result = factor_correlations(factors[code][chosen,0],x[chosen,j])
                    row['pearson_X'+str(j+1)] = result['pearson_r']
                    row['spearman_X'+str(j+1)] = result['spearman_r']
                metric_rows.append(row)
                dr, matrix = feature_diagnostics(features[code][chosen],groups,code,name,settings)
                redundancy.extend(dr)
                np.save(str(output/'features'/(code+'_'+name+'_correlations.npy')),matrix)
        for left,right in itertools.combinations('ABCDE',2):
            f,g = factors[left][:,0],factors[right][:,0]
            design = np.column_stack([np.ones(600),f[labels==0]])
            affine = np.linalg.lstsq(design,g[labels==0],rcond=None)[0]
            residual = g - np.column_stack([np.ones(1000),f])@affine
            for name,k in SPLITS:
                idx = labels==k
                pairs.append(dict(left=left,right=right,split=name,
                    **factor_correlations(f[idx],g[idx]),train_affine_intercept=float(affine[0]),
                    train_affine_slope=float(affine[1]),affine_RMSE=float(np.sqrt(np.mean(residual[idx]**2)))))
        write_csv(output/'label_free_metrics.csv',metric_rows)
        write_csv(output/'extracted_factors.csv',factor_rows)
        write_csv(output/'feature_redundancy.csv',redundancy)
        write_csv(output/'pca_group_loadings.csv',loadings)
        write_csv(output/'factor_pair_diagnostics.csv',pairs)
        save_json(output/'metadata.json',dict(settings=settings,source_verification=verification,
            reused=provenance,adjacency=mask.tolist(),adjacency_convention='source row -> target column',
            source_X_per_target_mlp=[int(p)+1 for p in parents],
            linear_coefficient_distribution='standard normal, zero intercept, raw X input, no NN weights',
            feature_groups='A: X1/X2/X3; B/C/D/E: target MLP X1/X2/X3',
            D_second_PCA_input='train-standardized per-MLP PC1 scores, omitting noninformative blocks',
            all_preprocessing_fit_rows=600,OLS_fit_rows=600,factor_variance_ddof=1,
            nn_training_calls=0,vae_training_calls=0,data_generation_calls=0,
            factor_sign='unchanged PCA output, no Z alignment',latent_read_attempts=audit['blocked_attempts'],
            redundancy='descriptive per-split correlations/SVD only, never feature selection',
            versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__)))
        if before != file_inventory(root): raise ValueError('Source experiment changed')
        save_json(output/'extraction_status.json',dict(completed=True,all_factors_frozen_before_Z=True,
            Z_read_attempts=audit['blocked_attempts'],new_condition_fits=['D','E'],reused_conditions=list('ABC'),
            source_files_unchanged=True,nn_training_calls=0,vae_training_calls=0,data_generation_calls=0))
        save_json(output/'extraction_manifest.json',{str(p.relative_to(output)):digest(p)
                  for p in output.rglob('*') if p.is_file()})
    print('All five factors frozen without Z:',output,flush=True)


def figures(output, metrics, factors, z, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    test = labels==2
    tests = [r for r in metrics if r['split']=='test']
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['absolute_pearson_r','x_reconstruction_raw_mse']):
        ax.bar(list('ABCDE'),[r[key] for r in tests]); ax.set_title(key)
    fig.tight_layout();fig.savefig(str(output/'figures/test_comparison.png'),dpi=160);plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(13,7))
    for ax,code in zip(axes.flat,'ABCDE'):
        row = next(r for r in tests if r['condition']==code)
        ax.scatter(z[test,0],factors[code][test,0],s=12,alpha=.6)
        ax.set(title='{} |r|={:.4f}'.format(code,row['absolute_pearson_r']),xlabel='True Z (evaluation only)',ylabel='Frozen factor')
    axes.flat[-1].axis('off')
    fig.tight_layout();fig.savefig(str(output/'figures/test_Z_scatter.png'),dpi=160);plt.close(fig)
    fig,axes = plt.subplots(1,5,figsize=(16,3.5))
    for ax,code in zip(axes,'ABCDE'):
        matrix = np.load(str(output/'features'/(code+'_train_correlations.npy')))
        ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm');ax.set_title(code+' train correlations')
    fig.tight_layout();fig.savefig(str(output/'figures/feature_correlations.png'),dpi=160);plt.close(fig)
    loads = read_csv(output/'pca_group_loadings.csv')
    fig,ax = plt.subplots(figsize=(8,4)); bottom=np.zeros(5)
    for group in range(1,4):
        vals=[float(next(r for r in loads if r['condition']==c and int(r['group'])==group)['squared_loading_share']) for c in 'ABCDE']
        ax.bar(list('ABCDE'),vals,bottom=bottom,label='group '+str(group));bottom+=vals
    ax.set(ylabel='Squared PC1 loading share',title='A: X groups; others: target MLP groups');ax.legend()
    fig.tight_layout();fig.savefig(str(output/'figures/group_loadings.png'),dpi=160);plt.close(fig)


def make_report(output,metrics,metadata):
    test = {r['condition']:r for r in metrics if r['split']=='test'}
    delta = lambda a,b: test[b]['absolute_pearson_r']-test[a]['absolute_pearson_r']
    lines=['# GraN-DAG学習効果・特徴量重複・MLP単位要約の比較', '',
        '既存の規格化tanh・sin・Z³、seed201、σ=1、学習600・検証200・テスト200の1条件。新規GraN-DAG学習0回、VAE学習0回、人工データ再生成0回。1回の探索的解析であり追加条件や調整は実施していない。', '',
        '## 比較条件と入力割当て', '',
        'A: 元Xの保存済みPCA。B: 同じ学習後マスクを適用した未学習NN H1の保存済みPCA。C: 学習済みNN H1の保存済みPCA。A/B/Cのモデル・因子得点・OLSを再fitせず再利用し、ハッシュと再変換による一致を確認した。',
        'D: 学習済み各MLPの10列を学習600件で定数判定・標準化しPC1に要約。その3得点を再び学習600件で標準化してPC1を抽出した。0分散ブロックは情報なしと記録し、最終PCAから除外する。',
        'E: RandomState(201)の標準正規係数30個を固定し、対応する単一入力Xから各MLP10個の線形特徴を作成。切片0。NN重み、非線形変換、Zやテスト結果による係数調整は使っていない。',
        '入力は各MLPにつき単一X: X1予測MLP←X2、X2予測MLP←X3、X3予測MLP←X1。全MLP結合で元Xの3変数すべてを含むが、1つのMLP内に複数Xを混合する構造ではない。行が入力元、列が予測対象。',
        'Eは入力割当て・30列・3グループをそろえた線形複製対照であり、非線形活性化を持つ未学習NNと完全同条件ではない。', '',
        '## 漏洩防止・再現性', '',
        '抽出と評価は別プロセス。すべての因子・OLS・診断を保存しハッシュを確定した後、評価プロセスでのみtrue_Zを読み込んだ。符号反転なし。PCAは常にPC1、Z相関に基づく条件・特徴選択なし。',
        '全標準化・定数判定・PCA・OLSは学習600件のみ。検証・テストはtransformと評価のみ。元モデル・保存H・マスク・サンプルID・分割の一致と既存実験ファイルの不変を確認した。', '',
        '## テスト200件の全条件', '',
        '|条件|内容|Pearson|絶対Pearson|Spearman|因子分散(ddof=1)|OLSによる元X再構成MSE|',
        '|---|---|---:|---:|---:|---:|---:|']
    for code in 'ABCDE':
        r=test[code]
        lines.append('|{}|{}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|'.format(code,NAMES[code],r['pearson_r'],r['absolute_pearson_r'],r['spearman_r'],r['factor_variance'],r['x_reconstruction_raw_mse']))
    lines += ['', '因子の尺度は条件で異なるため、分散の大小は情報の多さの比較にはならない。OLSは因子→元X3列、学習600件で切片付き最小二乗。A/B/Cは既存OLSを再利用し、D/Eのみfitした。', '',
        '|条件|因子とX1のPearson|因子とX2のPearson|因子とX3のPearson|', '|---|---:|---:|---:|']
    for code in 'ABCDE':
        r=test[code]
        lines.append('|{}|{:.6f}|{:.6f}|{:.6f}|'.format(code,r['pearson_X1'],r['pearson_X2'],r['pearson_X3']))
    lines += ['', '全分割の相関・MSE・各XとのPearson/Spearmanはmetrics.csv。符号はPCA出力のままであり、比較時は符号不定性を考慮する。', '',
        '## 学習効果とMLP単位要約', '',
        '同じマスクのB→Cで絶対相関差は{:+.6f}、MSE差は{:+.6f}。C→Dでは絶対相関差{:+.6f}、MSE差{:+.6f}。'.format(delta('B','C'),test['C']['x_reconstruction_raw_mse']-test['B']['x_reconstruction_raw_mse'],delta('C','D'),test['D']['x_reconstruction_raw_mse']-test['C']['x_reconstruction_raw_mse']),
        'Aに対する絶対相関差: B={:+.6f}、C={:+.6f}、D={:+.6f}、E={:+.6f}。'.format(*[delta('A',code) for code in 'BCDE']),
        'これらは今回指定された全条件の事後評価であり、Z相関で最良条件を選択する教師なし手続きではない。1シードでの小差に統計的一般性や安定性を主張しない。', '',
        '## 特徴量重複・線形対照の意味', '',
        '各ブロック内・ブロック間の平均/最大絶対相関、|r|≥0.95の割合、数値rank、participation rank、PC1分散割合をfeature_redundancy.csvへ保存。SVDは記述統計だけに使用し、因子や特徴の選択に使っていない。',
        'DのMLP別PC1分散説明率はD_block_pca.csv、最終PC1のグループ別二乗loading寄与はpca_group_loadings.csv。因子同士の相関・学習データで固定したアフィン対応の誤差はfactor_pair_diagnostics.csv。',
        'Eの各特徴はcXで、学習標準化後はsign(c)×標準化Xとなる。各Xを10回ずつ均等に複製するため、EのPCAはAと符号・定数倍率を除いて同じ因子になることが理論的に期待される。これは線形複製だけの効果を確認する設計上の性質であり、得られた結果を見て条件を変更していない。',
        '全30列を標準化しても、同じ入力を表す高相関な列が多いと、その方向がPCAで繰り返し重み付けされる。DはMLPごとに1得点・単位分散にそろえるが、ブロックごとの非線形な歪みや情報欠落を消すことは保証しない。', '',
        '## 有効性と因果グラフへの課題', '',
        '学習によるB→Cの変化、Dによる重複の調整、Eによる線形複製の対照を別々に判断する。Zに相関する因子を読み出せることだけでは、GraN-DAGの学習が元Xより優れた潜在情報を作るとは言えない。',
        '隠れ層と抽出因子はXの決定論的関数であり、新たな独立の因果変数や共通原因として扱うことはできない。今回のD/Eも因果構造を識別する方法ではない。潜在生成モデルへの拡張には、識別可能性の追加仮定、直接辺と潜在説明の競合、潜在崩壊の対策が残る。',
        '## 実行', '', '```bash', 'bash run_in_container.sh python -B latent_feature_controls.py --stage all', '```',
        '既存出力を上書きしない。A/B/Cの再fit、NNの学習、データ生成は呼び出さない。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def evaluate(settings):
    root,output=mapped(settings['source_dir']).resolve(),mapped(settings['output_dir']).resolve()
    if any((output/name).exists() for name in ('metrics.csv','report.md','evaluation_status.json')):
        raise FileExistsError('Refusing evaluation overwrite')
    if settings != json.loads((output/'config.json').read_text()): raise ValueError('Changed settings')
    status=json.loads((output/'extraction_status.json').read_text())
    if not status['all_factors_frozen_before_Z'] or status['Z_read_attempts']:
        raise ValueError('Clean frozen extraction required')
    manifest=json.loads((output/'extraction_manifest.json').read_text())
    for name,expected in manifest.items():
        if digest(output/name)!=expected: raise ValueError('Frozen artifact changed: '+name)
    before=file_inventory(root)
    if before != json.loads((output/'source_inventory_before.json').read_text()):
        raise ValueError('Source changed since extraction')
    ids=np.load(str(output/'sample_indices.npy'));labels=np.load(str(output/'sample_split.npy'))
    np.testing.assert_array_equal(ids,np.load(str(root/'data/sample_indices.npy')))
    np.testing.assert_array_equal(labels,np.load(str(root/'data/sample_split.npy')))
    factors={code:np.load(str(output/'factors'/(code+'.npy'))) for code in 'ABCDE'}
    # First permitted Z read: all five factors and reconstruction models are fixed.
    z=np.load(str(root/'data/true_Z.npy'))
    if z.shape!=(1000,1) or not np.isfinite(z).all(): raise ValueError('Invalid evaluation Z')
    metrics=[]
    for original in read_csv(output/'label_free_metrics.csv'):
        row={key:(float(value) if key not in ('condition','description','split') else value) for key,value in original.items()}
        idx=labels==dict(SPLITS)[row['split']]
        row.update(factor_correlations(factors[row['condition']][idx,0],z[idx,0]))
        metrics.append(row)
    write_csv(output/'metrics.csv',metrics)
    write_csv(output/'comparison_table.csv',[r for r in metrics if r['split']=='test'])
    metadata=json.loads((output/'metadata.json').read_text())
    figures(output,metrics,factors,z,labels)
    make_report(output,metrics,metadata)
    if before!=file_inventory(root): raise ValueError('Source files changed')
    for name,expected in manifest.items():
        if digest(output/name)!=expected: raise ValueError('Frozen extraction modified')
    save_json(output/'evaluation_status.json',dict(completed=True,Z_evaluation_only=True,separate_process=True,
        source_files_unchanged=True,all_frozen_hashes_verified=True,nn_training_calls=0,vae_training_calls=0,
        data_generation_calls=0,condition_changes=0))
    for r in metrics:
        if r['split']=='test': print(json.dumps(r,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['extract','evaluate'])
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    settings=json.loads(Path(args.config).read_text())
    (extract if args.stage=='extract' else evaluate)(settings)


if __name__=='__main__': main()
