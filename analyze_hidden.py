#!/usr/bin/env python3
"""Exploratory analysis of exported hidden layers; no GraN-DAG weights change."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def varying_columns(x):
    return x.std(axis=0) > 1e-10 * max(1., float(np.abs(x).max(initial=0)))


def correlations(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    denom = np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)[None, :]
    return np.divide(a.T @ b, denom, out=np.full(denom.shape, np.nan), where=denom > 1e-16)


def fit_ridge(x, y, alpha):
    mean = x.mean(0)
    scale = x.std(0)
    keep = varying_columns(x)
    z = (x[:, keep] - mean[keep]) / scale[keep]
    target_mean = y.mean()
    if z.shape[1]:
        weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y-target_mean))
    else:
        weights = np.empty(0)

    def predict(new):
        return target_mean + ((new[:, keep] - mean[keep]) / scale[keep]) @ weights
    return predict


def r2(y, pred):
    return float(1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2))


def probe(features, y, train, validation):
    """5-fold training-only alpha selection; final score on existing validation."""
    train_ids = np.flatnonzero(train)
    folds = np.array_split(np.random.RandomState(1729).permutation(train_ids), 5)
    choices = [1e-4, 1e-2, 1., 100.]
    losses = []
    for alpha in choices:
        errors = []
        for held in folds:
            fit_ids = np.setdiff1d(train_ids, held)
            predict = fit_ridge(features[fit_ids], y[fit_ids], alpha)
            errors.extend((y[held] - predict(features[held]))**2)
        losses.append(float(np.mean(errors)))
    alpha = choices[int(np.argmin(losses))]
    predict = fit_ridge(features[train], y[train], alpha)
    return {'alpha': alpha, 'train_r2': r2(y[train], predict(features[train])),
            'validation_r2': r2(y[validation], predict(features[validation]))}


def linear_cka(a, b):
    if not varying_columns(a).any() or not varying_columns(b).any():
        return None
    a, b = a - a.mean(0), b - b.mean(0)
    return float(np.sum((a.T @ b)**2) /
                 (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def analyze(data_path, directories, output):
    output.mkdir(parents=True, exist_ok=True)
    raw = np.load(str(data_path), allow_pickle=False)
    summaries, units, corr_rows, probes, comparisons = [], [], [], [], []
    stages = []
    for index, directory in enumerate(directories):
        with (directory / 'metadata.json').open() as stream:
            metadata = json.load(stream)
        ids = np.load(str(directory / 'sample_indices.npy'))
        split = np.load(str(directory / 'sample_split.npy'))
        train, validation = split == 0, split == 1
        if not train.any() or not validation.any():
            raise ValueError('Analysis needs both original training and validation rows')
        x = raw[ids]
        graph = np.load(str(directory / 'adjacency.npy'))
        names = [v['name'] for v in metadata['variables']]
        layers = [np.load(str(directory / layer['file'])) for layer in metadata['layers']]
        stage = 'train' if index == 0 else 'cam_pruned'
        stages.append((stage, ids, split, layers, graph, names))
        input_ranks = rankdata(x[train], axis=0)
        for layer_index, hidden in enumerate(layers):
            for target, name in enumerate(names):
                h = hidden[train, target, :]
                varying = varying_columns(h)
                variance = np.linalg.svd(h-h.mean(0), compute_uv=False)**2
                if varying.any():
                    pc95 = int(np.searchsorted(np.cumsum(variance)/variance.sum(), .95)+1)
                    effective_dim = float(variance.sum()**2 / (variance**2).sum())
                else:
                    pc95, effective_dim = 0, 0.
                parents = np.flatnonzero(graph[:, target])
                summaries.append({'stage': stage, 'layer': layer_index+1, 'target': name,
                                  'parents': ';'.join(names[p] for p in parents),
                                  'parent_count': len(parents), 'units': h.shape[1],
                                  'varying_units': int(varying.sum()), 'pc95_dimensions': pc95,
                                  'variance_participation_ratio': effective_dim})
                pearson = correlations(h, x[train])
                spearman = correlations(rankdata(h, axis=0), input_ranks)
                # Suppress undefined correlations for empirically constant units.
                pearson[~varying] = np.nan
                spearman[~varying] = np.nan
                for unit in range(h.shape[1]):
                    units.append({'stage': stage, 'layer': layer_index+1, 'target': name,
                                  'unit': unit+1, 'mean': float(h[:, unit].mean()),
                                  'std': float(h[:, unit].std()),
                                  'negative_fraction': float((h[:, unit] < 0).mean()),
                                  'varying': bool(varying[unit])})
                    for column, source in enumerate(names):
                        corr_rows.append({'stage': stage, 'layer': layer_index+1,
                                          'target': name, 'unit': unit+1,
                                          'observed_variable': source,
                                          'pearson': float(pearson[unit, column]),
                                          'spearman': float(spearman[unit, column]),
                                          'direct_parent': bool(graph[column, target])})
        for target, name in enumerate(names):
            parents = np.flatnonzero(graph[:, target])
            features = [('parents_linear', x[:, parents])]
            features += [('hidden_layer_{}'.format(k+1), h[:, target, :])
                         for k, h in enumerate(layers)]
            for feature_name, feature in features:
                result = probe(feature, x[:, target], train, validation)
                probes.append(dict(stage=stage, target=name, representation=feature_name, **result))

    a, b = stages
    if not np.array_equal(a[1], b[1]) or not np.array_equal(a[2], b[2]):
        raise ValueError('Stages must have matching source rows and splits')
    for k, (before, after) in enumerate(zip(a[3], b[3])):
        for j, name in enumerate(a[5]):
            selected = a[2] == 0
            comparisons.append({'target': name, 'layer': k+1,
                                'parents_before': int(a[4][:, j].sum()),
                                'parents_after': int(b[4][:, j].sum()),
                                'linear_cka': linear_cka(before[selected, j], after[selected, j]),
                                'relative_rms_change': float(np.linalg.norm(after[:, j]-before[:, j]) /
                                                             max(np.linalg.norm(before[:, j]), 1e-30))})
    write_csv(output / 'layer_summary.csv', summaries)
    write_csv(output / 'unit_summary.csv', units)
    write_csv(output / 'observed_correlations.csv', corr_rows)
    write_csv(output / 'linear_probes.csv', probes)
    write_csv(output / 'stage_comparison.csv', comparisons)
    make_plot(output, summaries, probes, comparisons, a[5])

    lookup = {(p['stage'], p['target'], p['representation']): p for p in probes}
    lines = [
        '# 隠れ層の探索的解析結果', '',
        '対象: 既存データ1000サンプル、10変数、隠れ層2層・各10ユニット。',
        '学習直後とCAM pruning後を比較。潜在変数の推定・復元は実行していない。', '',
        '## 計算方法', '',
        '- 定数ユニット判定・PCA・相関・CKAは元の学習800行で計算。',
        '- 定数判定: 標準偏差 <= 1e-10 × max(1, 当該MLP隠れ層の最大絶対値)。',
        '- PCA95: 平均を引いた未標準化活性の分散95%を説明する主成分数。潜在変数の数ではない。',
        '- 有効次元は固有値のparticipation ratio。(分散の合計)^2 / 分散の二乗和。',
        '- 線形プローブ: 観測対象値を親変数または隠れ層からRidge回帰で予測する補助解析。',
        '- 特徴量標準化とalpha選択は学習800行のみで実施。alphaは5-fold CVで選択。',
        '- R²は既存の検証200行で評価。0は検証平均と同等、負はそれより悪い。',
        '- この検証データはGraN-DAGの収束判定とCAM pruningの枝選択にも使用済み。独立した最終テストではない。',
        '- 補助回帰の係数はNN本体の係数ではなく、元のNNの予測精度を直接表さない。',
        '- CAM pruning後のNNは再学習されていない。マスク変更による活性を比較している。',
        '- CKAは表現類似度の指標。どちらかが定数表現なら未定義として空欄。',
        '- 相関は観測変数との関係を表すだけで、潜在変数や因果関係の発見を意味しない。', '',
        '## CAM pruning後の変数別結果', '',
        '| 予測対象 | 親 | 変動ユニット L1/L2 | PCA95 L1/L2 | 親の線形回帰 R² | 隠れ層2の線形プローブ R² |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for name in a[5]:
        rows = [s for s in summaries if s['stage'] == 'cam_pruned' and s['target'] == name]
        l1, l2 = rows
        base = lookup[('cam_pruned', name, 'parents_linear')]['validation_r2']
        h2 = lookup[('cam_pruned', name, 'hidden_layer_2')]['validation_r2']
        lines.append('| {} | {} | {}/{} | {}/{} | {:.3f} | {:.3f} |'.format(
            name, l1['parents'] or 'なし', l1['varying_units'], l2['varying_units'],
            l1['pc95_dimensions'], l2['pc95_dimensions'], base, h2))
    roots = [s['target'] for s in summaries
             if s['stage'] == 'cam_pruned' and s['layer'] == 2 and s['parent_count'] == 0]
    lines += ['', '## 今回の具体的な観察', '',
              'CAM pruning後に親を持たず、サンプルごとの隠れ活性が一定である対象: ' + ', '.join(roots) + '。',
              '親を持つ対象では、観測値の線形回帰よりも隠れ層2からの線形プローブの検証R²が高い。',
              'これは隠れ層に非線形な予測用特徴があることを示すが、学習の効果を分離するには未学習NNとの比較も必要。',
              '枝刈りで検証R²が必ず上がるわけではない。構造評価と特徴の予測能力を分けて評価する。',
              '前後の各値はlinear_probes.csvに保存。補助回帰は元NNの出力層を再学習したモデルとは異なる。', '',
              '## 読み取れることと限界', '',
              '親なしのMLPでは入力マスクがすべて0なので、隠れ層が一定になるのは構造上自然。',
              'Leaky ReLUの負値は小さな傾きを持ち、負値が多いだけでdead unitとは呼べない。',
              '単一の親から作った非線形表現でもPCA次元は2以上になり得るため、',
              'PCA次元を未知の原因の個数と解釈してはいけない。',
              '学習データで有用な特徴があることと、潜在変数が復元可能なことは別の検証課題。',
              '本データには潜在変数の正解値がないため、潜在情報の有無を直接検証できない。', '',
              '将来は潜在変数の正解を分離保存し、NN学習・枝選択に使わない独立データでプローブを評価する。',
              '元の観測入力・未学習NNの特徴・正解をシャッフルした対照とも比較する。',
              '今回の解析ではその潜在変数プローブは実行していない。', '',
              '## 出力', '',
              '- overview.png: 次元・ユニット変動・補助予測・表現比較。',
              '- layer_summary.csv: 各MLP・各層の分散と次元。',
              '- unit_summary.csv: 各ユニットの平均・標準偏差・負値割合。',
              '- observed_correlations.csv: 学習行におけるPearson/Spearman相関。定数はNaN。',
              '- linear_probes.csv: 親と隠れ層からの補助予測の学習・検証R²。',
              '- stage_comparison.csv: pruning前後のCKAと相対変化。', '',
              'CKAの参考: [Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a.html)。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


def make_plot(output, summary, probes, comparisons, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    idx = np.arange(len(names))
    for stage, label, color, offset in [('train', 'Before CAM', '#4775af', -.18),
                                         ('cam_pruned', 'After CAM', '#e78b39', .18)]:
        rows = [s for s in summary if s['stage'] == stage and s['layer'] == 2]
        axes[0, 0].bar(idx+offset, [r['pc95_dimensions'] for r in rows], .36, label=label, color=color)
        axes[0, 1].bar(idx+offset, [r['varying_units'] for r in rows], .36, label=label, color=color)
    axes[0, 0].set_title('Layer 2: PCs explaining 95% of variance')
    axes[0, 0].set_ylabel('PC count (not latent-variable count)')
    axes[0, 1].set_title('Layer 2: units varying across training samples')
    axes[0, 1].set_ylim(0, 11)
    for feature, label, offset, color in [('parents_linear', 'Linear probe of parents', -.18, '#777777'),
                                            ('hidden_layer_2', 'Linear probe of hidden layer 2', .18, '#56a080')]:
        rows = [p for p in probes if p['stage'] == 'cam_pruned' and p['representation'] == feature]
        axes[1, 0].bar(idx+offset, [p['validation_r2'] for p in rows], .36, label=label, color=color)
    axes[1, 0].axhline(0, color='black', linewidth=.6)
    axes[1, 0].set_title('After CAM: observed-target prediction on validation rows')
    axes[1, 0].set_ylabel('Validation R2 (exploratory; not latent recovery)')
    for layer, marker in [(1, 'o'), (2, 's')]:
        rows = [c for c in comparisons if c['layer'] == layer]
        axes[1, 1].plot(idx, [np.nan if r['linear_cka'] is None else r['linear_cka'] for r in rows],
                        marker=marker, linestyle='none', label='Layer {}'.format(layer))
    axes[1, 1].set_title('Before vs after CAM: linear CKA (blank = constant)')
    axes[1, 1].set_ylim(-.05, 1.05)
    for axis in axes.flat:
        axis.set_xticks(idx)
        axis.set_xticklabels(names)
        axis.legend(fontsize=8)
        axis.grid(axis='y', alpha=.2)
    fig.savefig(str(output / 'overview.png'), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/unpacked/data_p10_e10_n1000_GP/data1.npy'))
    parser.add_argument('--before', type=Path, default=Path('experiments/p10-run1/train/hidden'))
    parser.add_argument('--after', type=Path, default=Path('experiments/p10-run1/cam-pruning/cutoff_0.001000/hidden'))
    parser.add_argument('--output', type=Path, default=Path('experiments/p10-run1/hidden_analysis'))
    args = parser.parse_args()
    analyze(args.data, [args.before, args.after], args.output)


if __name__ == '__main__':
    main()
