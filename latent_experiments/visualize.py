"""PNG figures and a Japanese report for the frozen, held-out experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def metric_row(metrics, stage, kind, layer, scope, factor):
    return next(r for r in metrics if (r['stage'], r['representation'], r['layer'], r['scope'], r['latent']) ==
                (stage, kind, layer, scope, factor))


def make_figures(directory, x, z, labels, hidden, masks, observed, latent,
                 metrics, predictions, correlation_rows, pca_concat):
    d, k = x.shape[1], z.shape[1]
    columns = min(d, 5)
    blocks = int(np.ceil(d/columns))
    fig, axes = plt.subplots(k*blocks, columns, figsize=(3*columns, 2.7*k*blocks), squeeze=False)
    for factor in range(k):
        for j in range(d):
            axis = axes[factor*blocks+j//columns, j % columns]
            axis.scatter(z[:, factor], x[:, j], s=5, alpha=.3)
            axis.set_xlabel('Z'+str(factor+1))
            axis.set_ylabel('X'+str(j+1))
        for j in range(d, blocks*columns):
            axes[factor*blocks+j//columns, j % columns].axis('off')
    fig.suptitle('Generated observed variables vs true latent factors (all samples)', y=1.01)
    finish(fig, directory/'observed_vs_latent.png')

    n_layers = len(hidden['train'])
    fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 6), squeeze=False)
    for row, stage in enumerate(('train', 'cam')):
        for layer, values in enumerate(hidden[stage]):
            axis = axes[row, layer]
            for j in range(d):
                axis.hist(values[labels == 0, j, :].ravel(), bins=40, density=True,
                          histtype='step', label='MLP X'+str(j+1))
            axis.set_title('{} / hidden layer {}'.format(stage, layer+1))
            axis.set_xlabel('Post-activation value')
            axis.set_ylabel('Density (training rows)')
            axis.legend(fontsize=7)
    finish(fig, directory/'activation_distributions.png')

    fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 6), squeeze=False)
    for row, stage in enumerate(('train', 'cam')):
        for layer, values in enumerate(hidden[stage], 1):
            h = values.shape[2]
            matrix = np.full((k, d*h), np.nan)
            for entry in correlation_rows:
                if entry['stage'] == stage and entry['layer'] == layer:
                    factor = int(entry['latent'][1:])-1
                    target = int(entry['mlp'][1:])-1
                    matrix[factor, target*h+entry['unit']-1] = entry['pearson']
            axis = axes[row, layer-1]
            im = axis.imshow(matrix, aspect='auto', vmin=-1, vmax=1, cmap='coolwarm')
            axis.set_title('{} / L{}: hidden-Z Pearson r'.format(stage, layer))
            axis.set_xticks(np.arange(d)*h+(h-1)/2.)
            axis.set_xticklabels(['MLP X'+str(j+1) for j in range(d)], rotation=30)
            axis.set_yticks(range(k))
            axis.set_yticklabels(['Z'+str(j+1) for j in range(k)])
            fig.colorbar(im, ax=axis)
    finish(fig, directory/'hidden_latent_correlations.png')

    for factor in range(k):
        factor_name = 'Z'+str(factor+1)
        fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 7), squeeze=False)
        for row, stage in enumerate(('train', 'cam')):
            for layer in range(1, n_layers+1):
                coords = pca_concat[(stage, layer)]
                axis = axes[row, layer-1]
                im = axis.scatter(coords[labels == 2, 0], coords[labels == 2, 1],
                                  c=z[labels == 2, factor], cmap='viridis', s=14, alpha=.8)
                axis.set_title('{} / L{}: final-test PCA'.format(stage, layer))
                axis.set_xlabel('PC1 (PCA fit on training rows only)')
                axis.set_ylabel('PC2')
                fig.colorbar(im, ax=axis, label=factor_name)
        finish(fig, directory/('pca_'+factor_name+'.png'))

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        stage_names = ['train', 'cam', 'random_train_mask', 'random_cam_mask', 'random_full']
        short = ['Trained', 'CAM', 'Random\ntrain mask', 'Random\nCAM mask', 'Random\nfull mask']
        for row, score in enumerate(('test_r2', 'test_mse')):
            for layer in range(1, n_layers+1):
                width = .8/n_layers
                scores = [metric_row(metrics, stage, 'hidden_ridge', layer, 'concat', factor_name)[score]
                          for stage in stage_names]
                axes[row, 0].bar(np.arange(5)-.4+width*(layer-.5), scores, width,
                                 label='Hidden L'+str(layer))
            for kind, label, style in [('X_ridge', 'Original X / Ridge', '--'),
                                       ('X_extra_trees', 'Original X / ExtraTrees', ':')]:
                baseline = metric_row(metrics, 'X', kind, 0, 'all', factor_name)[score]
                axes[row, 0].axhline(baseline, linestyle=style, label=label, color='black')
            axes[row, 0].set_xticks(range(5))
            axes[row, 0].set_xticklabels(short, fontsize=8)
            axes[row, 0].set_title(factor_name+' / concatenated hidden features / '+score)
            for index, stage in enumerate(stage_names[:4]):
                scores = [metric_row(metrics, stage, 'hidden_ridge', n_layers,
                                      'X'+str(j+1), factor_name)[score] for j in range(d)]
                axes[row, 1].bar(np.arange(d)-.3+.2*index, scores, .2, label=short[index].replace('\n', ' '))
            axes[row, 1].set_xticks(range(d))
            axes[row, 1].set_xticklabels(['MLP X'+str(j+1) for j in range(d)], fontsize=8, rotation=20)
            axes[row, 1].set_title(factor_name+' / individual MLP, last layer / '+score)
            for axis in axes[row]:
                axis.legend(fontsize=7)
                axis.grid(axis='y', alpha=.2)
        finish(fig, directory/('probe_scores_'+factor_name+'.png'))

        selections = [('X', 'X_ridge', 0, 'all'), ('X', 'X_extra_trees', 0, 'all'),
                      ('train', 'hidden_ridge', n_layers, 'concat'), ('cam', 'hidden_ridge', n_layers, 'concat')]
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
        for axis, key in zip(axes, selections):
            rows = [r for r in predictions if (r['stage'], r['representation'], r['layer'], r['scope']) == key
                    and r['latent'] == factor_name]
            actual, pred = np.array([[r['true_z'], r['prediction']] for r in rows]).T
            axis.scatter(actual, pred, s=9, alpha=.5)
            limits = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
            axis.plot(limits, limits, 'k--', linewidth=.7)
            axis.set_title('{} / {}'.format(key[0], key[1]))
            axis.set_xlabel('True '+factor_name+' (final test)')
            axis.set_ylabel('Prediction')
        finish(fig, directory/('latent_predictions_'+factor_name+'.png'))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    full = nx.DiGraph()
    observed_nodes = ['X'+str(i+1) for i in range(d)]
    latent_nodes = ['Z'+str(i+1) for i in range(k)]
    full.add_nodes_from(observed_nodes+latent_nodes)
    full.add_edges_from((observed_nodes[a], observed_nodes[b]) for a, b in np.argwhere(observed))
    full.add_edges_from((latent_nodes[a], observed_nodes[b]) for a, b in np.argwhere(latent))
    position = {name: (i, 0) for i, name in enumerate(observed_nodes)}
    position.update({name: ((i+1)*max(1, d-1)/(k+1), 1) for i, name in enumerate(latent_nodes)})
    nx.draw_networkx(full, pos=position, ax=axes[0], node_color=['#8db5df']*d+['#edb366']*k,
                     node_size=600, font_size=8, arrowsize=15)
    axes[0].set_title('True full SEM (orange: unobserved Z)')
    for axis, stage in zip(axes[1:], ['train', 'cam']):
        graph = nx.DiGraph()
        graph.add_nodes_from(observed_nodes)
        graph.add_edges_from((observed_nodes[a], observed_nodes[b]) for a, b in np.argwhere(masks[stage]))
        nx.draw_networkx(graph, pos=nx.circular_layout(graph), ax=axis,
                         node_color='#8db5df', node_size=600, font_size=8, arrowsize=15)
        axis.set_title(stage+' inferred graph (observed X only)')
    for axis in axes:
        axis.axis('off')
    finish(fig, directory/'causal_graphs.png')


def make_report(root, config, metrics, graphs, units, reference):
    output = root/'analysis'
    status = json.loads((root/'training_status.json').read_text())
    k, d = config['data']['n_latent'], config['data']['n_observed']
    layers = config['training']['num_layers']
    lines = ['# 潜在変数情報の保持・読み出し実験', '',
             '## 実験条件と漏洩防止', '',
             '- 観測変数{}個、潜在変数{}個、サンプル{}行。'.format(d, k, config['data']['n_samples']),
             '- GraN-DAGにはXのみを入力。Zは別プロセスの回帰評価で初めて読み込む。',
             '- 学習{}行・検証{}行・独立最終テスト{}行。'.format(status['n_train'], status['n_validation'], status['n_final_test']),
             '- 既存train/to_dag/cam_pruningを再利用。損失、DAG制約、最適化方法は変更なし。',
             '- NN正規化は学習行のみ。CAMは学習+検証行のみ。最終テストXは学習完了後の凍結推論でのみ使用。',
             '- RidgeはStandardScalerを含むPipelineで、学習行内の交差検証によりalphaを選択。',
             '- Xの非線形回帰はExtraTrees。同じ学習行内CVで深さ・葉の最小行数を選択。',
             '- ZはGraN-DAGのグラフ・収束判断・ハイパーパラメータ選択に使用しない。',
             '- 未学習NNは実際の初期重みを保存。学習後/CAM後と同じマスクを付けた条件、および初期全結合マスク条件を評価。',
             '- 学習収束: {}、反復数: {}、最終h: {:.6g}。'.format(status['converged'], status['iterations'], status['final_h']),
             '- 初回worker所要時間: {:.1f}秒。'.format(status.get('original_worker_elapsed_seconds', status['worker_elapsed_seconds'])),
             '- 各層の出力形状: ({}, {}, {})。'.format(config['data']['n_samples'], d, config['training']['hid_dim']), '',
             '## 生成条件', '', '```json', json.dumps(config['data'], indent=2, ensure_ascii=False), '```', '',
             '観測係数の行=原因X・列=結果X。潜在係数は行=原因Z・列=結果X。',
             'Xi = Σk A[k,i] f(Zk) + Σj B[j,i] g(Xj) + εi。',
             'observed_adjacencyは直接のX間エッジのみ。未観測共通原因を含む全構造や周辺化グラフではない。', '',
             '## 最終テストでの予測結果（全MLP連結）', '',
             '| Z | 特徴 | 層 | R² | MSE |', '| --- | --- | --- | --- | --- |']
    names = {'train': '学習直後', 'to_dag': 'DAG化後（CAM前）', 'cam': 'CAM後', 'random_train_mask': '未学習/学習後マスク',
             'random_cam_mask': '未学習/CAM後マスク', 'random_full': '未学習/初期全結合'}
    for factor in range(1, k+1):
        name = 'Z'+str(factor)
        for kind in ('X_ridge', 'X_extra_trees'):
            row = metric_row(metrics, 'X', kind, 0, 'all', name)
            lines.append('| {} | {} | — | {:.4f} | {:.4f} |'.format(name, kind, row['test_r2'], row['test_mse']))
        for stage in names:
            for layer in range(1, layers+1):
                row = metric_row(metrics, stage, 'hidden_ridge', layer, 'concat', name)
                lines.append('| {} | {} | {} | {:.4f} | {:.4f} |'.format(
                    name, names[stage], layer, row['test_r2'], row['test_mse']))
    lines += ['', '## マスクで保持された入力の比較', '',
              '連結隠れ層でも、どのMLPにも入力されないX列は利用できない。',
              '次は各段階で少なくとも1つのMLPに入力されるX列の和集合からZをRidge予測した対照。',
              '| Z | 段階 | 利用可能X列数 | R² | MSE |', '| --- | --- | --- | --- | --- |']
    for factor in range(1, k+1):
        for stage in ('train', 'to_dag', 'cam'):
            row = metric_row(metrics, stage, 'input_union_ridge', 0, 'all_used_inputs', 'Z'+str(factor))
            lines.append('| Z{} | {} | {} | {:.4f} | {:.4f} |'.format(
                factor, names[stage], row['parents'], row['test_r2'], row['test_mse']))
    lines += ['', '## 各MLPと親の有無（第{}層、最終テスト）'.format(layers), '',
              '| Z | MLP | 親数 学習後→CAM後 | R² 学習後 | R² CAM後 | MSE 学習後 | MSE CAM後 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for factor in range(1, k+1):
        for j in range(1, d+1):
            a = metric_row(metrics, 'train', 'hidden_ridge', layers, 'X'+str(j), 'Z'+str(factor))
            b = metric_row(metrics, 'cam', 'hidden_ridge', layers, 'X'+str(j), 'Z'+str(factor))
            lines.append('| Z{} | X{} | {}→{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                factor, j, a['parents'], b['parents'], a['test_r2'], b['test_r2'], a['test_mse'], b['test_mse']))
    lines += ['', '## グラフ比較', '',
              '正解の全構造と推定観測グラフはfigures/causal_graphs.png。辺一覧はgraph_edges.csv。',
              '潜在交絡があるため通常のGraN-DAGの因果十分性仮定は満たされない。推定されたX間辺を直接因果と断定しない。',
              'graph_metrics.csvのSHD等は直接観測辺との記述的な差分で、潜在交絡を扱う精度指標ではない。', '']
    for graph in graphs:
        lines.append('- {}: 推定{}辺、正解の直接観測辺{}本、二値マスクはDAGか: {}。'.format(
            graph['stage'], graph['edges'], graph['direct_true_edges'], graph['is_acyclic']))
    lines += ['学習の制約は重み付き行列のhに適用されるため、微小重みを含む二値マスクには循環が残り得る。',
              '既存to_dag処理でDAG化した条件を別途保存・評価している。']
    if reference['applicable']:
        lines += ['', '## 線形ガウスの場合の理論上の参考値', '',
                  '真の線形生成式を既知とした母集団ベイズ最適予測の参考値。実測結果ではない。',
                  'Cov(Z|X) = (ΣZ^-1 + A diag(σeps^-2) Aᵀ)^-1。',
                  'MSE: '+str(reference['population_bayes_mse_per_z']),
                  'R²: '+str(reference['population_bayes_r2_per_z']),
                  '有限のテスト標本のR²は母集団参考値の上下に変動し得る。']
    lines += ['', '## 解釈上の注意', '',
              '- R²が高い場合、その特徴からZを読み出せる。未知の潜在変数を自動発見したことにはならない。',
              '- RidgeのR²が低くても、非線形な読み出しに使える情報が残る可能性がある。情報量そのものを測定したわけではない。',
              '- 隠れ層はマスクされたXの決定論的な変換。X以上の新しい情報を作るのではなく、読み出しやすさを比較している。',
              '- 個別MLPは自分自身のXを入力できない。X全列・全MLP連結との入力情報の違いに注意。parents_ridgeも併記している。',
              '- 親なしMLPの隠れ層は一定になり、サンプルごとのZを予測できない（回帰は学習Zの平均を返す）。',
              '- CAM後のNN重みは再学習しない。前後の差は入力マスク変更を含む。',
              '- train→CAMの差にはDAG化の閾値処理も含まれる。to_dag条件とCAM条件の差で枝刈り自体の影響を区別する。',
              '- 相関やPCAの色分かれ、主成分数だけでは潜在因子を同定できない。定数ユニットの相関はNaN。',
              '- 未学習NNとの差が小さい場合、学習の効果より入力情報やランダム特徴変換で説明できる可能性がある。',
              '- 入力マスクの和集合が縮む場合、連結特徴の低下には元のX列を利用できなくなる影響も含まれる。',
              '- 初回は1つのデータ・学習シード。優劣の一般化には独立生成データと複数シードが必要。',
              '- 最終テストは評価のみ。結果を見て設定を調整する場合、別の新規最終テストが必要。', '',
              '## 再実行', '', '```bash',
              'bash run_in_container.sh python latent_experiment.py --config configs/latent_linear.json',
              '# 同一条件のキャッシュを再利用する場合のみ --resume を追加', '```', '',
              '条件変更は新しいoutput_dirを使う。既存p10-run1結果は変更していない。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
