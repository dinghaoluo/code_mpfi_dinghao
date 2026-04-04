import json
from pathlib import Path

nb_path = Path('modelling_code/general_model_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

cell1 = ''.join(nb['cells'][1]['source'])
old_import = 'from scipy.stats import ttest_rel, wilcoxon\n'
new_import = 'from scipy.stats import linregress, ttest_rel, wilcoxon\n'
if old_import not in cell1:
    raise SystemExit('cell1 import pattern not found')
cell1 = cell1.replace(old_import, new_import)
nb['cells'][1]['source'] = cell1.splitlines(keepends=True)

cell6 = ''.join(nb['cells'][6]['source'])
old_preview = 'This preview collects the main ingredients of the final model on one reference population: LC activity, the run / reward / DA-release drives, the activity-dependent targeted extra DA-weight curve, the response-sorted run/reward weight matrix plus the DA-targeting indicator, both baseline-subtracted and per-cell normalized population heatmaps, and the PyrUp / PyrDown mean traces by class and response-strength tier.\n'
new_preview = 'This preview collects the main ingredients of the final model on one reference population: LC activity, the run / reward / DA-release drives, the activity-dependent targeted extra DA-weight curve, the response-sorted run/reward weight matrix plus the DA-targeting indicator, both baseline-subtracted and per-cell normalized population heatmaps, the PyrUp / PyrDown mean traces by class and response-strength tier, and response-rank diagnostics for the intrinsic recovery constant with regression overlays.\n'
if old_preview not in cell6:
    raise SystemExit('cell6 preview text not found')
cell6 = cell6.replace(old_preview, new_preview)
nb['cells'][6]['source'] = cell6.splitlines(keepends=True)

cell7 = ''.join(nb['cells'][7]['source'])
repls = {
    "fig = plt.figure(figsize=(7.4, 16.0), constrained_layout=True)\n": "fig = plt.figure(figsize=(7.4, 18.6), constrained_layout=True)\n",
    "outer = fig.add_gridspec(5, 2, height_ratios=[1.60, 0.92, 1.00, 0.88, 1.02], wspace=0.24, hspace=0.18)\n": "outer = fig.add_gridspec(6, 2, height_ratios=[1.60, 0.92, 1.00, 0.88, 1.02, 0.96], wspace=0.24, hspace=0.18)\n",
    "k = fig.add_subplot(outer[4, 0])\nl = fig.add_subplot(outer[4, 1])\n": "k = fig.add_subplot(outer[4, 0])\nl = fig.add_subplot(outer[4, 1])\nm = fig.add_subplot(outer[5, 0])\nn = fig.add_subplot(outer[5, 1])\n",
    "for ax in [a, b, c, d, e, f, f_targ, i, j, k, l]:\n": "for ax in [a, b, c, d, e, f, f_targ, i, j, k, l, m, n]:\n",
}
for old, new in repls.items():
    if old not in cell7:
        raise SystemExit(f'missing cell7 block: {old!r}')
    cell7 = cell7.replace(old, new)

needle = "def plot_response_tiers(ax, rates, resp_values, class_mask, stronger_is_higher, palette, title):\n"
insert = '''def plot_tau_rank_scatter(ax, tau_values, resp_values, class_mask, color, title, stronger_is_higher=True):
    idx = np.flatnonzero(class_mask)
    if idx.size == 0:
        ax.set_axis_off()
        return
    if stronger_is_higher:
        strength = resp_values[idx]
    else:
        strength = 1.0 / np.maximum(resp_values[idx], p.eps)
    ordered = idx[np.argsort(strength)[::-1]]
    ranks = np.arange(1, len(ordered) + 1)
    tau_ranked = tau_values[ordered]
    fit = linregress(ranks, tau_ranked)
    ax.scatter(ranks, tau_ranked, s=9, alpha=0.45, color=color, edgecolors='none')
    ax.plot(ranks, fit.intercept + fit.slope * ranks, color='black', linewidth=1.8)
    ax.set_xlim([1, len(ranks)])
    ax.set_xlabel('Neuron rank (high to low response)')
    ax.set_ylabel(r'Intrinsic $\\tau$ (s)')
    ax.set_title(title)
    ax.set_xticks([1, len(ranks)])
    ax.set_xticklabels(['High', 'Low'])
    ax.text(
        0.04,
        0.96,
        f'r = {fit.rvalue:.2f}\\np = {fit.pvalue:.1e}',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5),
    )


'''
if insert not in cell7:
    if needle not in cell7:
        raise SystemExit('plot_response_tiers insertion point not found')
    cell7 = cell7.replace(needle, insert + needle)

append_block = '''
plot_tau_rank_scatter(
    m,
    ref_pop['tau_intr'],
    resp,
    ref_base['classes']['is_up'],
    class_colors['is_up'],
    'PyrUp intrinsic tau by response rank',
    stronger_is_higher=True,
)

plot_tau_rank_scatter(
    n,
    ref_pop['tau_intr'],
    resp,
    ref_base['classes']['is_down'],
    class_colors['is_down'],
    'PyrDown intrinsic tau by response rank',
    stronger_is_higher=False,
)

'''
anchor = "plot_response_tiers(\n    l,\n    ref_base['rates'],\n    resp,\n    ref_base['classes']['is_down'],\n    stronger_is_higher=False,\n    palette=['thistle', 'mediumorchid', 'indigo'],\n    title='PyrDown by response strength',\n)\n\nsave_figure_bundle(fig, 'general_model_overview')\n"
replacement = "plot_response_tiers(\n    l,\n    ref_base['rates'],\n    resp,\n    ref_base['classes']['is_down'],\n    stronger_is_higher=False,\n    palette=['thistle', 'mediumorchid', 'indigo'],\n    title='PyrDown by response strength',\n)\n\n" + append_block + "save_figure_bundle(fig, 'general_model_overview')\n"
if anchor not in cell7:
    raise SystemExit('cell7 append anchor not found')
cell7 = cell7.replace(anchor, replacement)

nb['cells'][7]['source'] = cell7.splitlines(keepends=True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
