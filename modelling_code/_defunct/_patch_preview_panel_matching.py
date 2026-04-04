import json
from pathlib import Path

nb_path = Path('modelling_code/general_model_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

cell0 = ''.join(nb['cells'][0]['source'])
old_text = "with the first timestep initialized from the DA-free rate proxy.\n\n**3. Form the total CA1 drive**\n"
new_text = "with the first timestep initialized from the DA-free rate proxy. For readability, the preview panel labels this x-axis simply as firing rate (Hz), even though the model update itself uses the previous discrete timestep rate, $r_i(t_{k-1})$.\n\n**3. Form the total CA1 drive**\n"
if old_text not in cell0:
    raise SystemExit('cell0 insertion point not found')
cell0 = cell0.replace(old_text, new_text)
nb['cells'][0]['source'] = cell0.splitlines(keepends=True)

cell7 = ''.join(nb['cells'][7]['source'])
repls = {
    "a_gs = lc_gs[0, 0].subgridspec(1, 3, width_ratios=[0.18, 0.64, 0.18], wspace=0.0)\n": "a_gs = lc_gs[0, 0].subgridspec(1, 3, width_ratios=[0.16, 0.68, 0.16], wspace=0.0)\n",
    "e = fig.add_subplot(outer[1, 0])\n": "e_gs = outer[1, 0].subgridspec(1, 3, width_ratios=[0.16, 0.68, 0.16], wspace=0.0)\ne_left = fig.add_subplot(e_gs[0, 0])\ne = fig.add_subplot(e_gs[0, 1])\ne_right = fig.add_subplot(e_gs[0, 2])\n",
    "for ax in [a_left, a_right, a_pad]:\n": "for ax in [a_left, a_right, a_pad, e_left, e_right]:\n",
    "a.set_title('LC activity')\na.legend(frameon=False, fontsize=9, loc='upper right')\n": "a.set_title('LC activity')\na.legend(frameon=False, fontsize=9, loc='upper right')\nmatched_preview_aspect = 1.0\na.set_box_aspect(matched_preview_aspect)\n",
    "e.set_xlabel('Previous firing rate, $r_i(t_{k-1})$ (Hz)')\ne.set_ylabel('Targeted extra weight, $w_{\\mathrm{extra}}^{DA}$')\ne.set_title('Activity-dependent targeted DA weight')\n": "e.set_xlabel('Firing rate (Hz)')\ne.set_ylabel('DA sensitivity, $w_{\\mathrm{extra}}^{DA}$')\ne.set_title('Activity-dependent targeted DA weight')\ne.set_box_aspect(matched_preview_aspect)\n",
    "f.set_xticklabels(['w_Run', 'w_Rew.'])\n": "f.set_xticklabels(['$w_{\\mathrm{Run}}$', '$w_{\\mathrm{Rew}}$'])\n",
    "im0 = g.imshow(\n    heatmap,\n    aspect='auto',\n    cmap='viridis',\n    extent=[t_ref[t_mask][0], t_ref[t_mask][-1], heatmap.shape[0], 0],\n    vmin=-heat_lim,\n    vmax=heat_lim,\n)\n": "heatmap_extent = [t_ref[t_mask][0], t_ref[t_mask][-1] + p.dt, heatmap.shape[0], 0]\nim0 = g.imshow(\n    heatmap,\n    aspect='auto',\n    cmap='viridis',\n    extent=heatmap_extent,\n    vmin=-heat_lim,\n    vmax=heat_lim,\n)\n",
    "im1 = h.imshow(\n    norm_heatmap,\n    aspect='auto',\n    cmap='viridis',\n    extent=[t_ref[t_mask][0], t_ref[t_mask][-1], norm_heatmap.shape[0], 0],\n    vmin=0.0,\n    vmax=1.0,\n)\n": "norm_heatmap_extent = [t_ref[t_mask][0], t_ref[t_mask][-1] + p.dt, norm_heatmap.shape[0], 0]\nim1 = h.imshow(\n    norm_heatmap,\n    aspect='auto',\n    cmap='Greys',\n    extent=norm_heatmap_extent,\n    vmin=0.0,\n    vmax=1.0,\n)\n",
}
for old, new in repls.items():
    if old not in cell7:
        raise SystemExit(f'missing cell7 pattern: {old!r}')
    cell7 = cell7.replace(old, new)
nb['cells'][7]['source'] = cell7.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
