import json
from pathlib import Path

nb_path = Path('modelling_code/general_model_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

cell3 = ''.join(nb['cells'][3]['source'])
needle = "def plot_trace_pair(ax, t, left_traces, right_traces, left_color, right_color, left_label, right_label, title, show_xlabel=False, lower_floor=0.0):\n"
insert = '''def match_paired_pre_baseline(left_traces, right_traces, t, window):
    left = np.asarray(left_traces, dtype=float).copy()
    right = np.asarray(right_traces, dtype=float).copy()
    mask = window_mask(t, window)
    left_pre = np.nanmean(left[:, mask], axis=1, keepdims=True)
    right_pre = np.nanmean(right[:, mask], axis=1, keepdims=True)
    shared_pre = 0.5 * (left_pre + right_pre)
    return left + (shared_pre - left_pre), right + (shared_pre - right_pre)


'''
if insert not in cell3:
    if needle not in cell3:
        raise SystemExit('plot_trace_pair definition not found')
    cell3 = cell3.replace(needle, insert + needle)
nb['cells'][3]['source'] = cell3.splitlines(keepends=True)

cell11 = ''.join(nb['cells'][11]['source'])
old = "fig, ax_up, ax_down, ax_up_bar, ax_down_bar = build_experiment_axes()\n\nplot_trace_pair(\n    ax_up,\n    results['t'],\n    results['da_up_traces'],\n    results['non_da_up_traces'],\n"
new = "fig, ax_up, ax_down, ax_up_bar, ax_down_bar = build_experiment_axes()\n\nda_up_plot, non_da_up_plot = match_paired_pre_baseline(results['da_up_traces'], results['non_da_up_traces'], results['t'], p.pre_window)\nda_down_plot, non_da_down_plot = match_paired_pre_baseline(results['da_down_traces'], results['non_da_down_traces'], results['t'], p.pre_window)\n\nplot_trace_pair(\n    ax_up,\n    results['t'],\n    da_up_plot,\n    non_da_up_plot,\n"
if old not in cell11:
    raise SystemExit('exp2 first trace block not found')
cell11 = cell11.replace(old, new)
cell11 = cell11.replace("    results['da_down_traces'],\n    results['non_da_down_traces'],\n", "    da_down_plot,\n    non_da_down_plot,\n")
nb['cells'][11]['source'] = cell11.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
