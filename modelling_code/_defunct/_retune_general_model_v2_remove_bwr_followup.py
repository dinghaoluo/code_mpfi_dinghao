import json
from pathlib import Path

nb_path = Path('modelling_code/general_model_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

cell2 = ''.join(nb['cells'][2]['source'])
repls2 = {
    '    baseline_sd: float   = 0.60  # CA1 pyramidal baseline std\n': '    baseline_sd: float   = 0.70  # CA1 pyramidal baseline std\n',
    '    wR_mean: float = 0.56  # run-related coupling mean (absorbs the former run-drive amplitude)\n': '    wR_mean: float = 0.54  # run-related coupling mean (absorbs the former run-drive amplitude)\n',
    '    intrinsic_tau_mean: float   = 0.35  # mean intrinsic CA1 tau before baseline-dependent nudging\n': '    intrinsic_tau_mean: float   = 0.40  # mean intrinsic CA1 tau before baseline-dependent nudging\n',
    '    baseline_tau_coupling: float = 0.14  # lower-baseline cells receive slightly higher tau, globally across the population\n': '    baseline_tau_coupling: float = 0.12  # lower-baseline cells receive slightly higher tau, globally across the population\n',
    '    lc_to_da_gain: float = 5.00  # scales the DA release trace after the exponential transform\n': '    lc_to_da_gain: float = 4.60  # scales the DA release trace after the exponential transform\n',
}
for old, new in repls2.items():
    if old not in cell2:
        raise SystemExit(f'missing cell2 pattern: {old!r}')
    cell2 = cell2.replace(old, new)
nb['cells'][2]['source'] = cell2.splitlines(keepends=True)

cell7 = ''.join(nb['cells'][7]['source'])
old_layout = (
    "lc_gs = outer[0, 0].subgridspec(2, 1, height_ratios=[0.56, 0.44], hspace=0.0)\n"
    "a = fig.add_subplot(lc_gs[0, 0])\n"
    "a_pad = fig.add_subplot(lc_gs[1, 0])\n"
)
new_layout = (
    "lc_gs = outer[0, 0].subgridspec(2, 1, height_ratios=[0.56, 0.44], hspace=0.0)\n"
    "a_gs = lc_gs[0, 0].subgridspec(1, 3, width_ratios=[0.18, 0.64, 0.18], wspace=0.0)\n"
    "a_left = fig.add_subplot(a_gs[0, 0])\n"
    "a = fig.add_subplot(a_gs[0, 1])\n"
    "a_right = fig.add_subplot(a_gs[0, 2])\n"
    "a_pad = fig.add_subplot(lc_gs[1, 0])\n"
)
if old_layout not in cell7:
    raise SystemExit('missing cell7 layout block')
cell7 = cell7.replace(old_layout, new_layout)
old_pad = "a_pad.set_facecolor('white')\na_pad.axis('off')\n"
new_pad = "for ax in [a_left, a_right, a_pad]:\n    ax.set_facecolor('white')\n    ax.axis('off')\n"
if old_pad not in cell7:
    raise SystemExit('missing cell7 pad block')
cell7 = cell7.replace(old_pad, new_pad)
nb['cells'][7]['source'] = cell7.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
