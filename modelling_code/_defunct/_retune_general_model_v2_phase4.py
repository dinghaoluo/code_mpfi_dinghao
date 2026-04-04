import json
from pathlib import Path

nb_path = Path('modelling_code/general_model_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

cell2 = ''.join(nb['cells'][2]['source'])
replacements = {
    '    baseline_wR_coupling: float = 0.10  # lower-baseline cells receive slightly stronger run coupling, globally across the population\n': '    baseline_wR_coupling: float = 0.04  # lower-baseline cells receive slightly stronger run coupling, globally across the population\n',
    '    baseline_tau_coupling: float = 0.12  # lower-baseline cells receive slightly higher tau, globally across the population\n': '    baseline_tau_coupling: float = 0.14  # lower-baseline cells receive slightly higher tau, globally across the population\n',
    '    run_on_mid: float     = 0.12  # midpoint of the rising sigmoid\n': '    run_on_mid: float     = 0.08  # midpoint of the rising sigmoid\n',
    '    run_off_mid: float    = 1.90  # midpoint of the falling sigmoid\n': '    run_off_mid: float    = 1.85  # midpoint of the falling sigmoid\n',
    '    run_rise_scale: float = 0.06  # how sharp is the rise?\n': '    run_rise_scale: float = 0.05  # how sharp is the rise?\n',
}
for old, new in replacements.items():
    cell2 = cell2.replace(old, new)
nb['cells'][2]['source'] = cell2.splitlines(keepends=True)

cell7 = ''.join(nb['cells'][7]['source']).replace('fig = plt.figure(figsize=(8.4, 15.0), constrained_layout=True)\n', 'fig = plt.figure(figsize=(8.4, 16.0), constrained_layout=True)\n')
nb['cells'][7]['source'] = cell7.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
