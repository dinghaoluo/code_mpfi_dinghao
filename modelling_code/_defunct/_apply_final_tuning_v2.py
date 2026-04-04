import json
from pathlib import Path

nb_path = Path('modelling_code/general_model_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
cell2 = ''.join(nb['cells'][2]['source'])
repls = {
    '    frac_da_targ: float  = 0.35  # proportion of CA1 cells targeted by DA\n': '    frac_da_targ: float  = 0.25  # proportion of CA1 cells targeted by DA\n',
    '    da_targ_wR_bias: float = 0.02  # weak enrichment of DA-targeted cells among the more run-responsive cells\n': '    da_targ_wR_bias: float = 0.10  # enrichment of DA-targeted cells among the more run-responsive cells\n',
    '    da_half_rate: float  = 3.00  # r_1/2; midpoint of the targeted extra DA weight vs previous firing rate\n': '    da_half_rate: float  = 3.25  # r_1/2; midpoint of the targeted extra DA weight vs previous firing rate\n',
    '    da_rate_slope: float = 0.25  # k_r; steepness of the sigmoid\n': '    da_rate_slope: float = 0.30  # k_r; steepness of the sigmoid\n',
    '    wDA_global: float    = 0.05  # diffuse DA coupling shared across the population\n': '    wDA_global: float    = 0.04  # diffuse DA coupling shared across the population\n',
    '    da_block_scale: float = 0.35  # 1 - how much DA is blocked in exp 3\n': '    da_block_scale: float = 0.15  # 1 - how much DA is blocked in exp 3\n',
    '    lc_to_da_gain: float = 4.60  # scales the DA release trace after the exponential transform\n': '    lc_to_da_gain: float = 5.40  # scales the DA release trace after the exponential transform\n',
    '    da_ca1_delay: float  = 0.45  # additional delay from released DA to effective CA1 modulation\n': '    da_ca1_delay: float  = 1.00  # additional delay from released DA to effective CA1 modulation\n',
}
for old, new in repls.items():
    if old not in cell2:
        raise SystemExit(f'missing pattern: {old!r}')
    cell2 = cell2.replace(old, new)
nb['cells'][2]['source'] = cell2.splitlines(keepends=True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
