import json
import textwrap
from pathlib import Path

nb_path = Path('modelling_code/general_model_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

cell2 = textwrap.dedent("""
@dataclass
class PARAMS:
    # simulation grid
    dt: float     = 0.01  # time step
    t_pre: float  = 1.00  # simulation start time (negative) relative to run onset
    t_post: float = 6.00  # simulation end time relative to run onset

    # bootstrap controls
    n_bootstrap: int          = 20  # how many times to run bootstrapping for each exp?
    seed_start: int           = 0  # which seed to use by default (rng)
    lc_activation_fold: float = 2.40  # optogenetic LC stim. (exp 1) fold

    # population size
    n_cells: int = 1000  # population size

    # population priors
    baseline_mean: float = 1.23  # CA1 pyramidal baseline mean
    baseline_sd: float   = 0.60  # CA1 pyramidal baseline std

    wR_mean: float = 0.56  # run-related coupling mean (absorbs the former run-drive amplitude)
    wR_sd: float   = 0.15
    baseline_wR_coupling: float = 0.10  # lower-baseline cells receive slightly stronger run coupling, globally across the population
    wW_mean: float = 0.46  # reward-related coupling mean (absorbs the former reward-drive amplitude)
    wW_sd: float   = 1.15

    # additive DA drive
    frac_da_targ: float  = 0.35  # proportion of CA1 cells targeted by DA
    da_targ_wR_bias: float = 0.02  # weak enrichment of DA-targeted cells among the more run-responsive cells
    da_half_rate: float  = 3.00  # r_1/2; midpoint of the targeted extra DA weight vs previous firing rate
    da_rate_slope: float = 0.25  # k_r; steepness of the sigmoid
    wDA_global: float    = 0.05  # diffuse DA coupling shared across the population
    da_block_scale: float = 0.35  # 1 - how much DA is blocked in exp 3

    # cell-intrinsic recovery acting on the full latent CA1 state
    intrinsic_tau_mean: float   = 0.35  # mean intrinsic CA1 tau before baseline-dependent nudging
    baseline_tau_coupling: float = 0.12  # lower-baseline cells receive slightly higher tau, globally across the population
    intrinsic_tau_sd: float      = 0.05
    intrinsic_tau_min: float     = 0.05
    intrinsic_tau_max: float     = 1.00

    # output nonlinearity / output limits
    softplus_beta: float = 2.00
    max_rate: float      = 20.00  # capping CA1 pyramidal cell firing rate

    # run-drive shape
    run_on_mid: float     = 0.12  # midpoint of the rising sigmoid
    run_off_mid: float    = 2.00  # midpoint of the falling sigmoid
    run_rise_scale: float = 0.06  # how sharp is the rise?
    run_fall_scale: float = 0.28  # how sharp is the fall?

    # reward-drive shape
    reward_on_mid: float     = 0.68
    reward_off_mid: float    = 1.80
    reward_rise_scale: float = 0.14
    reward_fall_scale: float = 0.34

    # LC drive shape
    lc_baseline: float = 1.00  # baseline amplitude of LC activity
    lc_amp: float      = 1.50  # delta(peak, baseline) of LC phasic activity
    lc_mu: float       = 0.00  # centre (relative to run onset)
    lc_sigma: float    = 0.20  # sigma of Gaussian

    # LC -> DA release conversion (single transformation only)
    lc_to_da_gain: float = 4.80  # scales the DA release trace after the exponential transform
    da_kernel_tau: float = 2.50  # slow, flat decay for the single exponential LC -> DA kernel
    da_ca1_delay: float  = 0.40  # additional delay from released DA to effective CA1 modulation

    # population analysis
    pre_window: tuple  = (-1.00, -0.50)  # this does not matter much, since pre-run-onset activity is usually flat
    post_window: tuple = (0.50, 1.50)  # same as used in real experiments
    up_thresh: float   = 3 / 2  # same
    down_thresh: float = 2 / 3  # same

    # numerical safeguard
    eps: float = 1e-6


class_colors = {
    'is_up':    'firebrick',
    'is_other': 'grey',
    'is_down':  'purple'
}

condition_colors = {
    'baseline': '0.35',
    'lc': 'royalblue',
    'blocked': class_colors['is_down'],
    'da_targeted': class_colors['is_up'],
    'not_targeted': '0.45'
}
""").strip('\n') + '\n'

nb['cells'][2]['source'] = cell2.splitlines(keepends=True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
