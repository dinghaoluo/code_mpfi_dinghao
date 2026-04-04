import json
import re
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
    baseline_sd: float   = 0.50  # CA1 pyramidal baseline std

    wR_mean: float = 0.56  # run-related coupling mean (absorbs the former run-drive amplitude)
    wR_sd: float   = 0.22
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
    run_on_mid: float     = 0.14  # midpoint of the rising sigmoid
    run_off_mid: float    = 2.00  # midpoint of the falling sigmoid
    run_rise_scale: float = 0.06  # how sharp is the rise?
    run_fall_scale: float = 0.28  # how sharp is the fall?

    # reward-drive shape
    reward_on_mid: float     = 0.70
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

cell4 = textwrap.dedent("""
# simulation functions

def make_population(p, rng):
    b = rng.normal(p.baseline_mean, p.baseline_sd, p.n_cells)
    baseline_z = (p.baseline_mean - b) / max(p.baseline_sd, p.eps)

    wR_center = p.wR_mean + p.baseline_wR_coupling * baseline_z
    wR = rng.normal(wR_center, p.wR_sd, p.n_cells)
    wW = rng.normal(p.wW_mean, p.wW_sd, p.n_cells)

    tau_center = p.intrinsic_tau_mean + p.baseline_tau_coupling * baseline_z

    wR_z = (wR - p.wR_mean) / max(p.wR_sd, p.eps)
    targ_prob = np.clip(
        p.frac_da_targ + p.da_targ_wR_bias * wR_z,
        0.03,
        0.97,
    )
    da_targ = rng.random(p.n_cells) < targ_prob

    return {
        'b': b,
        'wR': wR,
        'wW': wW,
        'tau_intr': np.clip(
            rng.normal(tau_center, p.intrinsic_tau_sd, p.n_cells),
            p.intrinsic_tau_min,
            p.intrinsic_tau_max,
        ),
        'da_targ': da_targ,
    }


def simulate_population_condition(t, p, pop, da_scale=1.0):
    drives = make_drives(t, p)
    x = (
        pop['b'][:, None]
        + pop['wR'][:, None] * drives['R'][None, :]
        + pop['wW'][:, None] * drives['W'][None, :]
    )

    n_cells, n_t = x.shape
    alpha = np.clip(p.dt / pop['tau_intr'], 0.0, 1.0)

    latent_state = np.zeros((n_cells, n_t), dtype=float)
    rates = np.zeros((n_cells, n_t), dtype=float)

    rate_prev = np.clip(softplus(x[:, 0], p.softplus_beta), 0.0, p.max_rate)

    for k in range(n_t):
        wDA_extra = sigmoid(rate_prev, p.da_half_rate, p.da_rate_slope)
        wDA_t = p.wDA_global + pop['da_targ'].astype(float) * wDA_extra
        da_term = da_scale * wDA_t * drives['D'][k]
        state_target = x[:, k] + da_term

        if k == 0:
            latent_state[:, k] = state_target
        else:
            latent_state[:, k] = latent_state[:, k - 1] + alpha * (state_target - latent_state[:, k - 1])

        rates[:, k] = np.clip(softplus(latent_state[:, k], p.softplus_beta), 0.0, p.max_rate)
        rate_prev = rates[:, k]

    resp = response_strength(rates, t, p)
    classes = classify_cells(resp, p)
    return {
        't': t,
        'drives': drives,
        'rates': rates,
        'resp': resp,
        'classes': classes,
        'mean_traces': {
            'all': np.mean(rates, axis=0),
            'is_up': safe_mean_trace(rates, classes['is_up']),
            'is_other': safe_mean_trace(rates, classes['is_other']),
            'is_down': safe_mean_trace(rates, classes['is_down']),
        },
    }


def run_bootstrap_suite(p):
    t = np.arange(-p.t_pre, p.t_post, p.dt)
    post_mask = window_mask(t, p.post_window)

    p_lc = deepcopy(p)
    p_lc.lc_amp = p.lc_amp * p.lc_activation_fold

    base_all_traces = []
    lc_all_traces = []
    block_all_traces = []
    base_up_traces = []
    lc_up_traces = []
    block_up_traces = []
    base_down_traces = []
    lc_down_traces = []
    block_down_traces = []
    da_up_traces = []
    non_da_up_traces = []
    da_down_traces = []
    non_da_down_traces = []

    stats = {
        'base_up_pct': [],
        'base_down_pct': [],
        'lc_up_pct': [],
        'lc_down_pct': [],
        'block_up_pct': [],
        'block_down_pct': [],
        'p_up_da_targeted': [],
        'p_up_not_targeted': [],
        'p_down_da_targeted': [],
        'p_down_not_targeted': [],
        'post_rate_da_up': [],
        'post_rate_non_da_up': [],
    }

    for seed in range(p.seed_start, p.seed_start + p.n_bootstrap):
        rng = np.random.default_rng(seed)
        pop = make_population(p, rng)

        base  = simulate_population_condition(t, p, pop, da_scale=1.0)
        lc    = simulate_population_condition(t, p_lc, pop, da_scale=1.0)
        block = simulate_population_condition(t, p, pop, da_scale=p.da_block_scale)

        base_all_traces.append(base['mean_traces']['all'])
        lc_all_traces.append(lc['mean_traces']['all'])
        block_all_traces.append(block['mean_traces']['all'])
        base_up_traces.append(base['mean_traces']['is_up'])
        lc_up_traces.append(lc['mean_traces']['is_up'])
        block_up_traces.append(block['mean_traces']['is_up'])
        base_down_traces.append(base['mean_traces']['is_down'])
        lc_down_traces.append(lc['mean_traces']['is_down'])
        block_down_traces.append(block['mean_traces']['is_down'])

        da_mask = pop['da_targ']
        not_da_mask = ~da_mask
        da_up_mask = da_mask & base['classes']['is_up']
        non_da_up_mask = not_da_mask & base['classes']['is_up']
        da_down_mask = da_mask & base['classes']['is_down']
        non_da_down_mask = not_da_mask & base['classes']['is_down']
        da_up_traces.append(safe_mean_trace(base['rates'], da_up_mask))
        non_da_up_traces.append(safe_mean_trace(base['rates'], non_da_up_mask))
        da_down_traces.append(safe_mean_trace(base['rates'], da_down_mask))
        non_da_down_traces.append(safe_mean_trace(base['rates'], non_da_down_mask))

        stats['base_up_pct'].append(100 * np.mean(base['classes']['is_up']))
        stats['base_down_pct'].append(100 * np.mean(base['classes']['is_down']))
        stats['lc_up_pct'].append(100 * np.mean(lc['classes']['is_up']))
        stats['lc_down_pct'].append(100 * np.mean(lc['classes']['is_down']))
        stats['block_up_pct'].append(100 * np.mean(block['classes']['is_up']))
        stats['block_down_pct'].append(100 * np.mean(block['classes']['is_down']))
        stats['p_up_da_targeted'].append(100 * np.mean(base['classes']['is_up'][da_mask]))
        stats['p_up_not_targeted'].append(100 * np.mean(base['classes']['is_up'][not_da_mask]))
        stats['p_down_da_targeted'].append(100 * np.mean(base['classes']['is_down'][da_mask]))
        stats['p_down_not_targeted'].append(100 * np.mean(base['classes']['is_down'][not_da_mask]))
        stats['post_rate_da_up'].append(np.mean(base['rates'][da_up_mask][:, post_mask]) if np.any(da_up_mask) else np.nan)
        stats['post_rate_non_da_up'].append(np.mean(base['rates'][non_da_up_mask][:, post_mask]) if np.any(non_da_up_mask) else np.nan)

    stats = {key: np.asarray(value, dtype=float) for key, value in stats.items()}
    return {
        't': t,
        'params': p,
        'base_drives': make_drives(t, p),
        'base_all_traces': np.asarray(base_all_traces, dtype=float),
        'lc_all_traces': np.asarray(lc_all_traces, dtype=float),
        'block_all_traces': np.asarray(block_all_traces, dtype=float),
        'base_up_traces': np.asarray(base_up_traces, dtype=float),
        'lc_up_traces': np.asarray(lc_up_traces, dtype=float),
        'block_up_traces': np.asarray(block_up_traces, dtype=float),
        'base_down_traces': np.asarray(base_down_traces, dtype=float),
        'lc_down_traces': np.asarray(lc_down_traces, dtype=float),
        'block_down_traces': np.asarray(block_down_traces, dtype=float),
        'da_up_traces': np.asarray(da_up_traces, dtype=float),
        'non_da_up_traces': np.asarray(non_da_up_traces, dtype=float),
        'da_down_traces': np.asarray(da_down_traces, dtype=float),
        'non_da_down_traces': np.asarray(non_da_down_traces, dtype=float),
        'stats': stats,
    }
""").strip('\n') + '\n'

cell11 = textwrap.dedent("""
fig = plt.figure(figsize=(7.0, 4.0), constrained_layout=True)
fig.set_facecolor('white')
outer = fig.add_gridspec(3, 3, width_ratios=[4.6, 2.4, 2.4], height_ratios=[0.85, 1.30, 0.85], wspace=0.30, hspace=0.00)
trace_gs = outer[:, 0].subgridspec(2, 1, hspace=0.22)
ax_up = fig.add_subplot(trace_gs[0, 0])
ax_down = fig.add_subplot(trace_gs[1, 0])
ax_up_bar = fig.add_subplot(outer[1, 1])
ax_down_bar = fig.add_subplot(outer[1, 2])

for ax in [ax_up, ax_down, ax_up_bar, ax_down_bar]:
    ax.set_facecolor('white')

up_non_targ_color = 'lightcoral'
up_targ_color = class_colors['is_up']
down_non_targ_color = 'plum'
down_targ_color = class_colors['is_down']

plot_mean_sem(ax_up, results['t'], results['da_up_traces'], up_targ_color, 'DA-Up')
plot_mean_sem(ax_up, results['t'], results['non_da_up_traces'], up_non_targ_color, 'non-DA-Up')
ax_up.axvline(0, linestyle='--', color='red', linewidth=1)
ax_up.set_xlim([-1.0, 4.0])
ax_up.set_ylabel('Firing rate (Hz)')
ax_up.set_title('PyrUp')
set_trace_ylim(ax_up, results['da_up_traces'], results['non_da_up_traces'], pad_frac=0.10, lower_floor=0.0)
ax_up.legend(frameon=False, fontsize=8)
ax_up.spines['top'].set_visible(False)
ax_up.spines['right'].set_visible(False)
ax_up.set_xticklabels([])

plot_mean_sem(ax_down, results['t'], results['da_down_traces'], down_targ_color, 'DA-Up')
plot_mean_sem(ax_down, results['t'], results['non_da_down_traces'], down_non_targ_color, 'non-DA-Up')
ax_down.axvline(0, linestyle='--', color='red', linewidth=1)
ax_down.set_xlim([-1.0, 4.0])
ax_down.set_xlabel('Time from run onset (s)')
ax_down.set_ylabel('Firing rate (Hz)')
ax_down.set_title('PyrDown')
set_trace_ylim(ax_down, results['da_down_traces'], results['non_da_down_traces'], pad_frac=0.10, lower_floor=0.0)
ax_down.legend(frameon=False, fontsize=8)
ax_down.spines['top'].set_visible(False)
ax_down.spines['right'].set_visible(False)

pf.plot_bar_with_paired_scatter(
    ax_up_bar,
    results['stats']['p_up_not_targeted'],
    results['stats']['p_up_da_targeted'],
    colors=(up_non_targ_color, up_targ_color),
    title='PyrUp proportion',
    ylabel='Proportion (%)',
    xticklabels=('non-DA-Up', 'DA-Up'),
    ylim=paired_ylim(results['stats']['p_up_not_targeted'], results['stats']['p_up_da_targeted'], min_pad=3.0, anchor_floor=10.0),
)

pf.plot_bar_with_paired_scatter(
    ax_down_bar,
    results['stats']['p_down_not_targeted'],
    results['stats']['p_down_da_targeted'],
    colors=(down_non_targ_color, down_targ_color),
    title='PyrDown proportion',
    ylabel='Proportion (%)',
    xticklabels=('non-DA-Up', 'DA-Up'),
    ylim=paired_ylim(results['stats']['p_down_not_targeted'], results['stats']['p_down_da_targeted'], min_pad=3.0, anchor_floor=0.0),
)

for ext in ['.png', '.pdf']:
    fig.savefig(PLOT_SAVE_DIR / f'general_model_experiment_2_da_targeted_subsets{ext}', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()

print('DA-targeting summary (mean +/- SEM):')
print_summary_line('  P(PyrUp | DA-Up) (%)', results['stats']['p_up_da_targeted'])
print_summary_line('  P(PyrUp | non-DA-Up) (%)', results['stats']['p_up_not_targeted'])
print_paired_summary('  PyrUp proportion difference (%)', results['stats']['p_up_not_targeted'], results['stats']['p_up_da_targeted'])
print_summary_line('  P(PyrDown | DA-Up) (%)', results['stats']['p_down_da_targeted'])
print_summary_line('  P(PyrDown | non-DA-Up) (%)', results['stats']['p_down_not_targeted'])
print_paired_summary('  PyrDown proportion difference (%)', results['stats']['p_down_not_targeted'], results['stats']['p_down_da_targeted'])
print_summary_line('  Post-run rate, DA-Up (Hz)', results['stats']['post_rate_da_up'])
print_summary_line('  Post-run rate, non-DA-Up (Hz)', results['stats']['post_rate_non_da_up'])
print_paired_summary('  Post-run firing-rate difference (Hz)', results['stats']['post_rate_non_da_up'], results['stats']['post_rate_da_up'])
""").strip('\n') + '\n'

cell13 = textwrap.dedent("""
fig = plt.figure(figsize=(7.0, 4.0), constrained_layout=True)
fig.set_facecolor('white')
outer = fig.add_gridspec(3, 3, width_ratios=[4.6, 2.4, 2.4], height_ratios=[0.85, 1.30, 0.85], wspace=0.30, hspace=0.00)
trace_gs = outer[:, 0].subgridspec(2, 1, hspace=0.22)
ax_up = fig.add_subplot(trace_gs[0, 0])
ax_down = fig.add_subplot(trace_gs[1, 0])
ax_up_bar = fig.add_subplot(outer[1, 1])
ax_down_bar = fig.add_subplot(outer[1, 2])

for ax in [ax_up, ax_down, ax_up_bar, ax_down_bar]:
    ax.set_facecolor('white')

block_color = '0.45'

plot_mean_sem(ax_up, results['t'], results['base_up_traces'], class_colors['is_up'], 'Baseline')
plot_mean_sem(ax_up, results['t'], results['block_up_traces'], block_color, 'Blockade')
ax_up.axvline(0, linestyle='--', color='red', linewidth=1)
ax_up.set_xlim([-1.0, 4.0])
ax_up.set_ylabel('Firing rate (Hz)')
ax_up.set_title('PyrUp')
set_trace_ylim(ax_up, results['base_up_traces'], results['block_up_traces'], pad_frac=0.10, lower_floor=0.0)
ax_up.legend(frameon=False, fontsize=8)
ax_up.spines['top'].set_visible(False)
ax_up.spines['right'].set_visible(False)
ax_up.set_xticklabels([])

plot_mean_sem(ax_down, results['t'], results['base_down_traces'], class_colors['is_down'], 'Baseline')
plot_mean_sem(ax_down, results['t'], results['block_down_traces'], block_color, 'Blockade')
ax_down.axvline(0, linestyle='--', color='red', linewidth=1)
ax_down.set_xlim([-1.0, 4.0])
ax_down.set_xlabel('Time from run onset (s)')
ax_down.set_ylabel('Firing rate (Hz)')
ax_down.set_title('PyrDown')
set_trace_ylim(ax_down, results['base_down_traces'], results['block_down_traces'], pad_frac=0.10, lower_floor=0.0)
ax_down.legend(frameon=False, fontsize=8)
ax_down.spines['top'].set_visible(False)
ax_down.spines['right'].set_visible(False)

pf.plot_bar_with_paired_scatter(
    ax_up_bar,
    results['stats']['base_up_pct'],
    results['stats']['block_up_pct'],
    colors=(class_colors['is_up'], block_color),
    title='PyrUp proportion',
    ylabel='Proportion (%)',
    xticklabels=('Baseline', 'Blockade'),
    ylim=paired_ylim(results['stats']['base_up_pct'], results['stats']['block_up_pct'], min_pad=3.0, anchor_floor=10.0),
)

pf.plot_bar_with_paired_scatter(
    ax_down_bar,
    results['stats']['base_down_pct'],
    results['stats']['block_down_pct'],
    colors=(class_colors['is_down'], block_color),
    title='PyrDown proportion',
    ylabel='Proportion (%)',
    xticklabels=('Baseline', 'Blockade'),
    ylim=paired_ylim(results['stats']['base_down_pct'], results['stats']['block_down_pct'], min_pad=3.0, anchor_floor=10.0),
)

for ext in ['.png', '.pdf']:
    fig.savefig(PLOT_SAVE_DIR / f'general_model_experiment_3_partial_da_block{ext}', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()

print('Partial DA blockade summary (mean +/- SEM):')
print_summary_line('  Baseline PyrUp (%)', results['stats']['base_up_pct'])
print_summary_line('  Blockade PyrUp (%)', results['stats']['block_up_pct'])
print_paired_summary('  PyrUp proportion shift (%)', results['stats']['base_up_pct'], results['stats']['block_up_pct'])
print_summary_line('  Baseline PyrDown (%)', results['stats']['base_down_pct'])
print_summary_line('  Blockade PyrDown (%)', results['stats']['block_down_pct'])
print_paired_summary('  PyrDown proportion shift (%)', results['stats']['base_down_pct'], results['stats']['block_down_pct'])
""").strip('\n') + '\n'

for idx, text in {2: cell2, 4: cell4, 11: cell11, 13: cell13}.items():
    nb['cells'][idx]['source'] = text.splitlines(keepends=True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

pf_path = Path('utils/plotting_functions.py')
text = pf_path.read_text(encoding='utf-8')
text = text.replace('    y1 = top_y\n    y2 = y1\n', '    y1 = top_y\n    y2 = y1 + 0.06 * yrange\n')
text = text.replace("    ax.set_ylim(ylims[0], max(ylims[1], y2 + 0.08 * yrange))\n", "    ax.set_ylim(ylims[0], max(ylims[1], y2 + 0.10 * yrange))\n")
pf_path.write_text(text, encoding='utf-8')
