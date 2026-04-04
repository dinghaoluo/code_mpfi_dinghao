import json
import textwrap
from pathlib import Path


NB_PATH = Path("modelling_code/general_model_v2.ipynb")


CELL_2 = textwrap.dedent(
    """
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
        baseline_sd: float   = 0.35  # CA1 pyramidal baseline std

        wR_mean: float = 0.56  # run-related coupling mean (absorbs the former run-drive amplitude)
        wR_sd: float   = 0.42
        wW_mean: float = 0.46  # reward-related coupling mean (absorbs the former reward-drive amplitude)
        wW_sd: float   = 1.15

        # additive DA drive
        frac_da_targ: float          = 0.35  # proportion of CA1 cells targeted by DA
        da_targ_baseline_bias: float = 0.06  # weak enrichment of DA-targeted cells among lower-baseline cells
        da_targ_wW_bias: float       = 0.02  # weak enrichment of DA-targeted cells away from strongly reward-suppressed cells
        da_half_rate: float          = 3.00  # r_1/2; midpoint of the targeted extra DA weight vs previous firing rate
        da_rate_slope: float         = 0.25  # k_r; steepness of the sigmoid
        wDA_global: float            = 0.05  # diffuse DA coupling shared across the population
        da_block_scale: float        = 0.35  # 1 - how much DA is blocked in exp 3

        # cell-intrinsic recovery acting on the full latent CA1 state
        intrinsic_tau_mean: float    = 0.35  # mean intrinsic CA1 tau before baseline-dependent nudging
        baseline_tau_coupling: float = 0.10  # lower-baseline cells receive slightly higher tau, globally across the population
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
    """
).strip("\n") + "\n"


CELL_4 = textwrap.dedent(
    """
    # simulation functions

    def make_population(p, rng):
        b = rng.normal(p.baseline_mean, p.baseline_sd, p.n_cells)
        wR = rng.normal(p.wR_mean, p.wR_sd, p.n_cells)
        wW = rng.normal(p.wW_mean, p.wW_sd, p.n_cells)

        baseline_z = (p.baseline_mean - b) / max(p.baseline_sd, p.eps)
        tau_center = p.intrinsic_tau_mean + p.baseline_tau_coupling * baseline_z

        wW_z = (wW - p.wW_mean) / max(p.wW_sd, p.eps)
        targ_prob = np.clip(
            p.frac_da_targ
            + p.da_targ_baseline_bias * baseline_z
            - p.da_targ_wW_bias * wW_z,
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
    """
).strip("\n") + "\n"


CELL_7 = textwrap.dedent(
    """
    t_ref = results['t']
    rng_ref = np.random.default_rng(p.seed_start)
    ref_pop = make_population(p, rng_ref)

    p_ref_lc = deepcopy(p)
    p_ref_lc.lc_amp = p.lc_amp * p.lc_activation_fold

    ref_base = simulate_population_condition(t_ref, p, ref_pop, da_scale=1.0)
    ref_lc = simulate_population_condition(t_ref, p_ref_lc, ref_pop, da_scale=1.0)

    resp = ref_base['resp']
    sort_order = np.argsort(resp)[::-1]
    t_mask = (t_ref >= -1.0) & (t_ref <= 4.0)
    display_rates = ref_base['rates'][:, t_mask]
    heatmap = baseline_subtracted_traces(ref_base['rates'], t_ref, p.pre_window)[sort_order][:, t_mask]
    heat_lim = np.nanpercentile(np.abs(heatmap), 97)
    if not np.isfinite(heat_lim) or heat_lim <= 0:
        heat_lim = 1.0

    row_min = np.nanmin(display_rates, axis=1, keepdims=True)
    row_max = np.nanmax(display_rates, axis=1, keepdims=True)
    row_span = np.maximum(row_max - row_min, p.eps)
    norm_heatmap = ((display_rates - row_min) / row_span)[sort_order]

    weight_matrix = np.column_stack([
        ref_pop['wR'][sort_order],
        ref_pop['wW'][sort_order],
    ])
    weight_lim = np.nanpercentile(np.abs(weight_matrix), 98)
    if not np.isfinite(weight_lim) or weight_lim <= 0:
        weight_lim = 1.0

    da_targ_sorted = ref_pop['da_targ'][sort_order].astype(float)[:, None]

    r_grid = np.linspace(0.0, 8.0, 400)
    wDA_extra_grid = sigmoid(r_grid, p.da_half_rate, p.da_rate_slope)


    def split_response_tiers(resp_values, class_mask, stronger_is_higher=True):
        idx = np.flatnonzero(class_mask)
        if idx.size == 0:
            return []
        if stronger_is_higher:
            strength = resp_values[idx]
        else:
            strength = 1.0 / np.maximum(resp_values[idx], p.eps)
        ordered = idx[np.argsort(strength)]
        groups = np.array_split(ordered, 3)
        return [
            ('Low', groups[0]),
            ('Mid', groups[1]),
            ('High', groups[2]),
        ]


    def plot_response_tiers(ax, rates, resp_values, class_mask, stronger_is_higher, palette, title):
        tier_defs = split_response_tiers(resp_values, class_mask, stronger_is_higher=stronger_is_higher)
        tier_traces = []
        for (label, idx), color in zip(tier_defs, palette):
            if len(idx) == 0:
                continue
            traces = rates[idx]
            tier_traces.append(traces)
            plot_mean_sem(ax, t_ref, traces, color, label)
        ax.axvline(0, linestyle='--', color='red', linewidth=1)
        ax.set_xlim([-1.0, 4.0])
        ax.set_xlabel('Time from run onset (s)')
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title(title)
        if tier_traces:
            set_trace_ylim(ax, *tier_traces, pad_frac=0.10, lower_floor=0.0)
        ax.legend(frameon=False, fontsize=8, loc='upper right')


    fig = plt.figure(figsize=(8.4, 15.0), constrained_layout=True)
    fig.set_facecolor('white')
    outer = fig.add_gridspec(5, 2, height_ratios=[1.60, 0.92, 1.00, 0.88, 1.02], wspace=0.24, hspace=0.18)

    lc_gs = outer[0, 0].subgridspec(2, 1, height_ratios=[0.56, 0.44], hspace=0.0)
    a = fig.add_subplot(lc_gs[0, 0])
    a_pad = fig.add_subplot(lc_gs[1, 0])
    drive_gs = outer[0, 1].subgridspec(3, 1, hspace=0.15)
    b = fig.add_subplot(drive_gs[0, 0])
    c = fig.add_subplot(drive_gs[1, 0])
    d = fig.add_subplot(drive_gs[2, 0])
    e = fig.add_subplot(outer[1, 0])
    f_outer = outer[1, 1].subgridspec(1, 2, width_ratios=[1.20, 0.18], wspace=0.03)
    f = fig.add_subplot(f_outer[0, 0])
    f_targ = fig.add_subplot(f_outer[0, 1])
    g = fig.add_subplot(outer[2, 0])
    h = fig.add_subplot(outer[2, 1])
    i = fig.add_subplot(outer[3, 0])
    j = fig.add_subplot(outer[3, 1])
    k = fig.add_subplot(outer[4, 0])
    l = fig.add_subplot(outer[4, 1])

    for ax in [a, b, c, d, e, f, f_targ, i, j, k, l]:
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    a_pad.set_facecolor('white')
    a_pad.axis('off')

    for ax in [f, f_targ, g, h]:
        ax.set_facecolor('white')
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_visible(True)

    a.plot(t_ref, ref_base['drives']['L'], color='0.65', linewidth=2.2, label='Ctrl.')
    a.plot(t_ref, ref_lc['drives']['L'], color='royalblue', linewidth=2.5, label='Stim.')
    a.axvline(0, linestyle='--', color='red', linewidth=1)
    a.set_xlim([-1.0, 4.0])
    a.set_xlabel('Time from run onset (s)')
    a.set_ylabel('LC signal (a.u.)')
    a.set_title('LC activity')
    a.legend(frameon=False, fontsize=9, loc='upper right')

    for ax, key, color in [
        (b, 'R', 'goldenrod'),
        (c, 'W', 'magenta'),
        (d, 'D', 'darkgreen'),
    ]:
        ax.plot(t_ref, ref_base['drives'][key], color=color, linewidth=2.6)
        ax.axvline(0, linestyle='--', color='red', linewidth=1)
        ax.set_xlim([-1.0, 4.0])
        ax.set_ylabel('a.u.')

    b.set_xticklabels([])
    c.set_xticklabels([])
    d.set_xlabel('Time from run onset (s)')

    e.plot(r_grid, wDA_extra_grid, color='darkgreen', linewidth=2.5)
    e.axvline(p.da_half_rate, linestyle='--', color='0.55', linewidth=1)
    e.set_xlim([0.0, 8.0])
    e.set_ylim([0.0, 1.05])
    e.set_xlabel('Previous firing rate, $r_i(t_{k-1})$ (Hz)')
    e.set_ylabel('Targeted extra weight, $w_{\\mathrm{extra}}^{DA}$')
    e.set_title('Activity-dependent targeted DA weight')

    im_w = f.imshow(
        weight_matrix,
        aspect='auto',
        cmap='coolwarm',
        vmin=-weight_lim,
        vmax=weight_lim,
        interpolation='nearest',
    )
    f.set_xticks([0, 1])
    f.set_xticklabels(['wR', 'wW'])
    f.set_yticks([])
    f.set_ylabel('Cells (high to low response)')
    f.set_title('Couplings')
    cbar_w = fig.colorbar(im_w, ax=f, shrink=0.66, pad=0.015)
    cbar_w.set_label('Coupling (a.u.)')

    im_targ = f_targ.imshow(
        da_targ_sorted,
        aspect='auto',
        cmap='Greys',
        vmin=0.0,
        vmax=1.0,
        interpolation='nearest',
    )
    f_targ.set_xticks([0])
    f_targ.set_xticklabels(['$I_{\\mathrm{targ}}$'])
    f_targ.set_yticks([])
    f_targ.set_title('DA target')

    im0 = g.imshow(
        heatmap,
        aspect='auto',
        cmap='viridis',
        extent=[t_ref[t_mask][0], t_ref[t_mask][-1], heatmap.shape[0], 0],
        vmin=-heat_lim,
        vmax=heat_lim,
        interpolation='nearest',
    )
    g.axvline(0, linestyle='--', color='red', linewidth=1)
    g.set_xlim([-1.0, 4.0])
    g.set_xlabel('Time from run onset (s)')
    g.set_ylabel('Cells (high to low response)')
    g.set_title('Baseline-subtracted heatmap')
    g.set_box_aspect(0.74)
    cbar0 = fig.colorbar(im0, ax=g, shrink=0.62, pad=0.02)
    cbar0.set_label('Delta rate (Hz)')

    im1 = h.imshow(
        norm_heatmap,
        aspect='auto',
        cmap='viridis',
        extent=[t_ref[t_mask][0], t_ref[t_mask][-1], norm_heatmap.shape[0], 0],
        vmin=0.0,
        vmax=1.0,
        interpolation='nearest',
    )
    h.axvline(0, linestyle='--', color='red', linewidth=1)
    h.set_xlim([-1.0, 4.0])
    h.set_xlabel('Time from run onset (s)')
    h.set_ylabel('Cells (high to low response)')
    h.set_title('Per-cell normalized heatmap')
    h.set_box_aspect(0.74)
    cbar1 = fig.colorbar(im1, ax=h, shrink=0.62, pad=0.02)
    cbar1.set_label('Normalized rate')

    plot_mean_sem(i, t_ref, results['base_up_traces'], class_colors['is_up'], 'PyrUp')
    i.axvline(0, linestyle='--', color='red', linewidth=1)
    i.set_xlim([-1.0, 4.0])
    set_trace_ylim(i, results['base_up_traces'], pad_frac=0.10, lower_floor=0.0)
    i.set_xlabel('Time from run onset (s)')
    i.set_ylabel('Firing rate (Hz)')
    i.set_title('PyrUp population mean')
    i.legend(frameon=False, fontsize=9, loc='upper right')

    plot_mean_sem(j, t_ref, results['base_down_traces'], class_colors['is_down'], 'PyrDown')
    j.axvline(0, linestyle='--', color='red', linewidth=1)
    j.set_xlim([-1.0, 4.0])
    set_trace_ylim(j, results['base_down_traces'], pad_frac=0.10, lower_floor=0.0)
    j.set_xlabel('Time from run onset (s)')
    j.set_ylabel('Firing rate (Hz)')
    j.set_title('PyrDown population mean')
    j.legend(frameon=False, fontsize=9, loc='upper right')

    plot_response_tiers(
        k,
        ref_base['rates'],
        resp,
        ref_base['classes']['is_up'],
        stronger_is_higher=True,
        palette=['lightsalmon', 'indianred', 'firebrick'],
        title='PyrUp by response strength',
    )

    plot_response_tiers(
        l,
        ref_base['rates'],
        resp,
        ref_base['classes']['is_down'],
        stronger_is_higher=False,
        palette=['thistle', 'mediumorchid', 'indigo'],
        title='PyrDown by response strength',
    )

    for ext in ['.png', '.pdf']:
        fig.savefig(PLOT_SAVE_DIR / f'general_model_overview{ext}', dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()
    """
).strip("\n") + "\n"


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
for idx, text in {2: CELL_2, 4: CELL_4, 7: CELL_7}.items():
    nb["cells"][idx]["source"] = text.splitlines(keepends=True)
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
