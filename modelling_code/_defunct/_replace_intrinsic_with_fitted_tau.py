import json
from pathlib import Path

nb_path = Path('modelling_code/general_model_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

cell1 = ''.join(nb['cells'][1]['source'])
old_import = 'from scipy.stats import linregress, ttest_rel, wilcoxon\n'
new_import = 'from scipy.optimize import curve_fit\nfrom scipy.stats import linregress, ttest_rel, wilcoxon\n'
if old_import not in cell1:
    raise SystemExit('cell1 import line not found')
cell1 = cell1.replace(old_import, new_import)
nb['cells'][1]['source'] = cell1.splitlines(keepends=True)

cell6 = ''.join(nb['cells'][6]['source'])
old_text = 'This preview collects the main ingredients of the final model on one reference population: LC activity, the run / reward / DA-release drives, the activity-dependent targeted extra DA-weight curve, the response-sorted run/reward weight matrix plus the DA-targeting indicator, both baseline-subtracted and per-cell normalized population heatmaps, the PyrUp / PyrDown mean traces by class and response-strength tier, and response-rank diagnostics for the intrinsic recovery constant with regression overlays.\n'
new_text = 'This preview collects the main ingredients of the final model on one reference population: LC activity, the run / reward / DA-release drives, the activity-dependent targeted extra DA-weight curve, the response-sorted run/reward weight matrix plus the DA-targeting indicator, both baseline-subtracted and per-cell normalized population heatmaps, the PyrUp / PyrDown mean traces by class and response-strength tier, and response-rank diagnostics for fitted recovery tau from the final simulated firing-rate traces, with regression and fit-quality annotations.\n'
if old_text not in cell6:
    raise SystemExit('cell6 text not found')
cell6 = cell6.replace(old_text, new_text)
nb['cells'][6]['source'] = cell6.splitlines(keepends=True)

cell7 = ''.join(nb['cells'][7]['source'])
old_resp = "resp = ref_base['resp']\nsort_order = np.argsort(resp)[::-1]\n"
new_resp = "resp = ref_base['resp']\nfitted_recovery_tau, fitted_recovery_r2 = fit_population_recovery_taus(ref_base['rates'], t_ref, ref_base['classes'])\nsort_order = np.argsort(resp)[::-1]\n"
# apply later after helper injection so function exists before execution order? this is just source order, yes helper defs later but assignment earlier would fail. so insert after helper defs instead.

old_block = '''def plot_tau_rank_scatter(ax, tau_values, resp_values, class_mask, color, title, stronger_is_higher=True):
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
new_block = '''def fit_cell_recovery_tau(trace, t, baseline, is_up, tau_bounds=(0.05, 6.0)):
    post_mask = (t >= 0.0) & (t <= 4.0)
    t_post = t[post_mask]
    y_post = np.asarray(trace[post_mask], dtype=float)
    if y_post.size < 8:
        return np.nan, np.nan

    extremum_idx = int(np.argmax(y_post) if is_up else np.argmin(y_post))
    t_fit = t_post[extremum_idx:]
    y_fit = y_post[extremum_idx:]
    if y_fit.size < 8:
        return np.nan, np.nan

    amp0 = float(y_fit[0] - baseline)
    if (is_up and amp0 <= p.eps) or ((not is_up) and amp0 >= -p.eps):
        return np.nan, np.nan

    amp_bound = max(abs(amp0) * 2.0, 0.25)
    amp_lower, amp_upper = ((0.0, amp_bound) if is_up else (-amp_bound, 0.0))

    def model(tt_local, amp, tau):
        return baseline + amp * np.exp(-(tt_local - t_fit[0]) / tau)

    try:
        popt, _ = curve_fit(
            model,
            t_fit,
            y_fit,
            p0=[amp0, 1.0],
            bounds=([amp_lower, tau_bounds[0]], [amp_upper, tau_bounds[1]]),
            maxfev=5000,
        )
    except Exception:
        return np.nan, np.nan

    pred = model(t_fit, *popt)
    ss_res = float(np.sum((y_fit - pred) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, p.eps)
    return float(popt[1]), float(r2)


def fit_population_recovery_taus(rates, t, classes):
    baseline = np.mean(rates[:, window_mask(t, p.pre_window)], axis=1)
    fitted_tau = np.full(rates.shape[0], np.nan)
    fitted_r2 = np.full(rates.shape[0], np.nan)

    for idx in np.flatnonzero(classes['is_up']):
        fitted_tau[idx], fitted_r2[idx] = fit_cell_recovery_tau(rates[idx], t, baseline[idx], True)
    for idx in np.flatnonzero(classes['is_down']):
        fitted_tau[idx], fitted_r2[idx] = fit_cell_recovery_tau(rates[idx], t, baseline[idx], False)

    return fitted_tau, fitted_r2


def plot_tau_rank_scatter(ax, tau_values, r2_values, resp_values, class_mask, color, title, stronger_is_higher=True):
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
    r2_ranked = r2_values[ordered]
    valid = np.isfinite(tau_ranked)
    if np.sum(valid) < 3:
        ax.set_axis_off()
        return
    ranks_valid = ranks[valid]
    tau_valid = tau_ranked[valid]
    r2_valid = r2_ranked[valid]
    fit = linregress(ranks_valid, tau_valid)
    ax.scatter(ranks_valid, tau_valid, s=9, alpha=0.45, color=color, edgecolors='none')
    ax.plot(ranks_valid, fit.intercept + fit.slope * ranks_valid, color='black', linewidth=1.8)
    ax.set_xlim([1, len(ranks)])
    ax.set_xlabel('Neuron rank (high to low response)')
    ax.set_ylabel(r'Fitted recovery $\\tau$ (s)')
    ax.set_title(title)
    ax.set_xticks([1, len(ranks)])
    ax.set_xticklabels(['High', 'Low'])
    ax.text(
        0.04,
        0.96,
        f'r = {fit.rvalue:.2f}\\np = {fit.pvalue:.1e}\\nmedian $R^2$ = {np.nanmedian(r2_valid):.2f}\\nn = {len(ranks_valid)}',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5),
    )


'''
if old_block not in cell7:
    raise SystemExit('old tau block not found')
cell7 = cell7.replace(old_block, new_block)

anchor = "def plot_response_tiers(ax, rates, resp_values, class_mask, stronger_is_higher, palette, title):\n"
insertion = "fitted_recovery_tau, fitted_recovery_r2 = fit_population_recovery_taus(ref_base['rates'], t_ref, ref_base['classes'])\n\n\n" + anchor
if insertion.split(anchor)[0] not in cell7:
    if anchor not in cell7:
        raise SystemExit('plot_response_tiers anchor not found')
    cell7 = cell7.replace(anchor, insertion)

cell7 = cell7.replace(
    "    ref_pop['tau_intr'],\n    resp,\n",
    "    fitted_recovery_tau,\n    fitted_recovery_r2,\n    resp,\n",
    1,
)
cell7 = cell7.replace(
    "    'PyrUp intrinsic tau by response rank',\n",
    "    'PyrUp fitted recovery tau by response rank',\n",
    1,
)
cell7 = cell7.replace(
    "    ref_pop['tau_intr'],\n    resp,\n",
    "    fitted_recovery_tau,\n    fitted_recovery_r2,\n    resp,\n",
    1,
)
cell7 = cell7.replace(
    "    'PyrDown intrinsic tau by response rank',\n",
    "    'PyrDown fitted recovery tau by response rank',\n",
    1,
)

nb['cells'][7]['source'] = cell7.splitlines(keepends=True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
