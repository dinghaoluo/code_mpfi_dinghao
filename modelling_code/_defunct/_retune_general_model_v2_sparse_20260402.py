import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress


NB_PATH = Path("modelling_code/general_model_v2.ipynb")


def load_namespace():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    ns = {}
    for idx in [1, 2, 3, 4]:
        exec("".join(nb["cells"][idx]["source"]), ns)
    return ns


NS = load_namespace()
PARAMS = NS["PARAMS"]
window_mask = NS["window_mask"]
run_bootstrap_suite = NS["run_bootstrap_suite"]
make_population = NS["make_population"]
simulate_population_condition = NS["simulate_population_condition"]
match_paired_pre_baseline = NS["match_paired_pre_baseline"]


def split_response_tiers(resp_values, class_mask, eps, stronger_is_higher=True):
    idx = np.flatnonzero(class_mask)
    if idx.size == 0:
        return []
    if stronger_is_higher:
        strength = resp_values[idx]
    else:
        strength = 1.0 / np.maximum(resp_values[idx], eps)
    ordered = idx[np.argsort(strength)]
    groups = np.array_split(ordered, 3)
    return [
        ("Low", groups[0]),
        ("Mid", groups[1]),
        ("High", groups[2]),
    ]


def fit_cell_recovery_tau(trace, t, baseline, is_up, eps, tau_bounds=(0.05, 6.0)):
    post_mask = (t >= 0.0) & (t <= 4.0)
    t_post = t[post_mask]
    y_post = np.asarray(trace[post_mask], dtype=float)
    if y_post.size < 8:
        return np.nan

    extremum_idx = int(np.argmax(y_post) if is_up else np.argmin(y_post))
    t_fit = t_post[extremum_idx:]
    y_fit = y_post[extremum_idx:]
    if y_fit.size < 8:
        return np.nan

    amp0 = float(y_fit[0] - baseline)
    if (is_up and amp0 <= eps) or ((not is_up) and amp0 >= -eps):
        return np.nan

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
        return np.nan
    return float(popt[1])


def ramp_penalty(x, low=None, high=None, target=None, scale=1.0):
    if not np.isfinite(x):
        return 50.0
    if target is not None:
        return ((x - target) / scale) ** 2
    if low is not None and x < low:
        return ((low - x) / scale) ** 2
    if high is not None and x > high:
        return ((x - high) / scale) ** 2
    return 0.0


def preview_metrics(p):
    t = np.arange(-p.t_pre, p.t_post, p.dt)
    rng = np.random.default_rng(p.seed_start)
    pop = make_population(p, rng)
    base = simulate_population_condition(t, p, pop, da_scale=1.0)
    resp = base["resp"]
    classes = base["classes"]
    rates = base["rates"]
    pre_mask = window_mask(t, p.pre_window)
    post_mask = (t >= 0.0) & (t <= 4.0)

    out = {}

    up_baselines = []
    up_peaks = []
    up_peak_times = []
    up_end_offsets = []
    for _, idx in split_response_tiers(resp, classes["is_up"], p.eps, stronger_is_higher=True):
        if len(idx) == 0:
            up_baselines.append(np.nan)
            up_peaks.append(np.nan)
            up_peak_times.append(np.nan)
            up_end_offsets.append(np.nan)
            continue
        mean_trace = np.mean(rates[idx], axis=0)
        baseline = float(np.mean(mean_trace[pre_mask]))
        post_trace = mean_trace[post_mask]
        post_t = t[post_mask]
        peak_idx = int(np.argmax(post_trace))
        peak = float(post_trace[peak_idx])
        peak_t = float(post_t[peak_idx])
        end_offset = float(np.interp(4.0, t, mean_trace) - baseline)
        up_baselines.append(baseline)
        up_peaks.append(peak)
        up_peak_times.append(peak_t)
        up_end_offsets.append(end_offset)

    down_baselines = []
    for _, idx in split_response_tiers(resp, classes["is_down"], p.eps, stronger_is_higher=False):
        if len(idx) == 0:
            down_baselines.append(np.nan)
            continue
        mean_trace = np.mean(rates[idx], axis=0)
        down_baselines.append(float(np.mean(mean_trace[pre_mask])))

    baselines = np.mean(rates[:, pre_mask], axis=1)
    up_idx = np.flatnonzero(classes["is_up"])
    tau_fit = np.full(len(up_idx), np.nan)
    ordered_up = up_idx[np.argsort(resp[up_idx])[::-1]]
    for j, idx in enumerate(ordered_up):
        tau_fit[j] = fit_cell_recovery_tau(rates[idx], t, baselines[idx], True, p.eps)
    valid_tau = np.isfinite(tau_fit)
    if np.sum(valid_tau) >= 3:
        ranks = np.arange(1, len(ordered_up) + 1)[valid_tau]
        fit = linregress(ranks, tau_fit[valid_tau])
        tau_r = float(fit.rvalue)
        tau_p = float(fit.pvalue)
        tau_slope = float(fit.slope)
    else:
        tau_r = np.nan
        tau_p = np.nan
        tau_slope = np.nan

    out["up_baselines"] = up_baselines
    out["up_peaks"] = up_peaks
    out["up_peak_times"] = up_peak_times
    out["up_end_offsets"] = up_end_offsets
    out["down_baselines"] = down_baselines
    out["tau_r"] = tau_r
    out["tau_p"] = tau_p
    out["tau_slope"] = tau_slope
    return out


def experiment_metrics(p):
    p_eval = deepcopy(p)
    p_eval.n_bootstrap = 8
    p_eval.n_cells = 800
    results = run_bootstrap_suite(p_eval)
    t = results["t"]
    early_mask = (t >= 0.0) & (t < 0.8)
    late_mask = (t >= 1.0) & (t <= 4.0)

    da_up_plot, non_da_up_plot = match_paired_pre_baseline(
        results["da_up_traces"], results["non_da_up_traces"], t, p_eval.pre_window
    )
    da_down_plot, non_da_down_plot = match_paired_pre_baseline(
        results["da_down_traces"], results["non_da_down_traces"], t, p_eval.pre_window
    )

    metrics = {
        "base_up_pct": float(np.nanmean(results["stats"]["base_up_pct"])),
        "base_down_pct": float(np.nanmean(results["stats"]["base_down_pct"])),
        "exp1_up_early": float(np.nanmean(np.nanmean(results["lc_up_traces"][:, early_mask] - results["base_up_traces"][:, early_mask], axis=1))),
        "exp1_up_late": float(np.nanmean(np.nanmean(results["lc_up_traces"][:, late_mask] - results["base_up_traces"][:, late_mask], axis=1))),
        "exp1_down_late": float(np.nanmean(np.nanmean(results["lc_down_traces"][:, late_mask] - results["base_down_traces"][:, late_mask], axis=1))),
        "exp1_up_prop": float(np.nanmean(results["stats"]["lc_up_pct"] - results["stats"]["base_up_pct"])),
        "exp1_down_prop": float(np.nanmean(results["stats"]["lc_down_pct"] - results["stats"]["base_down_pct"])),
        "exp2_up_early": float(np.nanmean(np.nanmean(da_up_plot[:, early_mask] - non_da_up_plot[:, early_mask], axis=1))),
        "exp2_up_late": float(np.nanmean(np.nanmean(da_up_plot[:, late_mask] - non_da_up_plot[:, late_mask], axis=1))),
        "exp2_down_early": float(np.nanmean(np.nanmean(da_down_plot[:, early_mask] - non_da_down_plot[:, early_mask], axis=1))),
        "exp2_down_late": float(np.nanmean(np.nanmean(da_down_plot[:, late_mask] - non_da_down_plot[:, late_mask], axis=1))),
        "exp2_up_prop": float(np.nanmean(results["stats"]["p_up_da_targeted"] - results["stats"]["p_up_not_targeted"])),
        "exp2_down_prop": float(np.nanmean(results["stats"]["p_down_da_targeted"] - results["stats"]["p_down_not_targeted"])),
        "exp3_up_early": float(np.nanmean(np.nanmean(results["block_up_traces"][:, early_mask] - results["base_up_traces"][:, early_mask], axis=1))),
        "exp3_up_late": float(np.nanmean(np.nanmean(results["block_up_traces"][:, late_mask] - results["base_up_traces"][:, late_mask], axis=1))),
        "exp3_down_late": float(np.nanmean(np.nanmean(results["block_down_traces"][:, late_mask] - results["base_down_traces"][:, late_mask], axis=1))),
        "exp3_up_prop": float(np.nanmean(results["stats"]["block_up_pct"] - results["stats"]["base_up_pct"])),
        "exp3_down_prop": float(np.nanmean(results["stats"]["block_down_pct"] - results["stats"]["base_down_pct"])),
    }
    return metrics


def total_score(p):
    prev = preview_metrics(p)
    exp = experiment_metrics(p)

    score = 0.0

    up_baseline_targets = [1.50, 1.00, 0.50]
    for value, target in zip(prev["up_baselines"], up_baseline_targets):
        score += 7.0 * ramp_penalty(value, target=target, scale=0.22)

    for value in prev["down_baselines"]:
        score += 1.5 * ramp_penalty(value, target=2.0, scale=0.55)

    up_peaks = np.asarray(prev["up_peaks"], dtype=float)
    up_peak_times = np.asarray(prev["up_peak_times"], dtype=float)
    up_end_offsets = np.asarray(prev["up_end_offsets"], dtype=float)
    if np.sum(np.isfinite(up_peaks)) >= 2:
        score += 9.0 * ramp_penalty(float(np.nanmax(up_peaks) - np.nanmin(up_peaks)), high=0.18, scale=0.08)
    score += 10.0 * ramp_penalty(float(np.nanmean(up_peak_times)), target=1.02, scale=0.10)
    score += 4.0 * ramp_penalty(float(np.nanstd(up_peak_times)), high=0.10, scale=0.05)
    score += 5.0 * np.nanmean((up_end_offsets / 0.18) ** 2)

    score += 10.0 * ramp_penalty(prev["tau_r"], low=0.18, scale=0.10)
    score += 4.0 * ramp_penalty(prev["tau_p"], high=0.10, scale=0.08)
    score += 2.0 * ramp_penalty(prev["tau_slope"], low=0.0, scale=0.002)

    score += 6.0 * ramp_penalty(exp["base_up_pct"], target=28.0, scale=3.5)
    score += 5.0 * ramp_penalty(exp["base_down_pct"], target=10.5, scale=1.5)

    score += 12.0 * ramp_penalty(abs(exp["exp1_up_early"]), high=0.04, scale=0.02)
    score += 10.0 * ramp_penalty(exp["exp1_up_late"], low=0.18, scale=0.06)
    score += 4.0 * ramp_penalty(exp["exp1_up_late"], high=0.35, scale=0.08)
    score += 10.0 * ramp_penalty(abs(exp["exp1_down_late"]), high=0.03, scale=0.02)
    score += 6.0 * ramp_penalty(exp["exp1_up_prop"], low=1.5, scale=0.8)
    score += 6.0 * ramp_penalty(exp["exp1_down_prop"], high=-0.4, scale=0.5)

    score += 8.0 * ramp_penalty(abs(exp["exp2_up_early"]), high=0.04, scale=0.02)
    score += 5.0 * ramp_penalty(exp["exp2_up_late"], low=0.08, scale=0.05)
    score += 5.0 * ramp_penalty(exp["exp2_up_late"], high=0.24, scale=0.05)
    score += 10.0 * ramp_penalty(abs(exp["exp2_down_early"]), high=0.03, scale=0.02)
    score += 10.0 * ramp_penalty(abs(exp["exp2_down_late"]), high=0.03, scale=0.02)
    score += 6.0 * ramp_penalty(exp["exp2_up_prop"], low=3.5, scale=1.0)
    score += 6.0 * ramp_penalty(exp["exp2_down_prop"], high=-0.6, scale=0.5)

    score += 12.0 * ramp_penalty(abs(exp["exp3_up_early"]), high=0.04, scale=0.02)
    score += 8.0 * ramp_penalty(exp["exp3_up_late"], high=-0.05, scale=0.03)
    score += 8.0 * ramp_penalty(abs(exp["exp3_down_late"]), high=0.03, scale=0.02)
    score += 6.0 * ramp_penalty(exp["exp3_up_prop"], high=-0.5, scale=0.5)
    score += 6.0 * ramp_penalty(exp["exp3_down_prop"], low=0.2, scale=0.35)

    return score, prev, exp


def set_param(p, name, value):
    q = deepcopy(p)
    setattr(q, name, value)
    return q


def summarize(prev, exp):
    return {
        "up_baselines": [round(v, 3) for v in prev["up_baselines"]],
        "up_peaks": [round(v, 3) for v in prev["up_peaks"]],
        "up_peak_times": [round(v, 3) for v in prev["up_peak_times"]],
        "tau_r": round(prev["tau_r"], 3) if np.isfinite(prev["tau_r"]) else None,
        "tau_p": round(prev["tau_p"], 4) if np.isfinite(prev["tau_p"]) else None,
        "base_up_pct": round(exp["base_up_pct"], 2),
        "base_down_pct": round(exp["base_down_pct"], 2),
        "exp1_up_early": round(exp["exp1_up_early"], 3),
        "exp1_up_late": round(exp["exp1_up_late"], 3),
        "exp1_down_late": round(exp["exp1_down_late"], 3),
        "exp2_up_early": round(exp["exp2_up_early"], 3),
        "exp2_up_late": round(exp["exp2_up_late"], 3),
        "exp2_down_late": round(exp["exp2_down_late"], 3),
        "exp3_up_early": round(exp["exp3_up_early"], 3),
        "exp3_up_late": round(exp["exp3_up_late"], 3),
        "exp3_down_late": round(exp["exp3_down_late"], 3),
        "exp1_up_prop": round(exp["exp1_up_prop"], 2),
        "exp1_down_prop": round(exp["exp1_down_prop"], 2),
        "exp2_up_prop": round(exp["exp2_up_prop"], 2),
        "exp2_down_prop": round(exp["exp2_down_prop"], 2),
        "exp3_up_prop": round(exp["exp3_up_prop"], 2),
        "exp3_down_prop": round(exp["exp3_down_prop"], 2),
    }


def main():
    p = PARAMS()

    grid = {
        "baseline_mean": [1.10, 1.20, 1.30],
        "baseline_sd": [0.75, 0.90, 1.05],
        "wR_mean": [0.50, 0.54, 0.58],
        "wR_sd": [0.08, 0.12, 0.16],
        "wW_mean": [0.40, 0.46, 0.52],
        "intrinsic_tau_mean": [0.28, 0.34, 0.40],
        "baseline_tau_coupling": [0.02, 0.05, 0.08],
        "run_on_mid": [0.00, 0.04, 0.08],
        "run_off_mid": [1.55, 1.70, 1.85],
        "run_fall_scale": [0.28, 0.38, 0.48],
        "reward_on_mid": [0.60, 0.68, 0.76],
        "lc_to_da_gain": [4.8, 5.4, 6.0],
        "da_kernel_tau": [2.2, 2.5, 2.8],
        "da_ca1_delay": [0.75, 0.88, 1.00],
        "da_half_rate": [3.25, 3.75, 4.25],
        "da_rate_slope": [0.25, 0.35, 0.45],
        "wDA_global": [0.02, 0.04, 0.06],
        "frac_da_targ": [0.20, 0.25, 0.30],
        "da_targ_wR_bias": [0.02, 0.06, 0.10],
        "da_block_scale": [0.10, 0.15, 0.20],
    }

    score, prev, exp = total_score(p)
    print("initial", round(score, 3), summarize(prev, exp))

    for sweep in range(2):
        print(f"\nSWEEP {sweep + 1}")
        improved = False
        for name, values in grid.items():
            best_local = None
            for value in values:
                q = set_param(p, name, value)
                trial_score, trial_prev, trial_exp = total_score(q)
                payload = (trial_score, value, trial_prev, trial_exp)
                if best_local is None or trial_score < best_local[0]:
                    best_local = payload
            if best_local[0] + 1e-9 < score:
                score, best_value, prev, exp = best_local
                setattr(p, name, best_value)
                improved = True
                print(name, "->", best_value, "score", round(score, 3))
                print(summarize(prev, exp))
            else:
                print(name, "kept", getattr(p, name), "best_test", best_local[1], "score", round(best_local[0], 3))
        if not improved:
            break

    final = {
        "params": {name: getattr(p, name) for name in grid},
        "score": score,
        "preview": summarize(prev, exp),
    }
    out_path = Path("modelling_code/_defunct/_retune_general_model_v2_sparse_20260402_result.json")
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\nFINAL")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
