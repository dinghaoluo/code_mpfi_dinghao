# -*- coding: utf-8 -*-
"""
ca1 pyramidal population model with direct lc→pyramidal modulation and motivation coupling (v6).

this version matches the manuscript constraints and your plots:
- lc tonic ≈ 1 hz; onset burst is a centred gaussian (diameter 1 s, ±0.5 s), peak maps with time‑since‑reward r into 5–10 hz.
- ca1 pyramidal baselines are heterogeneous (gaussian around 3 hz; per‑cell baselines, not a fixed value).
- both up and down cells show an **anticipatory ramp** in the **−0.6..0 s** window, then a **post‑onset transient** that **peaks (up) or troughs (down) around ~1 s** and **returns to each cell’s own baseline by ~3 s with exponential decay**.
- only a subset of up cells are da‑modulated (axon‑proximal). da multiplies **only the post‑onset transient** of that subset.
- all multiplicative factors are bounded; firing rates are clipped to physiological caps.
"""

#%% imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrow


# ----------------------------
# helpers and kernels
# ----------------------------

def gaussian_centered(samp_freq: int, width_s: float = 1.0, eps: float = 0.01) -> np.ndarray:
    """
    centred gaussian with unit peak and diameter ≈ width_s.

    parameters:
    - samp_freq: sampling frequency (hz)
    - width_s: diameter (s) from −width_s/2 to +width_s/2
    - eps: relative edge value at ±width_s/2

    returns:
    - k: kernel with peak 1 at centre
    """
    n = int(round(width_s * samp_freq)) | 1  # odd length
    t = np.linspace(-width_s/2, +width_s/2, n)
    sigma = (width_s/2) / np.sqrt(2.0 * np.log(1.0/eps))
    k = np.exp(-0.5 * (t/sigma) ** 2)
    return k / k.max()


def exp_unit_area(samp_freq: int, tau_s: float = 2.8, dur_s: float = 6.0) -> np.ndarray:
    """
    causal exponential with unit area.

    parameters:
    - samp_freq: sampling frequency (hz)
    - tau_s: decay constant (s)
    - dur_s: duration (s)

    returns:
    - k: unit‑area kernel (t≥0)
    """
    n = int(round(dur_s * samp_freq)) + 1
    t = np.arange(n) / samp_freq
    k = np.exp(-t/tau_s)
    k /= np.trapz(k, t)
    return k


def embed_centered(k: np.ndarray, total_len: int, center_idx: int) -> np.ndarray:
    """
    embed centred kernel so its midpoint aligns to center_idx.

    parameters:
    - k: centred kernel (odd length)
    - total_len: output length
    - center_idx: alignment index

    returns:
    - v: embedded vector
    """
    v = np.zeros(total_len, float)
    half = len(k)//2
    s0 = max(0, center_idx-half)
    e0 = min(total_len, center_idx+half+1)
    ks = half - (center_idx - s0)
    ke = ks + (e0 - s0)
    v[s0:e0] = k[ks:ke]
    return v


def conv_causal_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    causal convolution with same length output.

    parameters:
    - x: input vector
    - k: causal kernel (t≥0)

    returns:
    - y: result cropped to len(x)
    """
    y = np.convolve(x, k, mode='full')
    return y[:len(x)]


def shift_zeropad(sig: np.ndarray, shift: int) -> np.ndarray:
    """
    zero‑padded shift: positive delays, negative advances. no wrap‑around.

    parameters:
    - sig: 1d array
    - shift: integer samples

    returns:
    - out: shifted array
    """
    out = np.zeros_like(sig)
    n = len(sig)
    if shift >= 0:
        ncopy = max(0, n-shift)
        if ncopy > 0:
            out[shift:shift+ncopy] = sig[:ncopy]
    else:
        s = -shift
        ncopy = max(0, n-s)
        if ncopy > 0:
            out[:ncopy] = sig[s:s+ncopy]
    return out


# ----------------------------
# parameters
# ----------------------------

@dataclass
class Params:
    """
    model parameters.

    parameters:
    - samp_freq: sampling frequency (hz)
    - pre_s/post_s: buffer around onset (s)
    - n_pyr: number of pyramidal cells
    - frac_up: fraction of up cells
    - frac_up_da: fraction of up cells that are da‑modulated
    - base_mu/base_sd: gaussian baseline for pyramidal cells (hz)
    - lc_*: lc firing characteristics
    - r_min/r_max: mapping range for time‑since‑reward r (s)
    - pre_ramp_dur: duration of anticipatory ramp (s)
    - pre_ramp_mu/sd: mean/sd ramp height at t=0 (hz)
    - up_amp_mu/sd: up transient amplitude at unit peak (hz)
    - down_amp_mu/sd: down transient magnitude at unit trough (hz)
    - up_tau_rise / up_tau_decay: up transient shape (s)
    - down_tau_rise / down_tau_recover: down transient shape (s)
    - peak_time_s: target time of extrema after onset (s) used for jittering
    - jitter_up_sd / jitter_down_sd: timing jitter (s)
    - da_gain: global da gain (dimensionless)
    - noise_sd: additive gaussian noise (hz)
    - max_fr_hz: hard cap on pyramidal rate (hz)
    """

    samp_freq: int = 1250
    pre_s: float = 1.0
    post_s: float = 7.0

    n_pyr: int = 500
    frac_up: float = 0.5
    frac_up_da: float = 0.35

    base_mu: float = 3.0
    base_sd: float = 0.6

    lc_baseline_hz: float = 1.0
    lc_peak_min_hz: float = 5.0
    lc_peak_max_hz: float = 10.0
    r_min: float = 0.0
    r_max: float = 5.0

    kL_width_s: float = 1.0
    kL_eps: float = 0.01
    kD_tau_s: float = 2.8
    kD_dur_s: float = 6.0

    pre_ramp_dur: float = 0.6
    pre_ramp_mu: float = 0.35
    pre_ramp_sd: float = 0.10

    up_amp_mu: float = 1.2
    up_amp_sd: float = 0.4
    down_amp_mu: float = 1.0
    down_amp_sd: float = 0.35

    up_tau_rise: float = 0.35
    up_tau_decay: float = 2.0
    down_tau_rise: float = 0.35
    down_tau_recover: float = 2.0

    peak_time_s: float = 1.1

    jitter_up_sd: float = 0.10
    jitter_down_sd: float = 0.12

    da_gain: float = 0.6
    noise_sd: float = 0.05
    max_fr_hz: float = 15.0

    def time_axis(self) -> Tuple[np.ndarray, int]:
        """
        build time axis and onset index.

        parameters:
        - none

        returns:
        - t: time axis (s)
        - onset_idx: index of run onset
        """
        n_pre = int(round(self.pre_s * self.samp_freq))
        n_post = int(round(self.post_s * self.samp_freq))
        n = n_pre + n_post
        t = (np.arange(n) - n_pre) / self.samp_freq
        return t, n_pre


# ----------------------------
# anticipatory ramp and post‑onset transients
# ----------------------------

def anticipatory_ramp(t: np.ndarray, onset_idx: int, dur_s: float, height_hz: float) -> np.ndarray:
    """
    linear ramp in [onset-dur_s, onset] reaching 'height_hz' at t=0; 0 elsewhere.

    parameters:
    - t: time axis (s)
    - onset_idx: index of run onset
    - dur_s: ramp duration (s)
    - height_hz: ramp height at onset (hz)

    returns:
    - r: ramp vector (hz)
    """
    r = np.zeros_like(t)
    t0 = t[onset_idx] - dur_s
    pre_mask = (t >= t0) & (t <= t[onset_idx])
    r[pre_mask] = height_hz * (t[pre_mask] - t0) / max(1e-9, dur_s)
    return r


def alpha_rise_exp_decay(t: np.ndarray, onset_idx: int, tau_rise: float, tau_decay: float) -> np.ndarray:
    """
    post‑onset transient with smooth rise then exponential decay (unit peak).

    parameters:
    - t: time axis (s)
    - onset_idx: onset index
    - tau_rise: rise time constant (s)
    - tau_decay: decay time constant (s)

    returns:
    - k: unit‑peak kernel for t≥0
    """
    dt = t - t[onset_idx]
    k = (1.0 - np.exp(-np.clip(dt, 0, None)/max(1e-9, tau_rise))) * np.exp(-np.clip(dt, 0, None)/max(1e-9, tau_decay))
    k /= max(1e-9, k.max())
    return k


# ----------------------------
# population construction
# ----------------------------

def make_population(p: Params, seed: int = 1) -> Dict[str, Any]:
    """
    construct per‑cell attributes and masks.

    parameters:
    - p: params
    - seed: rng seed

    returns:
    - pop: population dict
    """
    rng = np.random.default_rng(seed)
    t, onset_idx = p.time_axis()

    is_up = rng.random(p.n_pyr) < p.frac_up
    up_idx = np.where(is_up)[0]
    is_up_da = np.zeros(p.n_pyr, bool)
    if len(up_idx) > 0:
        n_da = int(round(p.frac_up_da * len(up_idx)))
        if n_da > 0:
            is_up_da[rng.choice(up_idx, n_da, replace=False)] = True

    baseline = rng.normal(p.base_mu, p.base_sd, size=p.n_pyr)

    pre_ramp_height = rng.normal(p.pre_ramp_mu, p.pre_ramp_sd, size=p.n_pyr).clip(0.0, None)
    up_amp = rng.normal(p.up_amp_mu, p.up_amp_sd, size=p.n_pyr).clip(0.2, None)
    down_amp = rng.normal(p.down_amp_mu, p.down_amp_sd, size=p.n_pyr).clip(0.2, None)

    # jitter extrema timing by shifting the post‑onset kernel
    jitter_up = (rng.normal(0.0, p.jitter_up_sd, size=p.n_pyr) * p.samp_freq).astype(int)
    jitter_down = (rng.normal(0.0, p.jitter_down_sd, size=p.n_pyr) * p.samp_freq).astype(int)

    kup = alpha_rise_exp_decay(t, onset_idx, p.up_tau_rise, p.up_tau_decay)
    kdown_post = alpha_rise_exp_decay(t, onset_idx, p.down_tau_rise, p.down_tau_recover)

    # anticipatory ramp (same form for all cells; amplitude varies per cell)
    ramp_unit = anticipatory_ramp(t, onset_idx, p.pre_ramp_dur, 1.0)  # unit height; scale per cell later

    return dict(
        t=t, onset_idx=onset_idx,
        is_up=is_up, is_up_da=is_up_da,
        baseline=baseline,
        pre_ramp_height=pre_ramp_height,
        up_amp=up_amp, down_amp=down_amp,
        jitter_up=jitter_up, jitter_down=jitter_down,
        kup=kup, kdown_post=kdown_post, ramp_unit=ramp_unit
    )


# ----------------------------
# lc and da
# ----------------------------

def lc_and_da(p: Params, length: int, onset_idx: int, r_seconds: float, inhibit: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    build lc (hz), raw da (a.u.) and normalised da (0..1) from r.

    parameters:
    - p: params
    - length: series length
    - onset_idx: onset index
    - r_seconds: time since last reward (s)
    - inhibit: onset suppression fraction (0..1)

    returns:
    - L: lc in hz
    - D: raw da a.u.
    - Dnorm: da normalised by 99th percentile
    """
    kL = gaussian_centered(p.samp_freq, p.kL_width_s, p.kL_eps)
    kD = exp_unit_area(p.samp_freq, p.kD_tau_s, p.kD_dur_s)

    r = np.clip(r_seconds, p.r_min, p.r_max)
    frac = (r - p.r_min) / max(1e-9, (p.r_max - p.r_min))
    peak = p.lc_peak_min_hz + frac * (p.lc_peak_max_hz - p.lc_peak_min_hz)
    amp = max(0.0, peak - p.lc_baseline_hz) * (1.0 - inhibit)

    L = np.full(length, p.lc_baseline_hz, float)
    L += amp * embed_centered(kL, length, onset_idx)

    burst = np.clip(L - p.lc_baseline_hz, 0.0, None)
    D = conv_causal_same(burst, kD)
    Dnorm = D / (np.percentile(D, 99) + 1e-9)
    return L, D, Dnorm


# ----------------------------
# simulate
# ----------------------------

def simulate_trial(p: Params, pop: Dict[str, Any], r_seconds: float, inhibit: float = 0.0, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    simulate one trial with anticipatory ramps and post‑onset transients.

    parameters:
    - p: params
    - pop: population dict
    - r_seconds: time since reward (s)
    - inhibit: lc onset suppression fraction (0..1)
    - seed: rng seed

    returns:
    - out: dict with t, L, D, rates, up/down means, masks
    """
    rng = np.random.default_rng(seed)

    t = pop['t']; onset_idx = pop['onset_idx']
    L, D, Dnorm = lc_and_da(p, len(t), onset_idx, r_seconds, inhibit)

    is_up = pop['is_up']; is_up_da = pop['is_up_da']
    baseline = pop['baseline']
    pre_height = pop['pre_ramp_height']
    up_amp = pop['up_amp']; down_amp = pop['down_amp']
    jitter_up = pop['jitter_up']; jitter_down = pop['jitter_down']
    kup = pop['kup']; kdown_post = pop['kdown_post']
    ramp_unit = pop['ramp_unit']

    n = len(baseline); nt = len(t)
    rates = np.zeros((n, nt), float)

    # precompute shifted post‑onset kernels
    kup_cells = np.vstack([shift_zeropad(kup, s) for s in jitter_up])
    kdown_cells = np.vstack([shift_zeropad(kdown_post, s) for s in jitter_down])

    # build per‑cell timecourses
    for i in range(n):
        ramp = pre_height[i] * ramp_unit  # pre‑onset ramp only
        if is_up[i]:
            post = up_amp[i] * kup_cells[i]
            # da multiplies post‑onset transient of the up‑DA subset only
            mod = 1.0 + (p.da_gain * (1.0 if is_up_da[i] else 0.0) * Dnorm)
            post *= mod
            r = baseline[i] + ramp + post
        else:
            post = -down_amp[i] * kdown_cells[i]  # negative transient, exponential recovery
            r = baseline[i] + ramp + post

        r += rng.normal(0.0, p.noise_sd, size=nt)
        rates[i] = np.clip(r, 0.0, p.max_fr_hz)

    up_mask = is_up
    down_mask = ~is_up
    mean_up = rates[up_mask].mean(axis=0) if up_mask.any() else np.zeros(nt)
    mean_down = rates[down_mask].mean(axis=0) if down_mask.any() else np.zeros(nt)

    return dict(t=t, onset_idx=onset_idx, L=L, D=D, rates=rates,
                mean_up=mean_up, mean_down=mean_down,
                masks=dict(is_up=is_up, is_up_da=is_up_da))


def run_conditions(p: Params, r_values: List[float], inhibit: float = 0.0, seed: int = 11) -> Dict[str, Any]:
    """
    simulate one population across several r conditions.

    parameters:
    - p: params
    - r_values: list of r in seconds
    - inhibit: lc onset suppression (0..1)

    returns:
    - out: dict with population and trials
    """
    pop = make_population(p, seed=seed)
    trials = [simulate_trial(p, pop, r, inhibit=inhibit, seed=seed+i) for i, r in enumerate(r_values)]
    return dict(params=p, population=pop, trials=trials)


# ----------------------------
# plotting
# ----------------------------

def plot_up_down(run: Dict[str, Any]) -> None:
    """
    plot lc (hz), da (a.u.), and population means for down (purple) and up (red).

    parameters:
    - run: output of run_conditions

    returns:
    - none
    """
    t = run['trials'][0]['t']
    ups = np.stack([o['mean_up'] for o in run['trials']]); mu_u = ups.mean(axis=0); se_u = ups.std(axis=0, ddof=1)/np.sqrt(len(run['trials']))
    dns = np.stack([o['mean_down'] for o in run['trials']]); mu_d = dns.mean(axis=0); se_d = dns.std(axis=0, ddof=1)/np.sqrt(len(run['trials']))
    L = run['trials'][0]['L']; D = run['trials'][0]['D']

    fig, axs = plt.subplots(3, 1, figsize=(5.4, 5.2), sharex=True)
    axs[0].plot(t, L, lw=2, color='k'); axs[0].set_ylabel('lc (hz)')
    axs[1].plot(t, D, lw=2, color='tab:green'); axs[1].set_ylabel('da (a.u.)')
    axs[2].plot(t, mu_d, lw=2.2, color='#79237a'); axs[2].fill_between(t, mu_d-se_d, mu_d+se_d, color='#79237a', alpha=0.25)
    axs[2].plot(t, mu_u, lw=2.2, color='#9c2f2f'); axs[2].fill_between(t, mu_u-se_u, mu_u+se_u, color='#9c2f2f', alpha=0.25)
    axs[2].set_xlabel('time from run onset (s)'); axs[2].set_ylabel('spike rate (hz)')
    for ax in axs: ax.axvline(0, color='k', lw=1, alpha=0.5); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); plt.show()


def draw_schematic(pop: Dict[str, Any]) -> None:
    """
    publication‑style schematic: circles for ca1 pyramids; up (red), down (purple), up‑DA outlined.

    parameters:
    - pop: population dict

    returns:
    - none
    """
    is_up = pop['is_up']; is_up_da = pop['is_up_da']
    n = len(is_up); cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n/cols))
    up_c, down_c, da_edge = '#9c2f2f', '#79237a', '#1e90ff'

    fig, ax = plt.subplots(figsize=(7.0, 3.8)); ax.set_axis_off()

    # lc node
    lc = Circle((0.08, 0.76), 0.05, transform=ax.transAxes, fc='black', ec='none'); ax.add_patch(lc)
    ax.text(0.08, 0.76, 'lc', color='w', ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(0.02, 0.92, 'baseline ≈ 1 hz burst peak 5–10 hz', transform=ax.transAxes, fontsize=8)

    # ca1 box
    box = FancyBboxPatch((0.22, 0.10), 0.74, 0.80, boxstyle='round,pad=0.015,rounding_size=0.02', fc='#fffaf0', ec='#8b4513', lw=1.2, transform=ax.transAxes); ax.add_patch(box)
    ax.text(0.59, 0.91, 'ca1 pyramidal cells (baselines ~ N(3 hz, 0.6^2))', transform=ax.transAxes, fontsize=10, color='#8b4513', ha='center')

    # lc→ca1 arrow
    ax.add_patch(FancyArrow(0.13, 0.74, 0.07, -0.18, width=0.005, length_includes_head=True, transform=ax.transAxes))

    # grid of cells
    gx0, gy0, gxw, gyh = 0.24, 0.14, 0.70, 0.72
    xs = np.linspace(gx0, gx0+gxw, cols, endpoint=False) + (gxw/cols)*0.5
    ys = np.linspace(gy0, gy0+gyh, rows, endpoint=False) + (gyh/rows)*0.5

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n: break
            x, y = xs[c], ys[rows-1-r]
            fc = up_c if is_up[idx] else down_c
            ec = da_edge if is_up_da[idx] else 'none'
            ax.add_patch(Circle((x, y), 0.012, transform=ax.transAxes, fc=fc, ec=ec, lw=1.2))
            idx += 1

    # legend
    ax.text(0.25, 0.16, 'up', color=up_c, transform=ax.transAxes, fontsize=9)
    ax.text(0.30, 0.16, 'down', color=down_c, transform=ax.transAxes, fontsize=9)
    ax.plot([0.36], [0.16], marker='o', ms=6, mfc=up_c, mec=da_edge, mew=1.2, transform=ax.transAxes)
    ax.text(0.38, 0.16, 'up (da‑modulated subset)', color='k', transform=ax.transAxes, fontsize=9)

    plt.tight_layout(); plt.show()


# ----------------------------
# demo
# ----------------------------

def demo() -> None:
    """
    run base and inhibition conditions; plot and draw schematic.

    parameters:
    - none

    returns:
    - none
    """
    p = Params()
    pop = make_population(p, seed=13)
    run = run_conditions(p, r_values=[0.5, 2.0, 3.5, 5.0], inhibit=0.0, seed=13)
    plot_up_down(run)
    run_inhib = run_conditions(p, r_values=[0.5, 2.0, 3.5, 5.0], inhibit=0.5, seed=13)
    plot_up_down(run_inhib)
    draw_schematic(pop)


if __name__ == '__main__':
    demo()
