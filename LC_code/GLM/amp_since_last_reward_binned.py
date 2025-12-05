# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 12:03:43 2025
Modified on Friday to get reward-aligned firing profiles 

Testing different time-since-last-reward bin widths (0.1–1.0 s)

@author: Dinghao Luo
"""


#%% imports
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import sem, wilcoxon
from matplotlib.cm import ScalarMappable

import GLM_functions as gf
from common import smooth_convolve, mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathLC


#%% parameters 
all_sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
LC_beh_stem   = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')
GLM_stem      = Path('Z:/Dinghao/code_dinghao/LC_ephys/GLM/tests')

SAMP_FREQ     = 1250
SAMP_FREQ_BEH = 1000

PRE_REW_S = 1.0            # reward-aligned window start
BASELINE_OFFSET = 0.5      # baseline = 0.5 s before trace end
SMOOTH_WINDOW   = 0.05     # ± smoothing for baseline averaging

TMIN = 0.5   # lower bound of t_since to include
TMAX = 8.0   # upper bound

BIN_WIDTHS = [0.1, 0.2, 0.4, 0.5, 1.0]  # seconds

MIN_TRIALS_PER_BIN = 3
MIN_CELLS_PER_BIN  = 5


#%% load LC cell properties
cell_prop = pd.read_pickle('Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% main loop
all_trials_by_cell = {}

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    # behaviour
    with open(LC_beh_stem / f'{recname}.pkl', 'rb') as f:
        beh = pickle.load(f)

    reward_times = beh['reward_times'][1:]
    run_onsets   = beh['run_onsets'][1:]
    trials_sts   = beh['trial_statements'][1:]

    # exclude opto trials
    opto_idx = [i for i, t in enumerate(trials_sts) if t[15] != '0']

    # reward alignment index
    rec_stem = Path(f'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}') / recname[:14] / recname
    aligned_rew = sio.loadmat(
        rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRew_msess1.mat'
        )['trialsRew'][0][0]
    rew_spike = aligned_rew['startLfpInd'][0][1:]

    # spike maps
    sess_stem = all_sess_stem / recname
    spike_maps = np.load(sess_stem / f'{recname}_smoothed_spike_map.npy',
                         allow_pickle=True)

    # eligible LC cells
    curr_cell_prop = cell_prop[cell_prop['sessname'] == recname]
    eligible_cells = [
        cluname for cluname, row in curr_cell_prop.iterrows()
        if row['identity'] != 'other' and row['run_onset_peak']
    ]
    if not eligible_cells:
        print('no LC RO-peak cells; skip')
        continue

    for cluname in eligible_cells:
        if cluname not in all_trials_by_cell:
            all_trials_by_cell[cluname] = []

    # valid trials matching GLM script logic
    valid_trials = [
        t for t, ro in enumerate(run_onsets[:-1])
        if t not in opto_idx and t-1 not in opto_idx and not np.isnan(ro)
        ]

    for ti in valid_trials:
        onset_time = run_onsets[ti] / SAMP_FREQ_BEH
        t_since = gf.time_since_last_reward(reward_times, onset_time, ti)

        if np.isnan(t_since) or not (TMIN <= t_since <= TMAX):
            continue

        for cluname in eligible_cells:
            clu_idx = int(cluname.split('clu')[-1]) - 2

            start_idx = rew_spike[ti] - int(PRE_REW_S * SAMP_FREQ)
            end_idx   = rew_spike[ti] + int(t_since * SAMP_FREQ)

            trace = spike_maps[clu_idx][start_idx:end_idx]

            if trace.size > 50:
                trace = smooth_convolve(trace, sigma=1250/10, axis=0)
                all_trials_by_cell[cluname].append((t_since, trace))


#%% loop over bins 
for BW in BIN_WIDTHS:
    print(f'\n========== BIN WIDTH = {BW:.3f} s ==========')

    # define bins
    n_bins = int((TMAX - TMIN) / BW)
    bin_edges   = np.arange(TMIN, TMAX + 1e-6, BW)
    bin_centres = TMIN + (np.arange(n_bins) + 0.5) * BW

    # ============================================================
    # 2A — Pooled reward-aligned curves (colour gradient curves)
    # ============================================================
    pooled_bin_traces = [[] for _ in range(n_bins)]

    for cluname, trials in all_trials_by_cell.items():
        for t_since, trace in trials:
            bi = int((t_since - TMIN) // BW)
            if 0 <= bi < n_bins:
                pooled_bin_traces[bi].append(trace)

    mean_curves = []
    xaxes = []

    for bi in range(n_bins-1):
        traces = pooled_bin_traces[bi]
        if len(traces) == 0:
            mean_curves.append(None)
            xaxes.append(None)
            continue

        L = min(len(tr) for tr in traces)
        arr = np.stack([tr[:L] for tr in traces])
        mean_prof = np.mean(arr, axis=0)
        xaxis = np.arange(L) / SAMP_FREQ - PRE_REW_S

        mean_curves.append(mean_prof)
        xaxes.append(xaxis)

    # ---- plot pooled curves ----
    fig, ax = plt.subplots(figsize=(3.0, 2.4))

    for bi in range(n_bins-1):
        if mean_curves[bi] is None:
            continue
        colour = plt.cm.Greens(0.3 + 0.6 * bi / max(1, (n_bins - 1)))
        ax.plot(xaxes[bi], mean_curves[bi], color=colour, lw=1.0)
        ax.axvline(xaxes[bi][-1], color=colour, ls='--', lw=0.8)

    ax.set(xlabel='Time from last reward (s)',
           ylabel='Firing rate (Hz)',
           title=f'{int(BW*1000)}-ms bins')
    ax.spines[['top', 'right']].set_visible(False)

    # colourbar
    norm = plt.Normalize(vmin=TMIN, vmax=TMAX)
    sm = ScalarMappable(norm=norm, cmap=plt.cm.Greens)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=.35)
    cbar.set_label('Time since last reward (s)', fontsize=8)

    plt.tight_layout()

    base = GLM_stem / f'test_rew_aligned_curves_{int(BW*1000)}ms'
    for ext in ['.png', '.pdf']:
        fig.savefig(base.with_suffix(ext), dpi=300, bbox_inches='tight')
    plt.close(fig)


    # ============================================================
    # 2B — Baseline amplitude per cell × bin, and stats
    # ============================================================
    cell_names = sorted(all_trials_by_cell.keys())
    n_cells = len(cell_names)
    baseline_mat = np.full((n_cells, n_bins), np.nan)

    for ci, cluname in enumerate(cell_names):
        trials = all_trials_by_cell[cluname]
        if len(trials) == 0:
            continue

        # group trials into bins
        cell_bin_traces = [[] for _ in range(n_bins)]
        for t_since, trace in trials:
            bi = int((t_since - TMIN) // BW)
            if 0 <= bi < n_bins:
                cell_bin_traces[bi].append(trace)

        # per-bin baseline computation
        for bi in range(n_bins):
            traces = cell_bin_traces[bi]
            if len(traces) < MIN_TRIALS_PER_BIN:
                continue

            L = min(len(tr) for tr in traces)
            arr = np.stack([tr[:L] for tr in traces])
            mean_prof = np.mean(arr, axis=0)
            xaxis = np.arange(L) / SAMP_FREQ - PRE_REW_S

            t_end  = xaxis[-1]
            t_base = t_end - BASELINE_OFFSET
            if t_base <= xaxis[0]:
                continue

            mask = (xaxis >= (t_base - SMOOTH_WINDOW)) & (xaxis <= (t_base + SMOOTH_WINDOW))
            idx = np.where(mask)[0]
            if idx.size == 0:
                idx = np.array([np.argmin(np.abs(xaxis - t_base))])

            baseline = np.nanmean(mean_prof[idx])
            baseline_mat[ci, bi] = baseline

    # per-cell slope of baseline vs t_since
    baseline_slopes = []
    for ci in range(n_cells):
        y = baseline_mat[ci, :]
        mask = ~np.isnan(y)
        if np.sum(mask) < 3:
            continue
        x = bin_centres[mask]
        y_sel = y[mask]
        a, b = np.polyfit(x, y_sel, 1)
        baseline_slopes.append(a)

    baseline_slopes = np.array(baseline_slopes)

    if baseline_slopes.size > 0:
        mean_slope = np.mean(baseline_slopes)
        sem_slope  = sem(baseline_slopes)
        _, p_w     = wilcoxon(baseline_slopes)
        print(f'baseline slope = {mean_slope:.3f} ± {sem_slope:.3f}, p={p_w:.2e}')
    else:
        print('not enough valid slopes')
        mean_slope = np.nan
        sem_slope  = np.nan
        p_w        = np.nan


    # ============================================================
    # 2C — Population baseline-vs-t_since curve (with stats text)
    # ============================================================
    baseline_mean_bin = np.nanmean(baseline_mat, axis=0)
    baseline_sem_bin  = np.nanstd(baseline_mat, axis=0, ddof=1) / \
                        np.sqrt(np.sum(~np.isnan(baseline_mat), axis=0))

    valid_bins = np.sum(~np.isnan(baseline_mat), axis=0) >= MIN_CELLS_PER_BIN

    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.errorbar(bin_centres[valid_bins],
                baseline_mean_bin[valid_bins],
                yerr=baseline_sem_bin[valid_bins],
                fmt='o-', lw=1, ms=3, color='forestgreen')

    ax.set(xlabel='Time since last reward (s)',
           ylabel='Baseline amplitude (Hz)',
           title=f'Baseline vs t_since ({int(BW*1000)}-ms bins)')
    ax.spines[['top', 'right']].set_visible(False)

    # ---- print stats ON the plot ----
    ax.text(0.02, 0.98,
            f'slope = {mean_slope:.3f} ± {sem_slope:.3f}\n'
            f'p = {p_w:.2e}',
            ha='left', va='top',
            transform=ax.transAxes,
            fontsize=7)

    plt.tight_layout()

    base = GLM_stem / f'baseline_vs_tsince_{int(BW*1000)}ms'
    for ext in ['.png', '.pdf']:
        fig.savefig(base.with_suffix(ext), dpi=300, bbox_inches='tight')
    plt.close(fig)
