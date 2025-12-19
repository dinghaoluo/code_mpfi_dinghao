# -*- coding: utf-8 -*-
"""
Created on 5 Dec 2025
Modified on 17 Dec 2025

Analyse baseline rates of binned ITI-traces
    Modified to also align to run onset 

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
GLM_stem      = Path('Z:/Dinghao/code_dinghao/LC_ephys/GLM')

SAMP_FREQ     = 1250
SAMP_FREQ_BEH = 1000

PRE_REW_S  = 1.0            # reward-aligned window start
PRE_RUN_S  = 3.0            # run-aligned window start 
POST_RUN_S = 5.0
BASELINE_OFFSET = 0.5      # baseline = 0.5 s before trace end
SMOOTH_WINDOW   = 0.05     # ± smoothing for baseline averaging to reduce noise 

TMIN = 0.5   # lower bound of t_since to include
TMAX = 8.0   # upper bound

BIN_WIDTH     = 0.5  # second
BIN_WIDTH_RUN = 1.0  # second

MIN_TRIALS_PER_BIN = 3
MIN_CELLS_PER_BIN  = 5


#%% load LC cell properties
cell_prop = pd.read_pickle('Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% main loop
all_trials_by_cell_rew = {}
all_trials_by_cell_run = {}

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    # load behaviour
    with open(LC_beh_stem / f'{recname}.pkl', 'rb') as f:
        beh = pickle.load(f)

    reward_times = beh['reward_times'][1:]
    run_onsets   = beh['run_onsets'][1:]
    trials_sts   = beh['trial_statements'][1:]

    # get opto trial idx
    opto_idx = [i for i, t in enumerate(trials_sts) if t[15] != '0']

    # reward-aligned spike index
    rec_stem = Path(f'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}') / recname[:14] / recname
    aligned_rew = sio.loadmat(
        rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRew_msess1.mat'
        )['trialsRew'][0][0]
    rew_spike = aligned_rew['startLfpInd'][0][1:]
    
    aligned_run = sio.loadmat(
        rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
        )['trialsRun'][0][0]
    run_spike = aligned_run['startLfpInd'][0][1:]

    # smoothed spike maps
    sess_stem = all_sess_stem / recname
    spike_maps = np.load(sess_stem / f'{recname}_smoothed_spike_map.npy', allow_pickle=True)

    # eligible LC cells
    curr_prop = cell_prop[cell_prop['sessname'] == recname]
    eligible_cells = [
        cluname for cluname, row in curr_prop.iterrows()
        if row['identity'] != 'other' and row['run_onset_peak']
        ]
    
    # if no cell, skip this session
    if not eligible_cells:
        print('No LC RO-peak cells; skip')
        continue

    # initiate dictionary entries 
    for cluname in eligible_cells:
        all_trials_by_cell_rew.setdefault(cluname, [])
        all_trials_by_cell_run.setdefault(cluname, [])

    # valid trials
    valid_trials = [
        t for t, ro in enumerate(run_onsets[:-1])
        if t not in opto_idx and t-1 not in opto_idx and not np.isnan(ro)
        ]

    # extract traces
    for ti in valid_trials:
        onset_time = run_onsets[ti] / SAMP_FREQ_BEH
        t_since = gf.time_since_last_reward(reward_times, onset_time, ti)

        # enforce ITI window (0.5–8.0 s)
        if np.isnan(t_since) or not (TMIN <= t_since <= TMAX):
            continue

        for cluname in eligible_cells:
            clu_idx = int(cluname.split('clu')[-1]) - 2

            # get reward trains 
            start_idx_rew = rew_spike[ti] - int(PRE_REW_S * SAMP_FREQ)
            end_idx_rew   = rew_spike[ti] + int(t_since * SAMP_FREQ)

            trace_rew = spike_maps[clu_idx][start_idx_rew : end_idx_rew]

            # require >50 samples to avoid noisy single bins
            if trace_rew.size <= 50:
                continue

            trace_rew = smooth_convolve(trace_rew, sigma=1250*0.2, axis=0)
            all_trials_by_cell_rew[cluname].append((t_since, trace_rew))
            
            
            # get run trains 
            start_idx_run = run_spike[ti] - int(PRE_RUN_S * SAMP_FREQ)
            end_idx_run   = run_spike[ti] + int(POST_RUN_S * SAMP_FREQ)

            trace_run = spike_maps[clu_idx][start_idx_run : end_idx_run]

            # require >50 samples to avoid noisy single bins
            if trace_run.size <= 50:
                continue

            trace_run = smooth_convolve(trace_run, sigma=1250*0.2, axis=0)
            all_trials_by_cell_run[cluname].append((t_since, trace_run))


#%% binning and baseline analysis
n_bins      = int((TMAX - TMIN) / BIN_WIDTH)
bin_edges   = np.arange(TMIN, TMAX + 1e-6, BIN_WIDTH)
bin_centres = TMIN + (np.arange(n_bins) + 0.5) * BIN_WIDTH

# pooled traces for reward-aligned plotting
pooled_bin_traces = [[] for _ in range(n_bins)]

for cluname, trials in all_trials_by_cell_rew.items():
    for t_since, trace in trials:
        bi = int((t_since - TMIN) // BIN_WIDTH)
        if 0 <= bi < n_bins:
            pooled_bin_traces[bi].append(trace)

mean_curves = []
xaxes = []

for bi in range(n_bins - 1):
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
    

#%% plot pooled reward-aligned curves
fig, ax = plt.subplots(figsize=(3.0, 2.4))

for bi in range(n_bins - 1):
    if mean_curves[bi] is None:
        continue
    colour = plt.cm.Greens(0.3 + 0.6 * bi / max(1, (n_bins - 1)))
    ax.plot(xaxes[bi], mean_curves[bi], color=colour, lw=1.0)
    ax.axvline(xaxes[bi][-1], color=colour, ls='--', lw=0.8)

ax.set(xlabel='Time from reward (s)', xticks=[0, 3, 6],
       ylabel='Firing rate (Hz)',
       title='LC ITI ramps')
ax.spines[['top', 'right']].set_visible(False)

norm = plt.Normalize(vmin=TMIN, vmax=TMAX)
sm = ScalarMappable(norm=norm, cmap=plt.cm.Greens)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=.35)
cbar.set_label('Time from reward (s)', fontsize=8)
cbar.set_ticks([0, 3, 6])

plt.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(GLM_stem / f'rew_to_run_rew_aligned_profiles_smoothed200ms{ext}', 
                dpi=300, bbox_inches='tight')


#%% baseline matrix per cell × bin
cell_names = sorted(all_trials_by_cell_rew.keys())
n_cells = len(cell_names)
baseline_mat = np.full((n_cells, n_bins), np.nan)

for ci, cluname in enumerate(cell_names):
    trials = all_trials_by_cell_rew[cluname]
    if len(trials) == 0:
        continue

    cell_bin_traces = [[] for _ in range(n_bins)]
    for t_since, trace in trials:
        bi = int((t_since - TMIN) // BIN_WIDTH)
        if 0 <= bi < n_bins:
            cell_bin_traces[bi].append(trace)

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

        baseline_mat[ci, bi] = np.nanmean(mean_prof[idx])


#%% per-cell slope of baseline
baseline_slopes = []

for ci in range(n_cells):
    y = baseline_mat[ci, :]
    mask = ~np.isnan(y)
    if np.sum(mask) < 3:
        continue
    x = bin_centres[mask]
    a, b = np.polyfit(x, y[mask], 1)
    baseline_slopes.append(a)

baseline_slopes = np.array(baseline_slopes)

if baseline_slopes.size > 0:
    mean_slope = np.mean(baseline_slopes)
    sem_slope  = sem(baseline_slopes)
    _, p_w     = wilcoxon(baseline_slopes)
else:
    mean_slope = np.nan
    sem_slope  = np.nan
    p_w        = np.nan

print(f'baseline slope = {mean_slope:.3f} ± {sem_slope:.3f}, p={p_w:.2e}')


#%% population baseline vs t_since
baseline_mean_bin = np.nanmean(baseline_mat, axis=0)
baseline_sem_bin  = np.nanstd(baseline_mat, axis=0, ddof=1) / \
                    np.sqrt(np.sum(~np.isnan(baseline_mat), axis=0))

valid_bins = np.sum(~np.isnan(baseline_mat), axis=0) >= MIN_CELLS_PER_BIN

fig, ax = plt.subplots(figsize=(2.6, 2.4))
ax.errorbar(bin_centres[valid_bins],
            baseline_mean_bin[valid_bins],
            yerr=baseline_sem_bin[valid_bins],
            fmt='o-', lw=1, ms=3, color='forestgreen')

ax.set(xlabel='Time from reward (s)', xticks=[3, 6],
       ylabel='Baseline amplitude (Hz)', yticks=[2, 3],
       title='Baseline vs t_since')
ax.spines[['top', 'right']].set_visible(False)

ax.text(0.02, 0.98,
        f'slope = {mean_slope:.3f} ± {sem_slope:.3f}\n'
        f'p = {p_w:.2e}',
        ha='left', va='top',
        transform=ax.transAxes,
        fontsize=7)

plt.tight_layout()
base = GLM_stem / 'baseline_vs_tsince'
for ext in ['.png', '.pdf']:
    fig.savefig(GLM_stem / f'baseline_vs_tsince_smoothed200ms{ext}', 
                dpi=300, bbox_inches='tight')
    


#%% pooled run-onset-aligned curves
n_bins_run = int((TMAX - TMIN) / BIN_WIDTH_RUN)

pooled_bin_traces_run = [[] for _ in range(n_bins_run)]

# collect traces into bins
for cluname, trials in all_trials_by_cell_run.items():
    for t_since, trace in trials:
        bi = int((t_since - TMIN) // BIN_WIDTH_RUN)
        if 0 <= bi < n_bins_run:
            pooled_bin_traces_run[bi].append(trace)

mean_curves_run = []
for bi in range(n_bins_run - 1):
    traces = pooled_bin_traces_run[bi]
    if len(traces) == 0:
        mean_curves_run.append(None)
        continue

    L = min(len(tr) for tr in traces)
    arr = np.stack([tr[:L] for tr in traces])
    mean_prof = np.mean(arr, axis=0)

    mean_curves_run.append(mean_prof)

xaxis_run = np.arange((PRE_RUN_S + POST_RUN_S) * SAMP_FREQ) / SAMP_FREQ - PRE_RUN_S


#%% plot pooled run-onset-aligned curves
fig, ax = plt.subplots(figsize=(3.0, 2.4))

for bi in range(n_bins_run - 1):
    if mean_curves_run[bi] is None:
        continue
    colour = plt.cm.Greens(0.3 + 0.6 * bi / max(1, (n_bins_run - 1)))
    ax.plot(xaxis_run, mean_curves_run[bi], color=colour, lw=1.0)

ax.set(xlabel='Time from run onset (s)', xticks=[-2, 0, 2, 4],
       ylabel='Firing rate (Hz)',
       title='LC ITI ramps (run-aligned)')
ax.spines[['top', 'right']].set_visible(False)

norm = plt.Normalize(vmin=TMIN, vmax=TMAX)
sm = ScalarMappable(norm=norm, cmap=plt.cm.Greens)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=.35)
cbar.set_label('Time since last reward (s)', fontsize=8)
cbar.set_ticks([0, 3, 6])

plt.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(GLM_stem / f'rew_to_run_run_aligned_profiles_smoothed200ms{ext}',
                dpi=300, bbox_inches='tight')