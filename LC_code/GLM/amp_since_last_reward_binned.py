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
from scipy.stats import sem, iqr, wilcoxon, linregress, ttest_1samp

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
BASELINE_ON  = 0.55      # before trace end
BASELINE_OFF = 0.45      # same 
BASELINE_LONG_ON  = 0.75  # for long baseline
BASELINE_LONG_OFF = 0.25  # same 

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

t_since_all          = []
baseline_fr_all      = []
phasic_fr_all        = []

t_since_long_all     = []  # new, 29 Jan 2026
baseline_long_fr_all = []
phasic_long_fr_all   = []

sess_regress_r_tsince_baseline = []
sess_regress_r_baseline_phasic = []

sess_regress_r_tsince_baseline_long = []
sess_regress_r_baseline_phasic_long = []

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    sess_t_since   = []
    sess_baseline  = []
    sess_phasic    = []
    
    sess_t_since_long  = []
    sess_baseline_long = []
    sess_phasic_long   = []

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
        
        curr_all_baselines      = []
        curr_all_phasic         = []
        
        curr_all_baselines_long = []  # new, 29 Jan 2026
        curr_all_phasic_long    = []

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
            
            # baseline fr 
            x_rew = np.arange(len(trace_rew)) / SAMP_FREQ - PRE_REW_S
            mask_base = (x_rew >= x_rew[-1] - BASELINE_ON) & (x_rew <= x_rew[-1] - BASELINE_OFF)
            baseline_fr = np.nanmean(trace_rew[mask_base])
            curr_all_baselines.append(baseline_fr)
            
            # phasic fr (run-aligned; ±0.25 s around run onset)
            x_run = np.arange(len(trace_run)) / SAMP_FREQ - PRE_RUN_S
            mask_phasic = (x_run >= -0.25) & (x_run <= 0.25)
            phasic_fr = np.nanmean(trace_run[mask_phasic])
            curr_all_phasic.append(phasic_fr)
            
            # long baseline fr, new, 29 Jan 2026
            if t_since > 1:
                # baseline fr 
                x_rew = np.arange(len(trace_rew)) / SAMP_FREQ - PRE_REW_S
                mask_base = (x_rew >= x_rew[-1] - BASELINE_LONG_ON) & (x_rew <= x_rew[-1] - BASELINE_LONG_OFF)
                baseline_long_fr = np.nanmean(trace_rew[mask_base])
                curr_all_baselines_long.append(baseline_long_fr)
                
                # phasic fr
                x_run = np.arange(len(trace_run)) / SAMP_FREQ - PRE_RUN_S
                mask_phasic = (x_run >= -0.25) & (x_run <= 0.25)
                phasic_long_fr = np.nanmean(trace_run[mask_phasic])
                curr_all_phasic_long.append(phasic_long_fr)

            
        t_since_all.append(t_since)
        baseline_fr_all.append(np.nanmean(curr_all_baselines))
        phasic_fr_all.append(np.nanmean(curr_all_phasic))
        
        if t_since > 1:
            t_since_long_all.append(t_since)  # new, 29 Jan 2026
            baseline_long_fr_all.append(np.nanmean(curr_all_baselines_long))  
            phasic_long_fr_all.append(np.nanmean(curr_all_phasic_long))
        
        # session level too 
        sess_t_since.append(t_since)
        sess_baseline.append(np.nanmean(curr_all_baselines))
        sess_phasic.append(np.nanmean(curr_all_phasic))
        
        if t_since > 1:
            sess_t_since_long.append(t_since)
            sess_baseline_long.append(np.nanmean(curr_all_baselines_long))
            sess_phasic_long.append(np.nanmean(curr_all_phasic_long))
        
    # --------------------
    # SESSION-LEVEL PLOTS
    # --------------------
    sess_t = np.array(sess_t_since)
    sess_b = np.array(sess_baseline)
    
    valid = (
        ~np.isnan(sess_t) &
        ~np.isnan(sess_b) &
        (sess_b > 0.01)
    )
    
    if np.sum(valid) < 5:
        print('Not enough valid trials for regression; skip session')
    else:
        x = sess_t[valid]
        y = sess_b[valid]
    
        slope, intercept, r, p, _ = linregress(x, y)
    
        sess_regress_r_tsince_baseline.append(r)
        
        # -------- plot single-session scatter --------
        fig, ax = plt.subplots(figsize=(2.3, 2.6))
    
        ax.scatter(x, y, s=10, color='forestgreen', ec='none', alpha=0.6)
    
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, intercept + slope * xfit, color='k', lw=1)
    
        ax.text(
            0.05, 0.95,
            f'r = {r:.2f}\np = {p:.3g}',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=8
        )
    
        ax.set(
            xlabel='Time since last reward (s)',
            ylabel='Baseline firing rate (Hz)',
            title=recname
        )
        ax.spines[['top', 'right']].set_visible(False)
    
        fig.tight_layout()
    
        for ext in ['.png', '.pdf']:
            fig.savefig(
                GLM_stem / 'rew_to_run_baseline_single_session' / f'{recname}{ext}',
                dpi=300,
                bbox_inches='tight'
            )
    
        plt.close()
        
        
    # (long) baseline vs rew-run interval 
    sess_t_long = np.array(sess_t_since_long)
    sess_b_long = np.array(sess_baseline_long)
    
    valid = (
        ~np.isnan(sess_t_long) &
        ~np.isnan(sess_b_long) &
        (sess_b_long > 0.01)
    )
    
    if np.sum(valid) < 5:
        print('Not enough valid trials for regression; skip session')
    else:
        x = sess_t_long[valid]
        y = sess_b_long[valid]
    
        slope, intercept, r, p, _ = linregress(x, y)
    
        sess_regress_r_tsince_baseline_long.append(r)
        
        # -------- plot single-session scatter --------
        fig, ax = plt.subplots(figsize=(2.3, 2.6))
    
        ax.scatter(x, y, s=10, color='forestgreen', ec='none', alpha=0.6)
    
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, intercept + slope * xfit, color='k', lw=1)
    
        ax.text(
            0.05, 0.95,
            f'r = {r:.2f}\np = {p:.3g}',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=8
        )
    
        ax.set(
            xlabel='Rew.-run interval (s)',
            ylabel='Baseline firing rate (Hz)',
            title=recname
        )
        ax.spines[['top', 'right']].set_visible(False)
    
        fig.tight_layout()
    
        for ext in ['.png', '.pdf']:
            fig.savefig(
                GLM_stem / 'rew_to_run_baseline_single_session_long' / f'{recname}{ext}',
                dpi=300,
                bbox_inches='tight'
            )
    
        plt.close()
        
        
    # baseline vs phasic 
    sess_p = np.array(sess_phasic)

    valid = (
        ~np.isnan(sess_b) &
        ~np.isnan(sess_p)
    )
    
    if np.sum(valid) < 5:
        print('Not enough trials for baseline–phasic regression')
    else:
        x = sess_b[valid]
        y = sess_p[valid]
    
        slope, intercept, r, p, _ = linregress(x, y)
    
        sess_regress_r_baseline_phasic.append(r)
    
        # -------- plot single-session scatter --------
        fig, ax = plt.subplots(figsize=(2.3, 2.6))
    
        ax.scatter(x, y, s=10, color='forestgreen', ec='none', alpha=0.6)
    
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, intercept + slope * xfit, color='k', lw=1)
    
        ax.text(
            0.05, 0.95,
            f'r = {r:.2f}\np = {p:.3g}',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=8
        )
    
        ax.set(
            xlabel='Baseline firing rate (Hz)',
            ylabel='Run-onset firing rate (Hz)',
            title=recname
        )
        ax.spines[['top', 'right']].set_visible(False)
    
        fig.tight_layout()
    
        for ext in ['.png', '.pdf']:
            fig.savefig(
                GLM_stem / 'baseline_vs_phasic_single_session' / f'{recname}{ext}',
                dpi=300,
                bbox_inches='tight'
            )
    
        plt.close()
        
    
    # baseline vs phasic 
    sess_p_long = np.array(sess_phasic_long)

    valid = (
        ~np.isnan(sess_b_long) &
        ~np.isnan(sess_p_long)
    )
    
    if np.sum(valid) < 5:
        print('Not enough trials for baseline–phasic regression')
    else:
        x = sess_b_long[valid]
        y = sess_p_long[valid]
    
        slope, intercept, r, p, _ = linregress(x, y)
    
        sess_regress_r_baseline_phasic_long.append(r)
    
        # -------- plot single-session scatter --------
        fig, ax = plt.subplots(figsize=(2.3, 2.6))
    
        ax.scatter(x, y, s=10, color='forestgreen', ec='none', alpha=0.6)
    
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, intercept + slope * xfit, color='k', lw=1)
    
        ax.text(
            0.05, 0.95,
            f'r = {r:.2f}\np = {p:.3g}',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=8
        )
    
        ax.set(
            xlabel='Baseline firing rate (Hz)',
            ylabel='Run-onset firing rate (Hz)',
            title=recname
        )
        ax.spines[['top', 'right']].set_visible(False)
    
        fig.tight_layout()
    
        for ext in ['.png', '.pdf']:
            fig.savefig(
                GLM_stem / 'baseline_vs_phasic_single_session_long' / f'{recname}{ext}',
                dpi=300,
                bbox_inches='tight'
            )
    
        plt.close()
    # -------------------------
    # SESSION-LEVEL PLOTS ENDS
    # -------------------------


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
    
    
#%% plot pooled reward-aligned curves (coarsened bins)
GROUP = 3  # number of fine bins to merge

fig, ax = plt.subplots(figsize=(2.4, 2.4))

n_groups = int(np.ceil((n_bins - 1) / GROUP))

for gi in range(n_groups):
    idx = range(gi * GROUP, min((gi + 1) * GROUP, n_bins - 1))

    curves = [mean_curves[i] for i in idx if mean_curves[i] is not None]
    xax    = [xaxes[i]      for i in idx if xaxes[i]      is not None]

    if len(curves) == 0:
        continue

    # equalise length
    L = min(len(c) for c in curves)
    arr = np.stack([c[:L] for c in curves])
    mean_prof = np.mean(arr, axis=0)

    # use the *latest* bin’s x-axis for alignment
    x = xax[-1][:L]
    x = x - x[-1]   # end-align to 0

    # colour by group centre t_since
    t_center = bin_centres[idx.start:idx.stop].mean()
    colour = plt.cm.Greens((t_center - TMIN) / (TMAX - TMIN))

    ax.plot(x, mean_prof, color=colour, lw=1.3)
    ax.axvline(0, color=colour, ls='--', lw=0.8)

ax.set(
    xlabel='Time to run onset (s)',
    ylabel='Firing rate (Hz)',
    xticks=[-3, 0],
    xlim=(-3, 0.2),
    title='LC ITI ramps'
)

ax.spines[['top', 'right']].set_visible(False)

norm = plt.Normalize(vmin=TMIN, vmax=TMAX)
sm = ScalarMappable(norm=norm, cmap=plt.cm.Greens)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=.35)
cbar.set_label('Time since last reward (s)', fontsize=8)
cbar.set_ticks([0, 3, 6])

plt.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        GLM_stem / f'rew_to_run_rew_aligned_profiles_smoothed200ms_end_aligned{ext}',
        dpi=300, bbox_inches='tight'
    )


#%% pooled run-aligned curves
GROUP = 3
pooled_bin_traces_run = [[] for _ in range(n_bins)]

# collect run-aligned traces using SAME bins
for cluname, trials in all_trials_by_cell_run.items():
    for t_since, trace in trials:
        bi = int((t_since - TMIN) // BIN_WIDTH)
        if 0 <= bi < n_bins:
            pooled_bin_traces_run[bi].append(trace)

# group bins (same as reward)
fig, ax = plt.subplots(figsize=(2.4, 2.4))

n_groups = int(np.ceil((n_bins - 1) / GROUP))

for gi in range(n_groups):
    idx = range(gi * GROUP, min((gi + 1) * GROUP, n_bins - 1))

    traces = [
        tr for i in idx for tr in pooled_bin_traces_run[i]
    ]

    if len(traces) == 0:
        continue

    L = min(len(tr) for tr in traces)
    arr = np.stack([tr[:L] for tr in traces])
    mean_prof = np.mean(arr, axis=0)

    # run-aligned x-axis
    x = np.arange(L) / SAMP_FREQ - PRE_RUN_S

    # colour by SAME t_since logic
    t_center = bin_centres[idx.start:idx.stop].mean()
    colour = plt.cm.Greens((t_center - TMIN) / (TMAX - TMIN))

    ax.plot(x, mean_prof, color=colour, lw=1.3)
    ax.axvline(0, color=colour, ls='--', lw=0.8)

ax.set(
    xlabel='Time from run onset (s)',
    ylabel='Firing rate (Hz)',
    xlim=(-1, 2),
    xticks=[-1, 0, 1, 2],
    ylim=(1.8, 4.6),
    title='LC ITI ramps (run-aligned)'
)

ax.spines[['top', 'right']].set_visible(False)

norm = plt.Normalize(vmin=TMIN, vmax=TMAX)
sm = ScalarMappable(norm=norm, cmap=plt.cm.Greens)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=.35)
cbar.set_label('Time since last reward (s)', fontsize=8)

plt.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        GLM_stem / f'rew_to_run_run_aligned_profiles_smoothed200ms_end_aligned{ext}',
        dpi=300, bbox_inches='tight'
    )


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
        t_base = t_end - BASELINE_OFF
        if t_base <= xaxis[0]:
            continue

        mask = (xaxis >= (t_base - BASELINE_ON)) & (xaxis <= (t_base - BASELINE_OFF))
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


#%% plot pooled run-onset-aligned curves
xaxis_run = np.arange((PRE_RUN_S + POST_RUN_S) * SAMP_FREQ) / SAMP_FREQ - PRE_RUN_S

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
    
    
#%% plot pooled run-onset-aligned curves (3 s window)
xaxis_run = np.arange(3 * SAMP_FREQ) / SAMP_FREQ - 1

fig, ax = plt.subplots(figsize=(2.6, 2.4))

for bi in range(n_bins_run - 1):
    if mean_curves_run[bi] is None:
        continue

    colour = plt.cm.Greens(0.3 + 0.6 * bi / max(1, (n_bins_run - 1)))

    ax.plot(
        xaxis_run,
        mean_curves_run[bi][2500:2500 + 3 * SAMP_FREQ],
        color=colour,
        lw=1.0
    )

ax.set(
    xlabel='Time from run onset (s)',
    xticks=[-1, 0, 1, 2],
    xlim=(-1, 2),
    ylabel='Firing rate (Hz)',
    title='LC ITI ramps (run-aligned)'
)

ax.spines[['top', 'right']].set_visible(False)

norm = plt.Normalize(vmin=TMIN, vmax=TMAX)
sm = ScalarMappable(norm=norm, cmap=plt.cm.Greens)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=.35)
cbar.set_label('Rew.-run interval (s)', fontsize=8)
cbar.set_ticks([0, 4, 8])

plt.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        GLM_stem / f'rew_to_run_run_aligned_profiles_smoothed200ms_3s{ext}',
        dpi=300,
        bbox_inches='tight'
    )
    
    
#%% scatter data wrangling 
t_since_all = np.array(t_since_all)
baseline_fr_all = np.array(baseline_fr_all)
phasic_fr_all = np.array(phasic_fr_all)

valid = (
    ~np.isnan(t_since_all) &
    ~np.isnan(baseline_fr_all) &
    ~np.isnan(phasic_fr_all)
)

t_since_v = t_since_all[valid]
baseline_v = baseline_fr_all[valid]
phasic_v = phasic_fr_all[valid]


#%% t_since vs baseline fr
filt_t, filt_base = [], []
for t, b in zip(t_since_all, baseline_fr_all):
    if not np.isnan(t) and not np.isnan(b) and b > 0.01:
        filt_t.append(t)
        filt_base.append(b)

slope, intercept, r, p, _ = linregress(filt_t, filt_base)

fig, ax = plt.subplots(figsize=(2.3, 2.6))
ax.scatter(filt_t, filt_base, color='forestgreen', s=5, edgecolor='none', alpha=0.5)

xfit = np.linspace(min(filt_t), max(filt_t), 100)
ax.plot(xfit, intercept + slope * xfit, color='k', lw=1)

ax.text(0.05, 0.95,
        f'$R = {r:.2f}$\n$p = {p:.3g}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=9)

ax.set_xlabel('Time since last reward (s)')
ax.set_ylabel('Baseline firing rate (Hz)')
ax.set_title('Baseline vs r-r interval')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
for ext in ['.pdf', '.png']:
    fig.savefig(GLM_stem / f'scatter_baseline_vs_tsince{ext}',
                dpi=300, bbox_inches='tight')
    

#%% (long) t_since vs baseline fr
filt_t_long, filt_base_long = [], []
for t, b in zip(t_since_long_all, baseline_long_fr_all):
    if not np.isnan(t) and not np.isnan(b) and b > 0.01:
        filt_t_long.append(t)
        filt_base_long.append(b)

slope, intercept, r, p, _ = linregress(filt_t_long, filt_base_long)

fig, ax = plt.subplots(figsize=(2.3, 2.6))
ax.scatter(filt_t_long, filt_base_long, color='forestgreen', s=5, edgecolor='none', alpha=0.5)

xfit = np.linspace(min(filt_t_long), max(filt_t_long), 100)
ax.plot(xfit, intercept + slope * xfit, color='k', lw=1)

ax.text(0.05, 0.95,
        f'$R = {r:.2f}$\n$p = {p:.3g}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=9)

ax.set_xlabel('Time since last reward (s)')
ax.set_ylabel('Baseline firing rate (Hz)')
ax.set_title('Baseline vs r-r interval (long)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set(xlim=(2,8))

fig.tight_layout()
for ext in ['.pdf', '.png']:
    fig.savefig(GLM_stem / f'scatter_baseline_vs_tsince_long{ext}',
                dpi=300, bbox_inches='tight')


#%% baseline fr vs phasic fr
filt_base, filt_phasic = [], []
for b, pfr in zip(baseline_fr_all, phasic_fr_all):
    if not np.isnan(b) and not np.isnan(pfr):
        filt_base.append(b)
        filt_phasic.append(pfr)

slope, intercept, r, p, _ = linregress(filt_base, filt_phasic)

fig, ax = plt.subplots(figsize=(2.3, 2.6))
ax.scatter(filt_base, filt_phasic, color='forestgreen', s=5, edgecolor='none', alpha=0.5)

xfit = np.linspace(min(filt_base), max(filt_base), 100)
ax.plot(xfit, intercept + slope * xfit, color='k', lw=1)

ax.text(0.05, 0.95,
        f'$R = {r:.2f}$\n$p = {p:.3g}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=9)

ax.set_xlabel('Baseline firing rate (Hz)')
ax.set_ylabel('Phasic firing rate (Hz)')
ax.set_title('Phasic vs baseline')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
for ext in ['.pdf', '.png']:
    fig.savefig(GLM_stem / f'scatter_phasic_vs_baseline{ext}',
                dpi=300, bbox_inches='tight')
    
    
#%% (long) baseline fr vs phasic fr
filt_base_long, filt_phasic_long = [], []
for b, pfr in zip(baseline_long_fr_all, phasic_long_fr_all):
    if not np.isnan(b) and not np.isnan(pfr):
        filt_base_long.append(b)
        filt_phasic_long.append(pfr)

slope, intercept, r, p, _ = linregress(filt_base_long, filt_phasic_long)

fig, ax = plt.subplots(figsize=(2.3, 2.6))
ax.scatter(filt_base_long, filt_phasic_long, color='forestgreen', s=5, edgecolor='none', alpha=0.5)

xfit = np.linspace(min(filt_base_long), max(filt_base_long), 100)
ax.plot(xfit, intercept + slope * xfit, color='k', lw=1)

ax.text(0.05, 0.95,
        f'$R = {r:.2f}$\n$p = {p:.3g}$',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=9)

ax.set_xlabel('Baseline firing rate (Hz)')
ax.set_ylabel('phasic long firing rate (Hz)')
ax.set_title('Phasic vs baseline (long)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
for ext in ['.pdf', '.png']:
    fig.savefig(GLM_stem / f'scatter_phasic_vs_baseline_long{ext}',
                dpi=300, bbox_inches='tight')
    
    
#%% violin plot summaries 
# rew-to-run vs baseline 
sess_regress_r_tsince_baseline = np.array(sess_regress_r_tsince_baseline)

tval, p_t = ttest_1samp(sess_regress_r_tsince_baseline, 0)
wstat, p_w = wilcoxon(sess_regress_r_tsince_baseline)

mean_r = np.nanmean(sess_regress_r_tsince_baseline)
sem_r  = sem(sess_regress_r_tsince_baseline)

# IQR
q25, q75 = np.percentile(sess_regress_r_tsince_baseline, [25, 75])
iqr_r = q75 - q25


fig, ax = plt.subplots(figsize=(1.6, 2.2))

parts = ax.violinplot(
    sess_regress_r_tsince_baseline,
    positions=[1],
    showmeans=False,
    showmedians=True,
    showextrema=False
)

for pc in parts['bodies']:
    pc.set_facecolor('forestgreen')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)

parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

# scatter individual sessions
ax.scatter(
    np.ones(len(sess_regress_r_tsince_baseline)),
    sess_regress_r_tsince_baseline,
    color='forestgreen',
    ec='none',
    s=10,
    alpha=0.6,
    zorder=3
)

# mean ± sem annotation
ymax = np.max(sess_regress_r_tsince_baseline)
ymin = np.min(sess_regress_r_tsince_baseline)

ax.text(
    1,
    ymax + 0.08*(ymax - ymin),
    f'Med = {np.median(sess_regress_r_tsince_baseline):.2f}\n'
    f'IQR = [{q25:.2f}, {q75:.2f}]\n'
    f'{mean_r:.2f} ± {sem_r:.2f}',
    ha='center',
    va='bottom',
    fontsize=7,
    color='forestgreen'
)

# significance text
ax.text(
    1,
    ymin - 0.15*(ymax - ymin),
    f't={tval:.2f}, p={p_t:.2e}\n'
    f'w={wstat:.2f}, p={p_w:.2e}',
    ha='center',
    va='top',
    fontsize=6.5
)

ax.set(
    xlim=(0.5, 1.5),
    xticks=[1],
    xticklabels=['Session r'],
    ylabel='Correlation (r)',
    title='Baseline vs t_since'
)

ax.spines[['top', 'right', 'bottom']].set_visible(False)

plt.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        GLM_stem / f'baseline_vs_tsince_r_violin{ext}',
        dpi=300,
        bbox_inches='tight'
    )
    
    
# (long) r-r to baseline 
sess_regress_r_tsince_baseline_long = np.array(sess_regress_r_tsince_baseline_long)

tval, p_t = ttest_1samp(sess_regress_r_tsince_baseline_long, 0)
wstat, p_w = wilcoxon(sess_regress_r_tsince_baseline_long)

mean_r = np.nanmean(sess_regress_r_tsince_baseline_long)
sem_r  = sem(sess_regress_r_tsince_baseline_long)

# IQR
q25, q75 = np.percentile(sess_regress_r_tsince_baseline_long, [25, 75])
iqr_r = q75 - q25


fig, ax = plt.subplots(figsize=(1.6, 2.2))

parts = ax.violinplot(
    sess_regress_r_tsince_baseline_long,
    positions=[1],
    showmeans=False,
    showmedians=True,
    showextrema=False
)

for pc in parts['bodies']:
    pc.set_facecolor('forestgreen')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)

parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

# scatter individual sessions
ax.scatter(
    np.ones(len(sess_regress_r_tsince_baseline_long)),
    sess_regress_r_tsince_baseline_long,
    color='forestgreen',
    ec='none',
    s=10,
    alpha=0.6,
    zorder=3
)

# mean ± sem annotation
ymax = np.max(sess_regress_r_tsince_baseline_long)
ymin = np.min(sess_regress_r_tsince_baseline_long)

ax.text(
    1,
    ymax + 0.08*(ymax - ymin),
    f'Med = {np.median(sess_regress_r_tsince_baseline_long):.2f}\n'
    f'IQR = [{q25:.2f}, {q75:.2f}]\n'
    f'{mean_r:.2f} ± {sem_r:.2f}',
    ha='center',
    va='bottom',
    fontsize=7,
    color='forestgreen'
)

# significance text
ax.text(
    1,
    ymin - 0.15*(ymax - ymin),
    f't={tval:.2f}, p={p_t:.2e}\n'
    f'w={wstat:.2f}, p={p_w:.2e}',
    ha='center',
    va='top',
    fontsize=6.5
)

ax.set(
    xlim=(0.5, 1.5),
    xticks=[1],
    xticklabels=['Session r'],
    ylabel='Correlation (r)',
    title='Baseline vs t_since'
)

ax.spines[['top', 'right', 'bottom']].set_visible(False)

plt.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        GLM_stem / f'baseline_long_vs_tsince_r_violin{ext}',
        dpi=300,
        bbox_inches='tight'
    )
    

# baseline vs phasic 
sess_regress_r_baseline_phasic = np.array(sess_regress_r_baseline_phasic)

tval, p_t = ttest_1samp(sess_regress_r_baseline_phasic, 0)
wstat, p_w = wilcoxon(sess_regress_r_baseline_phasic)

mean_r = np.nanmean(sess_regress_r_baseline_phasic)
sem_r  = sem(sess_regress_r_baseline_phasic)

# IQR
q25, q75 = np.percentile(sess_regress_r_baseline_phasic, [25, 75])
iqr_r = q75 - q25


fig, ax = plt.subplots(figsize=(1.6, 2.2))

parts = ax.violinplot(
    sess_regress_r_baseline_phasic,
    positions=[1],
    showmeans=False,
    showmedians=True,
    showextrema=False
)

for pc in parts['bodies']:
    pc.set_facecolor('forestgreen')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)

parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

ax.scatter(
    np.ones(len(sess_regress_r_baseline_phasic)),
    sess_regress_r_baseline_phasic,
    color='forestgreen',
    ec='none',
    s=10,
    alpha=0.6,
    zorder=3
)

ymax = np.max(sess_regress_r_baseline_phasic)
ymin = np.min(sess_regress_r_baseline_phasic)

ax.text(
    1,
    ymax + 0.08*(ymax - ymin),
    f'Med = {np.median(sess_regress_r_baseline_phasic):.2f}\n'
    f'IQR = [{q25:.2f}, {q75:.2f}]\n'
    f'{mean_r:.2f} ± {sem_r:.2f}',
    ha='center',
    va='bottom',
    fontsize=7,
    color='forestgreen'
)

ax.text(
    1,
    ymin - 0.15*(ymax - ymin),
    f't={tval:.2f}, p={p_t:.2e}\n'
    f'w={wstat:.2f}, p={p_w:.2e}',
    ha='center',
    va='top',
    fontsize=6.5
)

ax.set(
    xlim=(0.5, 1.5),
    xticks=[1],
    xticklabels=['Session r'],
    ylabel='Correlation (r)',
    title='Baseline vs run-onset FR'
)

ax.spines[['top', 'right', 'bottom']].set_visible(False)

plt.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        GLM_stem / f'baseline_vs_phasic_r_violin{ext}',
        dpi=300,
        bbox_inches='tight'
    )
    
    
# (long) baseline vs phasic 
sess_regress_r_baseline_phasic_long = np.array(sess_regress_r_baseline_phasic_long)

tval, p_t = ttest_1samp(sess_regress_r_baseline_phasic_long, 0)
wstat, p_w = wilcoxon(sess_regress_r_baseline_phasic_long)

mean_r = np.nanmean(sess_regress_r_baseline_phasic_long)
sem_r  = sem(sess_regress_r_baseline_phasic_long)

# IQR
q25, q75 = np.percentile(sess_regress_r_baseline_phasic_long, [25, 75])
iqr_r = q75 - q25


fig, ax = plt.subplots(figsize=(1.6, 2.2))

parts = ax.violinplot(
    sess_regress_r_baseline_phasic_long,
    positions=[1],
    showmeans=False,
    showmedians=True,
    showextrema=False
)

for pc in parts['bodies']:
    pc.set_facecolor('forestgreen')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)

parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

ax.scatter(
    np.ones(len(sess_regress_r_baseline_phasic)),
    sess_regress_r_baseline_phasic,
    color='forestgreen',
    ec='none',
    s=10,
    alpha=0.6,
    zorder=3
)

ymax = np.max(sess_regress_r_baseline_phasic_long)
ymin = np.min(sess_regress_r_baseline_phasic_long)

ax.text(
    1,
    ymax + 0.08*(ymax - ymin),
    f'Med = {np.median(sess_regress_r_baseline_phasic_long):.2f}\n'
    f'IQR = [{q25:.2f}, {q75:.2f}]\n'
    f'{mean_r:.2f} ± {sem_r:.2f}',
    ha='center',
    va='bottom',
    fontsize=7,
    color='forestgreen'
)

ax.text(
    1,
    ymin - 0.15*(ymax - ymin),
    f't={tval:.2f}, p={p_t:.2e}\n'
    f'w={wstat:.2f}, p={p_w:.2e}',
    ha='center',
    va='top',
    fontsize=6.5
)

ax.set(
    xlim=(0.5, 1.5),
    xticks=[1],
    xticklabels=['Session r'],
    ylabel='Correlation (r)',
    title='Baseline vs run-onset FR'
)

ax.spines[['top', 'right', 'bottom']].set_visible(False)

plt.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        GLM_stem / f'baseline_vs_phasic_long_r_violin{ext}',
        dpi=300,
        bbox_inches='tight'
    )