# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:08:34 2025
Modified on 27 Oct 2025

analyse the decay time constants of pyramidal cells in stim. vs ctrl. trials
(only REMAINERS included)

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import sem, linregress, ranksums, ttest_ind, wilcoxon, ttest_rel
import pandas as pd

from common import mpl_formatting
mpl_formatting()

import rec_list
pathHPCLCopt = rec_list.pathHPCLCopt
pathHPCLCtermopt = rec_list.pathHPCLCtermopt


#%% parameters
SAMP_FREQ = 1250  # Hz
TIME = np.arange(-SAMP_FREQ*3, SAMP_FREQ*7) / SAMP_FREQ  # 10 s

MAX_TIME = 10
MAX_SAMPLES = SAMP_FREQ * MAX_TIME

XAXIS = np.arange(-1*1250, 4*1250) / 1250
DELTA_THRES = 0.5  # Hz

# changed to 0.5-s bins, 21 Nov 2025
bin_edges = np.arange(3750 + int(-0.5*1250), 3750 + int(3.5*1250) + 1, 1250)
bin_labels = ['-0.5 - 0.5 s',
              ' 0.5 - 1.5 s',
              ' 1.5 - 2.5 s',
              ' 2.5 - 3.5 s']


#%% helpers
def _annotate_pvals(ax, pval_tuple, base_y=4.05, dy=0.12, star=False):
    p_ranksums, p_ttest_ind, p_wilcoxon, p_ttest_rel = pval_tuple
    all_tests = [
        ('RS', p_ranksums),
        ('tt_ind', p_ttest_ind),
        ('Wil', p_wilcoxon),
        ('tt_rel', p_ttest_rel)
    ]

    mids = (bin_edges[:-1] + bin_edges[1:]) / 2  
    mids = (mids - 3750) / 1250  # convert to seconds

    for row, (label, pvals) in enumerate(all_tests):
        y = base_y - row*dy
        for mid, p in zip(mids, pvals):

            if star:
                if p < 0.0001: text = '****'
                elif p < 0.001: text = '***'
                elif p < 0.01: text = '**'
                elif p < 0.05: text = '*'
                else: text = 'n.s.'
            else:
                text = f'{p:.1e}'

            ax.text(mid, y, text,
                    ha='center', va='bottom',
                    fontsize=3, color='k')


def _binwise_test(ctrl_traces, stim_traces, label):
    pvals_ranksums, pvals_ttest_ind = [], []
    pvals_wilcoxon, pvals_ttest_rel = [], []

    paired = len(ctrl_traces) == len(stim_traces)

    print(f'\nBinwise stats for {label}:')
    for b in range(len(bin_edges)-1):
        start, end = bin_edges[b], bin_edges[b+1]

        ctrl_bin = [np.mean(tr[start:end]) for tr in ctrl_traces]
        stim_bin = [np.mean(tr[start:end]) for tr in stim_traces]

        ctrl_mean = np.mean(ctrl_bin)
        ctrl_sem  = sem(ctrl_bin)
        stim_mean = np.mean(stim_bin)
        stim_sem  = sem(stim_bin)

        # always valid
        _, p_ranksums = ranksums(ctrl_bin, stim_bin, nan_policy='omit')
        pvals_ranksums.append(p_ranksums)

        _, p_ttest_ind = ttest_ind(ctrl_bin, stim_bin, nan_policy='omit')
        pvals_ttest_ind.append(p_ttest_ind)

        # fill NaN for missing values
        if paired:
            _, p_wilcoxon = wilcoxon(ctrl_bin, stim_bin, nan_policy='omit')
            _, p_ttest_rel = ttest_rel(ctrl_bin, stim_bin, nan_policy='omit')
        else:
            p_wilcoxon = np.nan
            p_ttest_rel = np.nan

        pvals_wilcoxon.append(p_wilcoxon)
        pvals_ttest_rel.append(p_ttest_rel)

        print(f'{bin_labels[b]}: {ctrl_mean:.4e}±{ctrl_sem:.4e}, {stim_mean:.4e}±{stim_sem:.4e}, p={p_wilcoxon:.4e}')

    return pvals_ranksums, pvals_ttest_ind, pvals_wilcoxon, pvals_ttest_rel


#%% path stems
all_sess_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions')
all_beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments')
ctrl_stim_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/run_onset_response/ctrl_stim')


#%% load data
print('Loading dataframes...')
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')
df_pyr = cell_profiles[cell_profiles['cell_identity'] == 'pyr']


#%% HPCLC
exp   = 'HPCLC'
paths = pathHPCLCopt

mean_prof_ctrl_all_ON, mean_prof_stim_all_ON   = [], []
mean_prof_ctrl_all_OFF, mean_prof_stim_all_OFF = [], []

mean_prof_ctrl_remain_ON, mean_prof_stim_remain_ON   = [], []
mean_prof_ctrl_remain_OFF, mean_prof_stim_remain_OFF = [], []

all_amp_remain_ON_delta_mean, all_amp_remain_OFF_delta_mean      = [], []
all_ctrl_stim_lick_time_delta, all_ctrl_stim_lick_distance_delta = [], []

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    train_path = all_sess_stem / recname / f'{recname}_all_trains_run.npy'
    trains = np.load(train_path, allow_pickle=True).item()

    if (all_beh_stem / 'HPCLC' / f'{recname}.pkl').exists():
        with open(all_beh_stem / 'HPCLC' / f'{recname}.pkl', 'rb') as f:
            beh = pickle.load(f)
    else:
        with open(all_beh_stem / 'HPCLCterm' / f'{recname}.pkl', 'rb') as f:
            beh = pickle.load(f)

    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds) if cond != '0']
    ctrl_idx = [trial+2 for trial in stim_idx]

    # behaviour deltas
    first_lick_times = [t[0][0] - s for t, s
                        in zip(beh['lick_times'], beh['run_onsets']) if t]
    ctrl_stim_lick_time_delta = np.median(
        [t for i, t in enumerate(first_lick_times) if i in stim_idx]
        ) - np.median(
        [t for i, t in enumerate(first_lick_times) if i in ctrl_idx]
        )
    all_ctrl_stim_lick_time_delta.append(ctrl_stim_lick_time_delta)

    first_lick_distances = [t[0] if type(t) != float and len(t) > 0 else np.nan
                            for t in beh['lick_distances_aligned']][1:]
    ctrl_stim_lick_distance_delta = np.mean(
        [t for i, t in enumerate(first_lick_distances) if i in stim_idx]
        ) - np.mean(
        [t for i, t in enumerate(first_lick_distances) if i in ctrl_idx]
        )
    all_ctrl_stim_lick_distance_delta.append(ctrl_stim_lick_distance_delta)

    curr_df_pyr = df_pyr[df_pyr['recname'] == recname]

    for idx, session in curr_df_pyr.iterrows():
        cluname = idx
        
        # ALL ON 
        if session['class_ctrl'] == 'run-onset ON':
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_ctrl_all_ON.append(mean_prof_ctrl)
        if session['class_stim'] == 'run-onset ON':
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_stim_all_ON.append(mean_prof_stim)
            
        # ALL OFF
        if session['class_ctrl'] == 'run-onset OFF':
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_ctrl_all_OFF.append(mean_prof_ctrl)
        if session['class_stim'] == 'run-onset OFF':
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_stim_all_OFF.append(mean_prof_stim)
        
        # REMAIN ON
        if session['class_ctrl'] == 'run-onset ON' and session['class_stim'] == 'run-onset ON':
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_ctrl_remain_ON.append(mean_prof_ctrl)
            mean_prof_stim_remain_ON.append(mean_prof_stim)

        # REMAIN OFF
        if session['class_ctrl'] == 'run-onset OFF' and session['class_stim'] == 'run-onset OFF':
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_ctrl_remain_OFF.append(mean_prof_ctrl)
            mean_prof_stim_remain_OFF.append(mean_prof_stim)

    # per-session amplitude delta
    if len(mean_prof_ctrl_remain_ON) > 0:
        union = sum((curr_df_pyr['class_ctrl'] == 'run-onset ON') &
                    (curr_df_pyr['class_stim'] == 'run-onset ON'))
        sess_mean_ctrl = np.mean(mean_prof_ctrl_remain_ON[-union:], axis=0)
        sess_mean_stim = np.mean(mean_prof_stim_remain_ON[-union:], axis=0)
        amp_remain_ON_delta_mean = np.mean(sess_mean_stim[3750+625:3750+1825]) - np.mean(sess_mean_ctrl[3750+625:3750+1825])
        all_amp_remain_ON_delta_mean.append(amp_remain_ON_delta_mean)

    if len(mean_prof_ctrl_remain_OFF) > 0:
        union = sum((curr_df_pyr['class_ctrl'] == 'run-onset OFF') &
                    (curr_df_pyr['class_stim'] == 'run-onset OFF'))
        sess_mean_ctrl = np.mean(mean_prof_ctrl_remain_OFF[-union:], axis=0)
        sess_mean_stim = np.mean(mean_prof_stim_remain_OFF[-union:], axis=0)
        amp_remain_OFF_delta_mean = np.mean(sess_mean_stim[3750+625:3750+1825]) - np.mean(sess_mean_ctrl[3750+625:3750+1825])
        all_amp_remain_OFF_delta_mean.append(amp_remain_OFF_delta_mean)


    ## single-session overlay plots
    save_path = ctrl_stim_stem / 'single_sessions_remainers'
    save_path.mkdir(exist_ok=True)

    # compute per-session average remainer traces
    if len(mean_prof_ctrl_remain_ON) > 0:
        union = sum((curr_df_pyr['class_ctrl'] == 'run-onset ON') &
                    (curr_df_pyr['class_stim'] == 'run-onset ON'))
        if union > 0:
            sess_mean_ctrl_ON = np.mean(mean_prof_ctrl_remain_ON[-union:], axis=0)
            sess_mean_stim_ON = np.mean(mean_prof_stim_remain_ON[-union:], axis=0)

            fig, ax = plt.subplots(figsize=(2.6, 2))
            ax.plot(XAXIS, sess_mean_ctrl_ON[2500:2500+5*1250],
                    label='ctrl.', color='firebrick')
            ax.plot(XAXIS, sess_mean_stim_ON[2500:2500+5*1250],
                    label='stim.', color='royalblue')
            for s in ['top', 'right']:
                ax.spines[s].set_visible(False)
            ax.set(xlabel='Time from run-onset (s)',
                   ylabel='Firing rate (Hz)',
                   xticks=[0, 2, 4])
            ax.set_title(f'{recname} ON remainers', fontsize=9)
            ax.legend(fontsize=6, frameon=False)
            fig.savefig(save_path / f'{recname}_remain_ON.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

    if len(mean_prof_ctrl_remain_OFF) > 0:
        union = sum((curr_df_pyr['class_ctrl'] == 'run-onset OFF') &
                    (curr_df_pyr['class_stim'] == 'run-onset OFF'))
        if union > 0:
            sess_mean_ctrl_OFF = np.mean(mean_prof_ctrl_remain_OFF[-union:], axis=0)
            sess_mean_stim_OFF = np.mean(mean_prof_stim_remain_OFF[-union:], axis=0)

            fig, ax = plt.subplots(figsize=(2.6, 2))
            ax.plot(XAXIS, sess_mean_ctrl_OFF[2500:2500+5*1250],
                    label='ctrl.', color='purple')
            ax.plot(XAXIS, sess_mean_stim_OFF[2500:2500+5*1250],
                    label='stim.', color='royalblue')
            for s in ['top', 'right']:
                ax.spines[s].set_visible(False)
            ax.set(xlabel='Time from run-onset (s)',
                   ylabel='Firing rate (Hz)',
                   xticks=[0, 2, 4])
            ax.set_title(f'{recname} OFF remainers', fontsize=9)
            ax.legend(fontsize=6, frameon=False)
            fig.savefig(save_path / f'{recname}_remain_OFF.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)


## group-level plots and stats
mean_ctrl_all_ON = np.mean(mean_prof_ctrl_all_ON, axis=0)[2500:2500+5*1250]
sem_ctrl_all_ON = sem(mean_prof_ctrl_all_ON, axis=0)[2500:2500+5*1250]
mean_stim_all_ON = np.mean(mean_prof_stim_all_ON, axis=0)[2500:2500+5*1250]
sem_stim_all_ON = sem(mean_prof_stim_all_ON, axis=0)[2500:2500+5*1250]

mean_ctrl_all_OFF = np.mean(mean_prof_ctrl_all_OFF, axis=0)[2500:2500+5*1250]
sem_ctrl_all_OFF = sem(mean_prof_ctrl_all_OFF, axis=0)[2500:2500+5*1250]
mean_stim_all_OFF = np.mean(mean_prof_stim_all_OFF, axis=0)[2500:2500+5*1250]
sem_stim_all_OFF = sem(mean_prof_stim_all_OFF, axis=0)[2500:2500+5*1250]

pvals_all_ON = _binwise_test(mean_prof_ctrl_all_ON, mean_prof_stim_all_ON, 'all ON')
pvals_all_OFF = _binwise_test(mean_prof_ctrl_all_OFF, mean_prof_stim_all_OFF, 'all OFF')

mean_ctrl_remain_ON = np.mean(mean_prof_ctrl_remain_ON, axis=0)[2500:2500+5*1250]
sem_ctrl_remain_ON = sem(mean_prof_ctrl_remain_ON, axis=0)[2500:2500+5*1250]
mean_stim_remain_ON = np.mean(mean_prof_stim_remain_ON, axis=0)[2500:2500+5*1250]
sem_stim_remain_ON = sem(mean_prof_stim_remain_ON, axis=0)[2500:2500+5*1250]

mean_ctrl_remain_OFF = np.mean(mean_prof_ctrl_remain_OFF, axis=0)[2500:2500+5*1250]
sem_ctrl_remain_OFF = sem(mean_prof_ctrl_remain_OFF, axis=0)[2500:2500+5*1250]
mean_stim_remain_OFF = np.mean(mean_prof_stim_remain_OFF, axis=0)[2500:2500+5*1250]
sem_stim_remain_OFF = sem(mean_prof_stim_remain_OFF, axis=0)[2500:2500+5*1250]

pvals_remain_ON = _binwise_test(mean_prof_ctrl_remain_ON, mean_prof_stim_remain_ON, 'remain ON')
pvals_remain_OFF = _binwise_test(mean_prof_ctrl_remain_OFF, mean_prof_stim_remain_OFF, 'remain OFF')


#%% plotting and stats 
## plot ON all 
fig, ax = plt.subplots(figsize=(2.6, 2))
ax.plot(XAXIS, mean_ctrl_all_ON, label='ctrl. all ON', color='firebrick')
ax.fill_between(XAXIS, mean_ctrl_all_ON + sem_ctrl_all_ON, mean_ctrl_all_ON - sem_ctrl_all_ON,
                color='firebrick', alpha=.15)
ax.plot(XAXIS, mean_stim_all_ON, label='stim. all ON', color=(87/255, 90/255, 187/255))
ax.fill_between(XAXIS, mean_stim_all_ON + sem_stim_all_ON, mean_stim_all_ON - sem_stim_all_ON,
                color=(87/255, 90/255, 187/255), alpha=.15)
_annotate_pvals(ax, pvals_all_ON, base_y=4.05, dy=0.12, star=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
       ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
ax.set_title(f'{exp}\nPyrUp all', fontsize=10)
ax.legend(fontsize=5, frameon=False)
for ext in ['.pdf', '.png']:
    fig.savefig(ctrl_stim_stem / f'{exp}_all_ON{ext}', 
                dpi=300, bbox_inches='tight')

## plot OFF all
fig, ax = plt.subplots(figsize=(2.6, 2))
ax.plot(XAXIS, mean_ctrl_all_OFF, label='ctrl. all OFF', color='purple')
ax.fill_between(XAXIS, mean_ctrl_all_OFF + sem_ctrl_all_OFF, mean_ctrl_all_OFF - sem_ctrl_all_OFF,
                color='purple', alpha=.15)
ax.plot(XAXIS, mean_stim_all_OFF, label='stim. all OFF', color=(78/255, 84/255, 206/255))
ax.fill_between(XAXIS, mean_stim_all_OFF + sem_stim_all_OFF, mean_stim_all_OFF - sem_stim_all_OFF,
                color=(78/255, 84/255, 206/255), alpha=.15)
_annotate_pvals(ax, pvals_all_OFF, base_y=4.05, dy=0.12, star=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
       ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
ax.set_title(f'{exp}\nPyrDown all', fontsize=10)
ax.legend(fontsize=5, frameon=False)
for ext in ['.pdf', '.png']:
    fig.savefig(ctrl_stim_stem / f'{exp}_all_OFF{ext}', 
                dpi=300, bbox_inches='tight')

## plot ON remainers
fig, ax = plt.subplots(figsize=(2.6, 2))
ax.plot(XAXIS, mean_ctrl_remain_ON, label='ctrl. remain ON', color='firebrick')
ax.fill_between(XAXIS, mean_ctrl_remain_ON + sem_ctrl_remain_ON, mean_ctrl_remain_ON - sem_ctrl_remain_ON,
                color='firebrick', alpha=.15)
ax.plot(XAXIS, mean_stim_remain_ON, label='stim. remain ON', color=(87/255, 90/255, 187/255))
ax.fill_between(XAXIS, mean_stim_remain_ON + sem_stim_remain_ON, mean_stim_remain_ON - sem_stim_remain_ON,
                color=(87/255, 90/255, 187/255), alpha=.15)
_annotate_pvals(ax, pvals_remain_ON, base_y=4.05, dy=0.12, star=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
       ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
ax.set_title(f'{exp}\nPyrUp remainers', fontsize=10)
ax.legend(fontsize=5, frameon=False)
for ext in ['.pdf', '.png']:
    fig.savefig(ctrl_stim_stem / f'{exp}_remain_ON{ext}', 
                dpi=300, bbox_inches='tight')

## plot OFF remainers
fig, ax = plt.subplots(figsize=(2.6, 2))
ax.plot(XAXIS, mean_ctrl_remain_OFF, label='ctrl. remain OFF', color='purple')
ax.fill_between(XAXIS, mean_ctrl_remain_OFF + sem_ctrl_remain_OFF, mean_ctrl_remain_OFF - sem_ctrl_remain_OFF,
                color='purple', alpha=.15)
ax.plot(XAXIS, mean_stim_remain_OFF, label='stim. remain OFF', color=(78/255, 84/255, 206/255))
ax.fill_between(XAXIS, mean_stim_remain_OFF + sem_stim_remain_OFF, mean_stim_remain_OFF - sem_stim_remain_OFF,
                color=(78/255, 84/255, 206/255), alpha=.15)
_annotate_pvals(ax, pvals_remain_OFF, base_y=4.05, dy=0.12, star=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
       ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
ax.set_title(f'{exp}\nPyrDown remainers', fontsize=10)
ax.legend(fontsize=5, frameon=False)
for ext in ['.pdf', '.png']:
    fig.savefig(ctrl_stim_stem / f'{exp}_remain_OFF{ext}', 
                dpi=300, bbox_inches='tight')


## regression: Δamp vs behaviour
# ON: Δamp vs distance
filt_amp, filt_dist = [], []
for amp, dist in zip(all_amp_remain_ON_delta_mean, all_ctrl_stim_lick_distance_delta):
    if not np.isnan(amp) and not np.isnan(dist) and -DELTA_THRES < amp < 2:
        filt_amp.append(amp)
        filt_dist.append(dist)
slope, intercept, r, p, _ = linregress(filt_amp, filt_dist)
fig, ax = plt.subplots(figsize=(2.4, 2.2))
ax.scatter(filt_amp, filt_dist, color='firebrick', s=30, alpha=0.8)
ax.plot(np.linspace(min(filt_amp), max(filt_amp), 100),
        intercept + slope*np.linspace(min(filt_amp), max(filt_amp), 100),
        color='k', lw=1)
ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$', transform=ax.transAxes,
        ha='left', va='top', fontsize=9)
ax.set_xlabel('ΔON (Hz)')
ax.set_ylabel('Δlick dist. (stim−ctrl)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
for ext in ['.pdf', '.png']:
    fig.savefig(ctrl_stim_stem / f'{exp}_remain_ON_delta_amp_dist{ext}', 
                dpi=300, bbox_inches='tight')

# ON: Δamp vs time
filt_amp, filt_time = [], []
for amp, time in zip(all_amp_remain_ON_delta_mean, all_ctrl_stim_lick_time_delta):
    if not np.isnan(amp) and not np.isnan(time) and time < 800 and -DELTA_THRES < amp < 2:
        filt_amp.append(amp)
        filt_time.append(time)
slope, intercept, r, p, _ = linregress(filt_amp, filt_time)
fig, ax = plt.subplots(figsize=(2.4, 2.2))
ax.scatter(filt_amp, filt_time, color='firebrick', s=30, alpha=0.8)
ax.plot(np.linspace(min(filt_amp), max(filt_amp), 100),
        intercept + slope*np.linspace(min(filt_amp), max(filt_amp), 100),
        color='k', lw=1)
ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$', transform=ax.transAxes,
        ha='left', va='top', fontsize=9)
ax.set_xlabel('ΔON (Hz)')
ax.set_ylabel('Δlick time (stim−ctrl)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
for ext in ['.pdf', '.png']:
    fig.savefig(ctrl_stim_stem / f'{exp}_remain_ON_delta_amp_time{ext}', 
                dpi=300, bbox_inches='tight')

# OFF: Δamp vs distance
filt_amp, filt_dist = [], []
for amp, dist in zip(all_amp_remain_OFF_delta_mean, all_ctrl_stim_lick_distance_delta):
    if not np.isnan(amp) and not np.isnan(dist) and -2 < amp < DELTA_THRES:
        filt_amp.append(amp)
        filt_dist.append(dist)
slope, intercept, r, p, _ = linregress(filt_amp, filt_dist)
fig, ax = plt.subplots(figsize=(2.4, 2.2))
ax.scatter(filt_amp, filt_dist, color='purple', s=30, alpha=0.8)
ax.plot(np.linspace(min(filt_amp), max(filt_amp), 100),
        intercept + slope*np.linspace(min(filt_amp), max(filt_amp), 100),
        color='k', lw=1)
ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$', transform=ax.transAxes,
        ha='left', va='top', fontsize=9)
ax.set_xlabel('ΔOFF (Hz)')
ax.set_ylabel('Δlick dist. (stim−ctrl)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
for ext in ['.pdf', '.png']:
    fig.savefig(ctrl_stim_stem / f'{exp}_remain_OFF_delta_amp_dist{ext}', 
                dpi=300, bbox_inches='tight')

# OFF: Δamp vs time
filt_amp, filt_time = [], []
for amp, time in zip(all_amp_remain_OFF_delta_mean, all_ctrl_stim_lick_time_delta):
    if not np.isnan(amp) and not np.isnan(time) and -400 < time < 1000 and -2 < amp < DELTA_THRES:
        filt_amp.append(amp)
        filt_time.append(time)
slope, intercept, r, p, _ = linregress(filt_amp, filt_time)
fig, ax = plt.subplots(figsize=(2.4, 2.2))
ax.scatter(filt_amp, filt_time, color='purple', s=30, alpha=0.8)
ax.plot(np.linspace(min(filt_amp), max(filt_amp), 100),
        intercept + slope*np.linspace(min(filt_amp), max(filt_amp), 100),
        color='k', lw=1)
ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$', transform=ax.transAxes,
        ha='left', va='top', fontsize=9)
ax.set_xlabel('ΔOFF (Hz)')
ax.set_ylabel('Δlick time (stim−ctrl)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
for ext in ['.pdf', '.png']:
    fig.savefig(ctrl_stim_stem / f'{exp}_remain_OFF_delta_amp_time{ext}', 
                dpi=300, bbox_inches='tight')