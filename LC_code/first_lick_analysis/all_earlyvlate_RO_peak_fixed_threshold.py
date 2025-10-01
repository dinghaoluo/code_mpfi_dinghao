# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:37:16 2023
Modified on Tue 22 Apr 2025:
    - reworking the script to calculate multiple other factors other than 
      peak amplitude
Modified on Thu Aug 14 2025:
    - added symmetric 7-bin (0–3.5 s) speed matching (±1.5 SD) identical to HPC script
      and applied matched trial indices for early vs late comparisons

loop over all cells for early v late trials

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import scipy.io as sio
from scipy.stats import sem, ranksums, ttest_ind

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathLC
pathnames = [p.split('\\')[-1] for p in paths]


#%% parameters
SAMP_FREQ = 1250
RUN_ONSET_BIN = 3750
BEF = 1   # s before run-onset
AFT = 4   # s after run-onset
WINDOW_HALF_SIZE = .5
RO_WINDOW = [
    int(RUN_ONSET_BIN - WINDOW_HALF_SIZE * SAMP_FREQ),
    int(RUN_ONSET_BIN + WINDOW_HALF_SIZE * SAMP_FREQ)
]

# matching params (mirror HPC script)
BIN_SIZE_MS = 500
TOTAL_LEN_MS = 3500
N_BINS = TOTAL_LEN_MS // BIN_SIZE_MS  # 7
MATCH_K = 1.5
MIN_MATCHED = 5

# speed plotting params
X_SEC = np.arange(3500) / 1000.0
X_SEC_PLOT = np.arange(4000) / 1000.0

YLIM_SPEED = (0, 70)

# colours
early_c = (168/255, 155/255, 202/255)
late_c  = (102/255, 83/255 , 162/255)


#%% helpers (mirror the HPC script)
def _compute_bin_speeds_7(trial_indices, speed_times,
                         n_bins=N_BINS, bin_size=BIN_SIZE_MS):
    """
    make a (n_trials x 7) matrix of mean speeds per 500 ms bin over 0–3500 ms
    """
    means = []
    valid = []
    total_len = n_bins * bin_size  # 3500 ms
    for t in trial_indices:
        try:
            sp = [pt[1] for pt in speed_times[t]]
            if len(sp) < total_len:
                continue
            s = np.asarray(sp[:total_len], dtype=float)
            m = s.reshape(n_bins, bin_size).mean(axis=1)
            means.append(m)
            valid.append(t)
        except Exception:
            continue
    if not means:
        return np.empty((0, n_bins)), []
    return np.vstack(means), valid

def _trial_bin_means(trial_idx_list, speed_times,
                     bin_size=BIN_SIZE_MS,
                     total_len=TOTAL_LEN_MS,
                     n_bins=N_BINS):
    """
    return (n_trials x 7) matrix of binned speeds (for quick stats/plots)
    """
    out = []
    for t in trial_idx_list:
        sp = [pt[1] for pt in speed_times[t]]
        if len(sp) < total_len:
            continue
        s = np.asarray(sp[:total_len], dtype=float).reshape(n_bins, bin_size).mean(axis=1)
        out.append(s)
    return np.vstack(out) if out else np.empty((0, n_bins))

def _get_profiles_and_spike_rates(trains, trials, RO_WINDOW,
                                 RUN_ONSET_BIN=RUN_ONSET_BIN,
                                 SAMP_FREQ=SAMP_FREQ,
                                 BEF=BEF, AFT=AFT):
    """
    extract peri-run-onset spike profiles and spike rates for a list of trials

    parameters:
    - trains: np.ndarray, shape (n_trials, n_timepoints), spike train array
    - trials: list[int], trial indices to include
    - RO_WINDOW: [int, int], index range for spike rate calculation around run-onset
    - RUN_ONSET_BIN: int
    - SAMP_FREQ: int
    - BEF: float
    - AFT: float

    returns:
    - profiles: list[np.ndarray], peri-run-onset spike profiles
    - spike_rates: list[float], mean spike rate within RO_WINDOW
    """
    profiles = []
    spike_rates = []
    for trial in trials:
        curr_train = trains[trial]
        profiles.append(curr_train[RUN_ONSET_BIN - BEF * SAMP_FREQ : RUN_ONSET_BIN + AFT * SAMP_FREQ])
        spike_rates.append(np.mean(curr_train[RO_WINDOW[0]:RO_WINDOW[1]]))
    return profiles, spike_rates

def _session_mean_speed(trial_list, speed_times, n=4000):
    """
    average speed trace over a set of trials for a single session (0–n ms)
    """
    arrs = []
    for t in trial_list:
        sp = [pt[1] for pt in speed_times[t]]
        if len(sp) >= n:
            arrs.append(np.asarray(sp[:n], dtype=float))
    if not arrs:
        return None
    return np.nanmean(np.vstack(arrs), axis=0)


#%% load cell table
print('loading data...')
cell_prop = pd.read_pickle(r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl')

# keys
clu_keys = list(cell_prop.index)
tagged_keys, putative_keys = [], []
tagged_RO_keys, putative_RO_keys = [], []
RO_keys = []
for clu in cell_prop.itertuples():
    if clu.identity == 'tagged':
        tagged_keys.append(clu.Index)
        if clu.run_onset_peak:
            tagged_RO_keys.append(clu.Index)
            RO_keys.append(clu.Index)
    if clu.identity == 'putative':
        putative_keys.append(clu.Index)
        if clu.run_onset_peak:
            putative_RO_keys.append(clu.Index)
            RO_keys.append(clu.Index)


#%% containers
# spiking
early_profiles = []
late_profiles = []
early_spike_rates = []
late_spike_rates = []

# speed (session-level)
sess_early_speed_means_raw = []   # pre-match
sess_late_speed_means_raw  = []
sess_early_speed_means     = []   # post-match
sess_late_speed_means      = []

recname = ''

for cluname in RO_keys:
    temp_recname = cluname.split(' ')[0]
    
    if temp_recname not in pathnames:
        continue
    
    if temp_recname != recname:
        recname = temp_recname
        
        print(f'\n{recname}')

        # load alignRun + behaviour flags
        alignRun = sio.loadmat(
            rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:-3]}\{recname}\{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
        )
        licks  = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
        starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
        tot_trial = licks.shape[0]

        behPar = sio.loadmat(
            rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:-3]}\{recname}\{recname}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
        )
        bad_idx = np.where(behPar['behPar'][0]['indTrBadBeh'][0] == 1)[1] - 1
        stim_idx = np.where(behPar['behPar'][0]['stimOn'][0] == 1)[1] - 1

        # first-lick time (append scalar, not extend), already in seconds
        first_licks = []
        for trial in range(tot_trial):
            lk = [l for l in licks[trial] if l - starts[trial] > .5 * SAMP_FREQ]
            if len(lk) == 0:
                first_licks.append(np.nan)
            else:
                first_licks.append((lk[0] - starts[trial]) / SAMP_FREQ)

        # raw early/late sets
        early_trials, late_trials = [], []
        for trial, t in enumerate(first_licks):
            if trial in bad_idx or trial in stim_idx or np.isnan(t):
                continue
            if t < 2.5:
                early_trials.append(trial)
            elif 2.5 < t < 3.5:
                late_trials.append(trial)

        print(f'found {len(early_trials)} early and {len(late_trials)} late trials (pre-match)')

        # trains
        all_trains = np.load(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}\{recname}_all_trains.npy',
            allow_pickle=True
        ).item()

        # behaviour pickle (to get speed_times_aligned)
        speed_times = None
        beh_try_paths = [
            Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC') / f'{recname}.pkl',
            Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LCterm') / f'{recname}.pkl',
        ]
        for beh_path in beh_try_paths:
            if beh_path.exists():
                with open(beh_path, 'rb') as f:
                    beh = pickle.load(f)
                speed_times = beh['speed_times_aligned'][1:]
                break

        # PRE-MATCHED session means (if we have speed)
        if speed_times is not None:
            e_mean_sp_raw = _session_mean_speed(early_trials, speed_times, n=4000)
            l_mean_sp_raw = _session_mean_speed(late_trials,  speed_times, n=4000)
            if e_mean_sp_raw is not None and l_mean_sp_raw is not None:
                sess_early_speed_means_raw.append(e_mean_sp_raw)
                sess_late_speed_means_raw.append(l_mean_sp_raw)

        # speed matching
        if speed_times is None:
            print('warning: behaviour pickle with speed_times_aligned not found; skipping speed matching')
            matched_early, matched_late = early_trials, late_trials
        else:
            E_bins, e_valid = _compute_bin_speeds_7(early_trials, speed_times)
            L_bins, l_valid = _compute_bin_speeds_7(late_trials,  speed_times)

            matched_early, matched_late = [], []
            if len(E_bins) and len(L_bins):
                e_mu = E_bins.mean(axis=0); e_sd = E_bins.std(axis=0, ddof=0)
                l_mu = L_bins.mean(axis=0); l_sd = L_bins.std(axis=0, ddof=0)

                e_low, e_high = e_mu - MATCH_K * e_sd, e_mu + MATCH_K * e_sd
                l_low, l_high = l_mu - MATCH_K * l_sd, l_mu + MATCH_K * l_sd

                l_mask_in_e = np.all((L_bins >= e_low) & (L_bins <= e_high), axis=1)
                e_mask_in_l = np.all((E_bins >= l_low) & (E_bins <= l_high), axis=1)

                matched_late  = [l_valid[i] for i in np.where(l_mask_in_e)[0]]
                matched_early = [e_valid[i] for i in np.where(e_mask_in_l)[0]]
            else:
                matched_early, matched_late = [], []

            print(f'{len(matched_early)} early and {len(matched_late)} late trials passed 7-bin speed filtering')

            # POST-MATCHED session means
            e_mean_sp = _session_mean_speed(matched_early, speed_times, n=4000)
            l_mean_sp = _session_mean_speed(matched_late,  speed_times, n=4000)
            if e_mean_sp is not None and l_mean_sp is not None:
                sess_early_speed_means.append(e_mean_sp)
                sess_late_speed_means.append(l_mean_sp)

        # stash for this session
        current_matched_early = matched_early
        current_matched_late  = matched_late
        
        # holder for this session's profiles
        curr_early_sess = []
        curr_late_sess = []

    # per-cluster work (spikes)
    trains = all_trains[cluname]
    if len(current_matched_early) >= MIN_MATCHED and len(current_matched_late) >= MIN_MATCHED:
        tmp_prof, tmp_rate = _get_profiles_and_spike_rates(trains, early_trials, RO_WINDOW)
        early_profiles.append(np.mean(tmp_prof, axis=0))
        early_spike_rates.extend(tmp_rate)
        
        curr_early_sess.append(early_profiles[-1])

        tmp_prof, tmp_rate = _get_profiles_and_spike_rates(trains, late_trials, RO_WINDOW)
        late_profiles.append(np.mean(tmp_prof, axis=0))
        late_spike_rates.extend(tmp_rate)
        
        curr_late_sess.append(late_profiles[-1])
        
        fig, ax = plt.subplots(figsize=(3.5,2))
        ax.plot(early_profiles[-1], label='early', c=early_c)
        ax.plot(late_profiles[-1], label='late', c=late_c)
        
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\single_cell_early_v_late\{cluname}.png',
            dpi=300,
            bbox_inches='tight'
            )
        
        plt.close()
        
    # curr sess profiles 
    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax.plot(np.mean(curr_early_sess, axis=0), label='early', c=early_c)
    ax.plot(np.mean(curr_late_sess, axis=0), label='late', c=late_c)
    
    ax.set(title=recname)
    
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\single_sess_early_v_late\{recname}.png',
        dpi=300,
        bbox_inches='tight'
        )


#%% PLOT: pre-matching session-averaged speed (mean±SEM across sessions)
if len(sess_early_speed_means_raw) and len(sess_late_speed_means_raw):
    E_raw = np.vstack(sess_early_speed_means_raw)
    L_raw = np.vstack(sess_late_speed_means_raw)

    E_raw_mean = np.mean(E_raw, axis=0)
    E_raw_sem  = sem(E_raw, axis=0)
    L_raw_mean = np.mean(L_raw, axis=0)
    L_raw_sem  = sem(L_raw, axis=0)

    fig, ax = plt.subplots(figsize=(2.1, 2.0))
    ax.plot(X_SEC_PLOT, E_raw_mean, c=early_c, label='early (<2.5 s)')
    ax.fill_between(X_SEC_PLOT, E_raw_mean+E_raw_sem, E_raw_mean-E_raw_sem,
                    color=early_c, edgecolor='none', alpha=.25)
    
    ax.plot(X_SEC_PLOT, L_raw_mean, c=late_c, label='late (2.5–3.5 s)')
    ax.fill_between(X_SEC_PLOT, L_raw_mean+L_raw_sem, L_raw_mean-L_raw_sem,
                    color=late_c, edgecolor='none', alpha=.25)
    
    ax.set(xlabel='Time from run onset (s)', xlim=(0, 4),
           ylabel='Speed (cm/s)', ylim=YLIM_SPEED,
           title='Pre-matching speed')
    ax.legend(frameon=False, fontsize=7)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\pre_matching_speed{ext}',
            dpi=300,
            bbox_inches='tight'
            )


#%% PLOT: post-matching session-averaged speed (mean±SEM across sessions)
if len(sess_early_speed_means) and len(sess_late_speed_means):
    E = np.vstack(sess_early_speed_means)
    L = np.vstack(sess_late_speed_means)

    E_mean = np.mean(E, axis=0)
    E_sem  = sem(E, axis=0)
    L_mean = np.mean(L, axis=0)
    L_sem  = sem(L, axis=0)

    fig, ax = plt.subplots(figsize=(2.1, 2.0))
    ax.plot(X_SEC_PLOT, E_mean, c=early_c, label='early (<2.5 s)')
    ax.fill_between(X_SEC_PLOT, E_mean+E_sem, E_mean-E_sem,
                    color=early_c, edgecolor='none', alpha=.25)

    ax.plot(X_SEC_PLOT, L_mean, c=late_c, label='late (2.5–3.5 s)')
    ax.fill_between(X_SEC_PLOT, L_mean+L_sem, L_mean-L_sem,
                    color=late_c, edgecolor='none', alpha=.25)

    # optional per-bin independent Welch t-tests (500 ms bins)
    n_bins = 7
    bin_size = 500  # ms
    E_bins = np.vstack([E[:, i*bin_size:(i+1)*bin_size].mean(axis=1) for i in range(n_bins)]).T
    L_bins = np.vstack([L[:, i*bin_size:(i+1)*bin_size].mean(axis=1) for i in range(n_bins)]).T

    pvals = np.ones(n_bins)
    for i in range(n_bins):
        if np.sum(np.isfinite(E_bins[:, i])) >= 2 and np.sum(np.isfinite(L_bins[:, i])) >= 2:
            _, p = ttest_ind(E_bins[:, i], L_bins[:, i], equal_var=False, nan_policy='omit')
            pvals[i] = p

    ymax = max((E_mean+E_sem).max(), (L_mean+L_sem).max())
    ymin = min((E_mean-E_sem).min(), (L_mean-L_sem).min())
    yr = ymax - ymin if ymax > ymin else 1.0
    bar_y  = ymax + 0.06 * yr
    text_y = ymax + 0.11 * yr
    for i in range(n_bins):
        x_left  = i * 0.5 + 0.1
        x_right = (i + 1) * 0.5 - 0.1
        ax.hlines(bar_y, x_left, x_right, color='k', lw=1)
        ax.text((x_left + x_right)/2, text_y, f'p={pvals[i]:.3f}',
                ha='center', va='bottom', fontsize=5)

    ax.set(xlabel='Time from run onset (s)', xlim=(0, 4),
           ylabel='Speed (cm/s)', ylim=YLIM_SPEED,
           title='Post-matching speed')
    ax.legend(frameon=False, fontsize=7)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\post_matching_speed{ext}',
            dpi=300,
            bbox_inches='tight'
            )


#%% PLOT: spiking (matched)
XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

early_mean = np.mean(early_profiles, axis=0) if len(early_profiles) else np.array([])
early_sem  = sem(early_profiles, axis=0)     if len(early_profiles) else np.array([])
late_mean  = np.mean(late_profiles, axis=0)  if len(late_profiles)  else np.array([])
late_sem   = sem(late_profiles, axis=0)      if len(late_profiles)  else np.array([])

if early_mean.size and late_mean.size:
    fig, ax = plt.subplots(figsize=(2.2, 2.1))
    ax.plot(XAXIS, early_mean, c=early_c, label='early (matched)')
    ax.fill_between(XAXIS, early_mean + early_sem, early_mean - early_sem,
                    color=early_c, edgecolor='none', alpha=.25)

    ax.plot(XAXIS, late_mean, c=late_c, label='late (matched)')
    ax.fill_between(XAXIS, late_mean + late_sem, late_mean - late_sem,
                    color=late_c, edgecolor='none', alpha=.25)

    plt.legend(fontsize=7, frameon=False)
    ax.set(xlabel='Time from run-onset (s)', xlim=(-1, 4),
           ylabel='Firing rate (Hz)')

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    # stats on ROI window using matched trials
    a = [x for x in early_spike_rates if not np.isnan(x)]
    b = [x for x in late_spike_rates if not np.isnan(x)]
    if len(a) and len(b):
        stat, p = ranksums(a, b)
        p_str = 'p<1e-4' if p < 1e-4 else f'p={p:.5g}'
        ax.set_title('\n' + f'early vs late (matched): {p_str}', fontsize=6)

    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\all_run_onset_mean_profiles_early_v_late{ext}',
            dpi=300,
            bbox_inches='tight'
            )
    
    print(
        f'early mean = {np.mean(a)}, sem = {sem(a)}\n'
        f'late mean = {np.mean(b)}, sem = {sem(b)}\n'
        f't = {stat}\n'
        f'p = {p}'
        )