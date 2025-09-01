# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 13:23:54 2025

Modified from LC_code/all_earlyvlate_RO_peak_fixed_threshold.py for processing
    LC axon-GCaMP recordings

@author: Dinghao Luo
"""

#%% imports
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import scipy.io as sio
from scipy.stats import sem, ranksums, ttest_ind

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLCHPCGCaMP


#%% parameters
BEH_SAMP_FREQ = 1000
SAMP_FREQ = 30
RUN_ONSET_BIN = SAMP_FREQ * 3

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
YLIM_SPEED = (0, 65)

# colours
early_c = (0.55, 0.65, 0.95)
late_c  = (0.20, 0.35, 0.65)


#%% helpers (mirror the HPC script)
def compute_bin_speeds_7(trial_indices, speed_times,
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

def get_profiles_and_spike_rates(trains, trials, RO_WINDOW,
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
        try:
            curr_train = trains[trial]
            profiles.append(curr_train[RUN_ONSET_BIN - BEF * SAMP_FREQ : RUN_ONSET_BIN + AFT * SAMP_FREQ])
            spike_rates.append(np.mean(curr_train[RO_WINDOW[0]:RO_WINDOW[1]]))
        except IndexError:
            continue
    return profiles, spike_rates

def _session_mean_speed(trial_list, speed_times, n=3500):
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


#%% load axon properties 
print('loading axon properties...')
axon_prop = pd.read_pickle(
    'Z:/Dinghao/code_dinghao/LCHPC_axon_GCaMP/LCHPC_axon_GCaMP_all_profiles.pkl'
    )

# we only care about the primary axons that have run onset peaks 
primary_axon_prop = axon_prop[
    (axon_prop['constituents'].notna()) &
    (axon_prop['run_onset_peak'] == True)
    ]


#%% containers
# activity
early_profiles = []
late_profiles = []
early_spike_rates = []
late_spike_rates = []

# speed (session-level)
sess_early_speed_means_raw = []   # pre-match
sess_late_speed_means_raw  = []
sess_early_speed_means     = []   # post-match
sess_late_speed_means      = []

for path in paths:
    recname = path.split('\\')[-1]
    print(f'\n{recname}')
    
    # load beh
    beh_path = rf'Z:/Dinghao/code_dinghao/behaviour/all_experiments/LCHPCGCaMP/{recname}.pkl'
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)
        
    # we need to be careful with the trials, since previously we eliminated 
    # trials that were too close to the start and end of the rec
    run_frames = np.asarray(beh['run_onset_frames'])
    valid_mask = (run_frames > 0)
    run_frames = run_frames[valid_mask]
    orig_idx = np.nonzero(valid_mask)[0]
    
    # monotonic ascension
    filtered_frames = []
    filtered_idx = []
    last = -np.inf
    for f, idx in zip(run_frames, orig_idx):
        if f > last:
            filtered_frames.append(f)
            filtered_idx.append(idx)
            last = f
    filtered_frames = np.array(filtered_frames)
    filtered_idx = np.array(filtered_idx)
    
    bef = int(BEF * SAMP_FREQ)
    aft = int(AFT * SAMP_FREQ)
    tot_frames = len(beh['frame_times'])
    
    head = np.searchsorted(filtered_frames, bef, side='left')
    tail = np.searchsorted(filtered_frames, tot_frames - aft, side='right')
    
    # this is used eventually 
    kept_frames = filtered_frames[head:tail]
    kept_trials = filtered_idx[head:tail]
    
    # get licks
    licks = [licks for trial, licks in enumerate(beh['lick_times_aligned'])
             if trial in kept_trials]
    tot_trials = len(licks)
    
    # get first licks
    first_licks = []
    for trial in range(tot_trials):
        curr_lks = licks[trial]
        if not isinstance(curr_lks, list):
            first_licks.append(np.nan)
        else:
            lk = [l for l in curr_lks
                  if l > .5 * BEH_SAMP_FREQ]
            if len(lk) == 0:
                first_licks.append(np.nan)
            else:
                first_licks.append((lk[0]) / BEH_SAMP_FREQ)
    
    # get bad trials 
    bad_trials = beh['bad_trials']
    
    # raw early/late sets
    early_trials, late_trials = [], []
    for trial, t in enumerate(first_licks):
        if t < 2.5:
            early_trials.append(trial)
        elif 2.5 < t < 3.5:
            late_trials.append(trial)

    print(f'found {len(early_trials)} early and {len(late_trials)} late trials (pre-match)')
    if len(early_trials) < 10 or len(late_trials) < 10:  # skip if not enough trials 
        continue
    
    # get keys of RO-peak axons 
    curr_axon_prop = primary_axon_prop[primary_axon_prop['recname'] == recname]
    curr_axon_keys = [s[s.find(' ')+1:]
                      for s in list(curr_axon_prop.index)]
    
    # all dFF
    all_dFF = np.load(
        rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions\{recname}\processed_data\RO_aligned_dict.npy',
        allow_pickle=True
    ).item()

    # behaviour pickle (to get speed_times_aligned)
    # speed_times = [speeds for trial, speeds in enumerate(beh['speed_times_aligned'])
    #                if trial in kept_trials]
    speed_times = beh['speed_times_aligned']

    # PRE-MATCHED session means (if we have speed)
    if speed_times is not None:
        e_mean_sp_raw = _session_mean_speed(early_trials, speed_times, n=3500)
        l_mean_sp_raw = _session_mean_speed(late_trials,  speed_times, n=3500)
        if e_mean_sp_raw is not None and l_mean_sp_raw is not None:
            sess_early_speed_means_raw.append(e_mean_sp_raw)
            sess_late_speed_means_raw.append(l_mean_sp_raw)

    # speed matching
    if speed_times is None:
        print('warning: behaviour pickle with speed_times_aligned not found; skipping speed matching')
        matched_early, matched_late = early_trials, late_trials
    else:
        E_bins, e_valid = compute_bin_speeds_7(early_trials, speed_times)
        L_bins, l_valid = compute_bin_speeds_7(late_trials,  speed_times)

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
        e_mean_sp = _session_mean_speed(matched_early, speed_times, n=3500)
        l_mean_sp = _session_mean_speed(matched_late,  speed_times, n=3500)
        if e_mean_sp is not None and l_mean_sp is not None:
            sess_early_speed_means.append(e_mean_sp)
            sess_late_speed_means.append(l_mean_sp)
            
        # if len(current_matched_early) >= MIN_MATCHED and len(current_matched_late) >= MIN_MATCHED:
        #     print('passed')
            
    # stash for this session
    current_matched_early = matched_early
    current_matched_late  = matched_late

    # per-cluster work (spikes)
    for roi in curr_axon_keys:
        dFF = all_dFF[roi]

        tmp_prof, tmp_rate = get_profiles_and_spike_rates(dFF, early_trials, RO_WINDOW)
        early_profiles.append(np.mean(tmp_prof, axis=0))
        early_spike_rates.extend(tmp_rate)

        tmp_prof, tmp_rate = get_profiles_and_spike_rates(dFF, late_trials, RO_WINDOW)
        late_profiles.append(np.mean(tmp_prof, axis=0))
        late_spike_rates.extend(tmp_rate)
        
        fig, ax = plt.subplots(figsize=(3.5,2))
        ax.plot(early_profiles[-1], label='early', c=early_c)
        ax.plot(late_profiles[-1], label='late', c=late_c)
        
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\first_lick_analysis\single_axon_early_v_late\{recname} {roi}',
            dpi=300,
            bbox_inches='tight'
            )
        
        plt.close()


#%% PLOT: pre-matching session-averaged speed (mean±SEM across sessions)
if len(sess_early_speed_means_raw) and len(sess_late_speed_means_raw):
    E_raw = np.vstack(sess_early_speed_means_raw)
    L_raw = np.vstack(sess_late_speed_means_raw)

    E_raw_mean = np.mean(E_raw, axis=0)
    E_raw_sem  = sem(E_raw, axis=0)
    L_raw_mean = np.mean(L_raw, axis=0)
    L_raw_sem  = sem(L_raw, axis=0)

    fig, ax = plt.subplots(figsize=(2.1, 2.0))
    ax.plot(X_SEC, E_raw_mean, c='grey', label='early (<2.5 s)')
    ax.fill_between(X_SEC, E_raw_mean+E_raw_sem, E_raw_mean-E_raw_sem,
                    color='grey', edgecolor='none', alpha=.25)

    ax.plot(X_SEC, L_raw_mean, c=late_c, label='late (2.5–3.5 s)')
    ax.fill_between(X_SEC, L_raw_mean+L_raw_sem, L_raw_mean-L_raw_sem,
                    color=late_c, edgecolor='none', alpha=.25)

    ax.set(xlabel='time from run onset (s)', xlim=(0, 3.5),
           ylabel='speed (cm/s)', ylim=YLIM_SPEED,
           title='pre-matching speed')
    ax.legend(frameon=False, fontsize=7)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\first_lick_analysis\pre_matching_speed{ext}',
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
    ax.plot(X_SEC, E_mean, c='grey', label='early (<2.5 s)')
    ax.fill_between(X_SEC, E_mean+E_sem, E_mean-E_sem,
                    color='grey', edgecolor='none', alpha=.25)

    ax.plot(X_SEC, L_mean, c=late_c, label='late (2.5–3.5 s)')
    ax.fill_between(X_SEC, L_mean+L_sem, L_mean-L_sem,
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

    ax.set(xlabel='time from run onset (s)', xlim=(0, 3.5),
           ylabel='speed (cm/s)', ylim=YLIM_SPEED,
           title='post-matching speed')
    ax.legend(frameon=False, fontsize=7)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\first_lick_analysis\post_matching_speed{ext}',
            dpi=300,
            bbox_inches='tight'
            )


#%% PLOT: spiking (matched)
XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

paired = [(e, l) for e, l in zip(early_profiles, late_profiles)
          if not (np.isnan(e).all() or np.isnan(l).all())]

e_arr = np.vstack([p[0] for p in paired])
l_arr = np.vstack([p[1] for p in paired])

early_mean = np.nanmean(e_arr, axis=0)
early_sem  = sem(e_arr, axis=0, nan_policy='omit')

late_mean  = np.nanmean(l_arr, axis=0)
late_sem   = sem(l_arr, axis=0, nan_policy='omit')

if early_mean.size and late_mean.size:
    fig, ax = plt.subplots(figsize=(2.2, 2.1))
    ax.plot(XAXIS, early_mean, c='grey', label='early (matched)')
    ax.fill_between(XAXIS, early_mean + early_sem, early_mean - early_sem,
                    color='grey', edgecolor='none', alpha=.25)

    ax.plot(XAXIS, late_mean, c=late_c, label='late (matched)')
    ax.fill_between(XAXIS, late_mean + late_sem, late_mean - late_sem,
                    color=late_c, edgecolor='none', alpha=.25)

    plt.legend(fontsize=7, frameon=False)
    ax.set(xlabel='time from run-onset (s)', xlim=(-1, 4),
           ylabel='spike rate (Hz)')

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
            rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\first_lick_analysis\all_run_onset_mean_profiles_early_v_late{ext}',
            dpi=300,
            bbox_inches='tight'
            )
    
    print(
        f'early mean = {np.mean(a)}, sem = {sem(a)}\n'
        f'late mean = {np.mean(b)}, sem = {sem(b)}\n'
        f't = {stat}\n'
        f'p = {p}'
        )