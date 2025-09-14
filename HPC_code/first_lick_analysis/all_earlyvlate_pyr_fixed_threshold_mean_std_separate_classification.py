# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 16:26:39 2025

Similar to ...fixed_threshold_mean_std.py but classifies PyrUP and PyrDOWN 
    separately for early- and late-1st-lick trials, similar to how ctrl.-stim.
    comparison works for LC-HPC stimulation experiment

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_ind

import support_HPC as support 
from plotting_functions import plot_violin_with_scatter
from common import mpl_formatting
mpl_formatting()

import rec_list
pathHPCLCopt = rec_list.pathHPCLCopt
pathHPCLCtermopt = rec_list.pathHPCLCtermopt

paths = pathHPCLCopt + pathHPCLCtermopt


#%% parameters 
SAMP_FREQ = 1250 
RUN_ONSET_BIN = 3750
BEF = 1  # seconds before run-onset
AFT = 4  # seconds after run-onset

# pre_post ratio thresholds 
run_onset_activated_thres = 2/3
run_onset_inhibited_thres = 3/2

# plotting colours 
early_c = (0.55, 0.65, 0.95)
late_c  = (0.20, 0.35, 0.65)

XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1


#%% path stems
mice_exp_stem = Path('Z:/Dinghao/MiceExp')
beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments')
ephys_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys')


#%% helper 
def _compute_bin_speeds_7(trial_indices, n_bins=7, bin_size=500):
    means = []
    valid = []
    total_len = n_bins * bin_size  # 3500
    for t in trial_indices:
        try:
            sp = [pt[1] for pt in speed_times[t]]
            if len(sp) < total_len:
                continue
            s = np.asarray(sp[:total_len], dtype=float)               # truncate to 0–3500 ms
            m = s.reshape(n_bins, bin_size).mean(axis=1)              # per-bin means (shape: 7,)
            means.append(m)
            valid.append(t)
        except:
            continue
    if len(means) == 0:
        return np.empty((0, n_bins)), []
    return np.vstack(means), valid    

def _get_profiles(trains, trials,
                 RUN_ONSET_BIN=RUN_ONSET_BIN,
                 SAMP_FREQ=SAMP_FREQ,
                 BEF=BEF, AFT=AFT):
    profiles = []
    for trial in trials:
        curr_train = trains[trial]
        profiles.append(curr_train[
            RUN_ONSET_BIN - BEF*SAMP_FREQ : RUN_ONSET_BIN + AFT*SAMP_FREQ
        ])
    return profiles

def _session_mean_speed(trial_list, speed_times, n=4000):
    arrs = []
    for t in trial_list:
        sp = [pt[1] for pt in speed_times[t]]
        if len(sp) >= n:
            arrs.append(np.asarray(sp[:n], dtype=float))
    if len(arrs) == 0:
        return None
    return np.nanmean(np.vstack(arrs), axis=0)

def _trial_bin_means(trial_idx_list, bin_size=500, total_len=3500, n_bins=7):
    out = []
    for t in trial_idx_list:
        sp = [pt[1] for pt in speed_times[t]]
        if len(sp) < total_len:
            continue
        s = np.asarray(sp[:total_len], dtype=float).reshape(n_bins, bin_size).mean(axis=1)
        out.append(s)
    return np.vstack(out) if len(out) else np.empty((0, 7))


#%% main
# ON cells 
early_profiles_ON = []
late_profiles_ON = []

# OFF cells 
early_profiles_OFF = []
late_profiles_OFF = []

# speed after speed matching for each session 
sess_early_speed_means = []
sess_late_speed_means = []

# speed BEFORE matching 
sess_early_speed_means_raw = []
sess_late_speed_means_raw = []

# main loop 
for path in paths:
    recname = Path(path).name
    
    run_path = mice_exp_stem / f'ANMD{recname[1:5]}' / recname[:-3] / recname / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
    alignRun = sio.loadmat(str(run_path))
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]

    beh_par_path = mice_exp_stem / f'ANMD{recname[1:5]}' / recname[:-3] / recname / f'{recname}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
    behPar = sio.loadmat(str(beh_par_path))
    bad_idx = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    good_idx = [t for t in range(tot_trial) if t not in bad_idx]

    beh_path = beh_stem / 'HPCLC' / f'{recname}.pkl' if path in pathHPCLCopt else beh_stem / 'HPCLCterm' / f'{recname}.pkl'
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)

    # behaviour parameters
    
    baseline_idx, stim_idx, ctrl_idx = support.get_trialtype_idx_MATLAB(beh_par_path)
    baseline_idx = baseline_idx[:-1]

    info_path = mice_exp_stem / f'ANMD{recname[1:5]}' / recname[:-3] / recname / f'{recname}_DataStructure_mazeSection1_TrialType1_Info.mat'
    beh_MATLAB = sio.loadmat(info_path)
    if 4 in beh_MATLAB['beh']['pulseMethod'][0][0]:  # reward stim should affect the next trial 
        pass
    else:
        stim_idx = [t - 1 for t in stim_idx]  # since we skipped trial 1 when extracting trains 
        ctrl_idx = [t - 1 for t in ctrl_idx]

    # get first-lick time
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] 
              if l-starts[trial] > .5*SAMP_FREQ]  # only if the animal does not lick in the first half a second (carry-over licks)
        
        if len(lk)==0:  # no licks in the current trial
            first_licks.append(np.nan)
        else:  # if there are licks, append relative time of first lick
            first_licks.extend(lk[0]-starts[trial])
    
    # convert first licks to seconds
    first_licks_sec = np.array(first_licks) / SAMP_FREQ

    early_trials = []
    late_trials = []
    for trial, t in enumerate(first_licks_sec):
        if trial in bad_idx or np.isnan(t): continue
        if t < 2.5:
            early_trials.append(trial)
        elif 2.5 < t < 3.5:
            late_trials.append(trial)

    if len(early_trials) < 5 or len(late_trials) < 5:
        continue
    
    speed_times = beh['speed_times_aligned'][1:]
    
    # speed BEFORE matching for visualisation 
    e_mean_sp_raw = _session_mean_speed(early_trials, speed_times, n=3500)
    l_mean_sp_raw = _session_mean_speed(late_trials,  speed_times, n=3500)
    if e_mean_sp_raw is not None and l_mean_sp_raw is not None:
        sess_early_speed_means_raw.append(e_mean_sp_raw)
        sess_late_speed_means_raw.append(l_mean_sp_raw)
    
    # speed matching begins here 
    matched_early = []
    matched_late = []
    
    # 7-bin speed extraction (0–3500 ms)
    E_bins, e_valid = _compute_bin_speeds_7(early_trials)  # shape: (nE, 7)
    L_bins, l_valid = _compute_bin_speeds_7(late_trials)   # shape: (nL, 7)
    
    matched_early, matched_late = [], []
    if len(E_bins) and len(L_bins):
        k = 1.5  # use ±1.5 SD for tighter fit 
    
        # early stats -> bounds that late must satisfy
        e_mu = E_bins.mean(axis=0)
        e_sd = E_bins.std(axis=0, ddof=0)
        e_low, e_high = e_mu - k*e_sd, e_mu + k*e_sd
    
        # late stats -> bounds that early must satisfy
        l_mu = L_bins.mean(axis=0)
        l_sd = L_bins.std(axis=0, ddof=0)
        l_low, l_high = l_mu - k*l_sd, l_mu + k*l_sd
    
        # masks
        l_mask_in_early_bounds = np.all((L_bins >= e_low) & (L_bins <= e_high), axis=1)
        e_mask_in_late_bounds  = np.all((E_bins >= l_low) & (E_bins <= l_high), axis=1)
    
        # mutually matched sets (symmetric)
        matched_late  = [l_valid[i] for i in np.where(l_mask_in_early_bounds)[0]]
        matched_early = [e_valid[i] for i in np.where(e_mask_in_late_bounds)[0]]
    else:
        matched_early, matched_late = [], []
    
    print(f'\n{recname}')
    print(f'{len(matched_early)} early and {len(matched_late)} late trials passed 7-bin speed filtering')
    
    # single unit work 
    all_trains_path = ephys_stem / 'all_sessions' / recname / f'{recname}_all_trains.npy'
    all_trains = np.load(all_trains_path, allow_pickle=True).item()
    
    # get pyr and int ID's and corresponding spike rates
    cell_identities, spike_rates = support.get_cell_info(info_path)
    
    curr_early_ON = []
    curr_late_ON = []
    curr_early_OFF = []
    curr_late_OFF = []
    
    for i, clu in enumerate(all_trains):
        if cell_identities[i] == 'int':  # interneurone filter
            continue 
        if spike_rates[i]<0.15 or spike_rates[i]>7:  # spike rate filter
            continue 
        
        trains = all_trains[clu]
        
        early_trains_mean = np.mean(trains[matched_early, :], axis=0)
        late_trains_mean  = np.mean(trains[matched_late, :], axis=0)
        
        early_ratio, early_type = support.classify_run_onset_activation_ratio(
            early_trains_mean, run_onset_activated_thres, run_onset_inhibited_thres
            )
        late_ratio, late_type   = support.classify_run_onset_activation_ratio(
            late_trains_mean, run_onset_activated_thres, run_onset_inhibited_thres
            )

        if early_type == 'run-onset ON':
            early_trains = _get_profiles(trains, matched_early)
            early_profiles_ON.extend(early_trains)
            curr_early_ON.extend(early_trains)
        if early_type == 'run-onset OFF':
            early_trains = _get_profiles(trains, matched_early)
            early_profiles_OFF.extend(early_trains)
            curr_early_OFF.extend(early_trains)
            
        if late_type == 'run-onset ON':
            late_trains  = _get_profiles(trains, matched_late)
            late_profiles_ON.extend(late_trains)
            curr_late_ON.extend(late_trains)
        if late_type == 'run-onset OFF':
            late_trains  = _get_profiles(trains, matched_late)
            late_profiles_OFF.extend(late_trains)
            curr_late_OFF.extend(late_trains)
        
    # single session plotting 
    if curr_early_ON:
        curr_early_mean = np.mean(curr_early_ON, axis=0)
        curr_early_sem = sem(curr_early_ON, axis=0)

        curr_late_mean = np.mean(curr_late_ON, axis=0)
        curr_late_sem = sem(curr_late_ON, axis=0)

        fig, ax = plt.subplots(figsize=(2.3, 2.0))
        ax.plot(XAXIS, curr_early_mean, c='grey', label='<2.5')
        ax.fill_between(XAXIS, curr_early_mean+curr_early_sem, curr_early_mean-curr_early_sem,
                        color='grey', edgecolor='none', alpha=.25)
        ax.plot(XAXIS, curr_late_mean, c=late_c, label='2.5~3.5')
        ax.fill_between(XAXIS, curr_late_mean+curr_late_sem, curr_late_mean-curr_late_sem,
                        color=late_c, edgecolor='none', alpha=.25)

        ax.legend(fontsize=5, frameon=False)
        ax.set(xlabel='Time from run onset (s)', xlim=(-1, 4),
               ylabel='Firing rate (Hz)')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)

        fig.tight_layout()
        plt.show()

        for ext in ['.png', '.pdf']:
            fig.savefig(
                ephys_stem / 'first_lick_analysis' / 'all_sessions_separate_identification' / f'{recname}_run_onset_ON_mean_profiles{ext}',
                dpi=300,
                bbox_inches='tight'
                )
    
        
    if curr_early_OFF:
        curr_early_mean = np.mean(curr_early_OFF, axis=0)
        curr_early_sem = sem(curr_early_OFF, axis=0)

        curr_late_mean = np.mean(curr_late_OFF, axis=0)
        curr_late_sem = sem(curr_late_OFF, axis=0)
        
        fig, ax = plt.subplots(figsize=(2.3, 2.0))
        ax.plot(XAXIS, curr_early_mean, c='grey', label='<2.5')
        ax.fill_between(XAXIS, curr_early_mean+curr_early_sem, curr_early_mean-curr_early_sem,
                        color='grey', edgecolor='none', alpha=.25)
        ax.plot(XAXIS, curr_late_mean, c=late_c, label='2.5~3.5')
        ax.fill_between(XAXIS, curr_late_mean+curr_late_sem, curr_late_mean-curr_late_sem,
                        color=late_c, edgecolor='none', alpha=.25)

        ax.legend(fontsize=5, frameon=False)
        ax.set(xlabel='Time from run onset (s)', xlim=(-1, 4),
               ylabel='Firing rate (Hz)')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)

        fig.tight_layout()
        plt.show()

        for ext in ['.png', '.pdf']:
            fig.savefig(
                ephys_stem / 'first_lick_analysis' / 'all_sessions_separate_identification' / f'{recname}_run_onset_OFF_mean_profiles{ext}',
                dpi=300,
                bbox_inches='tight'
                )


#%% mean spiking curves (ON) for early v late 
# stat
early_win_means = np.mean(np.array(early_profiles_ON)[:, 1250+625:1250+1825], axis=1)
late_win_means  = np.mean(np.array(late_profiles_ON)[:, 1250+625:1250+1825], axis=1)

# paired t-test
t_stat, p_val = ttest_ind(early_win_means, late_win_means, nan_policy='omit')

print(f'early mean = {np.mean(early_win_means)}, sem = {sem(early_win_means)}\n')
print(f'late mean = {np.mean(late_win_means)}, sem = {sem(late_win_means)}\n')
print(f'Paired t-test (0.5–1.5 s window): t = {t_stat:}, p = {p_val}')

XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

early_mean = np.mean(early_profiles_ON, axis=0)
early_sem = sem(early_profiles_ON, axis=0)

late_mean = np.mean(late_profiles_ON, axis=0)
late_sem = sem(late_profiles_ON, axis=0)

early_c = (0.55, 0.65, 0.95)
late_c  = (0.20, 0.35, 0.65)

fig, ax = plt.subplots(figsize=(2.3, 2.0))
ax.plot(XAXIS, early_mean, c='grey', label='<2.5')
ax.fill_between(XAXIS, early_mean+early_sem, early_mean-early_sem,
                color='grey', edgecolor='none', alpha=.25)
ax.plot(XAXIS, late_mean, c=late_c, label='2.5~3.5')
ax.fill_between(XAXIS, late_mean+late_sem, late_mean-late_sem,
                color=late_c, edgecolor='none', alpha=.25)

ax.hlines(3.5, .5, 1.5, color='k', lw=1)
ax.text((.5 + 1.5)/2, 3.5, f'p={p_val:.6f}',
        ha='center', va='bottom', fontsize=5)

ax.legend(fontsize=5, frameon=False)
ax.set(xlabel='time from run-onset (s)', xlim=(-1, 4),
       ylabel='spike rate (Hz)')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\all_run_onset_ON_mean_profiles_separate_identification{ext}',
                dpi=300,
                bbox_inches='tight')
    

#%% mean spiking curves (OFF) for early v late 
# stat
early_win_means = np.mean(np.array(early_profiles_OFF)[:, 1250+625:1250+1825], axis=1)
late_win_means  = np.mean(np.array(late_profiles_OFF)[:, 1250+625:1250+1825], axis=1)

# paired t-test
t_stat, p_val = ttest_ind(early_win_means, late_win_means, nan_policy='omit')

print(f'early mean = {np.mean(early_win_means)}, sem = {sem(early_win_means)}\n')
print(f'late mean = {np.mean(late_win_means)}, sem = {sem(late_win_means)}\n')
print(f'Paired t-test (0.5–1.5 s window): t = {t_stat:.3f}, p = {p_val:.4f}')

XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

early_mean = np.mean(early_profiles_OFF, axis=0)
early_sem = sem(early_profiles_OFF, axis=0)

late_mean = np.mean(late_profiles_OFF, axis=0)
late_sem = sem(late_profiles_OFF, axis=0)

early_c = (0.55, 0.65, 0.95)
late_c  = (0.20, 0.35, 0.65)

fig, ax = plt.subplots(figsize=(2.3, 2.0))
ax.plot(XAXIS, early_mean, c='grey', label='<2.5')
ax.fill_between(XAXIS, early_mean+early_sem, early_mean-early_sem,
                color='grey', edgecolor='none', alpha=.25)
ax.plot(XAXIS, late_mean, c=late_c, label='2.5~3.5')
ax.fill_between(XAXIS, late_mean+late_sem, late_mean-late_sem,
                color=late_c, edgecolor='none', alpha=.25)

ax.hlines(2.5, .5, 1.5, color='k', lw=1)
ax.text((.5 + 1.5)/2, 2.5, f'p={p_val:.6f}',
        ha='center', va='bottom', fontsize=5)

ax.legend(fontsize=5, frameon=False)
ax.set(xlabel='time from run-onset (s)', xlim=(-1, 4),
       ylabel='spike rate (Hz)')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\all_run_onset_OFF_mean_profiles_separate_identification{ext}',
                dpi=300,
                bbox_inches='tight')