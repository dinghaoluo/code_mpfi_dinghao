# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:56:30 2025
Modified on Fri 29 Aug 2025

replicate what we have done on LC run onset-peaking cells with HPC recordings
modified to use on Raphi's recordings 

temporary script due to cell profiles not saved 2 Sept 2025

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

import plotting_functions as pf
from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathHPC_Raphi
mazes = rec_list.pathHPC_Raphi_maze_sess


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
beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/HPCRaphi')
first_lick_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/first_lick_analysis_raphi')


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
            s = np.asarray(sp[:total_len], dtype=float)
            m = s.reshape(n_bins, bin_size).mean(axis=1)
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
for i, path in enumerate(paths):
    recname = path.split('\\')[-1]
    print(f'\n{recname}')
    
    alignRun = sio.loadmat(
        rf'Z:\Raphael_tests\mice_expdata\ANM{recname[1:4]}\{recname[:-3]}'
        rf'\{recname}\{recname}_DataStructure_mazeSection1_'
        rf'TrialType1_alignRun_msess{mazes[i]}.mat'
    )
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]

    behPar = sio.loadmat(
        rf'Z:\Raphael_tests\mice_expdata\ANM{recname[1:4]}\{recname[:-3]}'
        rf'\{recname}\{recname}_DataStructure_mazeSection1_'
        rf'TrialType1_behPar_msess{mazes[i]}.mat'
    )
    bad_idx = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    good_idx = [t for t in range(tot_trial) if t not in bad_idx]

    beh_path = beh_stem / f'{recname}.pkl'
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)

    # behaviour parameters
    (
        baseline_idx,
        stim_idx, 
        ctrl_idx
    ) = support.get_trialtype_idx_MATLAB(
        rf'{path}\{recname}_DataStructure_mazeSection1_TrialType1_behPar_msess{mazes[i]}.mat'
        )
    baseline_idx = baseline_idx[:-1]

    beh_MATLAB = sio.loadmat(rf'{path}\{recname}_DataStructure_mazeSection1_TrialType1_Info.mat')
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
        if trial in bad_idx or trial in stim_idx or np.isnan(t): continue
        if t < 2.5:
            early_trials.append(trial)
        elif 2.5 < t < 3.5:
            late_trials.append(trial)

    print(f'found {len(early_trials)} early trials, {len(late_trials)} late trials')

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
    
    if len(early_trials) < 10 or len(late_trials) < 10:
        print('skipped')
        continue

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
    
    print(f'{len(matched_early)} early and {len(matched_late)} late trials passed 7-bin speed filtering')
    if len(matched_early) < 10 or len(matched_late) < 10:
        print('skipped')
        continue

    # collect session means for later plotting 
    e_mean_sp = _session_mean_speed(matched_early, speed_times, n=3500)
    l_mean_sp = _session_mean_speed(matched_late, speed_times, n=3500)
    
    if e_mean_sp is not None and l_mean_sp is not None:
        sess_early_speed_means.append(e_mean_sp)
        sess_late_speed_means.append(l_mean_sp)
    
    # plot trace examples + per-bin independent t-tests (500 ms, 7 bins)
    fig, ax = plt.subplots(figsize=(2.8, 1.6))
    
    time_sec = np.arange(3500) / 1000.0
    for i in range(min(10, len(matched_early))):
        sp = [pt[1] for pt in speed_times[matched_early[i]]]
        ax.plot(time_sec, sp[:3500], color='grey', alpha=0.5)
    for i in range(min(10, len(matched_late))):
        sp = [pt[1] for pt in speed_times[matched_late[i]]]
        ax.plot(time_sec, sp[:3500], color=(0.2, 0.35, 0.65), alpha=0.5)
    
    E_b = _trial_bin_means(matched_early)  # shape: nE x 7
    L_b = _trial_bin_means(matched_late)   # shape: nL x 7
    
    # per-bin Welch t-test (independent, unequal variances)
    n_bins = 7
    bin_size = 500
    pvals = np.ones(n_bins)
    for i in range(n_bins):
        e = E_b[:, i] if E_b.size else np.array([])
        l = L_b[:, i] if L_b.size else np.array([])
        if e.size >= 2 and l.size >= 2:
            t, p = ttest_ind(e, l, equal_var=False, nan_policy='omit')
            pvals[i] = p
    
    # annotate bars + p-values above each 500 ms bin
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    bar_y  = ymax + 0.06 * yr
    text_y = ymax + 0.11 * yr
    
    for i in range(n_bins):
        x_left  = i * bin_size / 1000 + .1
        x_right = (i + 1) * bin_size / 1000 - .1
        ax.hlines(bar_y, x_left, x_right, color='k', lw=1)
        ax.text((x_left + x_right) / 2.0, text_y, f'p={pvals[i]:.3f}',
                ha='center', va='bottom', fontsize=5)
    
    ax.set(xlabel='time (s)', ylabel='speed (cm/s)', title=f'{recname} bin-filtered')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    
    vis_path = first_lick_stem/ 'single_session_speed_matching' / f'{recname}_bin_filtered_speed_traces'
    for ext in ['.pdf', '.png']:
        fig.savefig(f'{vis_path}{ext}', 
                    dpi=200)
    plt.close(fig)
    
    
    # single unit work 
    all_trains = np.load(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions_raphi\{recname}'
        rf'\{recname}_all_trains.npy',
        allow_pickle=True
    ).item()
    
    # get pyr and int ID's and corresponding spike rates
    cell_identities, spike_rates = support.get_cell_info(
        rf'{path}\{recname}_DataStructure_mazeSection1_TrialType1_Info.mat'
        )
    
    
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
        
        baseline_train = np.nanmean(trains[baseline_idx], axis=0)
        
        ratio, ratiotype = support.classify_run_onset_activation_ratio(
            baseline_train, run_onset_activated_thres, run_onset_inhibited_thres
            )

        if ratiotype == 'run-onset ON':
            temp_early = _get_profiles(trains, matched_early)
            temp_late = _get_profiles(trains, matched_late)
            
            if int(recname[1:4]) > 40:  # for some reason some sessions did not get scaled with SAMP_FREQ when extracting
                temp_early = [t*SAMP_FREQ for t in temp_early]
                temp_late = [t*SAMP_FREQ for t in temp_late]
            
            early_profiles_ON.extend(temp_early)
            late_profiles_ON.extend(temp_late)
            
            curr_early_ON.extend(temp_early)
            curr_late_ON.extend(temp_late)
            
        if ratiotype == 'run-onset OFF':
            temp_early = _get_profiles(trains, matched_early)
            temp_late = _get_profiles(trains, matched_late)
            
            if int(recname[1:4]) > 40:
                temp_early = [t*SAMP_FREQ for t in temp_early]
                temp_late = [t*SAMP_FREQ for t in temp_late]
            
            early_profiles_OFF.extend(temp_early)
            late_profiles_OFF.extend(temp_late)
            
            curr_early_OFF.extend(temp_early)
            curr_late_OFF.extend(temp_late)
        
    # single session plotting 
    if curr_early_ON:
        curr_early_mean = np.mean(curr_early_ON, axis=0)
        curr_early_sem = sem(curr_early_ON, axis=0)

        curr_late_mean = np.mean(curr_late_ON, axis=0)
        curr_late_sem = sem(curr_late_ON, axis=0)

        curr_early_amp = np.mean(np.array(curr_early_ON)[:, 1250+625:1250+1825], axis=1)
        curr_late_amp = np.mean(np.array(curr_late_ON)[:, 1250+625:1250+1825], axis=1)
        
        pf.plot_violin_with_scatter(curr_early_amp, curr_late_amp, 'grey', 'royalblue',
                                    paired=False, showscatter=True,
                                    ylim=(0, 10),
                                    save=True,
                                    title=f'{len(matched_early)} {len(matched_late)}\n{np.mean(curr_early_amp)} {np.mean(curr_late_amp)}',
                                    savepath=rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis_raphi\all_sessions\{recname}_run_onset_ON_mean_amp{ext}')

        fig, ax = plt.subplots(figsize=(2.3, 2.0))
        ax.plot(XAXIS, curr_early_mean, c='grey', label='<2.5')
        ax.fill_between(XAXIS, curr_early_mean+curr_early_sem, curr_early_mean-curr_early_sem,
                        color='grey', edgecolor='none', alpha=.25)
        ax.plot(XAXIS, curr_late_mean, c=late_c, label='2.5~3.5')
        ax.fill_between(XAXIS, curr_late_mean+curr_late_sem, curr_late_mean-curr_late_sem,
                        color=late_c, edgecolor='none', alpha=.25)

        ax.legend(fontsize=5, frameon=False)
        ax.set(xlabel='time from run onset (s)', xlim=(-1, 4),
               ylabel='firing rate (Hz)')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)

        fig.tight_layout()
        plt.show()

        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis_raphi\all_sessions\{recname}_run_onset_ON_mean_profiles.png',
                    dpi=300,
                    bbox_inches='tight')
    
        
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
        ax.set(xlabel='time from run onset (s)', xlim=(-1, 4),
               ylabel='firing rate (Hz)')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)

        fig.tight_layout()
        plt.show()

        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis_raphi\all_sessions\{recname}_run_onset_OFF_mean_profiles.png',
                    dpi=300,
                    bbox_inches='tight')


#%% BEFORE-matching session-averaged speed
E_raw = np.vstack(sess_early_speed_means_raw)  # n_sessions x 3500
L_raw = np.vstack(sess_late_speed_means_raw)

E_raw_mean = np.mean(E_raw, axis=0)
E_raw_sem  = sem(E_raw, axis=0)
L_raw_mean = np.mean(L_raw, axis=0)
L_raw_sem  = sem(L_raw, axis=0)

x_sec = np.arange(3500) / 1000.0

fig, ax = plt.subplots(figsize=(2.1, 2.0))
ax.plot(x_sec, E_raw_mean, c='grey', label='early (<2.5 s)')
ax.fill_between(x_sec, E_raw_mean+E_raw_sem, E_raw_mean-E_raw_sem,
                color='grey', edgecolor='none', alpha=.25)

late_c = (0.20, 0.35, 0.65)
ax.plot(x_sec, L_raw_mean, c=late_c, label='late (2.5–3.5 s)')
ax.fill_between(x_sec, L_raw_mean+L_raw_sem, L_raw_mean-L_raw_sem,
                color=late_c, edgecolor='none', alpha=.25)

ax.set(xlabel='time from run onset (s)', xlim=(0,3.5),
       ylabel='speed (cm/s)', ylim=(0, 65),
       title='pre-matching speed')
ax.legend(frameon=False, fontsize=7)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis_raphi\speed_pre_matched{ext}',
                dpi=300,
                bbox_inches='tight')
    

#%% average speed curves post-speed matching 
E = np.vstack(sess_early_speed_means)
L = np.vstack(sess_late_speed_means)

E_trace_mean = np.mean(E, axis=0)
E_trace_sem  = sem(E, axis=0)
L_trace_mean = np.mean(L, axis=0)
L_trace_sem  = sem(L, axis=0)

fig, ax = plt.subplots(figsize=(2.1, 2.0))
ax.plot(x_sec, E_trace_mean, c='grey', label='early (<2.5 s)')
ax.fill_between(x_sec, E_trace_mean+E_trace_sem, E_trace_mean-E_trace_sem,
                color='grey', edgecolor='none', alpha=.25)

late_c = (0.20, 0.35, 0.65)
ax.plot(x_sec, L_trace_mean, c=late_c, label='late (2.5–3.5 s)')
ax.fill_between(x_sec, L_trace_mean+L_trace_sem, L_trace_mean-L_trace_sem,
                color=late_c, edgecolor='none', alpha=.25)

# per-bin stats (500 ms bins)
bin_size = 500  # ms
n_bins = 7
E_bins = np.vstack([E[:, i*bin_size:(i+1)*bin_size].mean(axis=1) for i in range(n_bins)]).T
L_bins = np.vstack([L[:, i*bin_size:(i+1)*bin_size].mean(axis=1) for i in range(n_bins)]).T

pvals, tvals = [], []
dz = []
for i in range(n_bins):
    t, p = ttest_ind(E_bins[:, i], L_bins[:, i], equal_var=False, nan_policy='omit')
    tvals.append(t); pvals.append(p)
    d = E_bins[:, i] - L_bins[:, i]
    d = d[~np.isnan(d)]
    dz.append(np.mean(d) / (np.std(d, ddof=1) + 1e-12))
pvals = np.array(pvals)

# black bars + RAW p-values above each 0.5 s bin
ymax = max((E_trace_mean+E_trace_sem).max(), (L_trace_mean+L_trace_sem).max())
ymin = min((E_trace_mean-E_trace_sem).min(), (L_trace_mean-L_trace_sem).min())
yr = ymax - ymin if ymax > ymin else 1.0
bar_y  = ymax + 0.06 * yr
text_y = ymax + 0.11 * yr

for i in range(n_bins):
    x_left  = i * 0.5 + 0.1
    x_right = (i + 1) * 0.5 - 0.1
    ax.hlines(bar_y, x_left, x_right, color='k', lw=1)
    ax.text((x_left + x_right)/2, text_y, f'p={pvals[i]:.3f}',
            ha='center', va='bottom', fontsize=5)

ax.set(xlabel='time from run onset (s)', xlim=(0,3.5),
       ylabel='speed (cm/s)', ylim=(0,65),
       title='post-matching speed')
ax.legend(frameon=False, fontsize=7)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout()
plt.show()

# optional: console summary
edges_ms = [(i*500, (i+1)*500) for i in range(n_bins)]
print('\nindependent t-tests per 500 ms bin (raw p-values):')
for i, (lo, hi) in enumerate(edges_ms):
    print(f'  {lo:4d}–{hi:4d} ms: t={tvals[i]:6.3f}, p={pvals[i]:.4f}, dz={dz[i]:.3f}')

for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis_raphi\speed_post_matched{ext}',
                dpi=300,
                bbox_inches='tight')


#%% mean spiking curves (ON) for early v late 
# stat
early_win_means = np.mean(np.array(early_profiles_ON)[:, 1250+625:1250+1825], axis=1)
late_win_means  = np.mean(np.array(late_profiles_ON)[:, 1250+625:1250+1825], axis=1)

# paired t-test
t_stat, p_val = ttest_ind(early_win_means, late_win_means, nan_policy='omit')

print(f'early mean = {np.mean(early_win_means)}, sem = {sem(early_win_means)}\n')
print(f'late mean = {np.mean(late_win_means)}, sem = {sem(late_win_means)}\n')
print(f'Paired t-test (0.5–1.5 s window): t = {t_stat:}, p = {p_val}')

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
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis_raphi\all_run_onset_ON_mean_profiles{ext}',
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
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis_raphi\all_run_onset_OFF_mean_profiles{ext}',
                dpi=300,
                bbox_inches='tight')