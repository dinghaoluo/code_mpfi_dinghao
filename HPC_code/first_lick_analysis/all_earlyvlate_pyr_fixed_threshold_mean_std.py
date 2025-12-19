# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:56:30 2025

replicate what we have done on LC run onset-peaking cells with HPC recordings

@author: Dinghao Luo
"""


#%% imports
from pathlib import Path 

import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import sem, wilcoxon, ttest_ind

from plotting_functions import plot_violin_with_scatter
from common import mpl_formatting
mpl_formatting()

import rec_list
paths            = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt
bad_beh_paths    = rec_list.pathHPCbadbeh
recnames         = [Path(path).name for path in paths]
bad_beh_recnames = [Path(path).name for path in bad_beh_paths]


#%% parameters 
SAMP_FREQ = 1250 
RUN_ONSET_BIN = 3750
BEF = 1  # seconds before run-onset
AFT = 4  # seconds after run-onset

TIME = np.arange(-SAMP_FREQ*BEF, SAMP_FREQ*AFT)/SAMP_FREQ
X_SEC_PLOT = np.arange(4000) / 1000.0

# colours for speed
early_c = (168/255, 155/255, 202/255)
late_c  = (102/255, 83/255 , 162/255)


#%% path stems 
all_exp_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments')
first_lick_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/first_lick_analysis')


#%% helper 
def _annotate_pvals(ax, pvals,
                    start=-0.5, bin_size=1.0, fontsize=5,
                    y_level=None, star=True):
    if y_level is None:
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin if ymax > ymin else 1.0
        y_level = ymax + 0.05 * yr

    for i, p in enumerate(pvals):
        lo = start + i * bin_size
        hi = lo + bin_size
        mid = (lo + hi) / 2

        # decide text
        if star:
            if p < 0.0001: text = '****'
            elif p < 0.001: text = '***'
            elif p < 0.01: text = '**'
            elif p < 0.05: text = '*'
            else: text = 'n.s.'
        else:
            text = f'{p:.2e}'

        ax.hlines(y_level, lo+0.05, hi-0.05, color='k', lw=.75)
        ax.text(mid, y_level, text,
                ha='center', va='bottom', fontsize=fontsize, color='k')

def _binwise_test(early_profiles, late_profiles,
                  SAMP_FREQ=1250, BEF=1,
                  start=-0.5, end=3.5, bin_size=1.0,
                  label='profiles'):
    bins = np.arange(start, end, bin_size)
    n_bins = len(bins)
    pvals = []

    print(f'\nBinwise stats for {label}:')
    for b in range(n_bins):
        lo, hi = bins[b], bins[b] + bin_size
        lo_idx = int((lo + BEF) * SAMP_FREQ)
        hi_idx = int((hi + BEF) * SAMP_FREQ)

        e_means = np.mean(early_profiles[:, lo_idx:hi_idx], axis=1)
        l_means = np.mean(late_profiles[:, lo_idx:hi_idx], axis=1)
        
        e_means_means = np.mean(e_means)
        l_means_means = np.mean(l_means)
        e_means_sem = sem(e_means)
        l_means_sem = sem(l_means)

        _, p = wilcoxon(e_means, l_means, nan_policy='omit')
        pvals.append(p)
        print(f'  {lo:.1f}–{hi:.1f} s: e={e_means_means}, {e_means_sem}, l={l_means_means}, {l_means_sem},  p={p:.4e}')

    return pvals

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


#%% load data 
print('Loading dataframes...')
cell_profiles = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')
df_pyr = cell_profiles[cell_profiles['cell_identity'] == 'pyr']
pyrON = df_pyr[df_pyr['class'] == 'run-onset ON']
pyrOFF = df_pyr[df_pyr['class'] == 'run-onset OFF']


#%% main
# main container 
profiles = {}

# speed after speed matching for each session 
sess_early_speed_means = []
sess_late_speed_means = []

# speed BEFORE matching 
sess_early_speed_means_raw = []
sess_late_speed_means_raw = []

# cell loop
recname   = ''
skip_flag = False
current_session_ON_early  = []
current_session_ON_late   = []
current_session_OFF_early = []
current_session_OFF_late  = []

# combine ON and OFF and sort them first to loop over all cells all at once
all_valid_clunames = list(pyrON.index) + list(pyrOFF.index)
all_valid_clunames = sorted(all_valid_clunames,
                            key=lambda x: x.split(' ')[0])
for cluname in all_valid_clunames:
    temp_recname = cluname.split(' ')[0]
    
    if temp_recname != recname:
        # truncate data
        curr_profiles = {
            'early ON' : current_session_ON_early,
            'late ON'  : current_session_ON_late,
            'early OFF': current_session_OFF_early,
            'late OFF' : current_session_OFF_late
            }
        profiles[recname] = curr_profiles
            
        current_session_ON_early  = []
        current_session_ON_late   = []
        current_session_OFF_early = []
        current_session_OFF_late  = []
        
        recname = temp_recname
        if recname not in recnames:
            print(f'\n{recname}\nSkipped')
            skip_flag = True
            continue
        
        print(f'\n{recname}')
        skip_flag = False
        
        alignRun = sio.loadmat(
            rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:-3]}'
            rf'\{recname}\{recname}_DataStructure_mazeSection1_'
            r'TrialType1_alignRun_msess1.mat'
        )
        licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
        starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
        tot_trial = licks.shape[0]

        behPar = sio.loadmat(
            rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:-3]}'
            rf'\{recname}\{recname}_DataStructure_mazeSection1_'
            r'TrialType1_behPar_msess1.mat'
        )
        bad_idx  = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
        stim_idx = np.where(behPar['behPar'][0]['stimOn'][0]!=0)[1]-1

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

        if len(early_trials) < 10 or len(late_trials) < 10:
            print('Not enough trials; skipped')
            skip_flag = True
            continue

        all_trains = np.load(
            rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}'
            rf'\{recname}_all_trains_run.npy',
            allow_pickle=True
        ).item()

        try:
            beh_path = all_exp_stem/ 'HPCLC' / f'{recname}.pkl'
            with open(beh_path, 'rb') as f:
                beh = pickle.load(f)
        except FileNotFoundError:
            beh_path = all_exp_stem/ 'HPCLCterm' / f'{recname}.pkl'
            with open(beh_path, 'rb') as f:
                beh = pickle.load(f)

        speed_times = beh['speed_times_aligned'][1:]
        
        # speed BEFORE matching for visualisation 
        e_mean_sp_raw = _session_mean_speed(early_trials, speed_times, n=4000)
        l_mean_sp_raw = _session_mean_speed(late_trials,  speed_times, n=4000)
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
            k = 2
        
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
        
        if len(matched_early) < 5 or len(matched_late) < 5:
            print('Not enough speed-matched trials; skipped')
            skip_flag = True
            continue
        
        # collect session means for later plotting 
        e_mean_sp = _session_mean_speed(matched_early, speed_times, n=4000)
        l_mean_sp = _session_mean_speed(matched_late, speed_times, n=4000)
        
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
        
        # per-bin t test for sanity check 
        n_bins = 7
        bin_size = 500
        pvals = np.ones(n_bins)
        for i in range(n_bins):
            e = E_b[:, i] if E_b.size else np.array([])
            l = L_b[:, i] if L_b.size else np.array([])
            if e.size >= 2 and l.size >= 2:
                t, p = ttest_ind(e, l, equal_var=False)
                pvals[i] = p
        
        # skip session if any bin differs
        # if np.any(pvals < 0.05):
        #     print('Session rejected for unequal speeds (binwise t-test)')
        #     continue
        
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
        
        ax.set(xlabel='Time (s)', ylabel='Speed (cm/s)', title=f'{recname} bin-filtered')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        
        vis_path = first_lick_stem / 'single_session_speed_matching' / f'{recname}_bin_filtered_speed_traces'
        for ext in ['.pdf', '.png']:
            fig.savefig(f'{vis_path}{ext}', dpi=200)
        plt.close(fig)


    ## ---- accumulate data
    if skip_flag == False:
        trains = all_trains[cluname]
        if cluname in pyrON.index:
            early_profiles = np.nanmean(_get_profiles(trains, matched_early), axis=0)
            late_profiles  = np.nanmean(_get_profiles(trains, matched_late), axis=0)
            
            current_session_ON_early.append(early_profiles)
            current_session_ON_late.append(late_profiles)
        
        if cluname in pyrOFF.index:
            # print(f'OFF {cluname}')
            early_profiles = np.nanmean(_get_profiles(trains, matched_early), axis=0)
            late_profiles  = np.nanmean(_get_profiles(trains, matched_late), axis=0)
            
            current_session_OFF_early.append(early_profiles)
            current_session_OFF_late.append(late_profiles)
    ## ---- accumulate data ends 
    
    # after the for-loop ends, flush the last session
    curr_profiles = {
        'early ON' : current_session_ON_early,
        'late ON'  : current_session_ON_late,
        'early OFF': current_session_OFF_early,
        'late OFF' : current_session_OFF_late
        }
    profiles[recname] = curr_profiles


#%% BEFORE-matching session-averaged speed
E_raw = np.vstack(sess_early_speed_means_raw)  # n_sessions x 3500
L_raw = np.vstack(sess_late_speed_means_raw)

E_raw_mean = np.mean(E_raw, axis=0)
E_raw_sem  = sem(E_raw, axis=0)
L_raw_mean = np.mean(L_raw, axis=0)
L_raw_sem  = sem(L_raw, axis=0)

x_sec = np.arange(3500) / 1000.0

fig, ax = plt.subplots(figsize=(2.1, 2.0))
ax.plot(X_SEC_PLOT, E_raw_mean, c=early_c, label='early (<2.5 s)')
ax.fill_between(X_SEC_PLOT, E_raw_mean+E_raw_sem, E_raw_mean-E_raw_sem,
                color=early_c, edgecolor='none', alpha=.25)

ax.plot(X_SEC_PLOT, L_raw_mean, c=late_c, label='late (2.5–3.5 s)')
ax.fill_between(X_SEC_PLOT, L_raw_mean+L_raw_sem, L_raw_mean-L_raw_sem,
                color=late_c, edgecolor='none', alpha=.25)

ax.set(xlabel='Time from run onset (s)', xlim=(0, 4),
       ylabel='Speed (cm/s)', ylim=(0, 70),
       title='pre-matching speed')
ax.legend(frameon=False, fontsize=7)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(first_lick_stem / f'speed_pre_matched{ext}',
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
ax.plot(X_SEC_PLOT, E_trace_mean, c=early_c, label='early (<2.5 s)')
ax.fill_between(X_SEC_PLOT, E_trace_mean+E_trace_sem, E_trace_mean-E_trace_sem,
                color=early_c, edgecolor='none', alpha=.25)

ax.plot(X_SEC_PLOT, L_trace_mean, c=late_c, label='late (2.5–3.5 s)')
ax.fill_between(X_SEC_PLOT, L_trace_mean+L_trace_sem, L_trace_mean-L_trace_sem,
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
text_y = ymax + 0.08 * yr

for i in range(n_bins):
    x_left  = i * 0.5 + 0.1
    x_right = (i + 1) * 0.5 - 0.1
    ax.hlines(bar_y, x_left, x_right, color='k', lw=1)
    ax.text((x_left + x_right)/2, text_y, f'{pvals[i]:.3e}',
            ha='center', va='bottom', fontsize=3)

ax.set(xlabel='Time from run onset (s)', xlim=(0,4),
       ylabel='Speed (cm/s)', ylim=(0,70),
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
    fig.savefig(first_lick_stem / f'speed_post_matched{ext}',
                dpi=300,
                bbox_inches='tight')
    

#%% mean over all speeds for violinplot 
mean_E = np.mean(E, axis=1)
mean_L = np.mean(L, axis=1)

plot_violin_with_scatter(mean_E, mean_L,
                         early_c, late_c,
                         ylabel='Speed (cm/s)',
                         xticklabels=['Early', 'Late'],
                         print_statistics=True,
                         save=True,
                         savepath=first_lick_stem / 'speed_post_matched_violin')


#%% mean spiking curves (ON) for early v late 
# organise data
valid_profiles = {k: v for k, v in profiles.items() if v['early ON']
                  and k not in ['A068r-20231024-01',
                                'A071r-20230922-02',
                                'A071r-20230921-01']}
session_names  = list(valid_profiles.keys())

keys = [k for k, v in valid_profiles.items()]
nums = [len(v['early ON']) for k, v in valid_profiles.items()]
for key, num in zip(keys, nums):
    print(f'{key}: {num}')
for i in range(len(session_names)):
    early_ON  = [arr
                 for key, session in valid_profiles.items()
                 if key!=session_names[i]
                 for arr in session['early ON']]
    late_ON   = [arr
                 for key, session in valid_profiles.items()
                 if key!=session_names[i]
                 for arr in session['late ON']]
    early_OFF = [arr
                 for key, session in valid_profiles.items()
                 if key!=session_names[i]
                 for arr in session['early OFF']]
    late_OFF  = [arr
                 for key, session in valid_profiles.items()
                 if key!=session_names[i]
                 for arr in session['late OFF']]
    
    # stats
    early_win_means = np.mean(np.array(early_ON)[:, 1250+625:1250+1250], axis=1)
    late_win_means  = np.mean(np.array(late_ON)[:, 1250+625:1250+1250], axis=1)
    
    _, p_val = wilcoxon(early_win_means, late_win_means, nan_policy='omit')
    
    early_win_means = np.mean(np.array(early_OFF)[:, 1250+2500+625:1250+3750+625], axis=1)
    late_win_means  = np.mean(np.array(late_OFF)[:, 1250+2500+625:1250+3750+625], axis=1)
    
    _, p_val2 = wilcoxon(early_win_means, late_win_means, nan_policy='omit')
    
    # print(f'early mean = {np.mean(early_win_means)}, sem = {sem(early_win_means)}\n')
    # print(f'late mean = {np.mean(late_win_means)}, sem = {sem(late_win_means)}\n')
    # print(f'Wilcoxon test (0.5–1.5 s window): p = {p_val:.4e}')
    print(f'{keys[i]}, late - early = {round(np.mean(late_win_means) - np.mean(early_win_means), 3)} {p_val:.3g} {p_val2:.3g}')


#%% plotting
## ON
XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

early_mean = np.mean(early_ON, axis=0)
early_sem = sem(early_ON, axis=0)

late_mean = np.mean(late_ON, axis=0)
late_sem = sem(late_ON, axis=0)

fig, ax = plt.subplots(figsize=(2.3, 2.0))
ax.plot(XAXIS, early_mean, c='lightcoral', label='<2.5')
ax.fill_between(XAXIS, early_mean+early_sem, early_mean-early_sem,
                color='lightcoral', edgecolor='none', alpha=.25)
ax.plot(XAXIS, late_mean, c='firebrick', label='2.5~3.5')
ax.fill_between(XAXIS, late_mean+late_sem, late_mean-late_sem,
                color='firebrick', edgecolor='none', alpha=.25)

pvals = _binwise_test(np.array(early_ON),
                      np.array(late_ON),
                      SAMP_FREQ=SAMP_FREQ, BEF=BEF,
                      label='ON cells',
                      bin_size=1.0)
_annotate_pvals(ax, pvals, start=-0.5, bin_size=1.0, star=False)

ax.legend(fontsize=5, frameon=False)
ax.set(xlabel='Time from run onset (s)', xlim=(-1, 4), ylim=(1.4, 3.6),
       ylabel='Firing rate (Hz)')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()

# for ext in ['.png', '.pdf']:
#     fig.savefig(
#         first_lick_stem / f'all_run_onset_ON_mean_profiles{ext}',
#         dpi=300,
#         bbox_inches='tight')
    

## OFF
early_mean = np.mean(early_OFF, axis=0)
early_sem = sem(early_OFF, axis=0)

late_mean = np.mean(late_OFF, axis=0)
late_sem = sem(late_OFF, axis=0)

fig, ax = plt.subplots(figsize=(2.3, 2.0))
ax.plot(XAXIS, early_mean, c='violet', label='<2.5')
ax.fill_between(XAXIS, early_mean+early_sem, early_mean-early_sem,
                color='violet', edgecolor='none', alpha=.25)
ax.plot(XAXIS, late_mean, c='purple', label='2.5~3.5')
ax.fill_between(XAXIS, late_mean+late_sem, late_mean-late_sem,
                color='purple', edgecolor='none', alpha=.25)

pvals = _binwise_test(np.array(early_OFF),
                      np.array(late_OFF),
                      SAMP_FREQ=SAMP_FREQ, BEF=BEF,
                      label='OFF cells',
                      bin_size=1.0)
_annotate_pvals(ax, pvals, start=-0.5, bin_size=1.0, star=False)

ax.legend(fontsize=5, frameon=False)
ax.set(xlabel='Time from run onset (s)', xlim=(-1, 4), ylim=(1.15, 3.0),
       ylabel='Firing rate (Hz)')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()

# for ext in ['.png', '.pdf']:
#     fig.savefig(
#         first_lick_stem / f'all_run_onset_OFF_mean_profiles{ext}',
#         dpi=300,
#         bbox_inches='tight')