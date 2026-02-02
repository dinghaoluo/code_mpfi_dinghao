# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:29:38 2025

controls for LC run-onset peaks

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import pandas as pd 
from scipy.stats import sem, wilcoxon, ttest_1samp, linregress

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

import rec_list
paths = rec_list.pathLC

import plotting_functions as pf
from common import mpl_formatting
mpl_formatting()


#%% paths and parameters
LC_stem       = Path('Z:/Dinghao/code_dinghao/LC_ephys')
all_sess_stem = LC_stem / 'all_sessions'
LC_beh_stem   = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')

SAMP_FREQ     = 1_250 
SAMP_FREQ_BEH = 1_000
RUN_ONSET_BIN = SAMP_FREQ * 3
BEF = 1  # s, how much time before run-onset to get
AFT = 4  # same as above 
WINDOW_HALF_SIZE = .5

RO_WINDOW = [
    int(RUN_ONSET_BIN - WINDOW_HALF_SIZE * SAMP_FREQ), 
    int(RUN_ONSET_BIN + WINDOW_HALF_SIZE * SAMP_FREQ)
    ]  # window for spike summation, half a sec around run onsets

XAXIS_SPIKE_TIME = np.arange(SAMP_FREQ * (BEF + AFT)) / SAMP_FREQ - 1  # 5 seconds 
XAXIS_SPEED_TIME = np.arange(SAMP_FREQ_BEH * 4) / SAMP_FREQ_BEH  # 4 seconds 


#%% load data 
print('Loading data...')
cell_prop_path = LC_stem / 'LC_all_cell_profiles.pkl'
cell_prop = pd.read_pickle(cell_prop_path)

clu_keys = list(cell_prop.index)

tagged_keys = []; putative_keys = []
tagged_RO_keys = []; putative_RO_keys = []
RO_keys = []  # pooled run-onset bursting cells 
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
RO_keys = tagged_RO_keys + putative_RO_keys


#%% main 
all_high_speed_speed      = []
all_low_speed_speed       = []
all_high_init_speed_speed = []
all_low_init_speed_speed  = []
all_high_accel_speed      = []
all_low_accel_speed       = []

all_high_speed_curve      = []
all_low_speed_curve       = []
all_high_init_speed_curve = []
all_low_init_speed_curve  = []
all_high_accel_curve      = []
all_low_accel_curve       = []

all_high_speed_amp      = []
all_low_speed_amp       = []
all_high_init_speed_amp = []
all_low_init_speed_amp  = []
all_high_accel_amp      = []
all_low_accel_amp       = []

all_session_init_speed_FR_r = []
all_session_init_accel_FR_r = []
all_init_speeds = []
all_init_accel  = []
all_init_FR     = []

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    trains_path = all_sess_stem / recname / f'{recname}_all_trains_run.npy'
    trains = np.load(trains_path, allow_pickle=True).item()
    
    # break if no RO cells
    if not [clu for clu in trains.keys() if clu in RO_keys]:
        print('Session has no RunOn cell; skipped')
        continue
    
    with open(LC_beh_stem / f'{recname}.pkl', 'rb') as f:
        beh = pickle.load(f)
    
    # trial filtering (ignore bad trials & stim trials)
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds)
                if cond!='0']
    bad_idx = [trial for trial, bad in enumerate(beh['bad_trials'][1:])
               if bad]
    valid_trial_idx = [trial for trial in np.arange(len(stim_conds)) 
                       if trial not in stim_idx and trial not in bad_idx]
    
    # get speed, init. speed and accel.
    speed_trials      = [[t[1] for t in trial]
                         for trial in beh['speed_times_aligned'][1:]]
    mean_speed_trials = [np.mean(trial) for trial in speed_trials]
    high_speed_idx    = [trial for trial, speed in enumerate(mean_speed_trials)
                         if 45<speed<55]
    low_speed_idx     = [trial for trial, speed in enumerate(mean_speed_trials)
                         if 35<speed<45]
    
    init_speed_trials   = [np.mean(trial[:500]) for trial in speed_trials]
    high_init_speed_idx = [trial for trial, init_speed in enumerate(init_speed_trials)
                           if 35<init_speed<45]
    low_init_speed_idx  = [trial for trial, init_speed in enumerate(init_speed_trials)
                           if 25<init_speed<35]
    
    acceleration_trials = [np.mean(np.diff(trial[:500])) * 1_000
                           for trial in speed_trials]
    high_accel_idx      = [trial for trial, accel in enumerate(acceleration_trials)
                           if accel>80]
    low_accel_idx       = [trial for trial, accel in enumerate(acceleration_trials)
                           if 60<accel<80]
    
    # accumulate speed data 
    high_speed_speed      = []
    low_speed_speed       = []
    high_init_speed_speed = []
    low_init_speed_speed  = []
    high_accel_speed      = []
    low_accel_speed       = []
    for trial in valid_trial_idx:
        if trial in high_speed_idx:
            high_speed_speed.append(speed_trials[trial])
        if trial in low_speed_idx:
            low_speed_speed.append(speed_trials[trial])
        if trial in high_init_speed_idx:
            high_init_speed_speed.append(speed_trials[trial])
        if trial in low_init_speed_idx:
            low_init_speed_speed.append(speed_trials[trial])
        if trial in high_accel_idx:
            high_accel_speed.append(speed_trials[trial])
        if trial in low_accel_idx:
            low_accel_speed.append(speed_trials[trial])
    
    # pad speeds to be equal length
    if high_speed_speed and low_speed_speed:
        speed_speed_lengths = [len(speeds) for speeds in high_speed_speed + low_speed_speed]
        speed_speed_max_length = max(speed_speed_lengths)
        
        padded_high_speed_speed = np.full(
            (len(high_speed_speed), speed_speed_max_length), np.nan
            )
        padded_low_speed_speed = np.full(
            (len(low_speed_speed), speed_speed_max_length), np.nan
            )
        for i, lst in enumerate(high_speed_speed):
            n_valid = min(len(lst), speed_speed_max_length)
            padded_high_speed_speed[i, :n_valid] = lst[:n_valid]
        for i, lst in enumerate(low_speed_speed):
            n_valid = min(len(lst), speed_speed_max_length)
            padded_low_speed_speed[i, :n_valid] = lst[:n_valid]
        
        all_high_speed_speed.append(np.nanmean(padded_high_speed_speed, axis=0))
        all_low_speed_speed.append(np.nanmean(padded_low_speed_speed, axis=0))
    
    if high_init_speed_speed and low_init_speed_speed:
        init_speed_speed_lengths = [len(speeds) for speeds in high_init_speed_speed + low_init_speed_speed]
        init_speed_speed_max_length = max(init_speed_speed_lengths)
        
        padded_high_init_speed_speed = np.full(
            (len(high_init_speed_speed), init_speed_speed_max_length), np.nan
            )
        padded_low_init_speed_speed = np.full(
            (len(low_init_speed_speed), init_speed_speed_max_length), np.nan
            )
        for i, lst in enumerate(high_init_speed_speed):
            n_valid = min(len(lst), init_speed_speed_max_length)
            padded_high_init_speed_speed[i, :n_valid] = lst[:n_valid]
        for i, lst in enumerate(low_init_speed_speed):
            n_valid = min(len(lst), init_speed_speed_max_length)
            padded_low_init_speed_speed[i, :n_valid] = lst[:n_valid]
        
        all_high_init_speed_speed.append(np.nanmean(padded_high_init_speed_speed, axis=0))
        all_low_init_speed_speed.append(np.nanmean(padded_low_init_speed_speed, axis=0))
    
    if high_accel_speed and low_accel_speed:
        accel_speed_lengths = [len(speeds) for speeds in high_accel_speed + low_accel_speed]
        accel_speed_max_length = max(accel_speed_lengths)
        
        padded_high_accel_speed = np.full(
            (len(high_accel_speed), accel_speed_max_length), np.nan
            )
        padded_low_accel_speed = np.full(
            (len(low_accel_speed), accel_speed_max_length), np.nan
            )
        for i, lst in enumerate(high_accel_speed):
            n_valid = min(len(lst), accel_speed_max_length)
            padded_high_accel_speed[i, :n_valid] = lst[:n_valid]
        for i, lst in enumerate(low_accel_speed):
            n_valid = min(len(lst), accel_speed_max_length)
            padded_low_accel_speed[i, :n_valid] = lst[:n_valid]
        
        all_high_accel_speed.append(np.nanmean(padded_high_accel_speed, axis=0))
        all_low_accel_speed.append(np.nanmean(padded_low_accel_speed, axis=0))
    
    # accumulate spike data
    curr_high_speed_curve      = []
    curr_low_speed_curve       = []
    curr_high_init_speed_curve = []
    curr_low_init_speed_curve  = []
    curr_high_accel_curve      = []
    curr_low_accel_curve       = []
    
    curr_high_speed_amp      = []
    curr_low_speed_amp       = []
    curr_high_init_speed_amp = []
    curr_low_init_speed_amp  = []
    curr_high_accel_amp      = []
    curr_low_accel_amp       = []
    
    for clu in list(trains.keys()):
        if clu in RO_keys:
            # containers for individual cells             
            high_speed_curve      = []
            low_speed_curve       = []
            high_init_speed_curve = []
            low_init_speed_curve  = []
            high_accel_curve      = []
            low_accel_curve       = []
            
            high_speed_amp      = []
            low_speed_amp       = [] 
            high_init_speed_amp = []
            low_init_speed_amp  = []
            high_accel_amp      = []
            low_accel_amp       = []
            
            curr_train = trains[clu]
            
            for trial in valid_trial_idx:
                if trial in high_speed_idx:
                    high_speed_curve.append(curr_train[trial])
                    high_speed_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                if trial in low_speed_idx:
                    low_speed_curve.append(curr_train[trial])
                    low_speed_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                if trial in high_init_speed_idx:
                    high_init_speed_curve.append(curr_train[trial])
                    high_init_speed_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                if trial in low_init_speed_idx:
                    low_init_speed_curve.append(curr_train[trial])
                    low_init_speed_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                if trial in high_accel_idx:
                    high_accel_curve.append(curr_train[trial])
                    high_accel_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                if trial in low_accel_idx:
                    low_accel_curve.append(curr_train[trial])
                    low_accel_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                
            if high_speed_curve and low_speed_curve:
                curr_high_speed_curve.append(np.mean(high_speed_curve, axis=0))
                curr_low_speed_curve.append(np.mean(low_speed_curve, axis=0))
            if high_init_speed_curve and low_init_speed_curve:
                curr_high_init_speed_curve.append(np.mean(high_init_speed_curve, axis=0))
                curr_low_init_speed_curve.append(np.mean(low_init_speed_curve, axis=0))
            if high_accel_curve and low_accel_curve:
                curr_high_accel_curve.append(np.mean(high_accel_curve, axis=0))
                curr_low_accel_curve.append(np.mean(low_accel_curve, axis=0))
            
            if high_speed_amp and low_speed_amp:
                curr_high_speed_amp.append(np.mean(high_speed_amp))
                curr_low_speed_amp.append(np.mean(low_speed_amp))
            if high_init_speed_amp and low_init_speed_amp:
                curr_high_init_speed_amp.append(np.mean(high_init_speed_amp))
                curr_low_init_speed_amp.append(np.mean(low_init_speed_amp))
            if high_accel_amp and low_accel_amp:
                curr_high_accel_amp.append(np.mean(high_accel_amp))
                curr_low_accel_amp.append(np.mean(low_accel_amp))
    
    mean_high_speed_curve      = np.mean(curr_high_speed_curve, axis=0)
    mean_low_speed_curve       = np.mean(curr_low_speed_curve, axis=0)
    mean_high_init_speed_curve = np.mean(curr_high_init_speed_curve, axis=0)
    mean_low_init_speed_curve  = np.mean(curr_low_init_speed_curve, axis=0)
    mean_high_accel_curve      = np.mean(curr_high_accel_curve, axis=0)
    mean_low_accel_curve      = np.mean(curr_low_accel_curve, axis=0)
    
    if len(curr_high_speed_curve)>0 and len(curr_low_speed_curve)>0:
        all_high_speed_curve.append(np.mean(curr_high_speed_curve, axis=0))
        all_low_speed_curve.append(np.mean(curr_low_speed_curve, axis=0))
    if len(curr_high_init_speed_curve)>0 and len(curr_low_init_speed_curve)>0:
        all_high_init_speed_curve.append(np.mean(curr_high_init_speed_curve, axis=0))
        all_low_init_speed_curve.append(np.mean(curr_low_init_speed_curve, axis=0))
    if len(curr_high_accel_curve)>0 and len(curr_low_accel_curve)>0:
        all_high_accel_curve.append(np.mean(curr_high_accel_curve, axis=0))
        all_low_accel_curve.append(np.mean(curr_low_accel_curve, axis=0))
    
    mean_high_speed_amp      = np.mean(curr_high_speed_amp)
    mean_low_speed_amp       = np.mean(curr_low_speed_amp)
    mean_high_init_speed_amp = np.mean(curr_high_init_speed_amp)
    mean_low_init_speed_amp  = np.mean(curr_low_init_speed_amp)
    mean_high_accel_amp      = np.mean(curr_high_accel_amp)
    mean_low_accel_amp       = np.mean(curr_low_accel_amp)
    
    if not np.isnan(mean_high_speed_amp) and not np.isnan(mean_low_speed_amp):
        all_high_speed_amp.append(np.mean(curr_high_speed_amp))
        all_low_speed_amp.append(np.mean(curr_low_speed_amp))
    if not np.isnan(mean_high_init_speed_amp) and not np.isnan(mean_low_init_speed_amp):
        all_high_init_speed_amp.append(np.mean(curr_high_init_speed_amp))
        all_low_init_speed_amp.append(np.mean(curr_low_init_speed_amp))
    if not np.isnan(mean_high_accel_amp) and not np.isnan(mean_low_accel_amp):
        all_high_accel_amp.append(np.mean(curr_high_accel_amp))
        all_low_accel_amp.append(np.mean(curr_low_accel_amp))
    
    
    ## ---- correlations ---- ##
    session_mean_FR_per_trial    = []
    session_init_speed_per_trial = []
    session_init_accel_per_trial = []
    
    for trial in valid_trial_idx:
        session_init_speed_per_trial.append(init_speed_trials[trial])
        session_init_accel_per_trial.append(acceleration_trials[trial])
    
        fr_list = []
        for clu in trains.keys():
            if clu in RO_keys:
                fr_list.append(
                    np.mean(trains[clu][trial][RO_WINDOW[0]:RO_WINDOW[1]])
                )
        if len(fr_list) > 0:
            session_mean_FR_per_trial.append(np.mean(fr_list))
    
    if len(session_init_speed_per_trial) > 3:
        # append onto grand lists 
        all_init_speeds.extend(session_init_speed_per_trial)
        all_init_accel.extend(session_init_accel_per_trial)
        all_init_FR.extend(session_mean_FR_per_trial)
        
        # put into arrays for stats
        session_init_speed_per_trial = np.array(session_init_speed_per_trial, float)
        session_init_accel_per_trial = np.array(session_init_accel_per_trial, float)
        session_mean_FR_per_trial    = np.array(session_mean_FR_per_trial, float)
        
        # -------------------------
        # init speed vs pooled FR
        # -------------------------
        slope, intercept, r, p, _ = linregress(session_init_speed_per_trial, session_mean_FR_per_trial)
        all_session_init_speed_FR_r.append(r)
        
        xfit = np.linspace(session_init_speed_per_trial.min(), session_init_speed_per_trial.max(), 2)
        yfit = intercept + slope * xfit
        
        fig, ax = plt.subplots(figsize=(1.6, 1.6))
        ax.scatter(session_init_speed_per_trial, session_mean_FR_per_trial,
                   s=12, color='orange', ec='none', alpha=0.7)
        ax.plot(xfit, yfit, color='black', lw=1)
        
        ax.text(0.05, 0.95,
                f'r = {r:.2f}\np = {p:.3g}',
                transform=ax.transAxes,
                ha='left', va='top', fontsize=7)
        
        ax.set(xlabel='Init. speed (cm/s)',
               ylabel='Run-onset FR (Hz)',
               title=recname)
        ax.spines[['top', 'right']].set_visible(False)
        
        for ext in ['.png', '.pdf']:
            fig.savefig(
                rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\single_session_init_speed_FR\{recname}_init_speed_FR_corr{ext}',
                dpi=300, bbox_inches='tight'
            )
        plt.close(fig)
        
        # -------------------------
        # init accel vs pooled FR
        # -------------------------
        slope, intercept, r, p, _ = linregress(session_init_accel_per_trial, session_mean_FR_per_trial)
        all_session_init_accel_FR_r.append(r)
        
        xfit = np.linspace(session_init_accel_per_trial.min(), session_init_accel_per_trial.max(), 2)
        yfit = intercept + slope * xfit
        
        fig, ax = plt.subplots(figsize=(1.6, 1.6))
        ax.scatter(session_init_accel_per_trial, session_mean_FR_per_trial,
                   s=12, color='firebrick', ec='none', alpha=0.7)
        ax.plot(xfit, yfit, color='black', lw=1)
        
        ax.text(0.05, 0.95,
                f'r = {r:.2f}\np = {p:.3g}',
                transform=ax.transAxes,
                ha='left', va='top', fontsize=7)
        
        ax.set(xlabel='Init. accel. (cm/s²)',
               ylabel='Run-onset FR (Hz)',
               title=recname)
        ax.spines[['top', 'right']].set_visible(False)
        
        for ext in ['.png', '.pdf']:
            fig.savefig(
                rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\single_session_init_accel_FR\{recname}_init_accel_FR_corr{ext}',
                dpi=300, bbox_inches='tight'
            )
        plt.close(fig)
    
    else:
        print('Not enough trials: skipped')


#%% plotting - speed 
# ----------------------
# high v low MEAN SPEED
# ----------------------
padded_high_speed_speed = np.full((len(all_high_speed_speed), 4000), np.nan)
for i, lst in enumerate(all_high_speed_speed):
    n_valid = min(len(lst), 4000)
    padded_high_speed_speed[i, :n_valid] = lst[:n_valid]

padded_low_speed_speed = np.full((len(all_low_speed_speed), 4000), np.nan)
for i, lst in enumerate(all_low_speed_speed):
    n_valid = min(len(lst), 4000)
    padded_low_speed_speed[i, :n_valid] = lst[:n_valid]

high_speed_speed_mean = np.mean(padded_high_speed_speed, axis=0)
high_speed_speed_sem = sem(padded_high_speed_speed, axis=0)
low_speed_speed_mean = np.mean(padded_low_speed_speed, axis=0)
low_speed_speed_sem = sem(padded_low_speed_speed, axis=0)

fig, ax = plt.subplots(figsize=(1.8,1.4))

hsln, = ax.plot(XAXIS_SPEED_TIME, high_speed_speed_mean, color='orange')
ax.fill_between(XAXIS_SPEED_TIME, high_speed_speed_mean+high_speed_speed_sem,
                                  high_speed_speed_mean-high_speed_speed_sem,
                alpha=.3, color='orange', edgecolor='none')
lsln, = ax.plot(XAXIS_SPEED_TIME, low_speed_speed_mean, color='navajowhite')
ax.fill_between(XAXIS_SPEED_TIME, low_speed_speed_mean+low_speed_speed_sem,
                                  low_speed_speed_mean-low_speed_speed_sem,
                alpha=.3, color='navajowhite', edgecolor='none')

ax.set(xlabel='Time from run onset (s)', xticks=[0,2,4],
       ylabel='Speed (cm/s)', ylim=(0,75),
       title='High v low speed')
ax.legend([hsln, lsln], ['High', 'Low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_speed_speed{ext}',
                dpi=300, bbox_inches='tight')
    

# -----------------------
# high v low INIT. SPEED
# -----------------------
padded_high_init_speed_speed = np.full((len(all_high_init_speed_speed), 4000), np.nan)
for i, lst in enumerate(all_high_init_speed_speed):
    n_valid = min(len(lst), 4000)
    padded_high_init_speed_speed[i, :n_valid] = lst[:n_valid]

padded_low_init_speed_speed = np.full((len(all_low_init_speed_speed), 4000), np.nan)
for i, lst in enumerate(all_low_init_speed_speed):
    n_valid = min(len(lst), 4000)
    padded_low_init_speed_speed[i, :n_valid] = lst[:n_valid]

high_init_speed_speed_mean = np.mean(padded_high_init_speed_speed, axis=0)
high_init_speed_speed_sem  = sem(padded_high_init_speed_speed, axis=0)
low_init_speed_speed_mean  = np.mean(padded_low_init_speed_speed, axis=0)
low_init_speed_speed_sem   = sem(padded_low_init_speed_speed, axis=0)

fig, ax = plt.subplots(figsize=(1.8,1.4))

hsln, = ax.plot(XAXIS_SPEED_TIME, high_init_speed_speed_mean, color='orange')
ax.fill_between(XAXIS_SPEED_TIME, high_init_speed_speed_mean + high_init_speed_speed_sem,
                                  high_init_speed_speed_mean - high_init_speed_speed_sem,
                alpha=.3, color='orange', edgecolor='none')
lsln, = ax.plot(XAXIS_SPEED_TIME, low_init_speed_speed_mean, color='navajowhite')
ax.fill_between(XAXIS_SPEED_TIME, low_init_speed_speed_mean + low_init_speed_speed_sem,
                                  low_init_speed_speed_mean - low_init_speed_speed_sem,
                alpha=.3, color='navajowhite', edgecolor='none')

ax.set(xlabel='Time from run onset (s)', xticks=[0,2,4],
       ylabel='Speed (cm/s)', ylim=(0,75),
       title='High v low init. speed')
ax.legend([hsln, lsln], ['High', 'Low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_init_speed_speed{ext}',
                dpi=300, bbox_inches='tight')


# ------------------------
# high v low INIT. ACCEL.
# ------------------------
padded_high_accel_speed = np.full((len(all_high_accel_speed), 4000), np.nan)
for i, lst in enumerate(all_high_accel_speed):
    n_valid = min(len(lst), 4000)
    padded_high_accel_speed[i, :n_valid] = lst[:n_valid]

padded_low_accel_speed = np.full((len(all_low_accel_speed), 4000), np.nan)
for i, lst in enumerate(all_low_accel_speed):
    n_valid = min(len(lst), 4000)
    padded_low_accel_speed[i, :n_valid] = lst[:n_valid]

high_accel_speed_mean = np.mean(padded_high_accel_speed, axis=0)
high_accel_speed_sem = sem(padded_high_accel_speed, axis=0)
low_accel_speed_mean = np.mean(padded_low_accel_speed, axis=0)
low_accel_speed_sem = sem(padded_low_accel_speed, axis=0)

fig, ax = plt.subplots(figsize=(1.8,1.4))

hsln, = ax.plot(XAXIS_SPEED_TIME, high_accel_speed_mean, color='firebrick')
ax.fill_between(XAXIS_SPEED_TIME, high_accel_speed_mean+high_accel_speed_sem,
                                  high_accel_speed_mean-high_accel_speed_sem,
                alpha=.25, color='firebrick', edgecolor='none')
lsln, = ax.plot(XAXIS_SPEED_TIME, low_accel_speed_mean, color='lightcoral')
ax.fill_between(XAXIS_SPEED_TIME, low_accel_speed_mean+low_accel_speed_sem,
                                  low_accel_speed_mean-low_accel_speed_sem,
                alpha=.25, color='lightcoral', edgecolor='none')

ax.set(xlabel='Time from run onset (s)', 
       ylabel='Speed (cm/s)', ylim=(0,75),
       title='High v low accel')
ax.legend([hsln, lsln], ['High', 'Low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_accel_speed{ext}',
                dpi=300, bbox_inches='tight')


#%% plotting - spike curves 
# ----------------------
# high v low MEAN SPEED
# ----------------------
high_speed_curve_mean = np.mean(all_high_speed_curve, axis=0)[1250*2:1250*7]
high_speed_curve_sem = sem(all_high_speed_curve, axis=0)[1250*2:1250*7]
low_speed_curve_mean = np.mean(all_low_speed_curve, axis=0)[1250*2:1250*7]
low_speed_curve_sem = sem(all_low_speed_curve, axis=0)[1250*2:1250*7]

fig, ax = plt.subplots(figsize=(1.8,1.4))

hsln, = ax.plot(XAXIS_SPIKE_TIME, high_speed_curve_mean, color='orange')
ax.fill_between(XAXIS_SPIKE_TIME, high_speed_curve_mean+high_speed_curve_sem,
                                  high_speed_curve_mean-high_speed_curve_sem,
                alpha=.25, color='orange', edgecolor='none')
lsln, = ax.plot(XAXIS_SPIKE_TIME, low_speed_curve_mean, color='navajowhite')
ax.fill_between(XAXIS_SPIKE_TIME, low_speed_curve_mean+low_speed_curve_sem,
                                  low_speed_curve_mean-low_speed_curve_sem,
                alpha=.25, color='navajowhite', edgecolor='none')

ax.set(xlabel='Time from run onset (s)', 
       ylabel='Firing rate (Hz)', ylim=(1.5,5.7),
       title='High v low speed')
ax.legend([hsln, lsln], ['High', 'Low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_speed_spike_curves{ext}',
                dpi=300, bbox_inches='tight')


# -----------------------
# high v low INIT. SPEED
# -----------------------
high_init_speed_curve_mean = np.mean(all_high_init_speed_curve, axis=0)[1250*2:1250*7]
high_init_speed_curve_sem  = sem(all_high_init_speed_curve, axis=0)[1250*2:1250*7]
low_init_speed_curve_mean  = np.mean(all_low_init_speed_curve, axis=0)[1250*2:1250*7]
low_init_speed_curve_sem   = sem(all_low_init_speed_curve, axis=0)[1250*2:1250*7]

fig, ax = plt.subplots(figsize=(1.8,1.4))

hsln, = ax.plot(XAXIS_SPIKE_TIME, high_init_speed_curve_mean, color='orange')
ax.fill_between(XAXIS_SPIKE_TIME, high_init_speed_curve_mean + high_init_speed_curve_sem,
                                  high_init_speed_curve_mean - high_init_speed_curve_sem,
                alpha=.25, color='orange', edgecolor='none')
lsln, = ax.plot(XAXIS_SPIKE_TIME, low_init_speed_curve_mean, color='navajowhite')
ax.fill_between(XAXIS_SPIKE_TIME, low_init_speed_curve_mean + low_init_speed_curve_sem,
                                  low_init_speed_curve_mean - low_init_speed_curve_sem,
                alpha=.25, color='navajowhite', edgecolor='none')

ax.set(xlabel='Time from run onset (s)', 
       ylabel='Firing rate (Hz)', ylim=(1.5,5.7),
       title='High v low init. speed')
ax.legend([hsln, lsln], ['High', 'Low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_init_speed_spike_curves{ext}',
                dpi=300, bbox_inches='tight')
    

# ------------------------
# high v low INIT. ACCEL.
# ------------------------
high_accel_curve_mean = np.mean(all_high_accel_curve, axis=0)[1250*2:1250*7]
high_accel_curve_sem = sem(all_high_accel_curve, axis=0)[1250*2:1250*7]
low_accel_curve_mean = np.mean(all_low_accel_curve, axis=0)[1250*2:1250*7]
low_accel_curve_sem = sem(all_low_accel_curve, axis=0)[1250*2:1250*7]

fig, ax = plt.subplots(figsize=(1.8,1.4))

hsln, = ax.plot(XAXIS_SPIKE_TIME, high_accel_curve_mean, color='firebrick')
ax.fill_between(XAXIS_SPIKE_TIME, high_accel_curve_mean+high_accel_curve_sem,
                                  high_accel_curve_mean-high_accel_curve_sem,
                alpha=.25, color='firebrick', edgecolor='none')
lsln, = ax.plot(XAXIS_SPIKE_TIME, low_accel_curve_mean, color='lightcoral')
ax.fill_between(XAXIS_SPIKE_TIME, low_accel_curve_mean+low_accel_curve_sem,
                                  low_accel_curve_mean-low_accel_curve_sem,
                alpha=.25, color='lightcoral', edgecolor='none')

ax.set(xlabel='Time from run onset (s)', 
       ylabel='Firing rate (Hz)', ylim=(1.5,5.7),
       title='High v low accel')
ax.legend([hsln, lsln], ['High', 'low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_accel_spike_curves{ext}',
                dpi=300, bbox_inches='tight')


#%% plotting - amp
pf.plot_violin_with_scatter(all_low_speed_amp, all_high_speed_amp, 
                            'navajowhite', 'orange',
                            xticklabels=['Low speed', 'High speed'],
                            ylabel='Firing rate (Hz)',
                            ylim=(0, 10),
                            print_statistics=True,
                            save=True,
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\speed_35_45_55'
                            )

pf.plot_violin_with_scatter(all_low_init_speed_amp, all_high_init_speed_amp, 
                            'navajowhite', 'orange',
                            xticklabels=['Low init. speed', 'High init. speed'],
                            ylabel='Firing rate (Hz)',
                            ylim=(0, 10),
                            save=True,
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\init_speed_25_35_45'
                            )

pf.plot_violin_with_scatter(all_low_accel_amp, all_high_accel_amp, 
                            'lightcoral', 'firebrick',
                            xticklabels=['Low init. accel.', 'High init. accel.'],
                            ylabel='Firing rate (Hz)',
                            ylim=(0, 10),
                            save=True,
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\init_accel_60_80'
                            )


#%% corr between init speed and FR
rvals = np.array(all_session_init_speed_FR_r , float)
rvals = rvals[~np.isnan(rvals)]
n_sess = len(rvals)

median_r = np.median(rvals)
mean_r   = np.mean(rvals)
sem_r    = np.std(rvals, ddof=1) / np.sqrt(n_sess)

# IQR
q25, q75 = np.percentile(rvals, [25, 75])
iqr_r = q75 - q25

# stats
w_stat, p_w = wilcoxon(rvals, alternative='two-sided')
t_stat, p_t = ttest_1samp(rvals, popmean=0)

print(f'N sessions = {n_sess}')
print(f'Median r = {median_r:.3f}')
print(f'Mean r ± SEM = {mean_r:.3f} ± {sem_r:.3f}')
print(f'IQR = [{q25:.3f}, {q75:.3f}] (IQR = {iqr_r:.3f})')
print(f'Wilcoxon vs 0: W = {w_stat:.3f}, p = {p_w:.3g}')
print(f'T-test vs 0: T = {t_stat:.3f}, p = {p_t:.3g}')

fig, ax = plt.subplots(figsize=(2.0, 2.2))

parts = ax.violinplot(rvals, positions=[1],
                      showmeans=False, showmedians=True,
                      showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('orange')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)

parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

ax.scatter(np.ones(n_sess), rvals,
           s=12, color='orange', ec='none', alpha=0.55, zorder=3)

ax.axhline(0, color='gray', lw=1, ls='--')

ax.text(
    1.35, np.max(rvals),
    f'Median = {median_r:.2f}\n'
    f'IQR = [{q25:.2f}, {q75:.2f}]\n'
    f'{mean_r:.2f} ± {sem_r:.2f}\n'
    f'Wilc {p_w:.2e}\n'
    f'Ttest {p_t:.2e}',
    ha='left', va='top', fontsize=7, color='forestgreen'
)

ax.set(
    xlim=(0.5, 1.5),
    xticks=[1],
    xticklabels=['corr(init. speed, RO FR)'],
    ylim=(-0.5, 0.5),
    ylabel='Correlation (r)',
    title='Across-session corr.'
)

ax.spines[['top', 'right', 'bottom']].set_visible(False)
plt.tight_layout()
    

#%% corr between init accel and FR
rvals = np.array(all_session_init_accel_FR_r , float)
rvals = rvals[~np.isnan(rvals)]
n_sess = len(rvals)

median_r = np.median(rvals)
mean_r   = np.mean(rvals)
sem_r    = np.std(rvals, ddof=1) / np.sqrt(n_sess)

# IQR
q25, q75 = np.percentile(rvals, [25, 75])
iqr_r = q75 - q25

# stats
w_stat, p_w = wilcoxon(rvals, alternative='two-sided')
t_stat, p_t = ttest_1samp(rvals, popmean=0)

print(f'N sessions = {n_sess}')
print(f'Median r = {median_r:.3f}')
print(f'Mean r ± SEM = {mean_r:.3f} ± {sem_r:.3f}')
print(f'IQR = [{q25:.3f}, {q75:.3f}] (IQR = {iqr_r:.3f})')
print(f'Wilcoxon vs 0: W = {w_stat:.3f}, p = {p_w:.3g}')
print(f'T-test vs 0: T = {t_stat:.3f}, p = {p_t:.3g}')

fig, ax = plt.subplots(figsize=(2.0, 2.2))

parts = ax.violinplot(rvals, positions=[1],
                      showmeans=False, showmedians=True,
                      showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('darkred')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)

parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

ax.scatter(np.ones(n_sess), rvals,
           s=12, color='darkred', ec='none', alpha=0.55, zorder=3)

ax.axhline(0, color='gray', lw=1, ls='--')

ax.text(
    1.35, np.max(rvals),
    f'Median = {median_r:.2f}\n'
    f'IQR = [{q25:.2f}, {q75:.2f}]\n'
    f'{mean_r:.2f} ± {sem_r:.2f}\n'
    f'Wilc {p_w:.2e}\n'
    f'Ttest {p_t:.2e}',
    ha='left', va='top', fontsize=7, color='forestgreen'
)

ax.set(
    xlim=(0.5, 1.5),
    xticks=[1],
    xticklabels=['corr(init. accel, RO FR)'],
    ylim=(-0.5, 0.5),
    ylabel='Correlation (r)',
    title='Across-session corr.'
)

ax.spines[['top', 'right', 'bottom']].set_visible(False)
plt.tight_layout()
