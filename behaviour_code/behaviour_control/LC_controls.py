# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:29:38 2025

controls for LC run-onset peaks

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import sys 
import pandas as pd 
from scipy.stats import sem 

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf
from common import mpl_formatting
mpl_formatting()


#%% load data 
print('loading data...')
cell_prop = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )


#%% get keys for different categories of cells 
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


#%% parameters for processing
SAMP_FREQ = 1250 
RUN_ONSET_BIN = 3750
BEF = 1  # s, how much time before run-onset to get
AFT = 4  # same as above 
WINDOW_HALF_SIZE = .5

RO_WINDOW = [
    int(RUN_ONSET_BIN - WINDOW_HALF_SIZE * SAMP_FREQ), 
    int(RUN_ONSET_BIN + WINDOW_HALF_SIZE * SAMP_FREQ)
    ]  # window for spike summation, half a sec around run onsets

XAXIS_SPIKE_TIME = np.arange(1250*5) / 1250 - 1  # 5 seconds 
XAXIS_SPEED_TIME = np.arange(4000) / 1000  # 4 seconds 


#%% main 
all_high_speed_speed = []
all_low_speed_speed = []
all_high_accel_speed = []
all_low_accel_speed = []

all_high_speed_curve = []
all_low_speed_curve = []
all_high_accel_curve = []
all_low_accel_curve = []

all_high_speed_amp = []
all_low_speed_amp = []
all_high_accel_amp = []
all_low_accel_amp = []

for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    
    trains = np.load(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}\{recname}_all_trains_run.npy',
        allow_pickle=True
        ).item()
    
    with open(
            rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\LC\{recname}.pkl',
            'rb'
            ) as f:
        beh = pickle.load(f)
    
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds)
                if cond!='0']
    bad_idx = [trial for trial, bad in enumerate(beh['bad_trials'][1:])
               if bad]
    valid_trial_idx = [trial for trial in np.arange(len(stim_conds)) 
                       if trial not in stim_idx and trial not in bad_idx]
    
    speed_trials = [[t[1] for t in trial]
                    for trial in beh['speed_times_aligned'][1:]]
    mean_speed_trials = [np.mean(trial) for trial in speed_trials]
    high_speed_idx = [trial for trial, speed in enumerate(mean_speed_trials)
                      if 45<speed<55]
    low_speed_idx = [trial for trial, speed in enumerate(mean_speed_trials)
                     if 35<speed<45]
    
    acceleration_trials = [np.mean(np.diff(trial[:500])) * 1_000
                           for trial in speed_trials]
    high_accel_idx = [trial for trial, accel in enumerate(acceleration_trials)
                      if accel>80]
    low_accel_idx = [trial for trial, accel in enumerate(acceleration_trials)
                     if 60<accel<80]
    
    # accumulate speed data 
    high_speed_speed = []
    low_speed_speed = []
    high_accel_speed = []
    low_accel_speed = []
    for trial in valid_trial_idx:
        if trial in high_speed_idx:
            high_speed_speed.append(speed_trials[trial])
        if trial in low_speed_idx:
            low_speed_speed.append(speed_trials[trial])
        if trial in high_accel_idx:
            high_accel_speed.append(speed_trials[trial])
        if trial in low_accel_idx:
            low_accel_speed.append(speed_trials[trial])
            
    # process speeds to be equal length
    if len(high_speed_idx)>0 and len(low_speed_idx)>0:
        speed_speed_min_length = min([len(l) for l in high_speed_speed] + [len(l) for l in low_speed_speed])
        high_speed_speed = [l[:speed_speed_min_length] for l in high_speed_speed]
        low_speed_speed = [l[:speed_speed_min_length] for l in low_speed_speed]
        
        if high_speed_speed and low_speed_speed:
            all_high_speed_speed.append(np.mean(high_speed_speed, axis=0))
            all_low_speed_speed.append(np.mean(low_speed_speed, axis=0))
    
    if len(high_accel_idx)>0 and len(low_accel_idx)>0:
        accel_speed_min_length = min([len(l) for l in high_accel_speed] + [len(l) for l in low_accel_speed])
        high_accel_speed = [l[:accel_speed_min_length] for l in high_accel_speed]
        low_accel_speed = [l[:accel_speed_min_length] for l in low_accel_speed]
    
        if high_accel_speed and low_accel_speed:
            all_high_accel_speed.append(np.mean(high_accel_speed, axis=0))
            all_low_accel_speed.append(np.mean(low_accel_speed, axis=0))
    
    # accumulate spike data
    curr_high_speed_curve = []
    curr_low_speed_curve = []
    curr_high_accel_curve = []
    curr_low_accel_curve = []
    
    curr_high_speed_amp = []
    curr_low_speed_amp = []
    curr_high_accel_amp = []
    curr_low_accel_amp = []
    
    for clu in list(trains.keys()):
        if clu in RO_keys:
            # containers for individual cells             
            high_speed_curve = []
            low_speed_curve = []
            high_accel_curve = []
            low_accel_curve = []
            
            high_speed_amp = []
            low_speed_amp = [] 
            high_accel_amp = []
            low_accel_amp = []
            
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
            if high_accel_curve and low_accel_curve:
                curr_high_accel_curve.append(np.mean(high_accel_curve, axis=0))
                curr_low_accel_curve.append(np.mean(low_accel_curve, axis=0))
            
            if high_speed_amp and low_speed_amp:
                curr_high_speed_amp.append(np.mean(high_speed_amp))
                curr_low_speed_amp.append(np.mean(low_speed_amp))
            if high_accel_amp and low_accel_amp:
                curr_high_accel_amp.append(np.mean(high_accel_amp))
                curr_low_accel_amp.append(np.mean(low_accel_amp))
    
    mean_high_speed_curve = np.mean(curr_high_speed_curve, axis=0)
    mean_low_speed_curve = np.mean(curr_low_speed_curve, axis=0)
    mean_high_accel_curve = np.mean(curr_high_accel_curve, axis=0)
    mean_low_accel_curve = np.mean(curr_low_accel_curve, axis=0)
    
    if len(curr_high_speed_curve)>0 and len(curr_low_speed_curve)>0:
        all_high_speed_curve.append(np.mean(curr_high_speed_curve, axis=0))
        all_low_speed_curve.append(np.mean(curr_low_speed_curve, axis=0))
    if len(curr_high_accel_curve)>0 and len(curr_low_accel_curve)>0:
        all_high_accel_curve.append(np.mean(curr_high_accel_curve, axis=0))
        all_low_accel_curve.append(np.mean(curr_low_accel_curve, axis=0))
    
    mean_high_speed_amp = np.mean(curr_high_speed_amp)
    mean_low_speed_amp = np.mean(curr_low_speed_amp)
    mean_high_accel_amp = np.mean(curr_high_accel_amp)
    mean_low_accel_amp = np.mean(curr_low_accel_amp)
    
    if not np.isnan(mean_high_speed_amp) and not np.isnan(mean_low_speed_amp):
        all_high_speed_amp.append(np.mean(curr_high_speed_amp))
        all_low_speed_amp.append(np.mean(curr_low_speed_amp))
    if not np.isnan(mean_high_accel_amp) and not np.isnan(mean_low_accel_amp):
        all_high_accel_amp.append(np.mean(curr_high_accel_amp))
        all_low_accel_amp.append(np.mean(curr_low_accel_amp))


#%% plotting - speed 
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

hsln, = ax.plot(XAXIS_SPEED_TIME, high_speed_speed_mean, color='royalblue')
ax.fill_between(XAXIS_SPEED_TIME, high_speed_speed_mean+high_speed_speed_sem,
                                  high_speed_speed_mean-high_speed_speed_sem,
                alpha=.25, color='royalblue', edgecolor='none')
lsln, = ax.plot(XAXIS_SPEED_TIME, low_speed_speed_mean, color='lightsteelblue')
ax.fill_between(XAXIS_SPEED_TIME, low_speed_speed_mean+low_speed_speed_sem,
                                  low_speed_speed_mean-low_speed_speed_sem,
                alpha=.25, color='lightsteelblue', edgecolor='none')

ax.set(xlabel='time from run onset (s)', 
       ylabel='speed (cm·s$^{-1}$)', ylim=(0,75),
       title='high v low speed')
ax.legend([hsln, lsln], ['high', 'low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_speed_speed{ext}',
                dpi=300, bbox_inches='tight')


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

ax.set(xlabel='time from run onset (s)', 
       ylabel='speed (cm·s$^{-1}$)', ylim=(0,75),
       title='high v low accel')
ax.legend([hsln, lsln], ['high', 'low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_accel_speed{ext}',
                dpi=300, bbox_inches='tight')


#%% plotting - spike curves 
high_speed_curve_mean = np.mean(all_high_speed_curve, axis=0)[1250*2:1250*7]
high_speed_curve_sem = sem(all_high_speed_curve, axis=0)[1250*2:1250*7]
low_speed_curve_mean = np.mean(all_low_speed_curve, axis=0)[1250*2:1250*7]
low_speed_curve_sem = sem(all_low_speed_curve, axis=0)[1250*2:1250*7]

fig, ax = plt.subplots(figsize=(1.8,1.4))

hsln, = ax.plot(XAXIS_SPIKE_TIME, high_speed_curve_mean, color='royalblue')
ax.fill_between(XAXIS_SPIKE_TIME, high_speed_curve_mean+high_speed_curve_sem,
                                  high_speed_curve_mean-high_speed_curve_sem,
                alpha=.25, color='royalblue', edgecolor='none')
lsln, = ax.plot(XAXIS_SPIKE_TIME, low_speed_curve_mean, color='lightsteelblue')
ax.fill_between(XAXIS_SPIKE_TIME, low_speed_curve_mean+low_speed_curve_sem,
                                  low_speed_curve_mean-low_speed_curve_sem,
                alpha=.25, color='lightsteelblue', edgecolor='none')

ax.set(xlabel='time from run onset (s)', 
       ylabel='spike rate (Hz)', ylim=(1.5,5.7),
       title='high v low speed')
ax.legend([hsln, lsln], ['high', 'low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_speed_spike_curves{ext}',
                dpi=300, bbox_inches='tight')


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

ax.set(xlabel='time from run onset (s)', 
       ylabel='spike rate (Hz)', ylim=(1.5,5.7),
       title='high v low accel')
ax.legend([hsln, lsln], ['high', 'low'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\high_low_accel_spike_curves{ext}',
                dpi=300, bbox_inches='tight')


#%% plotting - amp
pf.plot_violin_with_scatter(all_low_speed_amp, all_high_speed_amp, 
                            'lightsteelblue', 'royalblue',
                            xticklabels=['low speed', 'high speed'],
                            ylabel='spike rate (Hz)',
                            ylim=(1, 9.8),
                            save=True,
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\speed_35_45_55'
                            )
pf.plot_violin_with_scatter(all_low_accel_amp, all_high_accel_amp, 
                            'lightcoral', 'firebrick',
                            xticklabels=['low accel.', 'high accel.'],
                            ylabel='spike rate (Hz)',
                            ylim=(1, 9.8),
                            save=True,
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\speed_controls\init_accel_60_80'
                            )