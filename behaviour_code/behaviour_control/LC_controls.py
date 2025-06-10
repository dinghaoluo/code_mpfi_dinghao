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


#%% main 
all_high_speed_amp = []
all_low_speed_amp = []
all_high_accel_amp = []
all_low_accel_amp = []

for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    
    trains = np.load(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}\{recname}_all_trains.npy',
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
    
    curr_high_speed_amp = []
    curr_low_speed_amp = []
    curr_high_accel_amp = []
    curr_low_accel_amp = []
    for clu in list(trains.keys()):
        if clu in RO_keys:
            high_speed_amp = []
            low_speed_amp = [] 
            high_accel_amp = []
            low_accel_amp = []
            curr_train = trains[clu]
            for trial in valid_trial_idx:
                if trial in high_speed_idx:
                    high_speed_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                if trial in low_speed_idx:
                    low_speed_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                if trial in high_accel_idx:
                    high_accel_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                if trial in low_accel_idx:
                    low_accel_amp.append(
                        np.mean(curr_train[trial][RO_WINDOW[0]:RO_WINDOW[1]])
                        )
                    
            if high_speed_amp and low_speed_amp:
                curr_high_speed_amp.append(np.mean(high_speed_amp))
                curr_low_speed_amp.append(np.mean(low_speed_amp))
            if high_accel_amp and low_accel_amp:
                curr_high_accel_amp.append(np.mean(high_accel_amp))
                curr_low_accel_amp.append(np.mean(low_accel_amp))
                
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


#%% plotting 
pf.plot_violin_with_scatter(all_low_speed_amp, all_high_speed_amp, 
                            'skyblue', 'steelblue',
                            xticklabels=['low speed', 'high speed'],
                            ylabel='spike rate (Hz)',
                            ylim=(1, 9.8),
                            save=True,
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\peak_amp_controls\speed_35_45_55'
                            )

pf.plot_violin_with_scatter(all_low_accel_amp, all_high_accel_amp, 
                            'skyblue', 'steelblue',
                            xticklabels=['low accel.', 'high accel.'],
                            ylabel='spike rate (Hz)',
                            ylim=(1, 9.8),
                            save=True,
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\peak_amp_controls\init_accel_60_80'
                            )