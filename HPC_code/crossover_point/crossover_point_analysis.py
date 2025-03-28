# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 17:51:40 2025

analyse the crossover point of ON and OFF cells at single-trial level 

@author: Dinghao Luo
"""

#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd 
import sys 

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt
# paths = rec_list.pathHPCLCopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from plotting_functions import plot_violin_with_scatter

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\HPC_code\utils')
import support_HPC as support


#%% parameters 
SAMP_FREQ = 1250  # Hz
TIME = np.arange(-SAMP_FREQ*3, SAMP_FREQ*7)/SAMP_FREQ  # 8 seconds 


#%% load data 
print('loading dataframes...')

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
    )
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 

beh_df = pd.concat((
    pd.read_pickle(
        r'Z:/Dinghao/code_dinghao/behaviour/all_HPCLC_sessions.pkl'
        ),
    pd.read_pickle(
        r'Z:/Dinghao/code_dinghao/behaviour/all_HPCLCterm_sessions.pkl'
        )
    ))


#%% main loop 
for path in paths:
    recname = path[-17:]
    print(recname)
    
    # get cells
    curr_df = df_pyr[df_pyr['recname']==recname]
    curr_ON_idx = [int(s.split(' ')[-3][3:])-2 for s in curr_df.index 
                   if curr_df.loc[s]['class']=='run-onset ON']
    curr_OFF_idx = [int(s.split(' ')[-3][3:])-2 for s in curr_df.index 
                   if curr_df.loc[s]['class']=='run-onset OFF']
    
    # filtering 
    if len(curr_ON_idx) < 5 or len(curr_OFF_idx) < 5:
        continue
    
    # get lick times 
    curr_beh_df = beh_df.loc[recname]  # subselect in read-only
    run_onsets = curr_beh_df['run_onsets'][1:]
    licks = [
        [(l-run_onset) for l in trial]
        if len(trial)!=0 else np.nan
        for trial, run_onset in zip(
                curr_beh_df['lick_times'][1:],
                run_onsets
                )
        ]
    first_licks = np.asarray(
        [next((l for l in trial if l > 1), np.nan)  # >1 to eliminate carry-over licks
        if isinstance(trial, list) else np.nan
        for trial in licks]
        )
    
    stim_trials = np.where(
        np.asarray([
            trial[15] for trial
            in curr_beh_df['trial_statements']
            ])!='0'
        )[0]
    ctrl_trials = stim_trials+2
    
    # get current session spike trains 
    clu_list, trains = support.load_train(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{}\{}_all_trains.npy'
        .format(recname, recname)
        )
    
    # trials 
    trials = np.arange(len(trains[0]))

    # main loop
    ON_mean_aligned = np.zeros((len(trials), 1250*6))
    OFF_mean_aligned = np.zeros((len(trials), 1250*6))
    for trial in trials:
        try:
            first_lick_bin = int(first_licks[trial]/1000*1250-3750)
            curr_ON_profiles = [trains[clu][trial] for clu in curr_ON_idx]
            curr_ON_mean = np.mean(curr_ON_profiles, axis=0)
            
            curr_OFF_profiles = [trains[clu][trial] for clu in curr_OFF_idx]
            curr_OFF_mean = np.mean(curr_OFF_profiles, axis=0)
            
            curr_ON_mean_aligned = curr_ON_mean[first_lick_bin-4*1250:first_lick_bin+2*1250]
            curr_OFF_mean_aligned = curr_OFF_mean[first_lick_bin-4*1250:first_lick_bin+2*1250]
            ON_mean_aligned[trial, :] = curr_ON_mean_aligned
            OFF_mean_aligned[trial, :] = curr_OFF_mean_aligned
        except ValueError:
            ON_mean_aligned[trial, :] = np.nan
            OFF_mean_aligned[trial, :] = np.nan
            
    ON_mean_aligned_mean = np.nanmean(ON_mean_aligned, axis=0)
    OFF_mean_aligned_mean = np.nanmean(OFF_mean_aligned, axis=0)

    fig, ax = plt.subplots(figsize=(3,2))
    ax.plot(np.arange(-1250*4, 1250*2)/1250,
            ON_mean_aligned_mean)
    ax.plot(np.arange(-1250*4, 1250*2)/1250,
            OFF_mean_aligned_mean)
    ax.set_title(recname)