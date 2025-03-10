# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:37:16 2023
Modified on Mon 10 Mar 17:45:12 2025:
    modified to work on HPC cells

loop over all cells for early v late trials

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import sys 

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt


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
early_profs = []
late_profs = []
early_ON_profs = []
late_ON_profs = []
early_OFF_profs = []
late_OFF_profs = []

for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    
    # load spike trains
    print('loading spike trains...')
    trains = np.load(
        r'Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions'
        rf'/{recname}/{recname}_all_trains.npy',
        allow_pickle=True
        ).item()
    
    # get lick times 
    print('extracting first lick times...')
    curr_beh_df = beh_df.loc[recname]  # subselect in read-only
    run_onsets = curr_beh_df['run_onsets'][1:]
    licks = [
        [(l-run_onset)/1000 for l in trial]  # convert from ms to s
        if len(trial)!=0 else np.nan
        for trial, run_onset in zip(
                curr_beh_df['lick_times'][1:],
                run_onsets
                )
        ]
    first_licks = np.asarray(
        [next((l for l in trial if l > 1), np.nan)  # >1 to prevent carry-over licks
        if isinstance(trial, list) else np.nan
        for trial in licks]
        )
    
    # get early and late lick trials (that are not stim. trials)
    stim_trials = np.where(
        np.asarray([
            trial[15] for trial
            in curr_beh_df['trial_statements']
            ])!='0'
        )[0]
    valid_trials = [i for i in range(len(first_licks)) 
                    if i not in stim_trials and not np.isnan(first_licks[i])]
    sorted_trials = sorted(valid_trials, 
                           key=lambda i: first_licks[i])
    
    early_trials = sorted_trials[:10]
    late_trials = sorted_trials[-10:]
    
    # get cell spiking data 
    print('extracting early and late 1st-lick spike trains...')
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    
    for cluname, session in curr_df_pyr.iterrows():
        train = trains[cluname]
        early_trains = [
            train[trial, :] for trial
            in early_trials
            ]
        late_trains = [
            train[trial, :] for trial
            in late_trials
            ]
        early_profs.append(np.mean(early_trains, axis=0))
        late_profs.append(np.mean(late_trains, axis=0))
        
        # ON and OFF only 
        if session['class']=='run-onset ON':
            early_ON_profs.append(np.mean(early_trains, axis=0))
            late_ON_profs.append(np.mean(late_trains, axis=0))
        elif session['class']=='run-onset OFF':
            early_OFF_profs.append(np.mean(early_trains, axis=0))
            late_OFF_profs.append(np.mean(late_trains, axis=0))
        

#%% plotting 
fig, ax = plt.subplots(figsize=(3,2))

ax.plot(np.mean(early_OFF_profs, axis=0))
ax.plot(np.mean(late_OFF_profs, axis=0), color='red')