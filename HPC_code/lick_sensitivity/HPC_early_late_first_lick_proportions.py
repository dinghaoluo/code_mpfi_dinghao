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
import scipy.io as sio
import pandas as pd 
import sys 
import matplotlib.cm as cm 

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from plotting_functions import plot_violin_with_scatter

# pre_post ratio thresholds 
run_onset_activated_thres = 0.80
run_onset_inhibited_thres = 1.25


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


#%% functions 
def classify_run_onset_activation_ratio(train, 
                                        run_onset_activated_thres,
                                        run_onset_inhibited_thres):
    """
    classify run-onset activation ratio based on pre- and post-run periods.

    parameters:
    - train: array of firing rates over time.
    - run_onset_activated_thres: threshold for classifying activation.
    - run_onset_inhibited_thres: threshold for classifying inhibition.
    - samp_freq: sampling frequency in Hz, default is 1250.
    - run_onset_bin: bin marking the run onset, default is 3750.

    returns:
    - ratio: pre/post activation ratio.
    - ratiotype: string indicating the activation class ('ON', 'OFF', 'unresponsive').
    """
    samp_freq = 1250
    run_onset_bin = 3750
    
    pre = np.nanmean(train[int(run_onset_bin-samp_freq*1.5):int(run_onset_bin-samp_freq*.5)])
    post = np.nanmean(train[int(run_onset_bin+samp_freq*.5):int(run_onset_bin+samp_freq*1.5)])
    ratio = pre/post
    if ratio < run_onset_activated_thres:
        ratiotype = 'run-onset ON'
    elif ratio > run_onset_inhibited_thres:
        ratiotype = 'run-onset OFF'
    else:
        ratiotype = 'run-onset unresponsive'
        
    return ratio, ratiotype


#%% main loop 
early_ON_props = []
late_ON_props = []
early_OFF_props = []
late_OFF_props = []

for path in paths:
    recname = path[-17:]
    
    # get lick times 
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
    
    # get bad trials 
    behPar = sio.loadmat(
        rf'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}/{recname[:14]}/{recname}'
        rf'/{recname}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
        )
    bad_beh_ind = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    
    # get early and late lick trials (that are not stim. trials)
    stim_trials = np.where(
        np.asarray([
            trial[15] for trial
            in curr_beh_df['trial_statements']
            ])!='0'
        )[0]
    valid_trials = [i for i in range(len(first_licks)) 
                    if i not in stim_trials 
                    and i not in bad_beh_ind 
                    and not np.isnan(first_licks[i])]
    
    if len(valid_trials) < 50:
        continue 
    
    # load spike trains
    print(f'\n{recname}')
    trains = np.load(
        r'Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions'
        rf'/{recname}/{recname}_all_trains.npy',
        allow_pickle=True
        ).item()
        
    sorted_trials = sorted(valid_trials, 
                           key=lambda i: first_licks[i])[10:-10]  # avoid extremities 
    
    early_trials = sorted_trials[:10]
    late_trials = sorted_trials[-10:]
    
    # get cell spiking data 
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    
    early_ON_count = 0
    late_ON_count = 0
    early_OFF_count = 0
    late_OFF_count = 0
    
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
        early_prof = np.mean(early_trains, axis=0)
        late_prof = np.mean(late_trains, axis=0)
        
        early_ratio, early_ratiotype = classify_run_onset_activation_ratio(
            early_prof, 
            run_onset_activated_thres, 
            run_onset_inhibited_thres
            )
        late_ratio, late_ratiotype = classify_run_onset_activation_ratio(
            late_prof, 
            run_onset_activated_thres, 
            run_onset_inhibited_thres
            )
        
        ratiotype = session['class']
        if early_ratiotype == 'run-onset ON':
            early_ON_count+=1
        elif early_ratiotype == 'run-onset OFF':
            early_OFF_count+=1
        if late_ratiotype == 'run-onset ON':
            late_ON_count+=1
        elif late_ratiotype == 'run-onset OFF':
            late_OFF_count+=1
        
        # plot_colours = [cm.Reds(i / (len(sorted_trials)-1)) 
        #                 for i in range(len(sorted_trials))]
        # if ratiotype!='run-onset unresponsive':
        #     fig, ax = plt.subplots(figsize=(3,2))
        #     for i, trial in enumerate(sorted_trials):
        #         ax.plot(np.arange(-1250, 1250*4)/1250,
        #                 train[trial, 3750-1250:3750+4*1250],
        #                 color=plot_colours[i],
        #                 linewidth=1)
        #     ax.set(xlabel='time from run-onset (s)',
        #            ylabel='spike rate (Hz)',
        #            title=f'{cluname}\n{ratiotype}')
            
        #     fig.savefig(
        #         r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_alignment'
        #         rf'\single_cell_ON_OFF\{cluname} {ratiotype}.png',
        #         dpi=300,
        #         bbox_inches='tight'
        #         )
        
    early_ON_props.append(early_ON_count/len(curr_df_pyr))
    early_OFF_props.append(early_OFF_count/len(curr_df_pyr))
    late_ON_props.append(late_ON_count/len(curr_df_pyr))
    late_OFF_props.append(late_OFF_count/len(curr_df_pyr))
        

#%% compute results 
plot_violin_with_scatter(early_ON_props, late_ON_props, 'orange', 'darkred')