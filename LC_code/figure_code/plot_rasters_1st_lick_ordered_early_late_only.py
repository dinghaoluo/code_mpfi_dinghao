# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:29:45 2023
Modified on Thu 19 Dec 2024 12:25:
    - improved readability 
    - attempts to modularise scripts

determine whether run-onset burst amplitude is correlated with early/late 1st-lick timing

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd
import sys 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% parameters 
samp_freq = 1250  # in Hz
run_onset_bin = 3750  # bin for run-onset
max_trial_length = 6  # in seconds 
time_bef = 1  # in seconds
time_aft = max_trial_length-time_bef
burst_window = .5  # in seconds, around the run-onset

xaxis = np.arange(max_trial_length * samp_freq) / samp_freq - time_bef


#%% load data 
all_rasters = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_rasters.npy',
    allow_pickle=True
    ).item()
all_trains = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_trains.npy',
    allow_pickle=True
    ).item()
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )

clu_list = list(cell_profiles.index)


#%% MAIN 
for cluname in clu_list:
    print(cluname)
    identity = cell_profiles.loc[cluname]['identity']
    
    rasters = all_rasters[cluname]
    trains = all_trains[cluname]
    
    align_run_file = sio.loadmat(
        r'Z:\Dinghao\MiceExp\ANMD{}\{}\{}\{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
        .format(
            cluname[1:5], cluname[:14], cluname[:17], cluname[:17]
            )
        )
    
    licks = align_run_file['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = align_run_file['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = align_run_file['trialsRun']['pumpLfpInd'][0][0][0][1:]
    tot_trials = licks.shape[0]
    for trial in range(tot_trials):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:
            pumps[trial] = np.nan
    
    beh_parameters_file = sio.loadmat(
        r'Z:\Dinghao\MiceExp\ANMD{}\{}\{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
        .format(
            cluname[1:5], cluname[:14], cluname[:17], cluname[:17]
            )
        )
    stim_on = beh_parameters_file['behPar']['stimOn'][0][0][0][1:]
    stim_idx = np.where(stim_on!=0)[0]+1  # +1 for legacy MATLAB indexing problem 
    bad_idx = np.where(beh_parameters_file['behPar'][0]['indTrBadBeh'][0]==1)[0]-1
    
    first_licks = []
    for trial in range(tot_trials):
        lk = [l for l in licks[trial] if l-starts[trial] > samp_freq]  # only if the animal does not lick in the first second (carry-over licks)
        if len(lk)==0:
            first_licks.append(np.nan)
        else:
            first_licks.extend(lk[0]-starts[trial])

    temp = list(np.arange(tot_trials))
    licks_ordered, temp_ordered = zip(*sorted(zip(first_licks, temp)))
    
    # pick out early and late trials for plotting, Dinghao 18 Sept 2024
    early_trials = []; late_trials = []
    for i in range(30):
        if temp_ordered[i] not in bad_idx and temp_ordered[i] not in stim_idx:
            if len(early_trials)<10:
                early_trials.append(temp_ordered[i])
        if temp_ordered[-(i+1)] not in bad_idx and temp_ordered[-(i+1)] not in stim_idx:  # this goes in the reverse direction 
            if len(late_trials)<10:
                late_trials.append(temp_ordered[-(i+1)])

    # plotting
    fig, axs = plt.subplots(2, 1, figsize=(3,2.5))
    
    early_prof = np.zeros((10, samp_freq * max_trial_length))
    late_prof = np.zeros((10, samp_freq * max_trial_length))
    
    for i, trial in enumerate(early_trials):
        curr_raster = rasters[trial]
        early_prof[i, :len(trains[trial][run_onset_bin-samp_freq:run_onset_bin+samp_freq*time_aft])] = \
            trains[trial][run_onset_bin-samp_freq:run_onset_bin+samp_freq*time_aft]*samp_freq
        curr_trial = np.where(curr_raster==1)[0]
        curr_trial = [(s - 3 * samp_freq) / samp_freq 
                      for s in curr_trial 
                      if s > run_onset_bin - samp_freq]  # starts from -1 s 
        
        axs[1].scatter(curr_trial, [i+1]*len(curr_trial),
                       color='grey', alpha=.7, s=.35)
        axs[1].plot([first_licks[trial]/samp_freq, first_licks[trial]/samp_freq],
                    [i, i+1],
                    linewidth=2, color='orchid')
        
    for i, trial in enumerate(reversed(late_trials)):
        curr_raster = rasters[trial]
        late_prof[i, :len(trains[trial][run_onset_bin-samp_freq:run_onset_bin+samp_freq*time_aft])] = \
            trains[trial][run_onset_bin-samp_freq:run_onset_bin+samp_freq*time_aft]*samp_freq
        curr_trial = np.where(curr_raster==1)[0]
        curr_trial = [(s - 3 * samp_freq) / samp_freq 
                      for s in curr_trial 
                      if s > run_onset_bin - samp_freq]  # starts from -1 s
        
        axs[0].scatter(curr_trial, [i+1]*len(curr_trial),
                       color='grey', alpha=.7, s=.35)
        axs[0].plot([first_licks[trial]/samp_freq, first_licks[trial]/samp_freq],
                    [i, i+1],
                    linewidth=2, color='orchid')
        
    e_mean = np.mean(early_prof, axis=0)
    l_mean = np.mean(late_prof, axis=0)
    max_y = max(max(e_mean), max(l_mean))
    min_y = min(min(e_mean), min(l_mean))
    
    axt1 = axs[1].twinx()
    axt1.plot(xaxis, np.mean(early_prof, axis=0), color='k')
    axt1.set(ylabel='spike rate (Hz)',
             ylim=(min_y, max_y))
    
    axt0 = axs[0].twinx()
    axt0.plot(xaxis, np.mean(late_prof, axis=0), color='k')
    axt0.set(ylabel='spike rate (Hz)',
             ylim=(min_y, max_y))
    
    for i in range(2):
        axs[i].set(xlabel='time from run-onset (s)', ylabel='trial #',
                   yticks=[1, 10], xticks=[0, 2, 4],
                   xlim=(-1, 5))
        for p in ['top', 'right']:
            axs[i].spines[p].set_visible(False)
    
    fig.suptitle(f'{cluname}\n{identity}')
    
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            r'Z:\Dinghao\code_dinghao\LC_ephys\lick_sensitivity\rasters_by_first_licks_earlyvlate_only\{}{}'
            .format(f'{cluname} {identity}', ext),
            dpi=300,
            bbox_inches='tight')
    
    plt.close(fig)