# -*- coding: utf-8 -*-
"""
Created on Mon 20 Nov 14:55:04 2023
Modified on 11 Sept 2025

quantify lick density etc 
modified to plot also the lick distributions (ctrl vs stim)

@author: Dinghao Luo 
"""


#%% imports 
from pathlib import Path 

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats import wilcoxon, ttest_rel

from plotting_functions import plot_violin_with_scatter
from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathLCopt


#%% path stems 
exp_stem = Path(r'Z:/Dinghao/MiceExp')


#%% parameters 
SAMP_FREQ = 1250  # Hz


#%% MAIN 
stim_licks = [] 
ctrl_licks = []

stim_std_med = []; stim_std_mean = []
ctrl_std_med = []; ctrl_std_mean = []

for path in paths:
    recname = Path(path).name
    print(recname)
    
    run_file_path = exp_stem / f'ANMD{recname[1:5]}' / recname[:14] / recname / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
    alignRun = sio.loadmat(run_file_path) 
    
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:
            pumps[trial] = [np.nan]
    
    beh_file_path = exp_stem / f'ANMD{recname[1:5]}' / recname[:14] / recname / f'{recname}_DataStructure_mazeSection1_TrialType1_Info.mat'
    behInfo = sio.loadmat(beh_file_path)['beh']
    
    stim_idx = np.squeeze(np.where(behInfo['pulseMethod'][0][0][0]!=0))-1
    ctrl_idx = np.arange(stim_idx[0])
    
    tot_trial = len(behInfo['pulseMethod'][0][0][0])-1
    
    all_licks = []
    for trial in range(tot_trial):
        # only if the animal does not lick in the first second (carry-over licks) and only include pre-consumption licks
        curr_start = starts[trial]
        
        curr_licks = [(l[0] - curr_start) / SAMP_FREQ for l in licks[trial] 
                      if (l[0] - curr_start) > SAMP_FREQ]
        all_licks.append(curr_licks)
        
    stim_licks.extend([lk for trial, lk in enumerate(all_licks) 
                       if trial in stim_idx and lk])
    ctrl_licks.extend([lk for trial, lk in enumerate(all_licks) 
                       if trial in ctrl_idx and lk])
        
    stim_std = [np.std(lk) for trial, lk in enumerate(all_licks) if trial in stim_idx]
    ctrl_std = [np.std(lk) for trial, lk in enumerate(all_licks) if trial in ctrl_idx]
    
    stim_std_med.append(np.nanmedian(stim_std))
    ctrl_std_med.append(np.nanmedian(ctrl_std))
    
    stim_std_mean.append(np.nanmean(stim_std))
    ctrl_std_mean.append(np.nanmean(ctrl_std))
    

#%% std comparison
savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\lick_std_summary_median'
plot_violin_with_scatter(ctrl_std_med, stim_std_med, 'grey', 'royalblue',
                         paired=True,
                         title='lick std. ctrl. v stim.', 
                         xticklabels=['ctrl.', 'stim.'],
                         ylabel='lick std.',
                         save=True, savepath=savepath, dpi=300)


#%% lick profiles 
plt.figure(figsize=(6,4))

# flatten lists of arrays → single 1D array of lick times
ctrl_all = np.concatenate(ctrl_licks)
stim_all = np.concatenate(stim_licks)

# define bins (e.g., 0–10 s in 0.25 s bins, adjust as needed)
bins = np.arange(0, 10, 0.05)

# normalised histograms → mean density
plt.hist(ctrl_all, bins=bins, density=True, histtype='step', 
         color='grey', label='ctrl')
plt.hist(stim_all, bins=bins, density=True, histtype='step', 
         color='royalblue', label='stim')

plt.xlabel('time from run onset (s)')
plt.ylabel('lick density (a.u.)')
plt.title('mean lick distribution (ctrl vs stim)')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()