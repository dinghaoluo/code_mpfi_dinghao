# -*- coding: utf-8 -*-
"""
Created on Mon 20 Nov 14:55:04 2023

quantify lick density etc 

@author: Dinghao Luo 
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
import scipy.io as sio
import sys 
from scipy.stats import wilcoxon, ttest_rel


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLCopt


#%% plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if (r'Z:\Dinghao\code_mpfi_dinghao\iutils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf


#%% MAIN 
stim_std_med = []; stim_std_mean = []
cont_std_med = []; cont_std_mean = []
for pathname in pathLC:
    sessname = pathname[-17:]
    print(sessname)

    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    alignRun = sio.loadmat(filename)    
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:
            pumps[trial] = [20000]
    
    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    behInfo = sio.loadmat(filename)['beh']
    stim_trial = np.squeeze(np.where(behInfo['pulseMethod'][0][0][0]!=0))-1
    stim_cont = np.arange(stim_trial[0])
    tot_trial = len(behInfo['pulseMethod'][0][0][0])-1
    
    all_licks = []
    for trial in range(tot_trial):
        # only if the animal does not lick in the first second (carry-over licks) and only include pre-consumption licks
        lk = [l[0] for l in licks[trial] if l-starts[trial]>1250]
        all_licks.append(lk-starts[trial])

    stim_std = [np.std(lk) for trial, lk in enumerate(all_licks) if trial in stim_trial]
    cont_std = [np.std(lk) for trial, lk in enumerate(all_licks) if trial in stim_cont]
    
    stim_std_med.append(np.nanmedian(stim_std)/1250)
    cont_std_med.append(np.nanmedian(cont_std)/1250)
    
    stim_std_mean.append(np.nanmean(stim_std)/1250)
    cont_std_mean.append(np.nanmean(cont_std)/1250)
    

#%% plotting
savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\lick_std_summary_median'
pf.plot_violin_with_scatter(cont_std_med, stim_std_med, 'grey', 'royalblue',
                            paired=True,
                            title='lick std. ctrl. v stim.', 
                            xticklabels=['ctrl.', 'stim.'],
                            ylabel='lick std.',
                            save=True, savepath=savepath, dpi=300)