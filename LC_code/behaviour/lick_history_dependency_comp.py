# -*- coding: utf-8 -*-
"""
Created on Fri 18 Aug 17:41:33 2023

analyse the history dependency of 1st lick timing in opto-sessions (all)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['font.family'] = 'Arial'
import scipy.io as sio
from scipy.stats import linregress  # median used 
import sys

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathOpt = rec_list.pathLCopt

# 0-2-0
sess_list = [sess[-17:] for sess in pathOpt]

n_bst = 1000  # hyperparameter for bootstrapping
print('\n**BOOTSTRAP # = {}**\n'.format(n_bst))


#%% MAIN 
all_curr_trials = []; all_prev_trials = []
all_pvals = []

for sessname in sess_list:
    print(sessname)
    
    infofilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    
    Info = sio.loadmat(infofilename)
    pulseMethod = Info['beh'][0][0]['pulseMethod'][0]
    
    # stim info
    tot_stims = len([t for t in pulseMethod if t!=0])
    stim_cond = pulseMethod[np.where(pulseMethod!=0)][0]  # check stim condition
    stim = [i for i, e in enumerate(pulseMethod) if e==stim_cond]
    
    # licks 
    lickfilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    alignRun = sio.loadmat(lickfilename)
    
    # ignore all 1st trials since it is before counting starts and is an empty cell
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]

    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:  # trials where no rewards were delivered or rewards were delivered after 16 s
            pumps[trial] = 20000
    
    first_licks = []
    
    for trial in range(tot_trial):
        lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
        if len(lk)!=0:  # append only if there is licks in this trial
            first_licks.append((lk[0]-starts[trial])/1250)
        else:
            first_licks.append(0)
    
    # filtering to remove outliers using 3 standard deviations
    mean_licks = np.mean(first_licks)
    std_licks = np.std(first_licks)
    upper = mean_licks + 2*std_licks
    lower = mean_licks - 2*std_licks
    if lower<=0:  # to make sure that trials with no licks (see above) do not get in the dataset
        lower = 0
 
    curr_trials_stim_temp = [first_licks[i-1] for i in stim]
    prev_trials_stim_temp = [first_licks[i-2] for i in stim]
    
    # filtering
    curr_trials_stim = []
    prev_trials_stim = []
    for i in range(len(stim)):
        if curr_trials_stim_temp[i]<upper and curr_trials_stim_temp[i]>lower and prev_trials_stim_temp[i]<upper and prev_trials_stim_temp[i]>lower:
            curr_trials_stim.append(curr_trials_stim_temp[i])
            prev_trials_stim.append(prev_trials_stim_temp[i])
    
    delta_stim = [curr_trials_stim[i]-prev_trials_stim[i] for i in range(len(curr_trials_stim))]
    results_stim = linregress(prev_trials_stim, delta_stim)
    slope_stim = results_stim[0]
    rvalue_stim = results_stim[2]
    
    non_stim_trials = np.where(pulseMethod==0)[0]
    all_slope_non_stim = []
    all_rvalue_non_stim = []
    for i in range(n_bst):
        selected_non_stim = non_stim_trials[np.random.randint(0, stim[0]-1, len(curr_trials_stim))]
        
        curr_trials_non_stim = [first_licks[i-1] for i in selected_non_stim]
        prev_trials_non_stim = [first_licks[i-2] for i in selected_non_stim]
        delta_non_stim = [curr_trials_non_stim[i]-prev_trials_non_stim[i] for i in range(len(curr_trials_stim))]
        results_non_stim = linregress(prev_trials_non_stim, delta_non_stim)
        slope_non_stim = results_non_stim[0]
        rvalue_non_stim = results_non_stim[2]
        
        all_slope_non_stim.append(slope_non_stim)
        all_rvalue_non_stim.append(rvalue_non_stim)
    
    
    fig, ax = plt.subplots(figsize=(4,2.5))
    ax.scatter([0.5]*n_bst, all_slope_non_stim, color='grey', s=3)
    ax.scatter(1.5, slope_stim, color='grey', s=3)
    plt.show()
    
    # fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\history_dependency\{}.png'.format(sessname),
    #             dpi=300,
    #             bbox_inches='tight')