# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:22:43 2024

correlates run-onset population drifts (single-trial) with behavioural 
    parameters 

@author: Dinghao Luo
"""


#%% imports 
import sys
import scipy.io as sio 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import linregress, zscore 


#%% run HPC-LC or HPC-LCterm
HPC_LC = 1

# load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
if HPC_LC:
    pathHPC = rec_list.pathHPCLCopt
elif not HPC_LC:
    pathHPC = rec_list.pathHPCLCtermopt


#%% main (single-trial act./inh. proportions)
all_act_props_zscore = []; all_act_props_baseline_zscore = []
all_licks_zscore = []; all_licks_baseline_zscore = []
for pathname in pathHPC:
    recname = pathname[-17:]
    if recname=='A063r-20230708-02' or recname=='A063r-20230708-01':    # lick detection problems 
        continue
    print(recname)
    
    trains = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
                          allow_pickle=True).item().values())
    tot_trial = len(trains[0])
    
    # determine if each cell is pyramidal or intern 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    tot_pyr = sum(pyr_id)
    
    # behaviour parameters
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2
    tot_trial = len(stimOn)
    tot_baseline = stim_trials[0]

    # for each trial we get act. and inh. proportions
    act_props = []; inh_props = []
    for trial in range(tot_trial):
        trial_act = 0; trial_inh = 0
        for i, if_pyr in enumerate(pyr_id):
            if if_pyr:
                # calculate pre/post ratio
                pp = trains[i][trial][1875:3125].mean()/trains[i][trial][4375:5625].mean()
                if pp>=1.25:
                    trial_inh+=1
                elif pp<=.8:
                    trial_act+=1
        act_props.append(trial_act/tot_pyr); inh_props.append(trial_inh/tot_pyr)
    act_props_baseline = act_props[:stim_trials[0]]
    inh_props_baseline = inh_props[:stim_trials[0]]
    
    # behaviour
    lickfilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(recname[1:5], recname[:14], recname[:17], recname[:17])
    alignRun = sio.loadmat(lickfilename)
    
    # ignore all 1st trials since it is before counting starts and is an empty cell
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
        if len(lk)!=0:  # append only if there is licks in this trial
            first_licks.append((lk[0]-starts[trial])/1250)
        else:
            first_licks.append(0)
    first_licks_baseline = first_licks[:stim_trials[0]]
    
    # filtering
    delete = []
    for trial in range(tot_trial):
        if first_licks[trial]==0 or first_licks[trial]>=8:
            delete.append(trial)
    first_licks = [v for i, v in enumerate(first_licks) if i not in delete]
    first_licks_baseline = [v for i, v in enumerate(first_licks_baseline) if i not in delete]
    act_props = [v for i, v in enumerate(act_props) if i not in delete]
    act_props_baseline = [v for i, v in enumerate(act_props_baseline) if i not in delete]
    
    # statistics 
    results = linregress(act_props, first_licks)
    pval = results[3]
    slope = results[0]; intercept = results[1]; rvalue = results[2]
    xmin = min(act_props); xmax = max(act_props)
    ymin = min(first_licks); ymax = max(first_licks)
    
    # plotting 
    fig, ax = plt.subplots(figsize=(3,3))
    ax.scatter(act_props, first_licks, s=2, c='grey')
    ax.plot([xmin-.1, xmax+.1], [intercept+(xmin-.1)*slope, intercept+(xmax+.1)*slope], 
            color='k', lw=1)
    ax.set(title='r={}\p={}'.format(round(slope, 5), round(pval, 5)),
           xlabel='prop. act.',
           ylabel='delay to 1st-licks (s)')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig('Z:\Dinghao\code_dinghao\HPC_all\population_v_1st_licks\{}_all_triasl_prop_act.png'.format(recname),
                dpi=120)
    plt.close()
    
    
    # statistics (baseline)
    results_baseline = linregress(act_props_baseline, first_licks_baseline)
    pval_baseline = results_baseline[3]
    slope_baseline = results_baseline[0]; intercept_baseline = results_baseline[1]; rvalue_baseline = results_baseline[2]
    xmin = min(act_props_baseline); xmax = max(act_props_baseline)
    ymin = min(first_licks_baseline); ymax = max(first_licks_baseline)
    
    # plotting (baseline)
    fig, ax = plt.subplots(figsize=(3,3))
    ax.scatter(act_props_baseline, first_licks_baseline, s=2, c='grey')
    ax.plot([xmin-.1, xmax+.1], [intercept_baseline+(xmin-.1)*slope_baseline, intercept_baseline+(xmax+.1)*slope_baseline], 
            color='k', lw=1)
    ax.set(title='r={}\p={}'.format(round(slope_baseline, 5), round(pval_baseline, 5)),
           xlabel='prop. act.',
           ylabel='delay to 1st-licks (s)')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig('Z:\Dinghao\code_dinghao\HPC_all\population_v_1st_licks\{}_baseline_prop_act.png'.format(recname),
                dpi=120)
    plt.close()
    
    all_licks_zscore.extend(zscore(first_licks))
    all_licks_baseline_zscore.extend(zscore(first_licks_baseline))
    all_act_props_zscore.extend(zscore(act_props))
    all_act_props_baseline_zscore.extend(zscore(act_props_baseline))