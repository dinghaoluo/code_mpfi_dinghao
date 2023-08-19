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

# n_bst = 1000  # hyperparameter for bootstrapping
# print('\n**BOOTSTRAP # = {}**\n'.format(n_bst))


#%% MAIN 
all_curr_trials = []; all_prev_trials = []
all_pvals = []

for sessname in sess_list:
    print(sessname)
    
    # infofilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    
    # Info = sio.loadmat(infofilename)
    # pulseMethod = Info['beh'][0][0]['pulseMethod'][0]
    
    # # stim info
    # tot_stims = len([t for t in pulseMethod if t!=0])
    # stim_cond = pulseMethod[np.where(pulseMethod!=0)][0]  # check stim condition
    # stim = [i for i, e in enumerate(pulseMethod) if e==stim_cond]
    
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
    
    curr_trials = first_licks[1:]
    prev_trials = first_licks[:-1]
    
    filt_curr_trials = []
    filt_prev_trials = []
    # filtering to remove outliers using 3 standard deviations
    mean_licks = np.mean(first_licks)
    std_licks = np.std(first_licks)
    upper = mean_licks + 2*std_licks
    lower = mean_licks - 2*std_licks
    if lower<=0:  # to make sure that trials with no licks (see above) do not get in the dataset
        lower = 0
    for i in range(1, len(first_licks)):
        if first_licks[i]<upper and first_licks[i]>lower and first_licks[i-1]<upper and first_licks[i-1]>lower:
            filt_curr_trials.append(first_licks[i])
            filt_prev_trials.append(first_licks[i-1])
    
    # linear regression comparing delta in lick times curr. v prev.
    delta = [curr_trials[i]-prev_trials[i] for i in range(len(curr_trials))]
    filt_delta = [filt_curr_trials[i]-filt_prev_trials[i] for i in range(len(filt_curr_trials))]
    results = linregress(filt_prev_trials, filt_delta)
    pval = results[3]
    slope = results[0]; intercept = results[1]; rvalue = [2]
    xmin = min(filt_prev_trials); xmax = max(filt_prev_trials)
    ymin = min(filt_delta); ymax = max(filt_delta)
    
    fig, ax = plt.subplots(figsize=(4,2.5))
    ax.set(xlabel='prev. trial t. 1st lick (s)',
           ylabel='change in t. 1st-lick\non curr. trial (s)',
           title='{}\n1st-lick time history dependency\npval={}, slope={}'.format(sessname, np.round(pval,4), np.round(slope,4)),
           xlim=(xmin-.5, xmax+.5), ylim=(ymin-.5, ymax+.5))
    ax.scatter(filt_prev_trials, filt_delta, color='grey', s=3)
    ax.plot([xmin-.1, xmax+.1], [intercept+(xmin-.1)*slope, intercept+(xmax+.1)*slope], color='k')
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\history_dependency\{}.png'.format(sessname),
                dpi=300,
                bbox_inches='tight')