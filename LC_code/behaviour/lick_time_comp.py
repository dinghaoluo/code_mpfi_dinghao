# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 13:26:35 2023

compare opto stim vs baseline licktime

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats import ranksums, wilcoxon  # median used 
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
all_licks_non_stim = []; all_licks_stim = []
all_mspeeds_non_stim = []; all_mspeeds_stim = []  # control 
all_accel_non_stim = []; all_accel_stim = []  # control 

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
    speeds = alignRun['trialsRun']['speed_MMsec'][0][0][0][1:]  # control 
    accels = alignRun['trialsRun']['accel_MMsecSq'][0][0][0][1:]  # control

    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:  # trials where no rewards were delivered or rewards were delivered after 16 s
            pumps[trial] = 20000
    
    first_licks = []
    licks_bef_rew = []
    mean_speeds = []
    for trial in range(tot_trial):

        # licks 
        lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
        if len(lk)!=0:  # append only if there is licks in this trial
            first_licks.append((lk[0]-starts[trial])/1250)
        else:
            first_licks.append(0)
        licks_bef_rew.append(len([l for l in lk if l-starts[trial]<pumps[trial]]))

        # mean speed
        ms = np.mean(speeds[trial])/10  # from mm/s to cm/s
        mean_speeds.append(ms)
    
    pval = []; pval_mspeeds = []; pval_bef_rew = []
    curr_licks_non_stim = []; curr_licks_stim = []
    curr_mspeeds_non_stim = []; curr_mspeeds_stim = []
    curr_maccels_non_stim = []; curr_maccels_stim = []
    for i in range(n_bst):
        # stim licks 
        licks_stim = [first_licks[i-1] for i in stim if first_licks[i-1]!=0]
        licks_br_stim = [licks_bef_rew[i-1] for i in stim]
        
        # select same number of non_stim to match 
        non_stim_trials = np.where(pulseMethod==0)[0]
        selected_non_stim = non_stim_trials[np.random.randint(0, 50, len(licks_stim))]

        licks_non_stim = [first_licks[i-1] for i in selected_non_stim]
        licks_br_non_stim = [licks_bef_rew[i-1] for i in selected_non_stim]

        mspeeds_non_stim = [mean_speeds[i-1] for i in selected_non_stim]
        mspeeds_stim = [mean_speeds[i-1] for i in stim]
        
        curr_licks_non_stim.append(licks_non_stim)
        curr_licks_stim.append(licks_stim)

        curr_mspeeds_non_stim.append(mspeeds_non_stim)
        curr_mspeeds_stim.append(mspeeds_stim)
        
        pval.append(ranksums(licks_non_stim, licks_stim)[1])
        pval_mspeeds.append(ranksums(mspeeds_non_stim, mspeeds_stim)[1])
        pval_bef_rew.append(ranksums(licks_br_non_stim, licks_br_stim)[1])
        
    if stim_cond==2:
        all_licks_non_stim.append(np.median(curr_licks_non_stim))
        all_licks_stim.append(np.median(curr_licks_stim))
        all_mspeeds_non_stim.append(np.median(curr_mspeeds_non_stim))
        all_mspeeds_stim.append(np.median(curr_mspeeds_stim))
    
    # licks plot 
    fig, ax = plt.subplots()
    for p in ['top', 'right', 'left']:
        ax.spines[p].set_visible(False)
    ax.set(title='{}, stim={}, p={}, {}'.format(sessname, stim_cond, round(np.mean(pval), 4), round(np.mean(pval_bef_rew), 4)),
           ylim=(0, 1.5),
           xlabel='t 1st lick (s)')
    ax.set_yticks([.5, 1])
    ax.set_yticklabels(['baseline', 'stim'])
    ax.scatter(licks_non_stim, [.5]*len(licks_stim), color='grey')
    ax.scatter(licks_stim, [1]*len(licks_stim), color='darkblue')
    ax.plot([np.median(licks_non_stim), np.median(licks_stim)], [.5, 1],
            color='grey', alpha=.5)
    fig.suptitle('t 1st licks')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}.png'.format(stim_cond, sessname),
                dpi=300,
                bbox_inches='tight')
    
    # mspeeds plot 
    fig, ax = plt.subplots()
    for p in ['top', 'right', 'left']:
        ax.spines[p].set_visible(False)
    ax.set(title='{}, stim={}, p={}'.format(sessname, stim_cond, round(np.mean(pval_mspeeds), 4)),
           ylim=(0, 1.5),
           xlabel='mean velocity (cm/s)')
    ax.set_yticks([.5, 1])
    ax.set_yticklabels(['baseline', 'stim'])
    ax.scatter(mspeeds_non_stim, [.5]*len(mspeeds_non_stim), color='grey')
    ax.scatter(mspeeds_stim, [1]*len(mspeeds_stim), color='darkblue')
    ax.plot([np.median(mspeeds_non_stim), np.median(mspeeds_stim)], [.5, 1],
            color='grey', alpha=.5)
    fig.suptitle('mean velocity')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}_control_velocity.png'.format(stim_cond, sessname),
                dpi=300,
                bbox_inches='tight')


#%% summary statistics 
res = 0; pval = 0; res_mspeeds = 0; pval_mspeeds = 0
res, pval = wilcoxon(all_licks_non_stim, all_licks_stim)
res_mspeeds, pval_mspeeds = wilcoxon(all_mspeeds_non_stim, all_mspeeds_stim)

# licks summary
fig, ax = plt.subplots(figsize=(3,4.5))
ax.plot([[1]*len(all_licks_stim), [2]*len(all_licks_stim)], [all_licks_non_stim, all_licks_stim], 
        color='grey', alpha=.25, linewidth=1)
ax.plot([1, 2], [np.median(all_licks_non_stim), np.mean(all_licks_stim)],
        color='royalblue', linewidth=2)
ymin = min(min(all_licks_stim), min(all_licks_non_stim))-.5
ymax = max(max(all_licks_stim), max(all_licks_non_stim))+.5
ax.set(xlim=(.5,2.5), ylim=(ymin,ymax),
       ylabel='t. 1st licks (s)',
       title='t. 1st licks non-stim v stim, p={}'.format(np.round(pval, 4)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['non-stim', 'stim'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
fig.suptitle('t 1st licks')
fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_wilc.png',
            dpi=300,
            bbox_inches='tight')

# mean speeds summary
fig, ax = plt.subplots(figsize=(3,4.5))
ax.plot([[1]*len(all_mspeeds_stim), [2]*len(all_mspeeds_stim)], [all_mspeeds_non_stim, all_mspeeds_stim], 
        color='grey', alpha=.25, linewidth=1)
ax.plot([1, 2], [np.median(all_mspeeds_non_stim), np.mean(all_mspeeds_stim)],
        color='royalblue', linewidth=2)
ymin_speed = min(min(all_mspeeds_stim), min(all_mspeeds_non_stim))-5
ymax_speed = max(max(all_mspeeds_stim), max(all_mspeeds_non_stim))+5
ax.set(xlim=(.5,2.5), ylim=(ymin_speed,ymax_speed),
       ylabel='mean velocity (cm/s)',
       title='mean velocity non-stim v stim, p={}'.format(np.round(pval_mspeeds, 4)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['non-stim', 'stim'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
fig.suptitle('mean velocity')
fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_wilc_control_velocity.png',
            dpi=300,
            bbox_inches='tight')