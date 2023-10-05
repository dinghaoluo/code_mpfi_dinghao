# -*- coding: utf-8 -*-
"""
Created on Wed 23 Aug 17:18:12 2023

compare opto stim vs baseline lickdist

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
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
comp_method = 'baseline'
print('\n**BOOTSTRAP # = {}**'.format(n_bst))
print('**comparison method: {}**\n'.format(comp_method))

# samp_freq = 1250  # Hz
# gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
# sigma_speed = samp_freq/100  # 10 ms
# gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
#               np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]


#%% MAIN 
all_licks_non_stim = []; all_licks_stim = []
# all_mspeeds_non_stim = []; all_mspeeds_stim = []  # control 
# all_accel_non_stim = []; all_accel_stim = []  # control 
# all_pspeeds_non_stim = []; all_pspeeds_stim = []  # control 
# all_initacc_non_stim = []; all_initacc_stim = []  # control 

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
    dist = alignRun['trialsRun']['xMM'][0][0][0][1:]  # distance at each sample
    
    first_licks = []
    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        # licks 
        lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
        if len(lk)!=0:  # append only if there is licks in this trial
            for i in range(len(lk)):
                ld = dist[trial][lk[0]-starts[trial]]/10                
                if ld > 30:  # filter out first licks before 30 (only starts counting at 30)
                    first_licks.append(dist[trial][lk[0]-starts[trial]]/10)
                    break
            if ld <= 30:
                first_licks.append(0)
        else:
            first_licks.append(0)
    
    # stim licks 
    licks_stim = [first_licks[i-1] for i in stim if first_licks[i-1]!=0 and first_licks[i+1]!=0]
    
    pval = []; 
    curr_licks_non_stim = []; curr_licks_stim = []

    for i in range(n_bst):
        # select same number of non_stim to match 
        non_stim_trials = np.where(pulseMethod==0)[0]
        if comp_method == 'baseline':
            selected_non_stim = non_stim_trials[np.random.randint(0, stim[0]-1, len(licks_stim))]
            licks_non_stim = []
            for t in selected_non_stim:
                if first_licks[t-1]!=0:
                    licks_non_stim.append(float(first_licks[t-1]))  # only compare trials with licks
                else:
                    licks_non_stim.append(float(first_licks[t]))
        elif comp_method == 'stim_cont': # stim_control 
            selected_non_stim = [i+2 for i in stim]
            licks_non_stim = [first_licks[i-1] for i in selected_non_stim if first_licks[i-1]!=0 and first_licks[i-3]!=0]
        
        curr_licks_non_stim.append(licks_non_stim)
        curr_licks_stim.append(licks_stim)
        
        pval.append(ranksums(licks_non_stim, licks_stim)[1])
        
    if stim_cond==2:
        all_licks_non_stim.append(np.median(curr_licks_non_stim))
        all_licks_stim.append(np.median(curr_licks_stim))
    
    
    # licks plot 
    fig, ax = plt.subplots()
    for p in ['top', 'right', 'left']:
        ax.spines[p].set_visible(False)
    ax.set(title='{}, stim={}'.format(sessname, stim_cond),
           ylim=(0, 1.5), xlim=(30, 225),
           xlabel='dist. 1st lick (cm)')
    ax.set_yticks([.5, 1])
    ax.set_yticklabels(['baseline', 'stim'])
    ax.scatter(licks_non_stim, [.5]*len(licks_stim), color='grey')
    ax.scatter(licks_stim, [1]*len(licks_stim), color='darkblue')
    ax.plot([np.median(licks_non_stim), np.median(licks_stim)], [.5, 1],
            color='grey', alpha=.5)
    fig.suptitle('dist. 1st licks')
    if comp_method == 'baseline':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_lickdist_0{}0\{}.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')
    elif comp_method == 'stim_cont':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_lickdist_0{}0_stim_cont\{}.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')
  
    
#%% summary statistics 
res = 0; pval = 0
res, pval = wilcoxon(all_licks_non_stim, all_licks_stim)

# licks summary
fig, ax = plt.subplots(figsize=(3,4.5))

bp = ax.boxplot([all_licks_non_stim, all_licks_stim],
                positions=[.5, 2],
                patch_artist=True,
                notch='True')

ax.scatter([.8]*len(all_licks_non_stim), 
           all_licks_non_stim, 
           s=10, c='grey', ec='none', lw=.5)

ax.scatter([1.7]*len(all_licks_stim), 
           all_licks_stim, 
           s=10, c='royalblue', ec='none', lw=.5)

colors = ['grey', 'royalblue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

bp['fliers'][0].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][1].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)

for median in bp['medians']:
    median.set(color='darkred',
                linewidth=1)

ax.plot([[.8]*len(all_licks_stim), [1.7]*len(all_licks_stim)], [all_licks_non_stim, all_licks_stim], 
        color='grey', alpha=.25, linewidth=1)
ax.plot([.8, 1.7], [np.median(all_licks_non_stim), np.median(all_licks_stim)],
        color='k', linewidth=2)
ymin = min(min(all_licks_stim), min(all_licks_non_stim))-.5
ymax = max(max(all_licks_stim), max(all_licks_non_stim))+.5
ax.set(xlim=(0,2.5), ylim=(ymin,ymax),
       ylabel='dist. 1st licks (cm)',
       title='dist. 1st licks non-stim v stim, p={}'.format(np.round(pval, 4)))
ax.set_xticks([.5, 2]); ax.set_xticklabels(['non-stim', 'stim'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
fig.suptitle('distance 1st licks')

if comp_method == 'baseline':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_lickdist_020\summary_wilc.png',
                dpi=500,
                bbox_inches='tight')
elif comp_method == 'stim_cont':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_lickdist_020_stim_cont\summary_wilc.png',
                dpi=500,
                bbox_inches='tight')