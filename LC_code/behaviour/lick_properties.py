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
from scipy.stats import wilcoxon


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLCopt


#%% MAIN 
stim_std_med = []
cont_std_med = []
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
    

#%% plotting 
pval = wilcoxon(stim_std_med, cont_std_med)[1]

fig, ax = plt.subplots(figsize=(3,4.5))
for p in ['top', 'right', 'bottom']:
        ax.spines[p].set_visible(False)
ax.set_xticklabels(['ctrl', 'stim'], minor=False)
ax.set(ylabel='std. licks (s)',
       title='p[ratio]={}'.format(round(pval,4)))

bp = ax.boxplot([cont_std_med, stim_std_med],
                positions=[.5, 2],
                patch_artist=True,
                notch='True')
colors = ['grey', 'royalblue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
bp['fliers'][0].set(marker ='v',
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

ax.scatter([.8]*len(cont_std_med), 
           cont_std_med, 
           s=10, c='grey', ec='none', lw=.5)
ax.scatter([1.7]*len(stim_std_med), 
           stim_std_med, 
           s=10, c='royalblue', ec='none', lw=.5)
ax.plot([[.8]*len(cont_std_med), [1.7]*len(stim_std_med)], [cont_std_med, stim_std_med], 
        color='grey', alpha=.25, linewidth=1)
ax.plot([.8, 1.7], [np.median(cont_std_med), np.median(stim_std_med)],
        color='k', linewidth=2)

fig.suptitle('mean std. licks')
fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_lickstd_020_summary_wilc.png',
            dpi=500,
            bbox_inches='tight')