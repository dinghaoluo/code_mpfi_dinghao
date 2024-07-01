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
    

#%% summary statistics median 
wilc_p = wilcoxon(cont_std_med, stim_std_med)[1]
ttest_p = ttest_rel(cont_std_med, stim_std_med)[1]

# licks summary
fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([cont_std_med, stim_std_med],
                   positions=[1, 2],
                   showextrema=False)

vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vp['bodies'][i].set_edgecolor('none')
    vp['bodies'][i].set_alpha(.75)
    b = vp['bodies'][i]
    # get the centre 
    m = np.mean(b.get_paths()[0].vertices[:,0])
    # make paths not go further right/left than the centre 
    if i==0:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
    if i==1:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)

ax.scatter([1.1]*len(cont_std_med), 
           cont_std_med, 
           s=10, c='grey', ec='none', lw=.5, alpha=.2)
ax.scatter([1.9]*len(stim_std_med), 
           stim_std_med, 
           s=10, c='royalblue', ec='none', lw=.5, alpha=.2)
ax.plot([[1.1]*len(cont_std_med), [1.9]*len(stim_std_med)], [cont_std_med, stim_std_med], 
        color='grey', alpha=.2, linewidth=1)

ax.plot([1.1, 1.9], [np.median(cont_std_med), np.median(stim_std_med)],
        color='grey', linewidth=2)
ax.scatter(1.1, np.median(cont_std_med), 
           s=30, c='grey', ec='none', lw=.5, zorder=2)
ax.scatter(1.9, np.median(stim_std_med), 
           s=30, c='royalblue', ec='none', lw=.5, zorder=2)
ymin = min(min(cont_std_med), min(stim_std_med))-.5
ymax = max(max(cont_std_med), max(stim_std_med))+.5
ax.set(xlim=(.5,2.5),
       ylabel='median std. licks (s)',
       title='med. std. licks ctrl v stim\nwilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['ctrl.', 'stim.'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)

fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\lick_std_summary_median.png',
            dpi=500,
            bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\lick_std_summary_median.pdf',
            bbox_inches='tight')


#%% summary statistics  mean
wilc_p = wilcoxon(cont_std_mean, stim_std_mean)[1]
ttest_p = ttest_rel(cont_std_mean, stim_std_mean)[1]

# licks summary
fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([cont_std_mean, stim_std_mean],
                   positions=[1, 2],
                   showextrema=False)

vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vp['bodies'][i].set_edgecolor('none')
    vp['bodies'][i].set_alpha(.75)
    b = vp['bodies'][i]
    # get the centre 
    m = np.mean(b.get_paths()[0].vertices[:,0])
    # make paths not go further right/left than the centre 
    if i==0:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
    if i==1:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)

ax.scatter([1.1]*len(cont_std_mean), 
           cont_std_mean, 
           s=10, c='grey', ec='none', lw=.5, alpha=.2)
ax.scatter([1.9]*len(stim_std_mean), 
           stim_std_mean, 
           s=10, c='royalblue', ec='none', lw=.5, alpha=.2)
ax.plot([[1.1]*len(cont_std_mean), [1.9]*len(stim_std_mean)], [cont_std_mean, stim_std_mean], 
        color='grey', alpha=.2, linewidth=1)

ax.plot([1.1, 1.9], [np.mean(cont_std_mean), np.mean(stim_std_mean)],
        color='grey', linewidth=2)
ax.scatter(1.1, np.mean(cont_std_mean), 
           s=30, c='grey', ec='none', lw=.5, zorder=2)
ax.scatter(1.9, np.mean(stim_std_mean), 
           s=30, c='royalblue', ec='none', lw=.5, zorder=2)
ymin = min(min(cont_std_mean), min(stim_std_mean))-.5
ymax = max(max(cont_std_mean), max(stim_std_mean))+.5
ax.set(xlim=(.5,2.5),
       ylabel='mean std. licks (s)',
       title='mean std. licks ctrl v stim\nwilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['ctrl.', 'stim.'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)

fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\lick_std_summary_mean.png',
            dpi=500,
            bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\lick_std_summary_mean.pdf',
            bbox_inches='tight')