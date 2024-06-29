# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:37:16 2023

loop over all cells for early v late trials

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import sys 
import scipy.io as sio
from scipy.stats import wilcoxon, ttest_rel


#%% plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load data 
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')

rasters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()
trains = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_info.npy',
                 allow_pickle=True).item()

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% obtain structure
clu_list = list(cell_prop.index)

tag_list = []; put_list = []
tag_rop_list = []; put_rop_list = []
rop_list = []
for clu in cell_prop.index:
    tg = cell_prop['tagged'][clu]
    pt = cell_prop['putative'][clu]
    rop = cell_prop['peakness'][clu]
    
    if tg:
        tag_list.append(clu)
    if pt:
        put_list.append(clu)
    
    if rop: 
        rop_list.append(clu)
        if tg:
            tag_rop_list.append(clu)
        if pt:
            put_rop_list.append(clu)


#%% parameters for processing 
window = [3750-313, 3750+313]  # window for spike summation, half a sec around run onsets


#%% main loop 
early = []  # 10 spike rates 
late = []  # 10 spike rates 
early_prof = []
late_prof = []

for cluname in rop_list:
    print(cluname)
    train = trains[cluname]
    raster = rasters[cluname]

    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    alignRun = sio.loadmat(filename)
    
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]
    
    behParf = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    behPar = sio.loadmat(behParf)
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stimOn_ind = np.where(stimOn!=0)[0]-1
    bad_beh_ind = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] if l-starts[trial] > 1250]  # only if the animal does not lick in the first second (carry-over licks)
        
        if len(lk)==0:
            first_licks.append(10000)
        else:
            first_licks.extend(lk[0]-starts[trial])

    # sort trials by first lick time
    temp = list(np.arange(tot_trial))
    licks_ordered, temp_ordered = zip(*sorted(zip(first_licks, temp)))
    
    early_trials = []
    late_trials = []
    
    for trial in temp_ordered[:40]:
        if trial not in stimOn_ind:
            early_trials.append(trial)
        if len(early_trials)>10:
            break
    for trial in reversed(temp_ordered[:40]):
        if trial not in stimOn_ind:
            late_trials.append(trial)
        if len(late_trials)>10:
            break
    
    for trial in early_trials:
        curr_raster = raster[trial]
        curr_train = train[trial]
        early.append(sum(curr_raster[window[0]:window[1]]))
        early_prof.append(curr_train[2500:5000])
    for trial in late_trials:
        curr_raster = raster[trial]
        curr_train = train[trial]
        late.append(sum(curr_raster[window[0]:window[1]]))
        late_prof.append(curr_train[2500:5000])
            

#%% statistics 
wilc_res, wilc_p = wilcoxon(early, late)
ttest_res, ttest_p = ttest_rel(early, late)
    
early_log = np.log(early)
late_log = np.log(late)

from numpy import inf 
early_log[early_log==-inf] = 0
late_log[late_log==-inf] = 0


#%% plotting 
early_colour = (.804,.267,.267); late_colour = (.545,0,0)

fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([early_log, late_log],
                   positions=[1, 2],
                   showextrema=False)

vp['bodies'][0].set_color(early_colour)
vp['bodies'][1].set_color(late_colour)
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

ax.scatter([1.1]*len(early), 
           early_log, 
           s=10, c=early_colour, ec='none', lw=.5, alpha=.05)
ax.scatter([1.9]*len(late), 
           late_log, 
           s=10, c=late_colour, ec='none', lw=.5, alpha=.05)
ax.plot([[1.1]*len(early), [1.9]*len(late)], [early_log, late_log], 
        color='grey', alpha=.05, linewidth=1)

ax.plot([1.1, 1.9], [np.mean(early_log), np.mean(late_log)],
        color='grey', linewidth=2)
ax.scatter(1.1, np.mean(early_log), 
           s=30, c=early_colour, ec='none', alpha=.75, lw=.5, zorder=2)
ax.scatter(1.9, np.mean(late_log), 
           s=30, c=late_colour, ec='none', lw=.5, zorder=2)
ymin = min(min(late_log), min(early_log))-.5
ymax = max(max(late_log), max(early_log))+.5
ax.set(xlim=(.5,2.5), ylim=(-.1,4.5),
       ylabel='spike rate (Hz)',
       title='early v late\nsingle-cell spike rate\nwilc_p={}\nttest_p={}'.format(round(wilc_p, 10), round(ttest_p, 10)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['early\n1st-lick', 'late\n1st-lick'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
    
fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_pooled_ROpeak_single_cell_earlyvlate.png',
            dpi=500,
            bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_pooled_ROpeak_single_cell_earlyvlate.pdf',
            bbox_inches='tight')

# plt.close()