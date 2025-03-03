# -*- coding: utf-8 -*-
"""
Created on Tue 21 Nov 11:22:03 2023

all good v bad trials with RO-peaking cells, slightly before run-onsets

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd
import sys 
from scipy.stats import sem, ttest_rel, wilcoxon
import matplotlib.pyplot as plt 

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% load data 
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
RO_peaking_keys = []
for cell in cell_prop.index:
    up = cell_prop['peakness'][cell]  # union-peakness
    pt = cell_prop['putative'][cell]  # putative
    tg = cell_prop['tagged'][cell]
    
    if up and pt:
        RO_peaking_keys.append(cell)
    if up and tg:  # since putative does not include tagged
        RO_peaking_keys.append(cell)
        
        
#%% main
bef_good = []
bef_bad = []

for pathname in pathLC:
    sessname = pathname[-17:]
    print(sessname)
    
    # import bad beh trial indices
    behPar = sio.loadmat(pathname+pathname[-18:]+
                         '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
    # -1 to account for MATLAB Python difference
    ind_bad_beh = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    # -1 to account for 0 being an empty trial
    ind_good_beh = np.arange(behPar['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
    # import stim trial indices
    stimOn = behPar['behPar']['stimOn'][0][0][0]
    first_stim = next((i for i, j in enumerate(stimOn) if j), None)
    if type(first_stim)==int:  # only the baseline trials
        ind_bad_beh = ind_bad_beh[ind_bad_beh<first_stim]
        ind_good_beh = ind_good_beh[ind_good_beh<first_stim]
    
    # import tagged cell spike trains from all_tagged_train
    if len(ind_bad_beh) >= 10:  # 10 bad trials at least, prevents contam.
        for name in RO_peaking_keys:
            if name[:17] == sessname:
                curr = all_train[name]  # train of current clu
                curr_good = np.zeros(len(ind_good_beh))
                curr_bad = np.zeros(len(ind_bad_beh))
                for i in range(len(ind_good_beh)):
                    curr_good[i] = np.mean(curr[ind_good_beh[i]][2500:3125])*1250  # take -1 to -.5 as the quantification period
                for i in range(len(ind_bad_beh)):
                    curr_bad[i] = np.mean(curr[ind_bad_beh[i]][2500:3125])*1250
                bef_good.append(np.mean(curr_good, axis=0))
                bef_bad.append(np.mean(curr_bad, axis=0))


#%% statistics 
pval = wilcoxon(bef_good, bef_bad)[1]


#%% plotting 
fig, ax = plt.subplots(figsize=(3,4))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticklabels(['good trials', 'bad trials'], minor=False)

ax.set(ylabel='avg. spike rate (Hz)',
       title='good v bad trials, bef run p={}'.format(round(pval, 3)))

bp = ax.boxplot([bef_good, bef_bad],
           positions=[.5, 2],
           patch_artist=True,
           notch='True')
colors = ['darkgreen', 'grey']
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
    
ax.scatter([[.8]*len(bef_good), [1.7]*len(bef_bad)], [bef_good, bef_bad], zorder=2,
           s=15, color='grey', edgecolor='k', alpha=.5)
ax.plot([[.8]*len(bef_good), [1.7]*len(bef_bad)], [np.mean(bef_good), np.mean(bef_bad)], zorder=2,
        color='grey', alpha=.5)

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_goodvbad_(alignedRun)_avg_pooled_ROpeaking_bef_run.png',
            dpi=500,
            bbox_inches='tight')