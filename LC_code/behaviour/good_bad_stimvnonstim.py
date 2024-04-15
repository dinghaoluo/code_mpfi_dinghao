# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:23:23 2023

compare good and bad trials percentage in stim and control 

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


#%% main 
all_stimOff_good = []
all_stimOn_good = []

for sessname in pathOpt:
    recname = sessname[-17:]
    behParf = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(recname[1:5], recname[:14], recname, recname)
    behPar = sio.loadmat(behParf)
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stimOn_ind = np.where(stimOn!=0)[0]-1
    bad_beh_ind = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    
    stimOff_tot = len(stimOn)-len(stimOn_ind)
    stimOn_tot = len(stimOn_ind)
    
    stimOff_bad = [s for s in bad_beh_ind if s not in stimOn_ind]
    stimOn_bad = [s for s in bad_beh_ind if s in stimOn_ind]
    
    stimOff_perc = 1-len(stimOff_bad)/stimOff_tot
    stimOn_perc = 1-len(stimOn_bad)/stimOn_tot
    
    all_stimOff_good.append(stimOff_perc)
    all_stimOn_good.append(stimOn_perc)
    
    
#%% stats and plotting 
results = ranksums(all_stimOff_good, all_stimOn_good)
pval = results[1]

fig, ax = plt.subplots(figsize=(3,4.5))

bp = ax.boxplot([all_stimOff_good, all_stimOn_good],
                positions=[.5, 2],
                patch_artist=True,
                notch='True')

ax.scatter([.8]*len(all_stimOff_good), 
           all_stimOff_good, 
           s=10, c='grey', ec='none', lw=.5)

ax.scatter([1.7]*len(all_stimOn_good), 
           all_stimOn_good, 
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

ax.plot([[.8]*len(all_stimOff_good), [1.7]*len(all_stimOn_good)], [all_stimOff_good, all_stimOn_good], 
        color='grey', alpha=.25, linewidth=1)
ax.plot([.8, 1.7], [np.median(all_stimOff_good), np.median(all_stimOn_good)],
        color='k', linewidth=2)
ymin = min(min(all_stimOn_good), min(all_stimOff_good))-.5
ymax = max(max(all_stimOn_good), max(all_stimOff_good))+.5
ax.set(xlim=(0,2.5), ylim=(ymin,ymax),
       ylabel='good trial percentage',
       title='stim v non-stim, p={}'.format(np.round(pval, 4)))
ax.set_xticks([.5, 2]); ax.set_xticklabels(['non-stim', 'stim'])
ax.set(ylim=(0, 1.05))
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
fig.suptitle('good trial percentage')

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\goodtrialperc.png',
            dpi=500,
            bbox_inches='tight')