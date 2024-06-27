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
results = wilcoxon(all_stimOff_good, all_stimOn_good)
pval = results[1]

fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([all_stimOff_good, all_stimOn_good],
                   positions=[1, 2],
                   showextrema=False, showmedians=True)

vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vp['bodies'][i].set_edgecolor('k')
vp['cmedians'].set(color='darkred', lw=2)

ax.scatter([1]*len(all_stimOff_good), 
           all_stimOff_good, 
           s=5, c='grey', ec='none', lw=.5)

ax.scatter([2]*len(all_stimOn_good), 
           all_stimOn_good, 
           s=5, c='royalblue', ec='none', lw=.5)

ax.plot([[1]*len(all_stimOff_good), [2]*len(all_stimOn_good)], [all_stimOff_good, all_stimOn_good], 
        color='grey', alpha=.25, linewidth=1)
ax.plot([1, 2], [np.median(all_stimOff_good), np.median(all_stimOn_good)],
        color='darkred', linewidth=2)
ymin = min(min(all_stimOn_good), min(all_stimOff_good))-.05
ymax = max(max(all_stimOn_good), max(all_stimOff_good))+.05
ax.set(xlim=(.5,2.5), ylim=(ymin,ymax),
       ylabel='good trial percentage',
       title='stim v non-stim\nrsums p={}'.format(np.round(pval, 6)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['ctrl', 'stim'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
# fig.suptitle('good trial percentage')

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\goodtrialperc.png',
            dpi=500,
            bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\goodtrialperc.pdf',
            bbox_inches='tight')