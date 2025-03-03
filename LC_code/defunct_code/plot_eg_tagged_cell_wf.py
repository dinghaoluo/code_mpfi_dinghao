# -*- coding: utf-8 -*-
"""
Created on Wed 30 Aug 17:29:13 2023

plot the waveform of a tagged cell

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
import sys
from scipy.stats import pearsonr

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise_to_all, normalise

use_sess = 'A032r-20220726-03'
use_clu = 3


#%% load data
avg_wf = np.load('Z:/Dinghao/code_dinghao/LC_by_sess/{}_avg_spk.npy'.format(use_sess),
                 allow_pickle=True).item()
avg_sem = np.load('Z:/Dinghao/code_dinghao/LC_by_sess/{}_avg_sem.npy'.format(use_sess),
                 allow_pickle=True).item()
tg_wf = np.load('Z:/Dinghao/code_dinghao/LC_tagged_by_sess/{}_tagged_spk.npy'.format(use_sess),
                allow_pickle=True).item()
tg_sem = np.load('Z:/Dinghao/code_dinghao/LC_tagged_by_sess/{}_tagged_sem.npy'.format(use_sess),
                allow_pickle=True).item()


#%% main 
avg_spk = avg_wf['{}'.format(use_clu)]
avg_sem = avg_sem['{}'.format(use_clu)]
tg_spk = tg_wf['{}'.format(use_clu)]
tg_sem = tg_wf['{}'.format(use_clu)]

fig, axs = plt.subplots(1,2,figsize=(4,2))

xaxis = np.arange(32)
axs[0].plot(avg_spk, c='black')
axs[0].fill_between(xaxis, avg_spk+avg_sem, avg_spk-avg_sem,
                    color='grey', alpha=.1)

axs[1].plot(tg_spk, c='royalblue')
# axs[1].fill_between(xaxis, tg_spk+tg_sem, tg_spk-tg_sem,
#                     color='royalblue', alpha=.1)

axs[0].set(xticks=[], xticklabels=[],
           yticks=[], yticklabels=[])
axs[1].set(xticks=[], xticklabels=[],
           yticks=[], yticklabels=[])

for sp in ['left', 'top', 'bottom', 'right']:
    axs[0].spines[sp].set_visible(False)
    axs[1].spines[sp].set_visible(False)
    
fig.suptitle('{} {}'.format(use_sess, use_clu))

fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\eg_tg_cell.png',
            dpi=300,
            bbox_inches='tight',
            transparent=False)