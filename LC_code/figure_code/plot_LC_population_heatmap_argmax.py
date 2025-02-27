# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:17:04 2023
Modified on Sat 13 Apr 11:00:14 2024: changed heatmap cmap 

Plot heatmap of average firing profiles based on argmax

@author: Dinghao Luo
"""


#%% imports 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import normalise, mpl_formatting
mpl_formatting()


#%% load data
avg_profile = np.load(
    r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_avg_sem.npy',
    allow_pickle=True).item()['all avg']

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )


#%% specify RO peaking putative Dbh cells
putative_keys = cell_profiles.index[cell_profiles['identity']=='putative'].tolist()
tagged_keys = cell_profiles.index[cell_profiles['identity']=='tagged'].tolist()


#%% data wrangling
putative_keys_sorted = sorted(
    putative_keys, 
    key=lambda cluname: np.argmax(avg_profile[cluname])
    )
putative_im_matrix = np.stack(
    [normalise(avg_profile[cluname][2500:2500+1250*4]) for cluname in putative_keys_sorted]
    )

tagged_keys_sorted = sorted(
    tagged_keys,
    key=lambda cluname: np.argmax(avg_profile[cluname])
    )
tagged_im_matrix = np.stack(
    [normalise(avg_profile[cluname][2500:2500+1250*4]) for cluname in tagged_keys_sorted])


#%% plotting 
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from run-onset (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('putative Dbh+ cells')

image = ax.imshow(putative_im_matrix, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(putative_keys_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 100, 200, 300])

plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\LC_putative_Dbh_ordered_heatmap{ext}',
        dpi=300,
        bbox_inches='tight')

plt.close(fig)

fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from run-onset (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('tagged Dbh+ cells')

image = ax.imshow(tagged_im_matrix, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(tagged_im_matrix)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 50])

plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\LC_tagged_Dbh_ordered_heatmap{ext}',
        dpi=300,
        bbox_inches='tight')

plt.close(fig)