# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:17:04 2023
Modified on Sat 13 Apr 11:00:14 2024: changed heatmap cmap 
Modified on 28 Feb 2025: installed wrapper 

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
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )


#%% main 
# filtering and sorting 
df_tagged = cell_profiles[cell_profiles['identity']=='tagged'].copy()  # .copy() as we need to sort the df 
df_tagged['mean_truncated'] = df_tagged['baseline_mean'].apply(
    lambda x: x[3750-1250:3750+1250*4]
    )
df_tagged['argmax'] = df_tagged['mean_truncated'].apply(np.argmax)
df_tagged_sorted = df_tagged.sort_values(by='argmax')
sorted_im_matrix_tagged = np.vstack(
    [normalise(train[3750-1250:3750+1250*4]) for train in 
     df_tagged_sorted['baseline_mean'].to_numpy()]
    )

df_putative = cell_profiles[cell_profiles['identity']=='putative'].copy()
df_putative['mean_truncated'] = df_putative['baseline_mean'].apply(
    lambda x: x[3750-1250:3750+1250*4]
    )
df_putative['argmax'] = df_putative['mean_truncated'].apply(np.argmax)
df_putative_sorted = df_putative.sort_values(by='argmax')
sorted_im_matrix_putative = np.vstack(
    [normalise(train[3750-1250:3750+1250*4]) for train in
     df_putative_sorted['baseline_mean'].to_numpy()]
    )


## plotting 
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from run-onset (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('tagged Dbh+ cells')

image = ax.imshow(sorted_im_matrix_tagged, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(df_tagged)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 50])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\LC_tagged_Dbh_ordered_heatmap{ext}',
        dpi=300,
        bbox_inches='tight')

plt.close(fig)


fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from run-onset (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('putative Dbh+ cells')

image = ax.imshow(sorted_im_matrix_putative, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(df_putative)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\LC_putative_Dbh_ordered_heatmap{ext}',
        dpi=300,
        bbox_inches='tight')