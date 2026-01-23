# -*- coding: utf-8 -*-
"""
Created on Mon 10 Mar 15:04:01 2025
Modified on 21 Jan 2026

plot run-onset ON and OFF cells for Raphi's data 

@author: Dinghao Luo
"""


#%% imports 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import PowerNorm
from scipy.stats import sem 
import pandas as pd

from common import mpl_formatting, normalise
mpl_formatting()

import rec_list
paths = rec_list.pathHPC_Raphi


#%% paths and parameters
HPC_stem       = Path('Z:/Dinghao/code_dinghao/HPC_ephys')
run_onset_stem = HPC_stem / 'run_onset_response_raphi'

RUN_ONSET_BIN = 3750  # in samples
SAMP_FREQ     = 1250  # in Hz

BEF = 1  # in seconds 
AFT = 4  # in seconds 

XAXIS = np.arange(-SAMP_FREQ * BEF, SAMP_FREQ * AFT) / SAMP_FREQ

PROF_WINDOW = [
    int(RUN_ONSET_BIN - SAMP_FREQ * BEF), 
    int(RUN_ONSET_BIN + SAMP_FREQ * AFT)
    ]


#%% load dataframe 
print('Loading dataframe...')

cell_profiles_path = HPC_stem / 'HPC_all_profiles_raphi.pkl'

cell_profiles = pd.read_pickle(cell_profiles_path)


# ------------
# scale (temp)
# ------------
profile_cols = [
    'prof_mean', 'prof_sem',
    'prof_stim_mean', 'prof_stim_sem',
    'prof_ctrl_mean', 'prof_ctrl_sem'
]

recnums = cell_profiles['recname'].str[1:4].astype(int)
scale_mask = recnums > 40

for col in profile_cols:
    cell_profiles.loc[scale_mask, col] = (
        cell_profiles.loc[scale_mask, col].apply(
            lambda x: x * SAMP_FREQ if isinstance(x, np.ndarray) else x
        )
    )

print(f'scaled {scale_mask.sum()} cells (recname > A040)')
# ------------
# scale (temp) ends
# ------------


df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 

df_ON  = df_pyr[df_pyr['class']=='run-onset ON']
df_OFF = df_pyr[df_pyr['class']=='run-onset OFF']


#%% statistics first 
# unique sessions
sessions = df_pyr['recname'].unique()
n_sessions = len(sessions)

# unique animals (Axxx from recname[:4])
animals = df_pyr['recname'].str[:4].unique()
n_animals = len(animals)

# ON OFF
n_ON  = len(df_ON)
n_OFF = len(df_OFF)
n_tot = len(df_pyr)



#%% sorting 
df_sorted = df_pyr.sort_values(by='pre_post')

pop_mat = df_sorted['prof_mean'].to_numpy()

pop_mat = np.asarray([normalise(cell[PROF_WINDOW[0] : PROF_WINDOW[1]])
    for cell in pop_mat
    if not np.isnan(normalise(cell[PROF_WINDOW[0] : PROF_WINDOW[1]])[0])])


#%% overall plot 
fig, ax = plt.subplots(figsize=(2.4,1.9))

gim = ax.imshow(pop_mat, aspect='auto', cmap='Greys', norm=PowerNorm(gamma=0.8),
                extent=(-1, 4, 0, pop_mat.shape[0]))
ax.set(title=f'{n_ON}/{n_tot} ON ({round(n_ON/n_tot, 4)})\n{n_OFF}/{n_tot} OFF ({round(n_OFF/n_tot, 4)})\nn_sess={n_sessions}, n_anm={n_animals}',
       xlabel='Time from run onset (s)',
       ylabel='Cell #', yticks=[4000, 8000, 12000])

plt.colorbar(gim, shrink=.5, ticks=[0, 1])

for ext in ['.png', '.pdf']:
    fig.savefig(
        run_onset_stem / f'all_run_onset_Greys{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    

#%% mean profiles 
ON_all = [cell.prof_mean[PROF_WINDOW[0] : PROF_WINDOW[1]] for cell in 
          df_ON.itertuples(index=False)]
ON_all_mean = np.mean(ON_all, axis=0)
ON_all_sem  = sem(ON_all, axis=0)

OFF_all = [cell.prof_mean[PROF_WINDOW[0] : PROF_WINDOW[1]] for cell in 
           df_OFF.itertuples(index=False)]
OFF_all_mean = np.mean(OFF_all, axis=0)
OFF_all_sem  = sem(OFF_all, axis=0)


# plotting 
fig, ax = plt.subplots(figsize=(2.6,2))

ON_ln, = ax.plot(XAXIS, ON_all_mean, lw=1, c='firebrick')
ax.fill_between(XAXIS,
                ON_all_mean + ON_all_sem, 
                ON_all_mean - ON_all_sem,
                color='firebrick', edgecolor='none', alpha=.3)
for p in ['top', 'right']:
    ax.spines[p].set_visible(False)
    
ax.set(xlabel='Time from run onset (s)', xticks=[0,2,4], xlim=(-1,4),
       ylabel='Firing rate (Hz)', yticks=[1, 1.5, 2, 2.5], ylim=(0.8, 2.6))

for ext in ['.png', '.pdf']:
    fig.savefig(run_onset_stem / f'all_ON_curve_raphi{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    
    
fig, ax = plt.subplots(figsize=(2.6,2))

OFF_ln, = ax.plot(XAXIS, OFF_all_mean, lw=1, c='purple')
ax.fill_between(XAXIS,
                OFF_all_mean + OFF_all_sem, 
                OFF_all_mean - OFF_all_sem,
                color='purple', edgecolor='none', alpha=.3)
for p in ['top', 'right']:
    ax.spines[p].set_visible(False)
    
ax.set(xlabel='Time from run onset (s)', xticks=[0,2,4], xlim=(-1,4),
       ylabel='Firing rate (Hz)', yticks=[1, 1.5, 2], ylim=(0.8, 2.6))

for ext in ['.png', '.pdf']:
    fig.savefig(run_onset_stem / f'all_OFF_curve_raphi{ext}',
        dpi=300,
        bbox_inches='tight'
        )

