# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 17:09:31 2025

plot profiles of run-onset ON/OFF cells based on raw spike rate change 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
from scipy.stats import sem 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys


#%% load dataframe 
print('loading dataframe...')
df = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')
df_pyr = df[(df['cell_identity']=='pyr') & (df['rectype']=='HPCLC')]


#%% functions
sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from plotting_functions import plot_violin_with_scatter
from common import normalise, mpl_formatting
mpl_formatting()

def compute_mean_and_sem(arr):
    '''take a 2D matrix'''
    '''returns mean and sem along axis=0'''
    return np.nanmean(arr, axis=0), sem(arr, axis=0)

def compute_raw_change(profile):
    pre_mean = np.mean(profile[3750 - 1875 : 3750 - 625])
    post_mean = np.mean(profile[3750 + 625 : 3750 + 1875])
    
    return post_mean - pre_mean


#%% parameters
xaxis = np.arange(-1, 4, 1/1250)


#%% calculate raw change 
df_pyr['raw_change'] = df_pyr['prof_mean_MATLAB'].apply(lambda prof: compute_raw_change(prof))
df_pyr['ctrl_raw_change'] = df_pyr['prof_ctrl_mean_MATLAB'].apply(lambda prof: compute_raw_change(prof))
df_pyr['stim_raw_change'] = df_pyr['prof_stim_mean_MATLAB'].apply(lambda prof: compute_raw_change(prof))


#%% sort dataframe by baseline pre-post ratios 
df_sorted = df_pyr.sort_values(by='raw_change', ascending=False)
df_sorted_ctrl = df_pyr.sort_values(by='ctrl_raw_change', ascending=False)
df_sorted_stim = df_pyr.sort_values(by='stim_raw_change', ascending=False)


#%% matrices 
pop_mat_ctrl = df_sorted_ctrl['prof_ctrl_mean'].to_numpy()
pop_mat_ctrl = np.asarray([normalise(cell[2500:2500+5*1250]) for cell in pop_mat_ctrl])

pop_mat_stim = df_sorted_stim['prof_stim_mean'].to_numpy()
pop_mat_stim = np.asarray([normalise(cell[2500:2500+5*1250]) for cell in pop_mat_stim])


#%% overall plot 
fig, ax = plt.subplots(figsize=(2.4,1.9))

cim = ax.imshow(pop_mat_ctrl, aspect='auto', cmap='Greys', interpolation='sinc',
                extent=(-1, 4, 0, pop_mat_ctrl.shape[0]))

ax.set(title='rate based -- ctrl.',
       xlabel='time from run-onset (s)',
       ylabel='cell #')

plt.colorbar(cim, shrink=.5, ticks=[0, 1])

for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\HPCLC_run_onset_ctrl_ratebased_Greys{}'.
        format(ext),
        dpi=300,
        bbox_inches='tight'
        )
    
fig, ax = plt.subplots(figsize=(2.4,1.9))

sim = ax.imshow(pop_mat_stim, aspect='auto', cmap='Greys', interpolation='sinc',
                extent=(-1, 4, 0, pop_mat_stim.shape[0]))
ax.set(title='rate based -- stim.',
       xlabel='time from run-onset (s)',
       ylabel='cell #')

plt.colorbar(sim, shrink=.5, ticks=[0, 1])

for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\HPCLC_run_onset_stim_ratebased_Greys{}'.
        format(ext),
        dpi=300,
        bbox_inches='tight'
        )