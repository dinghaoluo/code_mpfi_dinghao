# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:05:49 2025

Quantify alignment of LC cells to different behavioural landmarks 

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathLC


#%% path stems
sess_stem = Path(r'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions')


#%% parameters 
SAMP_FREQ  = 1250  # Hz


#%% load profiles 
cell_profiles = pd.read_pickle(r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')

cell_profiles_tagged   = cell_profiles[cell_profiles['identity']=='tagged']
cell_profiles_putative = cell_profiles[cell_profiles['identity']=='putative']


#%% helpers 
def _get_aligned_rate(
        trains_run, trains_rew, trains_cue, cluname
        ):
    '''this calculates the peak rates to given landmarks'''
    curr_trains_run = trains_run[cluname]
    curr_trains_rew = trains_rew[cluname]
    curr_trains_cue = trains_cue[cluname]
    
    return (
        np.max(np.mean(curr_trains_run, axis=0)), 
        np.max(np.mean(curr_trains_rew, axis=0)),
        np.max(np.mean(curr_trains_cue, axis=0))
        )
    

def _load_trains(recname):
    '''this loads trains aligned to run, rew and cue, in that order'''
    return (
        np.load(sess_stem / recname / f'{recname}_all_trains_run.npy', allow_pickle=True).item(),
        np.load(sess_stem / recname / f'{recname}_all_trains_rew.npy', allow_pickle=True).item(),
        np.load(sess_stem / recname / f'{recname}_all_trains_cue.npy', allow_pickle=True).item()
        )


#%% main 
rate_run_tagged   = []
rate_rew_tagged   = []
rate_cue_tagged   = []
rate_run_putative = []
rate_rew_putative = []
rate_cue_putative = []

for path in paths:
    recname = Path(path).name
    print(recname)
    
    # get spike trains
    trains_run, trains_rew, trains_cue = _load_trains(recname)
    
    # get clu list 
    curr_tagged   = cell_profiles_tagged[cell_profiles_tagged['sessname']==recname].index
    curr_putative = cell_profiles_putative[cell_profiles_putative['sessname']==recname].index
    
    # get distribution for each clu 
    for cluname in curr_tagged:
        run, rew, cue = _get_aligned_rate(trains_run, trains_rew, trains_cue, cluname)
        rate_run_tagged.append(run)
        rate_rew_tagged.append(rew)
        rate_cue_tagged.append(cue)
        
    for cluname in curr_putative:
        run, rew, cue = _get_aligned_rate(trains_run, trains_rew, trains_cue, cluname)
        rate_run_putative.append(run)
        rate_rew_putative.append(rew)
        rate_cue_putative.append(cue)


#%% statistics 
data = pd.DataFrame({
    'run': rate_run_tagged + rate_run_putative,
    'rew': rate_rew_tagged + rate_rew_putative,
    'cue': rate_cue_tagged + rate_cue_putative,
    'identity': ['tagged'] * len(rate_run_tagged) + ['putative'] * len(rate_run_putative)
})

# reshape wide → long
df_long = data.melt(id_vars='identity', var_name='event', value_name='rate')

plt.figure(figsize=(6, 4))

# bar plot with mean values
sns.barplot(
    data=df_long, y='event', x='rate', hue='identity',
    ci=None,  # we'll add error bars manually
    edgecolor='black', alpha=0.6
)

# add scatter points (jittered horizontally for visibility)
sns.stripplot(
    data=df_long, y='event', x='rate', hue='identity',
    dodge=True, jitter=True, alpha=0.5, zorder=1, linewidth=0
)

# remove duplicate legends
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], frameon=False)

# add error bars (mean ± sem)
from scipy.stats import sem

group_means = df_long.groupby(['event','identity'])['rate'].mean()
group_sems  = df_long.groupby(['event','identity'])['rate'].apply(sem)

y_coords = {'run':0, 'rew':1, 'cue':2}
offsets  = {'tagged':-0.2, 'putative':0.2}

for (event, ident), mean in group_means.items():
    sem_val = group_sems.loc[(event, ident)]
    y = y_coords[event] + offsets[ident]
    plt.errorbar(mean, y, xerr=sem_val, fmt='o', color='black', capsize=3, zorder=2)

plt.title('Peak rates of alignment by identity and event')
plt.xlabel('Peak firing rate (Hz)')
plt.ylabel('Event')
plt.tight_layout()
plt.show()