# -*- coding: utf-8 -*-
"""
Created on 29 Sept 12:12:54 2025

Compare LC RO peak amplitude between rewarded and unrewarded trials

@author: Dinghao Luo
"""


#%% imports
from pathlib import Path

import numpy as np
import pandas as pd
import pickle 
from scipy.stats import wilcoxon, sem
import matplotlib.pyplot as plt 
import scipy.io as sio

import plotting_functions as pf
from common import mpl_formatting 
mpl_formatting()

import rec_list
pathLC = rec_list.pathLC


#%% functions 
def get_trialtype_idx(beh_filename):
    behPar = sio.loadmat(beh_filename)
    stim_idx = np.where(behPar['behPar']['stimOn'][0][0][0]!=0)[0]
    
    if len(stim_idx)>0:
        return np.arange(1, stim_idx[0]), stim_idx, stim_idx+2  # stim_idx+2 are indices of control trials
    else:
        return np.arange(1, len(behPar['behPar']['stimOn'][0][0][0])), [], []  # if no stim trials


#%% parameters 
max_length = 12500  # max length for trial analysis
xaxis = np.arange(-3*1250, 7*1250, 1)/1250 

colour_tag = (70/255, 101/255, 175/255)
colour_put = (101/255, 82/255, 163/255)


#%% path stems 
LC_all_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys')
beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')


#%% load data 
cell_prop_path = LC_all_stem / 'LC_all_cell_profiles.pkl'
cell_prop = pd.read_pickle(cell_prop_path)


#%% MAIN
# get RO-peaking cells 
peak_rew_tag, peak_unrew_tag = [], []
prof_rew_tag, prof_unrew_tag = [], []

peak_rew_put, peak_unrew_put = [], []
prof_rew_put, prof_unrew_put = [], []

for path in pathLC:
    recname = Path(path).name
    print(f'\n{recname}')
    
    # load behaviour
    beh_path = beh_stem / f'{recname}.pkl'
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)
    
    # import stim trial indices
    baseline_idx, _, _ = get_trialtype_idx(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.
        format(path, recname)
        )
    
    # get series 
    reward_times = beh['reward_times'][1 : baseline_idx[-1]]
    unrewarded_idx = [trial for trial, x in enumerate(reward_times)
                      if np.isnan(x)]
    rewarded_idx   = [trial for trial, x in enumerate(reward_times)
                      if not np.isnan(x)]
    
    if len(unrewarded_idx) < 5: continue
    
    curr_cell_prop = cell_prop[cell_prop['sessname'] == recname]
    curr_RO = curr_cell_prop[curr_cell_prop['run_onset_peak'] == True]
    
    trains_path = LC_all_stem / 'all_sessions' / recname
    curr_trains = np.load(
        trains_path / f'{recname}_all_trains.npy', allow_pickle=True
        ).item()
    
    for cluname, row in curr_RO.iterrows():
        trains = curr_trains[cluname]
        curr_rew   = np.zeros([len(rewarded_idx), max_length])
        curr_unrew = np.zeros([len(unrewarded_idx), max_length])
        for i, trial in enumerate(rewarded_idx):
            if np.isnan(trains[trial][0]): continue 
            curr_length = len(trains[trial])
            curr_rew[i, :curr_length] = trains[trial][:max_length]
        for i, trial in enumerate(unrewarded_idx):
            if np.isnan(trains[trial][0]): continue 
            curr_length = len(trains[trial])
            curr_unrew[i, :curr_length] = trains[trial][:max_length]
            
        if row['identity'] == 'tagged':
            prof_rew_tag.append(np.mean(curr_rew, axis=0))
            peak_rew_tag.append(np.mean(prof_rew_tag[-1][3125:4375]))
            prof_unrew_tag.append(np.mean(curr_unrew, axis=0))
            peak_unrew_tag.append(np.mean(prof_unrew_tag[-1][3125:4375]))
        if row['identity'] == 'putative':
            prof_rew_put.append(np.mean(curr_rew, axis=0))
            peak_rew_put.append(np.mean(prof_rew_put[-1][3125:4375]))
            prof_unrew_put.append(np.mean(curr_unrew, axis=0))
            peak_unrew_put.append(np.mean(prof_unrew_put[-1][3125:4375]))


#%% tagged 
pval_wil = wilcoxon(peak_rew_tag, peak_unrew_tag)[1]

mean_rew_tag_profile = np.mean(prof_rew_tag, axis=0)
sem_rew_tag_profile = sem(prof_rew_tag, axis=0)
mean_unrew_tag_profile = np.mean(prof_unrew_tag, axis=0)
sem_unrew_tag_profile = sem(prof_unrew_tag, axis=0)

fig, ax = plt.subplots(figsize=(2,1.4))
p_rew_ln, = ax.plot(xaxis, mean_rew_tag_profile, color=colour_tag)
p_unrew_ln, = ax.plot(xaxis, mean_unrew_tag_profile, color='grey')

ax.fill_between(xaxis, mean_rew_tag_profile+sem_rew_tag_profile,
                       mean_rew_tag_profile-sem_rew_tag_profile,
                       color=colour_tag, alpha=.3, edgecolor='none')
ax.fill_between(xaxis, mean_unrew_tag_profile+sem_unrew_tag_profile,
                       mean_unrew_tag_profile-sem_unrew_tag_profile,
                       color='grey', alpha=.3, edgecolor='none')

# ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(title='good v bad trials (all Dbh+)',
       ylim=(1.5,5.6),
       xlim=(-1,4),xticks=[0,2,4],
       ylabel='Firing rate (Hz)',
       xlabel='Time from run onset (s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend([p_rew_ln, p_unrew_ln], 
          ['Good trial', 'Bad trial'], frameon=False, fontsize=8)

plt.plot([-.5,.5], [5.5, 5.5], c='k', lw=.5)
plt.text(0, 5.5, 'pval_wil={}'.format(round(pval_wil, 5)), ha='center', va='bottom', color='k', fontsize=5)

# for ext in ['.png', '.pdf']:
#     fig.savefig(f'Z:\Dinghao\code_dinghao\LC_ephys\LC_tagged_goodvunrew_ROpeaking{ext}',
#                 dpi=300,
#                 bbox_inches='tight')
    
    
#%% putative 
pval_wil = wilcoxon(peak_rew_put, peak_unrew_put)[1]

mean_rew_put_profile = np.mean(prof_rew_put, axis=0)
sem_rew_put_profile = sem(prof_rew_put, axis=0)
mean_unrew_put_profile = np.mean(prof_unrew_put, axis=0)
sem_unrew_put_profile = sem(prof_unrew_put, axis=0)

fig, ax = plt.subplots(figsize=(2,1.4))
p_rew_ln, = ax.plot(xaxis, mean_rew_put_profile, color=colour_put)
p_unrew_ln, = ax.plot(xaxis, mean_unrew_put_profile, color='grey')

ax.fill_between(xaxis, mean_rew_put_profile+sem_rew_put_profile,
                       mean_rew_put_profile-sem_rew_put_profile,
                       color=colour_put, alpha=.3, edgecolor='none')
ax.fill_between(xaxis, mean_unrew_put_profile+sem_unrew_put_profile,
                       mean_unrew_put_profile-sem_unrew_put_profile,
                       color='grey', alpha=.3, edgecolor='none')

# ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(title='good v bad trials (putative)',
       ylim=(1.6,5.6),
       xlim=(-1,4),xticks=[0,2,4],
       ylabel='Firing rate (Hz)',
       xlabel='Time from run onset (s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend([p_rew_ln, p_unrew_ln], 
          ['Good trial', 'Bad trial'], frameon=False, fontsize=8)

plt.plot([-.5,.5], [5, 5], c='k', lw=.5)
plt.text(0, 5, 'pval_wil={}'.format(round(pval_wil, 5)), ha='center', va='bottom', color='k', fontsize=5)

# for ext in ['.png', '.pdf']:
#     fig.savefig(f'Z:\Dinghao\code_dinghao\LC_ephys\LC_putative_goodvunrew_ROpeaking{ext}',
#                 dpi=300,
#                 bbox_inches='tight')


#%% statistics 
pf.plot_violin_with_scatter(peak_unrew_tag, peak_rew_tag, 
                            'grey', colour_tag, 
                            paired=True, 
                            xticklabels=['Bad\ntrials', 'Good\ntrials'], 
                            ylabel='Run onset peak amplitude', 
                            title='LC RO peak', 
                            save=False, 
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\LC_tagged_goodvunrew_ROpeaking_violin', 
                            ylim=(0,9))

pf.plot_violin_with_scatter(peak_unrew_put, peak_rew_put, 
                            'grey', colour_put,
                            paired=True, 
                            xticklabels=['Bad\ntrials', 'Good\ntrials'], 
                            ylabel='run onset peak amplitude', 
                            title='LC RO peak', 
                            save=False, 
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\LC_putative_goodvunrew_ROpeaking_violin', 
                            ylim=(0,9))