# -*- coding: utf-8 -*-
"""
Created on Sun June 11 13:12:54 2023
modified on 12 Dec 2024 to tidy up analysis

LC: visual and statistical comparison between good and bad trial RO peaks 

*use tagged + putative Dbh RO peaking cells*

***UPDATED GOOD/BAD TRIALS***
bad trial parameters 12 Dec 2024 (in the .pkl dataframe):
    rewarded == -1
    noFullStop
    licks before 90

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
def get_good_bad_idx(beh_series):
    bad_trial_map = beh_series['bad_trials']
    good_idx = [trial for trial, quality in enumerate(bad_trial_map) if not quality]
    bad_idx = [trial for trial, quality in enumerate(bad_trial_map) if quality]
    
    return good_idx, bad_idx

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
peak_good_tag, peak_bad_tag = [], []
prof_good_tag, prof_bad_tag = [], []

peak_good_put, peak_bad_put = [], []
prof_good_put, prof_bad_put = [], []

for path in pathLC:
    recname = Path(path).name
    print(recname)
    
    # load behaviour
    beh_path = beh_stem / f'{recname}.pkl'
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)
    
    # import bad beh trial indices
    behPar = sio.loadmat(path+path[-18:]+
                         '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
    # -1 to account for MATLAB Python difference
    bad_idx = np.where(behPar['behPar'][0]['indTrBadBeh'][0][0]==1)[0]-1
    # -1 to account for 0 being an empty trial
    good_idx = np.arange(behPar['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    good_idx = np.delete(good_idx, bad_idx)
    
    # import stim trial indices
    baseline_idx, _, _ = get_trialtype_idx(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.
        format(path, recname)
        )
    
    # import tagged cell spike trains from all_tagged_train
    if len(bad_idx) >= 10:  # 10 bad trials at least, prevents contam.
    
        curr_cell_prop = cell_prop[cell_prop['sessname'] == recname]
        curr_RO = curr_cell_prop[curr_cell_prop['run_onset_peak'] == True]
        
        trains_path = LC_all_stem / 'all_sessions' / recname
        curr_trains = np.load(
            trains_path / f'{recname}_all_trains.npy', allow_pickle=True
            ).item()
        
        for cluname, row in curr_RO.iterrows():
            trains = curr_trains[cluname]
            curr_good = np.zeros([len(good_idx), max_length])
            curr_bad = np.zeros([len(bad_idx), max_length])
            for i, trial in enumerate(good_idx):
                curr_length = len(trains[trial])
                curr_good[i, :curr_length] = trains[trial][:max_length]
            for i, trial in enumerate(bad_idx):
                curr_length = len(trains[trial])
                curr_bad[i, :curr_length] = trains[trial][:max_length]
                
            if row['identity'] == 'tagged':
                prof_good_tag.append(np.mean(curr_good, axis=0))
                peak_good_tag.append(np.mean(prof_good_tag[-1][3125:4375]))
                prof_bad_tag.append(np.mean(curr_bad, axis=0))
                peak_bad_tag.append(np.mean(prof_bad_tag[-1][3125:4375]))
            if row['identity'] == 'putative':
                prof_good_put.append(np.mean(curr_good, axis=0))
                peak_good_put.append(np.mean(prof_good_put[-1][3125:4375]))
                prof_bad_put.append(np.mean(curr_bad, axis=0))
                peak_bad_put.append(np.mean(prof_bad_put[-1][3125:4375]))


#%% tagged 
pval_wil = wilcoxon(peak_good_tag, peak_bad_tag)[1]

mean_good_tag_profile = np.mean(prof_good_tag, axis=0)
sem_good_tag_profile = sem(prof_good_tag, axis=0)
mean_bad_tag_profile = np.mean(prof_bad_tag, axis=0)
sem_bad_tag_profile = sem(prof_bad_tag, axis=0)

fig, ax = plt.subplots(figsize=(2,1.4))
p_good_ln, = ax.plot(xaxis, mean_good_tag_profile, color=colour_tag)
p_bad_ln, = ax.plot(xaxis, mean_bad_tag_profile, color='grey')

ax.fill_between(xaxis, mean_good_tag_profile+sem_good_tag_profile,
                       mean_good_tag_profile-sem_good_tag_profile,
                       color=colour_tag, alpha=.3, edgecolor='none')
ax.fill_between(xaxis, mean_bad_tag_profile+sem_bad_tag_profile,
                       mean_bad_tag_profile-sem_bad_tag_profile,
                       color='grey', alpha=.3, edgecolor='none')

# ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(title='good v bad trials (all Dbh+)',
       ylim=(1.5,5.6),
       xlim=(-1,4),xticks=[0,2,4],
       ylabel='Firing rate (Hz)',
       xlabel='Time from run onset (s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend([p_good_ln, p_bad_ln], 
          ['Good trial', 'Bad trial'], frameon=False, fontsize=8)

plt.plot([-.5,.5], [5.5, 5.5], c='k', lw=.5)
plt.text(0, 5.5, 'pval_wil={}'.format(round(pval_wil, 5)), ha='center', va='bottom', color='k', fontsize=5)

for ext in ['.png', '.pdf']:
    fig.savefig(f'Z:\Dinghao\code_dinghao\LC_ephys\LC_tagged_goodvbad_ROpeaking{ext}',
                dpi=300,
                bbox_inches='tight')
    
    
#%% putative 
pval_wil = wilcoxon(peak_good_put, peak_bad_put)[1]

mean_good_put_profile = np.mean(prof_good_put, axis=0)
sem_good_put_profile = sem(prof_good_put, axis=0)
mean_bad_put_profile = np.mean(prof_bad_put, axis=0)
sem_bad_put_profile = sem(prof_bad_put, axis=0)

fig, ax = plt.subplots(figsize=(2,1.4))
p_good_ln, = ax.plot(xaxis, mean_good_put_profile, color=colour_put)
p_bad_ln, = ax.plot(xaxis, mean_bad_put_profile, color='grey')

ax.fill_between(xaxis, mean_good_put_profile+sem_good_put_profile,
                       mean_good_put_profile-sem_good_put_profile,
                       color=colour_put, alpha=.3, edgecolor='none')
ax.fill_between(xaxis, mean_bad_put_profile+sem_bad_put_profile,
                       mean_bad_put_profile-sem_bad_put_profile,
                       color='grey', alpha=.3, edgecolor='none')

# ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(title='good v bad trials (putative)',
       ylim=(1.6,5.6),
       xlim=(-1,4),xticks=[0,2,4],
       ylabel='Firing rate (Hz)',
       xlabel='Time from run onset (s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend([p_good_ln, p_bad_ln], 
          ['Good trial', 'Bad trial'], frameon=False, fontsize=8)

plt.plot([-.5,.5], [5, 5], c='k', lw=.5)
plt.text(0, 5, 'pval_wil={}'.format(round(pval_wil, 5)), ha='center', va='bottom', color='k', fontsize=5)

for ext in ['.png', '.pdf']:
    fig.savefig(f'Z:\Dinghao\code_dinghao\LC_ephys\LC_putative_goodvbad_ROpeaking{ext}',
                dpi=300,
                bbox_inches='tight')


#%% statistics 
pf.plot_violin_with_scatter(peak_bad_tag, peak_good_tag, 
                            'grey', colour_tag, 
                            paired=True, 
                            xticklabels=['Bad\ntrials', 'Good\ntrials'], 
                            ylabel='Run onset peak amplitude', 
                            title='LC RO peak', 
                            print_statistics=True,
                            save=True, 
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\LC_tagged_goodvbad_ROpeaking_violin', 
                            ylim=(0,9))

pf.plot_violin_with_scatter(peak_bad_put, peak_good_put, 
                            'grey', colour_put,
                            paired=True, 
                            xticklabels=['Bad\ntrials', 'Good\ntrials'], 
                            ylabel='run onset peak amplitude', 
                            title='LC RO peak', 
                            print_statistics=True,
                            save=True, 
                            savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\LC_putative_goodvbad_ROpeaking_violin', 
                            ylim=(0,9))