# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:48:22 2025

quantify stim effects; based on HPC_all_stim_effects.py

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pandas as pd 
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import sys 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

from plotting_functions import plot_violin_with_scatter


#%% load dataframe 
print('loading dataframe...')
df = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_pyr_stim_effects.pkl')


#%% parameters
PRE = 1       # seconds before run-onset
POST = 4
BIN_SIZE = 0.4
ALPHA = 0.05

TIME_AXIS = np.arange(-PRE, POST, BIN_SIZE)[:-1]  # 12.5 bins, actually...
POST_MASK = (TIME_AXIS >= 0) & (TIME_AXIS <= 2.5)


#%% initialise new dataframe
cols = ['rectype', 'recname',
        'response_class', 'n_sig', 'n_act', 'n_inh',
        'mod_strength', 'mod_strength_norm']
df_new = pd.DataFrame(columns=cols)


#%% loop through cells and classify
print('classifying...')
for cluname, row in df.iterrows():
    pvals = np.array(row['pvals'])
    delta = np.array(row['delta'])

    sig = (pvals < ALPHA) & POST_MASK
    act = sig & (delta > 0)
    inh = sig & (delta < 0)

    n_act = act.sum()
    n_inh = inh.sum()
    n_sig = sig.sum()

    if n_sig == 0:
        label = 'unchanged'
        mod_strength = 0
        mod_strength_norm = 0

    else:
        if n_act > n_inh:
            label = 'activated'
            used_bins = act
        elif n_inh > n_act:
            label = 'inhibited'
            used_bins = inh
        else:
            # tie-break by first significant bin
            act_idx = np.where(act)[0]
            inh_idx = np.where(inh)[0]
            if len(act_idx) == 0 or len(inh_idx) == 0:
                label = 'mixed'
                used_bins = sig
            elif act_idx[0] < inh_idx[0]:
                label = 'activated'
                used_bins = act
            else:
                label = 'inhibited'
                used_bins = inh

        mod_strength = np.mean(np.abs(delta[used_bins]))
        baseline_rate = np.mean(row['mean_ctrl'][:np.sum(TIME_AXIS < 0)])
        baseline_rate = max(baseline_rate, 1e-3)
        mod_strength_norm = mod_strength / baseline_rate

    df_new.loc[cluname] = np.array([
        row['rectype'],
        row['recname'],
        label,
        n_sig,
        n_act,
        n_inh,
        mod_strength,
        mod_strength_norm
    ], dtype='object')


#%% test 
df_term = df_new[df_new['rectype']=='HPCLCterm']


#%% summary
print('\n--- Summary ---')
summary = df_term['response_class'].value_counts()
print('Response class counts:')
print(summary)

print('\nModulation strength (mean ± SEM):')
for cls in ['activated', 'inhibited']:
    vals = df_term[df_term['response_class'] == cls]['mod_strength'].astype(float)
    if len(vals) > 0:
        print(f'{cls}: {vals.mean():.3f} ± {vals.sem():.3f} Hz (n={len(vals)})')
    else:
        print(f'{cls}: no data')

#%% statistical comparison
act_vals = df_term[df_term['response_class'] == 'activated']['mod_strength'].astype(float)
inh_vals = df_term[df_term['response_class'] == 'inhibited']['mod_strength'].astype(float)

if len(act_vals) > 1 and len(inh_vals) > 1:
    print('\nStatistical comparison (mod_strength):')
    tval, pval = ttest_ind(act_vals, inh_vals, equal_var=False)
    print(f't-test: t = {tval:.3f}, p = {pval:.3e}')
    
    uval, pval_u = mannwhitneyu(act_vals, inh_vals, alternative='two-sided')
    print(f'Mann–Whitney U: U = {uval}, p = {pval_u:.3e}')
else:
    print('\nNot enough data for statistical comparison.')
    

#%% average firing profiles for activated and inhibited cells
print('\nComputing population firing rate profiles...')

def get_mean_sem_profiles(df_subset, cond='stim'):
    traces = np.stack(df_subset[f'mean_{cond}'].values)
    return traces.mean(axis=0), traces.std(axis=0) / np.sqrt(traces.shape[0])

# filter by class
activated_cells = df_term[df_term['response_class'] == 'activated'].index
inhibited_cells = df_term[df_term['response_class'] == 'inhibited'].index

# look up firing rate traces from the original dataframe
df_activated = df.loc[activated_cells]
df_inhibited = df.loc[inhibited_cells]

# get mean and SEM
mean_act_ctrl, sem_act_ctrl = get_mean_sem_profiles(df_activated, 'ctrl')
mean_act_stim, sem_act_stim = get_mean_sem_profiles(df_activated, 'stim')
mean_inh_ctrl, sem_inh_ctrl = get_mean_sem_profiles(df_inhibited, 'ctrl')
mean_inh_stim, sem_inh_stim = get_mean_sem_profiles(df_inhibited, 'stim')


#%% plot
fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.4), sharey=True)

# activated
axs[0].plot(TIME_AXIS, mean_act_ctrl, label='ctrl', color='grey')
axs[0].fill_between(TIME_AXIS, mean_act_ctrl - sem_act_ctrl, mean_act_ctrl + sem_act_ctrl,
                    color='grey', alpha=0.3)
axs[0].plot(TIME_AXIS, mean_act_stim, label='stim', color='crimson')
axs[0].fill_between(TIME_AXIS, mean_act_stim - sem_act_stim, mean_act_stim + sem_act_stim,
                    color='crimson', alpha=0.3)
axs[0].set(title='Activated cells', xlabel='Time (s)', ylabel='Firing rate (Hz)')
axs[0].axvline(0, ls='--', c='k', lw=0.5)

# inhibited
axs[1].plot(TIME_AXIS, mean_inh_ctrl, label='ctrl', color='grey')
axs[1].fill_between(TIME_AXIS, mean_inh_ctrl - sem_inh_ctrl, mean_inh_ctrl + sem_inh_ctrl,
                    color='grey', alpha=0.3)
axs[1].plot(TIME_AXIS, mean_inh_stim, label='stim', color='navy')
axs[1].fill_between(TIME_AXIS, mean_inh_stim - sem_inh_stim, mean_inh_stim + sem_inh_stim,
                    color='navy', alpha=0.3)
axs[1].set(title='Inhibited cells', xlabel='Time (s)')
axs[1].axvline(0, ls='--', c='k', lw=0.5)

for ax in axs:
    ax.set_xlim(-1, 4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axs[1].legend(loc='upper right', frameon=False)

fig.tight_layout()


#%% mod strength plot 
plot_violin_with_scatter(act_vals, inh_vals, 'crimson', 'navy',
                         paired=False,
                         showscatter=True,
                         ylabel='Δ spike rate (Hz)',
                         xticklabels=['act.', 'inh.'])
