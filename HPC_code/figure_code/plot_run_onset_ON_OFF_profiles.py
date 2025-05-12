# -*- coding: utf-8 -*-
"""
Created on Mon 10 Mar 15:04:01 2025

plot run-onset ON and OFF cells 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import sem 
import pandas as pd
import sys 
import seaborn as sns

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, normalise_to_all
from plotting_functions import plot_violin_with_scatter
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLC = rec_list.pathHPCLCopt
pathHPCLCterm = rec_list.pathHPCLCtermopt


#%% parameters
run_onset_bin = 3750  # in samples
samp_freq = 1250  # in Hz
time_bef = 1  # in seconds 
time_aft = 4  # in seconds 
xaxis = np.arange(-samp_freq*time_bef, samp_freq*time_aft) / samp_freq
prof_window = (run_onset_bin-samp_freq*time_bef, run_onset_bin+samp_freq*time_aft)


#%% load dataframe 
print('loading dataframe...')
cell_profiles = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 

df_ON = df_pyr[df_pyr['class']=='run-onset ON']
df_OFF = df_pyr[df_pyr['class']=='run-onset OFF']


#%% plain and simple first--just the mean profiles 
ON_all = [cell.prof_ctrl_mean[2500:3750+4*1250] for cell in 
          df_pyr[df_pyr['class_ctrl']=='run-onset ON']
          .itertuples(index=False)]
ON_all_mean = np.mean(ON_all, axis=0)
ON_all_sem = sem(ON_all, axis=0)

OFF_all = [cell.prof_ctrl_mean[2500:3750+4*1250] for cell in 
           df_pyr[df_pyr['class_ctrl']=='run-onset OFF']
           .itertuples(index=False)]
OFF_all_mean = np.mean(OFF_all, axis=0)
OFF_all_sem = sem(OFF_all, axis=0)

fig, ax = plt.subplots(figsize=(2.6,2))

ON_ln, = ax.plot(xaxis, ON_all_mean, lw=0.8, c='firebrick')
ax.fill_between(xaxis,
                ON_all_mean + ON_all_sem, 
                ON_all_mean - ON_all_sem,
                color='firebrick', edgecolor='none', alpha=.25)
for p in ['top', 'right']:
    ax.spines[p].set_visible(False)
    
ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
       ylabel='spike rate (Hz)', yticks=[1, 2, 3, 4], ylim=(.9, 4.1))

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_ON_curve{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    
    
fig, ax = plt.subplots(figsize=(2.6,2))

ON_ln, = ax.plot(xaxis, OFF_all_mean, lw=.8, c='purple')
ax.fill_between(xaxis,
                OFF_all_mean + OFF_all_sem, 
                OFF_all_mean - OFF_all_sem,
                color='purple', edgecolor='none', alpha=.25)
for p in ['top', 'right']:
    ax.spines[p].set_visible(False)
    
ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
       ylabel='spike rate (Hz)', yticks=[1, 2, 3, 4], ylim=(.9, 4.1))

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_OFF_curve{ext}',
        dpi=300,
        bbox_inches='tight'
        )


#%% plot ctrl v stim profiles
ON_all_ctrl = [cell.prof_mean for cell in 
               df_pyr[df_pyr['class_ctrl']=='run-onset ON']
               .itertuples(index=False)]
ON_all_stim = [cell.prof_stim_mean for cell in 
               df_pyr[df_pyr['class_stim']=='run-onset ON']
               .itertuples(index=False)]
OFF_all_ctrl = [cell.prof_mean for cell in 
               df_pyr[df_pyr['class_ctrl']=='run-onset OFF']
               .itertuples(index=False)]
OFF_all_stim = [cell.prof_stim_mean for cell in 
               df_pyr[df_pyr['class_stim']=='run-onset OFF']
               .itertuples(index=False)]

ON_all_ctrl_mean = np.mean(ON_all_ctrl, axis=0)
ON_all_stim_mean = np.mean(ON_all_stim, axis=0)
OFF_all_ctrl_mean = np.mean(OFF_all_ctrl, axis=0)
OFF_all_stim_mean = np.mean(OFF_all_stim, axis=0)

ON_all_ctrl_sem = sem(ON_all_ctrl, axis=0)
ON_all_stim_sem = sem(ON_all_stim, axis=0)
OFF_all_ctrl_sem = sem(OFF_all_ctrl, axis=0)
OFF_all_stim_sem = sem(OFF_all_stim, axis=0)

# ON cells 
fig, ax = plt.subplots(figsize=(2.1,1.5))
ON_stim_ln, = ax.plot(
    xaxis, 
    ON_all_stim_mean[prof_window[0]:prof_window[1]], 
    color='firebrick', linewidth=1, zorder=10)
ax.fill_between(
    xaxis, 
    ON_all_stim_mean[prof_window[0]:prof_window[1]]+ON_all_stim_sem[prof_window[0]:prof_window[1]],
    ON_all_stim_mean[prof_window[0]:prof_window[1]]-ON_all_stim_sem[prof_window[0]:prof_window[1]],
    alpha=.25, color='firebrick', edgecolor='none', zorder=10)
ON_ctrl_ln, = ax.plot(
    xaxis, 
    ON_all_ctrl_mean[prof_window[0]:prof_window[1]], 
    color='grey', linewidth=1)
ax.fill_between(
    xaxis, 
    ON_all_ctrl_mean[prof_window[0]:prof_window[1]]+ON_all_ctrl_sem[prof_window[0]:prof_window[1]],
    ON_all_ctrl_mean[prof_window[0]:prof_window[1]]-ON_all_ctrl_sem[prof_window[0]:prof_window[1]],
    alpha=.25, color='grey', edgecolor='none')

ax.legend(
    [ON_stim_ln, ON_ctrl_ln], ['stim.', 'ctrl.'], 
    frameon=False, fontsize=6)

ax.set(title='run-onset ON',
       xlabel='time from run-onset (s)', ylabel='spike rate (Hz)',
       xlim=(-time_bef, time_aft), xticks=(0,2,4))
ax.title.set_fontsize(10)
for p in ['top', 'right']:
    ax.spines[p].set_visible(False)

for ext in ('.png', '.pdf'):
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\ON_ctrl_stim{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    
# OFF cells 
fig, ax = plt.subplots(figsize=(2.2,1.5))
OFF_stim_ln, = ax.plot(
    xaxis, 
    OFF_all_stim_mean[prof_window[0]:prof_window[1]], 
    color='purple', linewidth=1, zorder=10)
ax.fill_between(
    xaxis, 
    OFF_all_stim_mean[prof_window[0]:prof_window[1]]+OFF_all_stim_sem[prof_window[0]:prof_window[1]],
    OFF_all_stim_mean[prof_window[0]:prof_window[1]]-OFF_all_stim_sem[prof_window[0]:prof_window[1]],
    alpha=.25, color='purple', edgecolor='none', zorder=10)
OFF_ctrl_ln, = ax.plot(
    xaxis, 
    OFF_all_ctrl_mean[prof_window[0]:prof_window[1]], 
    color='grey', linewidth=1)
ax.fill_between(
    xaxis, 
    OFF_all_ctrl_mean[prof_window[0]:prof_window[1]]+OFF_all_ctrl_sem[prof_window[0]:prof_window[1]],
    OFF_all_ctrl_mean[prof_window[0]:prof_window[1]]-OFF_all_ctrl_sem[prof_window[0]:prof_window[1]],
    alpha=.25, color='grey', edgecolor='none')

ax.legend(
    [OFF_stim_ln, OFF_ctrl_ln], ['stim.', 'ctrl.'], 
    frameon=False, fontsize=6)

ax.set(title='run-onset OFF',
       xlabel='time from run-onset (s)', ylabel='spike rate (Hz)',
       xlim=(-time_bef, time_aft), xticks=(0,2,4))

for p in ['top', 'right']:
    ax.spines[p].set_visible(False)

for ext in ('.png', '.pdf'):
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\OFF_ctrl_stim{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    

#%% transition matrices 
class_labels = {
    'run-onset OFF': 'OFF',
    'run-onset ON': 'ON',
    'run-onset unresponsive': 'unresponsive'}

transition_matrix = pd.crosstab(
    df_pyr['class_ctrl'], df_pyr['class_stim'], 
    normalize='index'
    ).rename(index=class_labels, columns=class_labels)

fig, ax = plt.subplots(figsize=(4,4))

sns.heatmap(transition_matrix, annot=True, 
            cmap='viridis', fmt=".2f", cbar=False,
            ax=ax)

ax.set_xlabel('stim. class', fontsize=10)
ax.set_ylabel('ctrl. class', fontsize=10)
ax.set_title('cell class transition (ctrl.→stim.)', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', labelsize=9, rotation=45)
ax.tick_params(axis='y', labelsize=9)

for ext in ('.png', '.pdf'):
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response'
                f'\ctrl_stim\cell_class_transition_matrix_ctrl_stim{ext}',
                dpi=300,
                bbox_inches='tight')


# LC-opt
df_pyr_LCopt = df_pyr[df_pyr['rectype']=='HPCLC']
transition_matrix_LCopt = pd.crosstab(
    df_pyr_LCopt['class_ctrl'], df_pyr_LCopt['class_stim'], 
    normalize='index'
    ).rename(index=class_labels, columns=class_labels)

fig, ax = plt.subplots(figsize=(4,4))

sns.heatmap(transition_matrix_LCopt, annot=True, 
            cmap='viridis', fmt=".2f", cbar=False,
            ax=ax)

ax.set_xlabel('stim. class', fontsize=10)
ax.set_ylabel('ctrl. class', fontsize=10)
ax.set_title('cell class transition LCopt (ctrl.→stim.)', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', labelsize=9, rotation=45)
ax.tick_params(axis='y', labelsize=9)

for ext in ('.png', '.pdf'):
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response'
                f'\ctrl_stim\cell_class_transition_matrix_ctrl_stim_LCopt{ext}',
                dpi=300,
                bbox_inches='tight')

# LC-opt
df_pyr_LCtermopt = df_pyr[df_pyr['rectype']=='HPCLCterm']
transition_matrix_LCtermopt = pd.crosstab(
    df_pyr_LCtermopt['class_ctrl'], df_pyr_LCtermopt['class_stim'], 
    normalize='index'
    ).rename(index=class_labels, columns=class_labels)

fig, ax = plt.subplots(figsize=(4,4))

sns.heatmap(transition_matrix_LCtermopt, annot=True, 
            cmap='viridis', fmt=".2f", cbar=False,
            ax=ax)

ax.set_xlabel('stim. class', fontsize=10)
ax.set_ylabel('ctrl. class', fontsize=10)
ax.set_title('cell class transition LCtermopt (ctrl.→stim.)', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', labelsize=9, rotation=45)
ax.tick_params(axis='y', labelsize=9)

for ext in ('.png', '.pdf'):
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response'
                f'\ctrl_stim\cell_class_transition_matrix_ctrl_stim_LCtermopt{ext}',
                dpi=300,
                bbox_inches='tight')
    

#%% pre-post ratio for remaining ON and new ON cells
# step 1: identity ON-ON and extract pre-post 
df_ON_ON = df_pyr[(df_pyr['class']=='run-onset ON') &
                  (df_pyr['class_stim']=='run-onset ON')]
pre_post_ON_ON = pd.to_numeric(df_ON_ON['pre_post_stim']).to_numpy()

# step 2: identify other/OFF-ON and extract pre-post 
df_other_ON = df_pyr[(df_pyr['class']!='run-onset ON') &
                     (df_pyr['class_stim']=='run-onset ON')]
pre_post_other_ON = pd.to_numeric(df_other_ON['pre_post_stim']).to_numpy()

plot_violin_with_scatter(
    pre_post_ON_ON, pre_post_other_ON,
    'firebrick', 'darkorange',
    showscatter=True,
    paired=False,
    ylabel='pre-post ratio',
    xticklabels=('cons.\nON', 'new\nON'),
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\new_ON_cells_pre_post'
    )



#%% persistent vs newly-induced 
df_pyr_sorted = df_pyr.sort_values(by='pre_post_ctrl')

ON_pers_ctrl = [cell.prof_ctrl_mean for cell in 
                df_pyr_sorted[df_pyr_sorted['class_ctrl']=='run-onset ON']
                .itertuples(index=False)]
ON_pers_stim = [cell.prof_stim_mean for cell in 
                df_pyr_sorted[df_pyr_sorted['class_ctrl']=='run-onset ON']
                .itertuples(index=False)]
ON_pers_concat = [np.concatenate((ON_pers_ctrl[i], ON_pers_stim[i]))
                  for i in range(len(ON_pers_ctrl))]
ON_pers_ctrl_norm = [normalise_to_all(ON_pers_ctrl[i], ON_pers_concat[i])
                     for i in range(len(ON_pers_ctrl))]
ON_pers_stim_norm = [normalise_to_all(ON_pers_stim[i], ON_pers_concat[i])
                     for i in range(len(ON_pers_ctrl))]

ON_new_ctrl = [cell.prof_ctrl_mean for cell in 
               df_pyr_sorted[(df_pyr_sorted['class_ctrl']!='run-onset ON') &
                             (df_pyr_sorted['class_stim']=='run-onset ON')]
               .itertuples(index=False)]
ON_new_stim = [cell.prof_stim_mean for cell in 
               df_pyr_sorted[(df_pyr_sorted['class_ctrl']!='run-onset ON') &
                             (df_pyr_sorted['class_stim']=='run-onset ON')]
               .itertuples(index=False)]
ON_new_concat = [np.concatenate((ON_new_ctrl[i], ON_new_stim[i]))
                 for i in range(len(ON_new_ctrl))]
ON_new_ctrl_norm = [normalise_to_all(ON_new_ctrl[i], ON_new_concat[i])
                    for i in range(len(ON_new_ctrl))]
ON_new_stim_norm = [normalise_to_all(ON_new_stim[i], ON_new_concat[i])
                    for i in range(len(ON_new_ctrl))]

OFF_all_ctrl = [cell.prof_mean for cell in 
               df_pyr[df_pyr['class_ctrl']=='run-onset OFF']
               .itertuples(index=False)]
OFF_all_stim = [cell.prof_stim_mean for cell in 
               df_pyr[df_pyr['class_stim']=='run-onset OFF']
               .itertuples(index=False)]


#%% plotting 
fig, axs = plt.subplots(2,1, figsize=(2,3))

axs[0].imshow(ON_pers_ctrl_norm, cmap='viridis', interpolation='none',
              extent=(-1, 4, 0, len(ON_pers_ctrl)))
axs[0].set_aspect(.003)
axs[1].imshow(ON_pers_stim_norm, cmap='viridis', interpolation='none',
              extent=(-1, 4, 0, len(ON_pers_ctrl)))
axs[1].set_aspect(.003)

fig.savefig(r'C:\Users\luod\OneDrive - Max Planck Florida Institute for Neuroscience\Desktop\persistent_matrix.png',
            dpi=300, bbox_inches='tight')


fig, axs = plt.subplots(2,1, figsize=(2,3))

axs[0].imshow(ON_new_ctrl_norm, cmap='viridis', interpolation='none',
              extent=(-1, 4, 0, len(ON_new_ctrl)))
axs[0].set_aspect(.005)
axs[1].imshow(ON_new_stim_norm, cmap='viridis', interpolation='none',
              extent=(-1, 4, 0, len(ON_new_ctrl)))
axs[1].set_aspect(.005)

fig.savefig(r'C:\Users\luod\OneDrive - Max Planck Florida Institute for Neuroscience\Desktop\new_matrix.png',
            dpi=300, bbox_inches='tight')