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
from common import mpl_formatting
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
ON_all = [cell.prof_mean[2500:3750+4*1250] for cell in 
          df_pyr[df_pyr['class']=='run-onset ON']
          .itertuples(index=False)]
ON_all_mean = np.mean(ON_all, axis=0)
ON_all_sem = sem(ON_all, axis=0)

OFF_all = [cell.prof_mean[2500:3750+4*1250] for cell in 
           df_pyr[df_pyr['class']=='run-onset OFF']
           .itertuples(index=False)]
OFF_all_mean = np.mean(OFF_all, axis=0)
OFF_all_sem = sem(OFF_all, axis=0)

fig, ax = plt.subplots(figsize=(2,1.4))

ON_ln, = ax.plot(xaxis, ON_all_mean, lw=0.8, c='firebrick')
ax.fill_between(xaxis,
                ON_all_mean + ON_all_sem, 
                ON_all_mean - ON_all_sem,
                color='firebrick', edgecolor='none', alpha=.25)
for p in ['top', 'right']:
    ax.spines[p].set_visible(False)
    
ax.set(title='run-onset ON',
       ylabel='spike rate (Hz)',
       xlabel='time from run-onset (s)')

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_ON_curve{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    
    
fig, ax = plt.subplots(figsize=(2,1.4))

ON_ln, = ax.plot(xaxis, OFF_all_mean, lw=.8, c='purple')
ax.fill_between(xaxis,
                OFF_all_mean + OFF_all_sem, 
                OFF_all_mean - OFF_all_sem,
                color='purple', edgecolor='none', alpha=.25)
for p in ['top', 'right']:
    ax.spines[p].set_visible(False)
    
ax.set(title='run-onset OFF',
       ylabel='spike rate (Hz)',
       xlabel='time from run-onset (s)')

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_OFF_curve{ext}',
        dpi=300,
        bbox_inches='tight'
        )


#%% plot ctrl v stim profiles
ON_all_ctrl = [cell.prof_mean for cell in 
               df_pyr[df_pyr['class']=='run-onset ON']
               .itertuples(index=False)]
ON_all_stim = [cell.prof_stim_mean for cell in 
               df_pyr[df_pyr['class_stim']=='run-onset ON']
               .itertuples(index=False)]
OFF_all_ctrl = [cell.prof_mean for cell in 
               df_pyr[df_pyr['class']=='run-onset OFF']
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
    
plt.close()

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
    
plt.close()


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

plt.close()


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

plt.close()


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

plt.close()
    

#%% pre-post ratio for remaining ON and new ON cells
# step 1: identity ON-ON and extract pre-post 
df_ON_ON = df_pyr[(df_pyr['class_ctrl']=='run-onset ON') &
                  (df_pyr['class_stim']=='run-onset ON')]
pre_post_ON_ON = pd.to_numeric(df_ON_ON['pre_post_stim']).to_numpy()

# step 2: identify other/OFF-ON and extract pre-post 
df_other_ON = df_pyr[(df_pyr['class_ctrl']!='run-onset ON') &
                     (df_pyr['class_stim']=='run-onset ON')]
pre_post_other_ON = pd.to_numeric(df_other_ON['pre_post_stim']).to_numpy()

plot_violin_with_scatter(
    pre_post_ON_ON, pre_post_other_ON,
    'firebrick', 'darkorange',
    paired=False,
    ylabel='pre-post ratio',
    xticklabels=('cons.\nON', 'new\nON'),
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\new_ON_cells_pre_post'
    )


#%% are these new ON cells more correlated with licking?
import scipy.io as sio 
from tqdm import tqdm

beh_df = pd.concat((
    pd.read_pickle(
        r'Z:/Dinghao/code_dinghao/behaviour/all_HPCLC_sessions.pkl'
        ),
    pd.read_pickle(
        r'Z:/Dinghao/code_dinghao/behaviour/all_HPCLCterm_sessions.pkl'
        )
    ))

recname = ''

early_profs, late_profs, early_mid_profs, late_mid_profs = [], [], [], []
for clu in tqdm(df_other_ON.itertuples(),
                total=len(df_other_ON)):
    if clu.recname != recname:
        recname = clu.recname
    
        # get lick times 
        curr_beh_df = beh_df.loc[recname]  # subselect in read-only
        run_onsets = curr_beh_df['run_onsets'][1:]
        licks = [
            [(l-run_onset)/1000 for l in trial]  # convert from ms to s
            if len(trial)!=0 else np.nan
            for trial, run_onset in zip(
                    curr_beh_df['lick_times'][1:],
                    run_onsets
                    )
            ]
        first_licks = np.asarray(
            [next((l for l in trial if l > 1), np.nan)  # >1 to prevent carry-over licks
            if isinstance(trial, list) else np.nan
            for trial in licks]
            )
        
        # get bad trials 
        behPar = sio.loadmat(
            rf'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}/{recname[:14]}/{recname}'
            rf'/{recname}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
            )
        bad_beh_ind = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
        
        # get early and late lick trials (that are not stim. trials)
        stim_trials = np.where(
            np.asarray([
                trial[15] for trial
                in curr_beh_df['trial_statements']
                ])!='0'
            )[0]
        valid_trials = [i for i in range(len(first_licks)) 
                        if i not in stim_trials 
                        and i not in bad_beh_ind 
                        and not np.isnan(first_licks[i])]
    
        # load spike trains
        if len(valid_trials) > 50:
            trains = np.load(
                r'Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions'
                rf'/{recname}/{recname}_all_trains.npy',
                allow_pickle=True
                ).item()
        
            sorted_trials = sorted(valid_trials, 
                                   key=lambda i: first_licks[i])[10:-10]  # avoid extremities 
            
            early_trials = sorted_trials[:10]
            early_mid_trials = sorted_trials[:int(len(sorted_trials)/2)]
            late_trials = sorted_trials[-10:]
            late_mid_trials = sorted_trials[int(len(sorted_trials)/2):]
        
    if len(valid_trials) < 50:
        continue 
    
    train = trains[clu.Index]
    early_trains = [
        train[trial, :] for trial
        in early_trials
        ]
    late_trains = [
        train[trial, :] for trial
        in late_trials
        ]
    early_mid_trains = [
        train[trial, :] for trial
        in early_mid_trials
        ]
    late_mid_trains = [
        train[trial, :] for trial
        in late_mid_trials
        ]
    
    early_profs.append(np.mean(early_trains, axis=0))
    late_profs.append(np.mean(late_trains, axis=0))
    early_mid_profs.append(np.mean(early_mid_trains, axis=0))
    late_mid_profs.append(np.mean(late_mid_trains, axis=0))
    
early_profs_mean = np.mean(early_profs, axis=0)
late_profs_mean = np.mean(late_profs, axis=0)
early_mid_profs_mean = np.mean(early_profs, axis=0)
late_mid_profs_mean = np.mean(late_profs, axis=0)

from scipy.stats import sem
early_profs_sem = sem(early_profs, axis=0)
late_profs_sem = sem(late_profs, axis=0)
early_mid_profs_sem = sem(early_profs, axis=0)
late_mid_profs_sem = sem(late_profs, axis=0)

early_ratios = [np.nanmean(prof[3750-1250:3750])/np.nanmean(prof[3750:3750+1250])
                if sum(prof[3750:3750+1250])>0 else 1
                for prof in early_profs]
late_ratios = [np.nanmean(prof[3750-1250:3750])/np.nanmean(prof[3750:3750+1250])
                if sum(prof[3750:3750+1250])>0 else 1
                for prof in late_profs]
outlier_mask = [i for i in range(len(early_ratios))
                if early_ratios[i] > 10 or late_ratios[i] > 10]
early_ratios = [v for i, v in enumerate(early_ratios) 
                if i not in outlier_mask]
late_ratios = [v for i, v in enumerate(late_ratios) 
                  if i not in outlier_mask]

# early_mid_ON_ratios = [np.nanmean(prof[3750-1250:3750])/np.nanmean(prof[3750:3750+1250])
#                        if sum(prof[3750:3750+1250])>0 else 1
#                        for prof in early_mid_ON_profs]
# late_mid_ON_ratios = [np.nanmean(prof[3750-1250:3750])/np.nanmean(prof[3750:3750+1250])
#                       if sum(prof[3750:3750+1250])>0 else 1
#                       for prof in late_mid_ON_profs]

# outlier_mid_mask = [i for i in range(len(early_mid_ON_ratios))
#                     if early_mid_ON_ratios[i] > 5 or late_mid_ON_ratios[i] > 5]

# early_mid_ON_ratios = [v for i, v in enumerate(early_mid_ON_ratios) 
#                        if i not in outlier_mid_mask]
# late_mid_ON_ratios = [v for i, v in enumerate(late_mid_ON_ratios) 
#                       if i not in outlier_mid_mask]


plot_violin_with_scatter(early_ratios, late_ratios, 'orange', 'darkred')

xaxis = np.arange(-1250, 1250*4)/1250

fig, ax = plt.subplots(figsize=(3,2))

ax.plot(xaxis, 
        early_mid_profs_mean[3750-1250:3750+1250*4])
ax.plot(xaxis,
        late_mid_profs_mean[3750-1250:3750+1250*4], 
        color='red')
ax.fill_between(
    xaxis,
    early_profs_mean[3750-1250:3750+1250*4]+early_profs_sem[3750-1250:3750+1250*4],
    early_profs_mean[3750-1250:3750+1250*4]-early_profs_sem[3750-1250:3750+1250*4],
    alpha=.35
    )
ax.fill_between(
    xaxis,
    late_profs_mean[3750-1250:3750+1250*4]+late_profs_sem[3750-1250:3750+1250*4],
    late_profs_mean[3750-1250:3750+1250*4]-late_profs_sem[3750-1250:3750+1250*4],
    alpha=.35, color='red', edgecolor='none'
    )

