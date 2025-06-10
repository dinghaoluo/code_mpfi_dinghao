# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:38:31 2025

Quantify the effects of LC stimulation on CA1 pyramidal population using 
    CI

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import pickle
import sys 
import os 
import matplotlib.pyplot as plt

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

from plotting_functions import plot_violin_with_scatter


#%% load data 
print('loading dataframes...')

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
    )
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']


#%% parameters 
SAMP_FREQ = 1250  # Hz 
PRE = 1  # s 
POST = 4  # s 
SEM_THRES = 1.96  # for 95% CI
THRESHOLD = 0.2  # s 
THRES_SAMP = int(THRESHOLD * SAMP_FREQ)
FUZZY = .90  # how many time bins need to be consecutively above/below 

taxis = np.linspace(-PRE, POST, (PRE+POST) * SAMP_FREQ)


#%% initialise dataframe
sess = {
    'rectype': [],
    'recname': [],
    'delta': [],
    'label': [],
    'mean_ctrl': [],
    'std_ctrl': [],
    'mean_stim': [],
    'std_stim': [],
    'above': [],
    'below': [],
    'ctrl_ratio': [],
    'stim_ratio': [],
    'shuf_label': []
    }
df = pd.DataFrame(sess)


#%% main
all_ctrl_stim_lick_time_delta = []
all_ctrl_stim_lick_distance_delta = []

for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')

    trains = np.load(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}\{recname}_all_trains.npy',
        allow_pickle=True
        ).item()

    if os.path.exists(
            rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl'):
        with open(rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl', 'rb') as f:
            beh = pickle.load(f)
    else:
        with open(rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLCterm\{recname}.pkl', 'rb') as f:
            beh = pickle.load(f)

    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds) if cond != '0']
    ctrl_idx = [trial + 2 for trial in stim_idx]

    pulse_times = beh['pulse_times'][1:]
    run_onsets = beh['run_onsets'][1:len(pulse_times)+1]
    pulse_onsets = [p[0] - r if p else [] for p, r in zip(pulse_times, run_onsets)]
    filtered_stim_idx = []
    filtered_ctrl_idx = []
    for stim_i, ctrl_i in zip(stim_idx, ctrl_idx):
        offset = pulse_onsets[stim_i]
        if isinstance(offset, (int, float)) and offset <= SAMP_FREQ:
            filtered_stim_idx.append(stim_i)
            filtered_ctrl_idx.append(ctrl_i)
    print(f'filtered out {len(stim_idx) - len(filtered_stim_idx)} bad trial pairs')

    if len(filtered_stim_idx) == 0:
        print('no trials left; abort')
        continue

    # first lick distances and time 
    first_lick_times = [t[0][0] - s for t, s
                        in zip(beh['lick_times'], beh['run_onsets'])
                        if t]
    ctrl_stim_lick_time_delta = np.mean(
        [t for i, t in enumerate(first_lick_times) if i in stim_idx]
        ) - np.mean(
            [t for i, t in enumerate(first_lick_times) if i in ctrl_idx]
            )
    all_ctrl_stim_lick_time_delta.append(ctrl_stim_lick_time_delta)
            
    first_lick_distances = [t[0]
                            if type(t)!=float and len(t)>0
                            else np.nan
                            for t 
                            in beh['lick_distances_aligned']
                            ][1:]
    ctrl_stim_lick_distance_delta = np.mean(
        [t for i, t in enumerate(first_lick_distances) if i in stim_idx]
        ) - np.mean(
            [t for i, t in enumerate(first_lick_distances) if i in ctrl_idx]
            )
    all_ctrl_stim_lick_distance_delta.append(ctrl_stim_lick_distance_delta)

    curr_df_pyr = df_pyr[df_pyr['recname'] == recname]

    for cluname, row in tqdm(curr_df_pyr.iterrows(),
                             total=len(curr_df_pyr)):
        train = trains[cluname]
        stim_train = train[filtered_stim_idx, 3750 - PRE * SAMP_FREQ: 3750 + POST * SAMP_FREQ]
        ctrl_train = train[filtered_ctrl_idx, 3750 - PRE * SAMP_FREQ: 3750 + POST * SAMP_FREQ]

        # get mean and std 
        mean_stim = stim_train.mean(axis=0)
        sem_stim = stim_train.std(axis=0) / np.sqrt(stim_train.shape[0])
        mean_ctrl = ctrl_train.mean(axis=0)
        sem_ctrl = ctrl_train.std(axis=0) / np.sqrt(ctrl_train.shape[0])
        
        # delta 
        delta = mean_stim - mean_ctrl
        
        # classification using N_STD thresholds 
        lower = np.clip(mean_ctrl - SEM_THRES * sem_ctrl, 0, None)  # clip at 0
        upper = mean_ctrl + SEM_THRES * sem_ctrl
        above = mean_stim > upper
        below = mean_stim < lower
        
        # threshold it
        conv_kernel = np.ones(THRES_SAMP, dtype=int)
        is_activated = np.any(
            np.convolve(above[PRE*SAMP_FREQ:-PRE*SAMP_FREQ].astype(int),
                        conv_kernel,
                        mode='valid') >= THRES_SAMP * FUZZY
        )
        is_inhibited = np.any(
            np.convolve(below[PRE*SAMP_FREQ:-PRE*SAMP_FREQ].astype(int),
                        conv_kernel,
                        mode='valid') >= THRES_SAMP * FUZZY
        )
        
        # assign labels
        if is_activated and is_inhibited:
            # find first timepoint (relative to PRE) where activation/inhibition crosses threshold
            act_conv = np.convolve(above[PRE*SAMP_FREQ:].astype(int), conv_kernel, mode='valid')
            inh_conv = np.convolve(below[PRE*SAMP_FREQ:].astype(int), conv_kernel, mode='valid')
        
            # find first index where condition is satisfied
            act_time = np.argmax(act_conv >= THRES_SAMP * FUZZY)
            inh_time = np.argmax(inh_conv >= THRES_SAMP * FUZZY)
        
            # resolve based on who crosses threshold first
            if act_time < inh_time:
                label = 'activated'
            elif inh_time < act_time:
                label = 'inhibited'
            else:
                label = 'ambiguous'  # rare case: exact tie, optional
        elif is_activated:
            label = 'activated'
        elif is_inhibited:
            label = 'inhibited'
        else:
            label = 'unchanged'
            
        # thresholdj mean firing rate 
        baseline_firing = mean_ctrl[:PRE*SAMP_FREQ].mean()
        if baseline_firing < 1:
            label = 'ambiguous'

        # calculate pre-post ratios
        train_ext = train[:, 3750 - int(1.5 * SAMP_FREQ): 3750 + int(1.5 * SAMP_FREQ)]
        stim_ext = train_ext[filtered_stim_idx]
        ctrl_ext = train_ext[filtered_ctrl_idx]
        
        mean_stim_pre = stim_ext[:, :SAMP_FREQ].mean(axis=0)
        mean_stim_post = stim_ext[:, -SAMP_FREQ:].mean(axis=0)
        mean_ctrl_pre = ctrl_ext[:, :SAMP_FREQ].mean(axis=0)
        mean_ctrl_post = ctrl_ext[:, -SAMP_FREQ:].mean(axis=0)
        
        stim_ratio = (mean_stim_post.mean() + 1e-3) / (mean_stim_pre.mean() + 1e-3)
        ctrl_ratio = (mean_ctrl_post.mean() + 1e-3) / (mean_ctrl_pre.mean() + 1e-3)
        
        # collect shuffled stim/ctrl trains across 10 shuffles
        combined_idx = filtered_stim_idx + filtered_ctrl_idx
                
        np.random.shuffle(combined_idx)  # in-place
        shuf_trains = train[combined_idx, 3750 - PRE*SAMP_FREQ : 3750 + POST*SAMP_FREQ]
        
        n_half = shuf_trains.shape[0] // 2
        
        shuf_stim_train = shuf_trains[:n_half]
        shuf_ctrl_train = shuf_trains[n_half:]
        
        # mean and sem
        mean_shuf_stim = shuf_stim_train.mean(axis=0)
        sem_shuf_stim = shuf_stim_train.std(axis=0) / np.sqrt(len(shuf_stim_train))
        mean_shuf_ctrl = shuf_ctrl_train.mean(axis=0)
        sem_shuf_ctrl = shuf_ctrl_train.std(axis=0) / np.sqrt(len(shuf_ctrl_train))
        
        # CI-based threshold
        lower = np.clip(mean_shuf_ctrl - SEM_THRES * sem_shuf_ctrl, 0, None)
        upper = mean_shuf_ctrl + SEM_THRES * sem_shuf_ctrl
        above = mean_shuf_stim > upper
        below = mean_shuf_stim < lower
        
        # classify
        conv_kernel = np.ones(THRES_SAMP, dtype=int)
        is_activated = np.any(
            np.convolve(above[PRE*SAMP_FREQ:-PRE*SAMP_FREQ].astype(int), conv_kernel, mode='valid')
            >= THRES_SAMP * FUZZY
        )
        is_inhibited = np.any(
            np.convolve(below[PRE*SAMP_FREQ:-PRE*SAMP_FREQ].astype(int), conv_kernel, mode='valid')
            >= THRES_SAMP * FUZZY
        )
        
        # assign label
        if is_activated and not is_inhibited:
            shuf_label = 'activated'
        elif is_inhibited and not is_activated:
            shuf_label = 'inhibited'
        elif is_activated and is_inhibited:
            shuf_label = 'ambiguous'
        else:
            shuf_label = 'unchanged'
                
        # store everything
        df.loc[cluname] = np.array([
            'HPCLC' if path in rec_list.pathHPCLCopt else 'HPCLCterm',
            cluname.split(' ')[0],
            delta,
            label,
            mean_ctrl,
            sem_ctrl,
            mean_stim,
            sem_stim,
            above,
            below,
            ctrl_ratio,
            stim_ratio,
            shuf_label
        ], dtype='object')
        
        ## plotting 
        # make the figure
        plt.figure(figsize=(3, 2.4))
        plt.plot(taxis, mean_ctrl, label='ctrl mean', c='grey', lw=1)
        plt.fill_between(taxis, lower, upper, color='grey', edgecolor='k', alpha=.2)
        plt.plot(taxis, mean_stim, label='stim mean', color='royalblue', lw=1)

        # mark run‐onset
        plt.axvline(0, color='k', linestyle=':')

        plt.xlabel('time from run-onset (s)')
        plt.ylabel('spike rate (Hz)')
        plt.title(f'{cluname} {label}')
        plt.legend(frameon=False, fontsize='small')
        plt.tight_layout()

        # either show or save
        if path in rec_list.pathHPCLCopt:
            outdir = 'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_curves\HPC_LC_pyr_stim_effects_sem'
        else:
            outdir = 'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_curves\HPC_LCterm_pyr_stim_effects_sem'
        plt.savefig(rf'{outdir}/{cluname} {label}.png', dpi=300, bbox_inches='tight')
        plt.close()
        

#%% Summary & group-level breakdown
print('\n--- Summary ---')
summary = df['label'].value_counts()
print('Response class counts:')
print(summary)

# modulation strength
print('\nModulation strength (mean ± SEM):')
mod_strengths = {}
for cls in ['activated', 'inhibited']:
    vals = df[df['label'] == cls]['delta'].apply(lambda d: np.mean(np.abs(d)))
    mod_strengths[cls] = vals
    if len(vals) > 0:
        print(f'{cls}: {vals.mean():.3f} ± {vals.sem():.3f} Hz (n={len(vals)})')
    else:
        print(f'{cls}: no data')

df_HPCLC = df[df['rectype'] == 'HPCLC']
df_HPCLCterm = df[df['rectype'] == 'HPCLCterm']
df_session_counts = df.groupby(['recname', 'label']).size().unstack(fill_value=0)
df_session_props = df_session_counts.div(df_session_counts.sum(axis=1), axis=0)
plot_violin_with_scatter(df_session_props['activated'], df_session_props['inhibited'], 
                         'darkorange', 'purple',
                         showscatter=True,
                         ylabel='proportion of cells',
                         xticklabels=['act.', 'inh.'],
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\population_ctrl_stim\act_inh_violin')

df_session_counts_shuf = df.groupby(['recname', 'shuf_label']).size().unstack(fill_value=0)
df_session_props_shuf = df_session_counts_shuf.div(df_session_counts_shuf.sum(axis=1), axis=0)
plot_violin_with_scatter(df_session_props_shuf['activated'], df_session_props_shuf['inhibited'],
                         'darkorange', 'grey',
                         ylabel='prop. activated',
                         xticklabels=['real', 'shuf.'],
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\population_ctrl_stim\act_shuf_violin')


#%% Group mean traces
df_act = df[df['label'] == 'activated']
df_inh = df[df['label'] == 'inhibited']

full_time = np.arange((PRE + POST) * SAMP_FREQ) / SAMP_FREQ - PRE

fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.6))

for ax, group, title, color, extrema_func in zip(
    axs,
    [df_act, df_inh],
    ['activated', 'inhibited'],
    ['darkorange', 'purple'],
    [np.argmax, np.argmin]  # max for activated, min for inhibited
):
    if len(group) == 0:
        ax.set_title(f'{title} (n=0)')
        continue

    ctrl = np.stack(group['mean_ctrl'].values)
    stim = np.stack(group['mean_stim'].values)
    mean_ctrl = ctrl.mean(axis=0)
    mean_stim = stim.mean(axis=0)
    sem_ctrl = ctrl.std(axis=0) / np.sqrt(len(ctrl))
    sem_stim = stim.std(axis=0) / np.sqrt(len(stim))

    ax.plot(full_time, mean_ctrl, color='navajowhite', label='ctrl')
    ax.fill_between(full_time, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color='navajowhite', alpha=0.3)
    ax.plot(full_time, mean_stim, color=color, label='stim')
    ax.fill_between(full_time, mean_stim - sem_stim, mean_stim + sem_stim, color=color, alpha=0.3)

    # # add vertical line at peak (activated) or trough (inhibited)
    # stim_idx = extrema_func(mean_stim)
    # stim_time = full_time[stim_idx]
    # ax.axvline(stim_time, ls='--', color=color, lw=0.75)
    # ax.text(stim_time + 0.05, ax.get_ylim()[1] * 1,
    #         f'{stim_time:.2f}s', va='top', ha='left',
    #         color=color, fontsize=7)

    ax.set(title=title, xlabel='time from run-onset (s)', xlim=(-1, 4), xticks=[0, 2, 4])
    ax.axvline(0, ls='--', c='k', lw=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axs[0].set_ylabel('firing rate (Hz)')
axs[1].legend(frameon=False, fontsize=7)
fig.suptitle('all stim.')
fig.tight_layout()
plt.show()
for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\population_ctrl_stim\all_stim_act_inh{ext}',
                dpi=300,
                bbox_inches='tight')


df_HPCLC_act = df_HPCLC[df_HPCLC['label'] == 'activated']
df_HPCLC_inh = df_HPCLC[df_HPCLC['label'] == 'inhibited']

fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.6))

for ax, group, title, color, extrema_func in zip(
    axs,
    [df_HPCLC_act, df_HPCLC_inh],
    ['activated', 'inhibited'],
    ['darkorange', 'purple'],
    [np.argmax, np.argmin]  # max for activated, min for inhibited
):
    if len(group) == 0:
        ax.set_title(f'{title} (n=0)')
        continue

    ctrl = np.stack(group['mean_ctrl'].values)
    stim = np.stack(group['mean_stim'].values)
    mean_ctrl = ctrl.mean(axis=0)
    mean_stim = stim.mean(axis=0)
    sem_ctrl = ctrl.std(axis=0) / np.sqrt(len(ctrl))
    sem_stim = stim.std(axis=0) / np.sqrt(len(stim))

    ax.plot(full_time, mean_ctrl, color='grey', label='ctrl')
    ax.fill_between(full_time, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color='grey', alpha=0.3)
    ax.plot(full_time, mean_stim, color=color, label='stim')
    ax.fill_between(full_time, mean_stim - sem_stim, mean_stim + sem_stim, color=color, alpha=0.3)

    # add vertical line at peak (activated) or trough (inhibited)
    stim_idx = extrema_func(mean_stim)
    stim_time = full_time[stim_idx]
    ax.axvline(stim_time, ls='--', color=color, lw=0.75)
    ax.text(stim_time + 0.05, ax.get_ylim()[1] * 1,
            f'{stim_time:.2f}s', va='top', ha='left',
            color=color, fontsize=7)

    ax.set(title=title, xlabel='time from run-onset (s)', xlim=(-1, 4), xticks=[0, 2, 4])
    ax.axvline(0, ls='--', c='k', lw=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axs[0].set_ylabel('firing rate (Hz)')
axs[1].legend(frameon=False, fontsize=7)
fig.suptitle('HPCLC')
fig.tight_layout()
plt.show()


df_HPCLCterm_act = df_HPCLCterm[df_HPCLCterm['label'] == 'activated']
df_HPCLCterm_inh = df_HPCLCterm[df_HPCLCterm['label'] == 'inhibited']

full_time = np.arange((PRE + POST) * SAMP_FREQ) / SAMP_FREQ - PRE
fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.6))

for ax, group, title, color, extrema_func in zip(
    axs,
    [df_HPCLCterm_act, df_HPCLCterm_inh],
    ['activated', 'inhibited'],
    ['darkorange', 'purple'],
    [np.argmax, np.argmin]  # max for activated, min for inhibited
):
    if len(group) == 0:
        ax.set_title(f'{title} (n=0)')
        continue

    ctrl = np.stack(group['mean_ctrl'].values)
    stim = np.stack(group['mean_stim'].values)
    mean_ctrl = ctrl.mean(axis=0)
    mean_stim = stim.mean(axis=0)
    sem_ctrl = ctrl.std(axis=0) / np.sqrt(len(ctrl))
    sem_stim = stim.std(axis=0) / np.sqrt(len(stim))

    ax.plot(full_time, mean_ctrl, color='grey', label='ctrl')
    ax.fill_between(full_time, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color='grey', alpha=0.3)
    ax.plot(full_time, mean_stim, color=color, label='stim')
    ax.fill_between(full_time, mean_stim - sem_stim, mean_stim + sem_stim, color=color, alpha=0.3)

    # add vertical line at peak (activated) or trough (inhibited)
    stim_idx = extrema_func(mean_stim)
    stim_time = full_time[stim_idx]
    ax.axvline(stim_time, ls='--', color=color, lw=0.75)
    ax.text(stim_time + 0.05, ax.get_ylim()[1] * 1,
            f'{stim_time:.2f}s', va='top', ha='left',
            color=color, fontsize=7)

    ax.set(title=title, xlabel='time from run-onset (s)', xlim=(-1, 4), xticks=[0, 2, 4])
    ax.axvline(0, ls='--', c='k', lw=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axs[0].set_ylabel('firing rate (Hz)')
axs[1].legend(frameon=False, fontsize=7)
fig.suptitle('HPCLCterm')
fig.tight_layout()
plt.show()


#%% Activation vs inhibition timing histogram
act_times_HPCLC, inh_times_HPCLC = [], []

for _, row in df_HPCLC.iterrows():
    above, below = row['above'], row['below']
    if above is None or below is None:
        continue

    conv_kernel = np.ones(THRES_SAMP, dtype=int)
    act_conv = np.convolve(above[PRE*SAMP_FREQ:].astype(int), conv_kernel, mode='valid')
    inh_conv = np.convolve(below[PRE*SAMP_FREQ:].astype(int), conv_kernel, mode='valid')

    if np.any(act_conv >= THRES_SAMP * FUZZY):
        idx = np.argmax(act_conv >= THRES_SAMP * FUZZY)
        act_times_HPCLC.append(taxis[PRE*SAMP_FREQ + idx])

    if np.any(inh_conv >= THRES_SAMP * FUZZY):
        idx = np.argmax(inh_conv >= THRES_SAMP * FUZZY)
        inh_times_HPCLC.append(taxis[PRE*SAMP_FREQ + idx])

fig, ax = plt.subplots(figsize=(4, 2.5))
bins = np.arange(0.1, 3.51, 0.04)
ax.hist(act_times_HPCLC, bins=bins, alpha=0.6, label='activation', color='darkorange')
ax.hist(inh_times_HPCLC, bins=bins, alpha=0.6, label='inhibition', color='purple')
ax.axvline(0, color='k', linestyle='--', lw=0.75)
ax.set_xlim(0.1, 3.5)
ax.set_xlabel('time from run-onset (s)')
ax.set_ylabel('number of cells')
ax.set_title('HPCLC activation vs inhibition timing')
ax.legend(frameon=False)
fig.tight_layout()
plt.show()


act_times_HPCLCterm, inh_times_HPCLCterm = [], []

for _, row in df_HPCLCterm.iterrows():
    above, below = row['above'], row['below']
    if above is None or below is None:
        continue

    conv_kernel = np.ones(THRES_SAMP, dtype=int)
    act_conv = np.convolve(above[PRE*SAMP_FREQ:].astype(int), conv_kernel, mode='valid')
    inh_conv = np.convolve(below[PRE*SAMP_FREQ:].astype(int), conv_kernel, mode='valid')

    if np.any(act_conv >= THRES_SAMP * FUZZY):
        idx = np.argmax(act_conv >= THRES_SAMP * FUZZY)
        act_times_HPCLCterm.append(taxis[PRE*SAMP_FREQ + idx])

    if np.any(inh_conv >= THRES_SAMP * FUZZY):
        idx = np.argmax(inh_conv >= THRES_SAMP * FUZZY)
        inh_times_HPCLCterm.append(taxis[PRE*SAMP_FREQ + idx])

fig, ax = plt.subplots(figsize=(4, 2.5))
bins = np.arange(0.1, 3.51, 0.04)
ax.hist(act_times_HPCLCterm, bins=bins, alpha=0.6, label='activation', color='darkorange')
ax.hist(inh_times_HPCLCterm, bins=bins, alpha=0.6, label='inhibition', color='purple')
ax.axvline(0, color='k', linestyle='--', lw=0.75)
ax.set_xlim(0.1, 3.5)
ax.set_xlabel('time from run-onset (s)')
ax.set_ylabel('number of cells')
ax.set_title('HPCLCterm activation vs inhibition timing')
ax.legend(frameon=False)
fig.tight_layout()
plt.show()


#%% Compare ctrl/stim ratios within activated and inhibited cells
from scipy.stats import ttest_rel, wilcoxon

# extract
act_ctrl_HPCLC = df_HPCLC[df_HPCLC['label'] == 'activated']['ctrl_ratio'].astype(float)
act_stim_HPCLC = df_HPCLC[df_HPCLC['label'] == 'activated']['stim_ratio'].astype(float)
inh_ctrl_HPCLC = df_HPCLC[df_HPCLC['label'] == 'inhibited']['ctrl_ratio'].astype(float)
inh_stim_HPCLC = df_HPCLC[df_HPCLC['label'] == 'inhibited']['stim_ratio'].astype(float)

# stats
tval_act, pval_act = ttest_rel(act_ctrl_HPCLC, act_stim_HPCLC)
wstat_act, pval_act_wil = wilcoxon(act_ctrl_HPCLC, act_stim_HPCLC)
tval_inh, pval_inh = ttest_rel(inh_ctrl_HPCLC, inh_stim_HPCLC)
wstat_inh, pval_inh_wil = wilcoxon(inh_ctrl_HPCLC, inh_stim_HPCLC)

# plot
fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

axs[0].scatter(act_ctrl_HPCLC, act_stim_HPCLC, color='darkorange', s=2, alpha=0.7)
axs[0].plot([0, 6], [0, 6], 'k--', lw=0.75)
axs[0].set_title('activated')
axs[0].set_xlabel('ctrl pre/post')
axs[0].set_ylabel('stim pre/post')
axs[0].text(0.05, 5.8,
            f't = {tval_act:.2f}, p = {pval_act:.2g}\nW = {wstat_act:.0f}, p = {pval_act_wil:.2g}',
            ha='left', va='top', fontsize=8)

axs[1].scatter(inh_ctrl_HPCLC, inh_stim_HPCLC, color='purple', s=2, alpha=0.7)
axs[1].plot([0, 6], [0, 6], 'k--', lw=0.75)
axs[1].set_title('inhibited')
axs[1].set_xlabel('ctrl pre/post')
axs[1].text(0.05, 5.8,
            f't = {tval_inh:.2f}, p = {pval_inh:.2g}\nW = {wstat_inh:.0f}, p = {pval_inh_wil:.2g}',
            ha='left', va='top', fontsize=8)

for ax in axs:
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.suptitle('HPCLC')
plt.show()


# extract
act_ctrl_HPCLCterm = df_HPCLCterm[df_HPCLCterm['label'] == 'activated']['ctrl_ratio'].astype(float)
act_stim_HPCLCterm = df_HPCLCterm[df_HPCLCterm['label'] == 'activated']['stim_ratio'].astype(float)
inh_ctrl_HPCLCterm = df_HPCLCterm[df_HPCLCterm['label'] == 'inhibited']['ctrl_ratio'].astype(float)
inh_stim_HPCLCterm = df_HPCLCterm[df_HPCLCterm['label'] == 'inhibited']['stim_ratio'].astype(float)

# stats
tval_act, pval_act = ttest_rel(act_ctrl_HPCLCterm, act_stim_HPCLCterm)
wstat_act, pval_act_wil = wilcoxon(act_ctrl_HPCLCterm, act_stim_HPCLCterm)
tval_inh, pval_inh = ttest_rel(inh_ctrl_HPCLCterm, inh_stim_HPCLCterm)
wstat_inh, pval_inh_wil = wilcoxon(inh_ctrl_HPCLCterm, inh_stim_HPCLCterm)

# plot
fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

axs[0].scatter(act_ctrl_HPCLCterm, act_stim_HPCLCterm, color='darkorange', s=2, alpha=0.7)
axs[0].plot([0, 6], [0, 6], 'k--', lw=0.75)
axs[0].set_title('activated')
axs[0].set_xlabel('ctrl pre/post')
axs[0].set_ylabel('stim pre/post')
axs[0].text(0.05, 5.8,
            f't = {tval_act:.2f}, p = {pval_act:.2g}\nW = {wstat_act:.0f}, p = {pval_act_wil:.2g}',
            ha='left', va='top', fontsize=8)

axs[1].scatter(inh_ctrl_HPCLCterm, inh_stim_HPCLCterm, color='purple', s=2, alpha=0.7)
axs[1].plot([0, 6], [0, 6], 'k--', lw=0.75)
axs[1].set_title('inhibited')
axs[1].set_xlabel('ctrl pre/post')
axs[1].text(0.05, 5.8,
            f't = {tval_inh:.2f}, p = {pval_inh:.2g}\nW = {wstat_inh:.0f}, p = {pval_inh_wil:.2g}',
            ha='left', va='top', fontsize=8)

for ax in axs:
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.suptitle('HPCLCterm')
plt.show()


#%% Plot mean delta (stim − ctrl) for activated and inhibited cells
def plot_mean_delta(df_group, title_prefix):
    df_act = df_group[df_group['label'] == 'activated']
    df_inh = df_group[df_group['label'] == 'inhibited']
    full_time = np.arange((PRE + POST) * SAMP_FREQ) / SAMP_FREQ - PRE

    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.6))

    for ax, group, title, color in zip(
        axs,
        [df_act, df_inh],
        ['activated', 'inhibited'],
        ['darkorange', 'purple']
    ):
        if len(group) == 0:
            ax.set_title(f'{title} (n=0)')
            continue

        deltas = np.stack(group['delta'].values)
        mean_delta = deltas.mean(axis=0)
        sem_delta = deltas.std(axis=0) / np.sqrt(len(deltas))

        ax.plot(full_time, mean_delta, color=color)
        ax.fill_between(full_time, mean_delta - sem_delta, mean_delta + sem_delta,
                        color=color, alpha=0.3)
        ax.axhline(0, ls='--', c='k', lw=0.75)
        ax.axvline(0, ls='--', c='k', lw=0.75)

        ax.set(title=title, xlabel='time from run-onset (s)',
               xlim=(-1, 4), xticks=[0, 2, 4])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].set_ylabel('Δ firing rate (Hz)')
    fig.suptitle(f'{title_prefix} Δ (stim − ctrl)')
    fig.tight_layout()
    plt.show()

# plot for HPCLC and HPCLCterm
plot_mean_delta(df_HPCLC, 'HPCLC')
plot_mean_delta(df_HPCLCterm, 'HPCLCterm')


#%% ON vs DOWN classification
def compare_on_off_sessionwise(df_group, title_prefix):
    # filter activated and inhibited cells
    df_act = df_group[df_group['label'] == 'activated']
    df_inh = df_group[df_group['label'] == 'inhibited']

    # group by recname
    on_props = []
    off_props = []
    sessions = sorted(set(df_act['recname']) | set(df_inh['recname']))

    for rec in sessions:
        act_ratios = df_act[df_act['recname'] == rec]['stim_ratio'].astype(float)
        inh_ratios = df_inh[df_inh['recname'] == rec]['stim_ratio'].astype(float)

        if len(act_ratios) == 0 or len(inh_ratios) == 0:
            continue  # skip incomplete sessions

        prop_on = (act_ratios > 1.5).mean()
        prop_off = (inh_ratios < 2/3).mean()

        on_props.append(prop_on)
        off_props.append(prop_off)

    # stats
    on_props = np.array(on_props)
    off_props = np.array(off_props)
    tval, pval_t = ttest_rel(on_props, off_props)
    stat, pval_w = wilcoxon(on_props, off_props)

    # plot
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(['ON (act.)', 'OFF (inh.)'],
           [on_props.mean() * 100, off_props.mean() * 100],
           yerr=[on_props.std() / np.sqrt(len(on_props)) * 100,
                 off_props.std() / np.sqrt(len(off_props)) * 100],
           color=['darkorange', 'purple'], alpha=0.8, capsize=3)

    ax.set_ylabel('% of cells per session')
    ax.set_ylim(0, 100)
    ax.set_title(f'{title_prefix}\nON vs OFF (session mean ± SEM)\n'
                 f't = {tval:.2f}, p = {pval_t:.2g}  |  W = {stat:.0f}, p = {pval_w:.2g}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

compare_on_off_sessionwise(df_HPCLC, 'HPCLC')
compare_on_off_sessionwise(df_HPCLCterm, 'HPCLCterm')


#%% regression of activation amplitude vs lick delay/distance
import matplotlib.pyplot as plt
from scipy.stats import linregress

# extract per-session activation amplitude
df_act = df[df['label'] == 'activated']
session_delta_amplitudes = df_act.groupby('recname')['delta'].apply(
    lambda dlist: np.mean([np.mean(d[int(1.5*SAMP_FREQ): int(-2.5*SAMP_FREQ)]) for d in dlist])
)

# ensure matching session order
recnames = sorted(session_delta_amplitudes.index)
delta_time = np.array([
    delta for rec, delta in zip(df['recname'].unique(), all_ctrl_stim_lick_time_delta)
    if rec in session_delta_amplitudes.index
])
delta_dist = np.array([
    delta for rec, delta in zip(df['recname'].unique(), all_ctrl_stim_lick_distance_delta)
    if rec in session_delta_amplitudes.index
])
amplitudes = session_delta_amplitudes.values

# filter first 
valid_mask = ~np.isnan(delta_dist) & ~np.isnan(amplitudes)

# regression: amplitude vs lick time delta
slope_time, intercept_time, r_time, p_time, _ = linregress(delta_time, amplitudes)

# regression: amplitude vs lick distance delta
slope_dist, intercept_dist, r_dist, p_dist, _ = linregress(delta_dist[valid_mask], amplitudes[valid_mask])

# plot time-based regression
fig, ax = plt.subplots(figsize=(3.6, 3))
ax.scatter(delta_time, amplitudes, color='crimson')
x_vals = np.linspace(min(delta_time), max(delta_time), 100)
ax.plot(x_vals, intercept_time + slope_time * x_vals, 'k--', lw=1)
ax.set_xlabel('lick latency (stim − ctrl)')
ax.set_ylabel('activation amplitude (Hz)')
ax.set_title(f'vs lick delay\nr = {r_time:.2f}, p = {p_time:.3g}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.show()

# plot distance-based regression
fig, ax = plt.subplots(figsize=(3.6, 3))
ax.scatter(delta_dist, amplitudes, color='navy')
x_vals = np.linspace(min(delta_dist), max(delta_dist), 100)
ax.plot(x_vals, intercept_dist + slope_dist * x_vals, 'k--', lw=1)
ax.set_xlabel('lick distance (stim − ctrl)')
ax.set_ylabel('activation amplitude (Hz)')
ax.set_title(f'vs lick distance\nr = {r_dist:.2f}, p = {p_dist:.3g}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.show()
