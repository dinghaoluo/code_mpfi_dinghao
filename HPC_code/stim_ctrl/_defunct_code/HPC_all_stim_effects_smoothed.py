# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:10:45 2025

Quantify the effects of LC stimulation on CA1 pyramidal population using bin-by-bin t-tests on smoothed spike trains.

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
from scipy.stats import ttest_rel, wilcoxon

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


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
N_STD = 2  # 2 std for 95% CI
WINDOW = [0, 4]  # post-onset window for classification


#%% initialise dataframe
sess = {
    'rectype': [],
    'recname': [],
    'n_sig': [],
    'n_act': [],
    'n_inh': [],
    'pvals': [],
    'delta': [],
    'label': [],
    'mean_ctrl': [],
    'mean_stim': [],
    'ctrl_ratio': [],
    'stim_ratio': []
    }
df = pd.DataFrame(sess)

#%% main
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

    curr_df_pyr = df_pyr[df_pyr['recname'] == recname]

    for cluname, row in tqdm(curr_df_pyr.iterrows(),
                             total=len(curr_df_pyr)):
        train = trains[cluname]
        stim_train = train[filtered_stim_idx, 3750 - PRE * SAMP_FREQ: 3750 + POST * SAMP_FREQ]
        ctrl_train = train[filtered_ctrl_idx, 3750 - PRE * SAMP_FREQ: 3750 + POST * SAMP_FREQ]

        # only test within post-onset window
        start_idx = int(PRE * SAMP_FREQ)
        end_idx = int((PRE + WINDOW[1]) * SAMP_FREQ)
        stim_win = stim_train[:, start_idx:end_idx]
        ctrl_win = ctrl_train[:, start_idx:end_idx]

        delta = stim_win.mean(axis=0) - ctrl_win.mean(axis=0)
        tvals, pvals = ttest_rel(stim_win, ctrl_win, axis=0)

        sig = pvals < ALPHA
        act = sig & (delta > 0)
        inh = sig & (delta < 0)

        n_sig = sig.sum()
        n_act = act.sum()
        n_inh = inh.sum()

        if n_sig < 1250 * 0.01:
            label = 'unchanged'
        elif n_act > n_inh:
            label = 'activated'
        elif n_inh > n_act:
            label = 'inhibited'
        else:
            label = 'mixed'

        mean_ctrl = ctrl_train.mean(axis=0)
        mean_stim = stim_train.mean(axis=0)
        
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

        # store everything
        df.loc[cluname] = np.array([
            'HPCLC' if path in rec_list.pathHPCLCopt else 'HPCLCterm',
            cluname.split(' ')[0],
            n_sig,
            n_act,
            n_inh,
            pvals,
            delta,
            label,
            mean_ctrl,
            mean_stim,
            ctrl_ratio,
            stim_ratio
        ], dtype='object')

        # plotting
        full_time = np.arange((PRE + POST) * SAMP_FREQ) / SAMP_FREQ - PRE
        sem_ctrl = ctrl_train.std(axis=0) / np.sqrt(ctrl_train.shape[0])
        sem_stim = stim_train.std(axis=0) / np.sqrt(stim_train.shape[0])

        fig, ax = plt.subplots(figsize=(2.2, 1.8))
        ax.plot(full_time, mean_ctrl, color='grey', label='ctrl')
        ax.fill_between(full_time, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color='grey', alpha=0.3)
        ax.plot(full_time, mean_stim, color='royalblue', label='stim')
        ax.fill_between(full_time, mean_stim - sem_stim, mean_stim + sem_stim, color='royalblue', alpha=0.3)

        for i, (a, ih) in enumerate(zip(act, inh)):
            t0 = full_time[start_idx + i]
            t1 = full_time[start_idx + i + 1] if i + 1 < len(act) else t0 + 1/SAMP_FREQ
            if a:
                color = 'crimson'
            elif ih:
                color = 'navy'
            else:
                continue
            ax.axvspan(t0, t1, ymin=0.95, ymax=0.98, color=color, lw=.5, alpha=0.7)

        ax.set(xlabel='time from run-onset (s)', ylabel='firing rate (Hz)', xlim=(-1, 4), xticks=[0, 2, 4])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=False, fontsize=7)
        ax.set_title(cluname)
        fig.tight_layout()

        save_path = (
            rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_curves\HPC_LC_pyr_stim_effects\{cluname}.png'
            if path in rec_list.pathHPCLCopt else
            rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_curves\HPC_LCterm_pyr_stim_effects\{cluname}.png'
        )
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

#%% save
outpath = r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_pyr_stim_effects.pkl'
df.to_pickle(outpath)
print(f'saved to {outpath}')

#%% summary
print('\n--- Summary ---')
summary = df['label'].value_counts()
print('Response class counts:')
print(summary)

print('\nModulation strength (mean ± SEM):')
mod_strengths = {}
for cls in ['activated', 'inhibited']:
    vals = df[df['label'] == cls]['delta'].apply(lambda d: np.mean(np.abs(d)))
    mod_strengths[cls] = vals
    if len(vals) > 0:
        print(f'{cls}: {vals.mean():.3f} ± {vals.sem():.3f} Hz (n={len(vals)})')
    else:
        print(f'{cls}: no data')

#%% statistical comparison
from scipy.stats import ttest_ind, mannwhitneyu

act_vals = mod_strengths['activated']
inh_vals = mod_strengths['inhibited']

if len(act_vals) > 1 and len(inh_vals) > 1:
    print('\nStatistical comparison (modulation strength):')
    tval, pval = ttest_ind(act_vals, inh_vals, equal_var=False)
    print(f't-test: t = {tval:.3f}, p = {pval:.3e}')
    
    uval, pval_u = mannwhitneyu(act_vals, inh_vals, alternative='two-sided')
    print(f'Mann–Whitney U: U = {uval}, p = {pval_u:.3e}')
else:
    print('\nNot enough data for statistical comparison.')


#%% population mean plot
activated = df[df['label'] == 'activated']
inhibited = df[df['label'] == 'inhibited']
activated_term = df[(df['label'] == 'activated') & (df['rectype'] == 'HPCLCterm')]
inhibited_term = df[(df['label'] == 'inhibited') & (df['rectype'] == 'HPCLCterm')]

full_time = np.arange((PRE + POST) * SAMP_FREQ) / SAMP_FREQ - PRE

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)

for ax, group, title, color in zip(axs,
                                   [activated_term, inhibited_term],
                                   ['Activated cells', 'Inhibited cells'],
                                   ['crimson', 'navy']):
    if len(group) == 0:
        ax.set_title(f'{title} (n=0)')
        continue
    ctrl_traces = np.stack(group['mean_ctrl'])
    stim_traces = np.stack(group['mean_stim'])
    mean_ctrl = ctrl_traces.mean(axis=0)
    mean_stim = stim_traces.mean(axis=0)
    sem_ctrl = ctrl_traces.std(axis=0) / np.sqrt(ctrl_traces.shape[0])
    sem_stim = stim_traces.std(axis=0) / np.sqrt(stim_traces.shape[0])
    ax.plot(full_time, mean_ctrl, color='grey', label='ctrl')
    ax.fill_between(full_time, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color='grey', alpha=0.3)
    ax.plot(full_time, mean_stim, color=color, label='stim')
    ax.fill_between(full_time, mean_stim - sem_stim, mean_stim + sem_stim, color=color, alpha=0.3)
    ax.set(title=title, xlabel='time from run-onset (s)', xlim=(-1, 4), xticks=[0, 2, 4])
    ax.axvline(0, ls='--', c='k', lw=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axs[0].set_ylabel('firing rate (Hz)')
axs[1].legend(frameon=False, fontsize=7)
fig.tight_layout()


#%% session by session plot
outdir_root = 'Z:/Dinghao/code_dinghao/HPC_ephys/population_ctrl_stim'

# group and plot by session (recname) within each rectype
for rectype in ['HPCLC', 'HPCLCterm']:
    df_rectype = df[df['rectype'] == rectype]
    if rectype == 'HPCLC':
        rectype_dir = os.path.join(outdir_root, 'HPC_LC_pyr_act_inh')
    else:
        rectype_dir = os.path.join(outdir_root, 'HPC_LCterm_pyr_act_inh')
    os.makedirs(rectype_dir, exist_ok=True)
    
    for recname in df_rectype['recname'].unique():
        df_sess = df_rectype[df_rectype['recname'] == recname]
        activated_sess = df_sess[df_sess['label'] == 'activated']
        inhibited_sess = df_sess[df_sess['label'] == 'inhibited']

        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)

        for ax, group, title, color in zip(axs,
                                           [activated_sess, inhibited_sess],
                                           ['Activated cells', 'Inhibited cells'],
                                           ['crimson', 'navy']):
            if len(group) == 0:
                ax.set_title(f'{title} (n=0)')
                continue
            ctrl_traces = np.stack(group['mean_ctrl'])
            stim_traces = np.stack(group['mean_stim'])
            mean_ctrl = ctrl_traces.mean(axis=0)
            mean_stim = stim_traces.mean(axis=0)
            sem_ctrl = ctrl_traces.std(axis=0) / np.sqrt(ctrl_traces.shape[0])
            sem_stim = stim_traces.std(axis=0) / np.sqrt(stim_traces.shape[0])
            ax.plot(full_time, mean_ctrl, color='grey', label='ctrl')
            ax.fill_between(full_time, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color='grey', alpha=0.3)
            ax.plot(full_time, mean_stim, color=color, label='stim')
            ax.fill_between(full_time, mean_stim - sem_stim, mean_stim + sem_stim, color=color, alpha=0.3)
            ax.set(title=title, xlabel='time (s)', xlim=(-1, 4), xticks=[0, 2, 4])
            ax.axvline(0, ls='--', c='k', lw=0.75)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axs[0].set_ylabel('firing rate (Hz)')
        axs[1].legend(frameon=False, fontsize=7)
        fig.suptitle(f'{recname} ({rectype})', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        savepath = os.path.join(rectype_dir, f'{recname}_session_mean_curves.png')
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        

#%% compare ratios
act_ctrl_ratios = activated_term['ctrl_ratio'].astype(float)
act_stim_ratios = activated_term['stim_ratio'].astype(float)

inh_ctrl_ratios = inhibited_term['ctrl_ratio'].astype(float)
inh_stim_ratios = inhibited_term['stim_ratio'].astype(float)

# perform statistical tests
tval_act, pval_act = ttest_rel(act_ctrl_ratios, act_stim_ratios)
stat_act, pval_act_wil = wilcoxon(act_ctrl_ratios, act_stim_ratios)

tval_inh, pval_inh = ttest_rel(inh_ctrl_ratios, inh_stim_ratios)
stat_inh, pval_inh_wil = wilcoxon(inh_ctrl_ratios, inh_stim_ratios)

# plot
fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

# activated
axs[0].scatter(act_ctrl_ratios, act_stim_ratios, color='crimson', s=2, alpha=0.7)
axs[0].plot([0, 6], [0, 6], 'k--', lw=0.75)
axs[0].set_title('Activated')
axs[0].set_xlabel('ctrl pre/post')
axs[0].set_ylabel('stim pre/post')
axs[0].text(0.05, 5.8,
            f't = {tval_act:.2f}, p = {pval_act:.2g}\nW = {stat_act:.0f}, p = {pval_act_wil:.2g}',
            ha='left', va='top', fontsize=8)

# inhibited
axs[1].scatter(inh_ctrl_ratios, inh_stim_ratios, color='navy', s=2, alpha=0.7)
axs[1].plot([0, 6], [0, 6], 'k--', lw=0.75)
axs[1].set_title('Inhibited')
axs[1].set_xlabel('ctrl pre/post')
axs[1].text(0.05, 5.8,
            f't = {tval_inh:.2f}, p = {pval_inh:.2g}\nW = {stat_inh:.0f}, p = {pval_inh_wil:.2g}',
            ha='left', va='top', fontsize=8)

# aesthetics
for ax in axs:
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.tight_layout()
plt.show()