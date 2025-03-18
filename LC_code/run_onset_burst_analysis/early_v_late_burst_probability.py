# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 17:01:51 2025

compare burst probability between early- v late-lick trials 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import mat73
import sys 
import scipy.io as sio
import pandas as pd 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting 
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC


#%% load behaviour dataframe 
beh_df = pd.read_pickle(r'Z:/Dinghao/code_dinghao/behaviour/all_LC_sessions.pkl')
df = pd.read_pickle(r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% parameters
time_window = 1  # seconds before/after run-onset


#%% functions 
def compute_burst_probability(burst_times, run_onsets, window=time_window):
    """
    computes burst probability as the fraction of run-onset trials 
    that contain at least one burst in the specified time window.

    parameters:
    - burst_times: numpy array of burst timestamps in seconds
    - run_onsets: list of run onset times in seconds
    - window: time window (default: ±1s around run-onset)

    returns:
    - burst_prob: probability of burst occurrence per trial
    """
    if len(burst_times) == 0:
        return 0  # no bursts at all

    burst_present = np.array([
        np.any((burst_times >= onset - window) & (burst_times <= onset + window))
        for onset in run_onsets
    ])

    return np.mean(burst_present)  # fraction of trials with bursts


#%% main
burst_prob_early = []
burst_prob_late = []

for path in paths:
    recname = path[-17:]
    print(f'processing {recname}')

    # load burst and spike data
    sess_folder = rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}'
    burst_dict = np.load(
        rf'{sess_folder}\{recname}_all_bursts.npy', 
        allow_pickle=True
        ).item()
    spike_dict = np.load(
        rf'{sess_folder}\{recname}_all_spikes.npy', 
        allow_pickle=True
        ).item()

    # get aligned run-onset timepoints 
    run_onsets_aligned = mat73.loadmat(
        rf'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}/{recname[:14]}/{recname}'
        rf'/{recname}_BehavElectrDataLFP.mat'
    )['Laps']['startLfpInd'][1:] / 1250  # convert to seconds 
    
    # get lick times
    curr_beh_df = beh_df.loc[recname]
    run_onsets = curr_beh_df['run_onsets'][1:]

    # get first licks
    licks = [
        [(l - run_onset) / 1000 for l in trial]  # convert from ms to s
        if len(trial) != 0 else np.nan
        for trial, run_onset in zip(curr_beh_df['lick_times'][1:], run_onsets)
    ]
    first_licks = np.asarray([
        next((l for l in trial if l > 1), np.nan)  # >1s to prevent carry-over licks
        if isinstance(trial, list) else np.nan
        for trial in licks
    ])

    # get valid early & late trials
    behPar = sio.loadmat(
        rf'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}/{recname[:14]}/{recname}'
        rf'/{recname}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
    )
    bad_beh_ind = np.where(behPar['behPar'][0]['indTrBadBeh'][0] == 1)[1] - 1

    # get early and late lick trials (that are not stim. trials)
    stim_trials = np.where(
        np.asarray([trial[15] for trial in curr_beh_df['trial_statements']]) != '0'
    )[0]
    valid_trials = [
        i for i in range(len(first_licks))
        if i not in stim_trials and not np.isnan(first_licks[i])
    ]
    if len(valid_trials) < 50:
        continue

    sorted_trials = sorted(valid_trials, key=lambda i: first_licks[i])
    
    NUM_SAMP_TRIALS = int(len(sorted_trials)/10)  # sample 20% of early and late 1st-lick trials
    early_trial_onsets = [run_onsets_aligned[i] for i in sorted_trials[:NUM_SAMP_TRIALS]]
    late_trial_onsets = [run_onsets_aligned[i] for i in sorted_trials[-NUM_SAMP_TRIALS:]]

    # process each neuron
    for cluname in burst_dict.keys():
        identity = df.loc[cluname]['identity']
        if identity not in ['tagged', 'putative']:
            continue

        bursts = burst_dict[cluname]  # list of (start_idx, end_idx) tuples
        spike_times = np.array(spike_dict[cluname]) / 20000  # convert to seconds

        # convert burst indices to times
        burst_times = []
        for start_idx, end_idx in bursts:
            burst_times.append(spike_times[start_idx:end_idx + 1])
        burst_times = np.concatenate(burst_times) if burst_times else np.array([])

        # compute for early and late trials
        prob_early = compute_burst_probability(burst_times, early_trial_onsets)
        prob_late = compute_burst_probability(burst_times, late_trial_onsets)

        # store per-cell values
        burst_prob_early.append(prob_early)
        burst_prob_late.append(prob_late)


#%% plotting 
from plotting_functions import plot_violin_with_scatter

# define colours
early_colour = (.804, .267, .267)  # early trials
late_colour = (.545, 0, 0)  # late trials

plot_violin_with_scatter(burst_prob_early, burst_prob_late, 
                         early_colour, late_colour,
                         xticklabels=('early\n$1^{st}$-lick', 'late\n$1^{st}$-lick'),
                         ylabel='burst probability',
                         showmeans=True,
                         showmedians=False,
                         showscatter=False,
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\burst_probability')


#%% boxplot 
import matplotlib.pyplot as plt
import scipy.stats as stats

burst_prob_early = np.array(burst_prob_early)
burst_prob_late = np.array(burst_prob_late)

pval = stats.wilcoxon(burst_prob_early, burst_prob_late, nan_policy='omit')[1]

fig, ax = plt.subplots(figsize=(2.2, 3))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)

ax.set(xticks=[1, 2], xticklabels=['Early', 'Late'],
       ylabel='burst probability', yticks=[0, 0.3, 0.6],
       title=f'wilc_p = {round(pval, 4)}')

bp = ax.boxplot([burst_prob_early, burst_prob_late],
                positions=[1, 2], widths=0.3,
                patch_artist=True,
                notch=True)

for patch, color in zip(bp['boxes'], [early_colour, late_colour]):
    patch.set_facecolor(color)

for median in bp['medians']:
    median.set(color='k', linewidth=1.5)

bp['fliers'][0].set(marker='v', color='#e7298a', markersize=5, alpha=0.5)
bp['fliers'][1].set(marker='o', color='#e7298a', markersize=5, alpha=0.5)

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\burst_probability_box{ext}',
        dpi=300,
        bbox_inches='tight'
        )