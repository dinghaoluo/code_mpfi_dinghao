# -*- coding: utf-8 -*-
"""
Created on Fri 18 Aug 17:41:33 2023
Modified on Fri 31 Oct 2025

analyse trial-to-trial correction of 1st-lick timing
(error defined as first-lick-to-reward interval)

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathLC + \
        rec_list.pathHPCLCopt + \
        rec_list.pathHPCLCtermopt + \
        rec_list.pathHPC_Raphi


#%% parameters 
SAMP_FREQ = 1250  # Hz


#%% main
sess_names, sess_eta, sess_r = [], [], []

for path in paths:
    recname = Path(path).name
    print(recname)

    # behaviour file
    beh_try_paths = [
        Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/HPCLC') / f'{recname}.pkl',
        Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/HPCLCterm') / f'{recname}.pkl',
        Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC') / f'{recname}.pkl',
        Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/HPCRaphi') / f'{recname}.pkl'
    ]
    for beh_path in beh_try_paths:
        if beh_path.exists():
            if beh_path.stat().st_size == 0:
                print(f'{recname}: empty file {beh_path.name}; skipped this path.')
                continue
            try:
                with open(beh_path, 'rb') as f:
                    beh = pickle.load(f)
                break
            except EOFError:
                print(f'{recname}: corrupted pickle file {beh_path.name}; skipped this path.')
                continue
    else:
        print(f'{recname}: no valid behaviour file found; skipped.')
        continue
    
    # process trial statements to get baseline trials 
    trial_statements = beh['trial_statements'][1:]
    opto_idx = [idx for idx, trial in enumerate(trial_statements) if trial[15] != '0']
    if opto_idx:
        end_baseline = opto_idx[0]
    else:
        end_baseline = len(trial_statements)
    
    # process speeds 
    speed_times = beh['speed_times_aligned'][1:end_baseline]
    trial_speeds = [np.nanmean([speed[1] for speed in trial])
                    for trial in speed_times]
    sess_mean_speed = np.nanmean(trial_speeds)
    sess_std_speed  = np.nanstd(trial_speeds)
    valid_speed_mask = (trial_speeds > sess_mean_speed - sess_std_speed) & \
                       (trial_speeds < sess_mean_speed + sess_std_speed)
    
    # process licks 
    lick_times   = beh['lick_times_aligned'][1:end_baseline]
    first_licks = np.array([
        trial[0]/SAMP_FREQ if isinstance(trial, (list, np.ndarray)) and len(trial) > 0 else np.nan
        for trial in lick_times
    ])
    if np.nanmean(first_licks) < 2:  # if animal licks super early all the time, skip
        print('mean lick < 2 s; skipped.')
        continue
    sess_mean_lick = np.nanmean(first_licks)
    sess_std_lick  = np.nanstd(first_licks)
    valid_lick_mask = (first_licks > sess_mean_lick - sess_std_lick) & \
                      (first_licks < sess_mean_lick + sess_std_lick)
    
    # process rewards 
    reward_times = np.array([trial/SAMP_FREQ if not np.isnan(trial) else np.nan
                             for trial in beh['reward_times_aligned'][1:end_baseline]])

    # filtering
    valid_trial_mask = valid_speed_mask & valid_lick_mask
    if np.sum(valid_trial_mask) < 10:
        print('insufficient valid trials; skipped.')
        continue

    ## -- REGRESSION STARTS HERE -- ##
    # session reference = typical lick time
    sess_mean = np.nanmean(first_licks)

    # build consecutive pairs, but only where BOTH trials are valid
    e_n_list = []
    e_np1_list = []
    for i in range(len(first_licks) - 1):
        f0 = first_licks[i]
        f1 = first_licks[i+1]
        if np.isfinite(f0) and np.isfinite(f1):
            # error = lick - session_mean
            e_n_list.append(f0 - sess_mean)
            e_np1_list.append(f1 - sess_mean)

    e_n   = np.array(e_n_list, dtype=float)
    e_np1 = np.array(e_np1_list, dtype=float)

    # final outlier clamp on *pair* level
    pair_mask = (
        np.isfinite(e_n) & np.isfinite(e_np1) &
        (e_n   > -10) & (e_n   < 10) &
        (e_np1 > -10) & (e_np1 < 10)
    )
    e_n, e_np1 = e_n[pair_mask], e_np1[pair_mask]

    if e_n.size < 3:
        print('insufficient finite pairs after filtering')
        continue

    # regression: e_{n+1} = α + (1–η)*e_n
    slope, intercept, r, p, _ = linregress(e_n, e_np1)
    eta = 1 - slope

    # store session summary
    sess_names.append(recname)
    sess_eta.append(eta)

    # plot
    fig, ax = plt.subplots(figsize=(2.8, 3))
    ax.scatter(e_n, e_np1, color='grey', s=6, alpha=0.6)
    xlo, xhi = np.min(e_n), np.max(e_n)
    ax.plot(
        [xlo, xhi],
        intercept + slope * np.array([xlo, xhi]),
        'k', lw=1.5
    )
    ax.set(
        xlabel='digression from mean (n) (s)',
        ylabel='digression from mean (n+1) (s)',
        title=f'{recname}\nupdate gain η={eta:.3f}, r={r:.3f}, p={p:.3g}'
    )
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()

    fig.savefig(
        f'Z:/Dinghao/code_dinghao/history_dependency/single_session/{recname}_digression_from_mean.png',
        dpi=300, bbox_inches='tight'
    )
    
    # trial-to-trial correction plot
    Δe = e_np1 - e_n
    
    # correlation between Δe and eₙ (should be negative if correcting)
    valid_mask = np.isfinite(e_n) & np.isfinite(Δe)
    if np.sum(valid_mask) > 2:
        r_corr = np.corrcoef(e_n[valid_mask], Δe[valid_mask])[0, 1]
    else:
        r_corr = np.nan

    # store session summary
    sess_r.append(r_corr)
    
    # bin by current error
    bins = np.linspace(np.nanmin(e_n), np.nanmax(e_n), 15)
    bin_idx = np.digitize(e_n, bins)
    bin_means = [np.nanmean(Δe[bin_idx == i]) for i in range(1, len(bins))]
    bin_sems  = [np.nanstd(Δe[bin_idx == i]) / np.sqrt(np.sum(bin_idx == i))
                 for i in range(1, len(bins))]
    
    # plot
    fig, ax = plt.subplots(figsize=(2.8, 3))
    ax.errorbar(bins[:-1], bin_means, yerr=bin_sems, fmt='-o',
                color='black', markersize=3, lw=1)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.axvline(0, color='k', ls='--', lw=0.5)
    ax.set(
        xlabel='error(n) (s)',
        ylabel='Δerror (error(n+1)−error(n))',
        title=f'{recname}\ntrial-to-trial correction'
    )
    plt.tight_layout()
    plt.show()
    
    fig.savefig(
        f'Z:/Dinghao/code_dinghao/history_dependency/single_session/{recname}_deltaE_vs_E.png',
        dpi=300, bbox_inches='tight'
    )