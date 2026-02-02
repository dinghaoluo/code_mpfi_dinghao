# -*- coding: utf-8 -*-
"""
Created on Wed 13 Nov 16:53:32 2024

summarise pharmacological experiments with SCH23390

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, wilcoxon

from common import mpl_formatting, replace_outlier, smooth_convolve
mpl_formatting()

from plotting_functions import plot_violin_with_scatter

import behaviour_functions as bf
import rec_list


#%% paths and parameters
pathNEblocker = rec_list.pathAlphaBlocker
sessNEblocker = rec_list.sessAlphaBlocker

pharmacology_stem = Path('Z:/Dinghao/code_dinghao/pharmacology')

tot_distance = 2200  # mm
XAXIS = np.arange(tot_distance) / 10  # cm


#%% main containers
mean_speeds_baseline = []
mean_speeds_drug     = []

mean_licks_baseline  = []
mean_licks_drug      = []

reward_percentages_baseline = []
reward_percentages_drug     = []


#%% load data
for i, pathname in enumerate(pathNEblocker):
    sessname = pathname[-13:]
    print(sessname)

    sesslist = sessNEblocker[i]

    for j, sess in enumerate(sesslist):
        suffix = f'-0{sess}'
        txtpath = f'{pathname}{suffix}T.txt'

        file = bf.process_behavioural_data(txtpath)

        # reward percentage
        reward_times = file['reward_times'][1:-1]
        rewarded = [1 if not np.isnan(t) else 0 for t in reward_times]

        if j == 0:
            reward_percentages_baseline.append(np.mean(rewarded))
        elif j == 1:
            reward_percentages_drug.append(np.mean(rewarded))

        # speed
        speed_dist = np.array([
            replace_outlier(np.array(trial))
            for trial in file['speed_distances_aligned']
            if len(trial) > 0
        ])

        if j == 0:
            mean_speeds_baseline.append(np.mean(speed_dist, axis=0) * 1.8)
        elif j == 1:
            mean_speeds_drug.append(np.mean(speed_dist, axis=0) * 1.8)

        # licks
        lick_dist = np.array([
            smooth_convolve(np.array(trial), sigma=10) * 10
            for trial in file['lick_maps']
            if len(trial) > 0
        ])

        if j == 0:
            mean_licks_baseline.append(np.mean(lick_dist, axis=0))
        elif j == 1:
            mean_licks_drug.append(np.mean(lick_dist, axis=0))


#%% convert to arrays
mean_speeds_baseline = np.array(mean_speeds_baseline)
mean_speeds_drug     = np.array(mean_speeds_drug)

mean_licks_baseline  = np.array(mean_licks_baseline)
mean_licks_drug      = np.array(mean_licks_drug)

reward_percentages_baseline = np.array(reward_percentages_baseline)
reward_percentages_drug     = np.array(reward_percentages_drug)


#%% SPEED PLOT
fig, ax = plt.subplots(figsize=(2, 1.7))

ms_baseline = np.mean(mean_speeds_baseline, axis=0)
ms_drug     = np.mean(mean_speeds_drug, axis=0)

ss_baseline = sem(mean_speeds_baseline, axis=0)
ss_drug     = sem(mean_speeds_drug, axis=0)

speed_means_baseline = np.mean(mean_speeds_baseline, axis=1)
speed_means_drug     = np.mean(mean_speeds_drug, axis=1)

speed_mean_baseline = np.mean(speed_means_baseline)
speed_sem_baseline  = sem(speed_means_baseline)
speed_mean_drug     = np.mean(speed_means_drug)
speed_sem_drug      = sem(speed_means_drug)

speed_med_baseline = np.median(speed_means_baseline)
speed_p25_baseline = np.percentile(speed_means_baseline, 25)
speed_p75_baseline = np.percentile(speed_means_baseline, 75)

speed_med_drug = np.median(speed_means_drug)
speed_p25_drug = np.percentile(speed_means_drug, 25)
speed_p75_drug = np.percentile(speed_means_drug, 75)

_, p_speed = wilcoxon(speed_means_baseline, speed_means_drug)

print(
    f'Speed baseline: mean {speed_mean_baseline:.4f} ± {speed_sem_baseline:.4f}, '
    f'median {speed_med_baseline:.4f} [{speed_p25_baseline:.4f}, {speed_p75_baseline:.4f}]'
)
print(
    f'Speed drug    : mean {speed_mean_drug:.4f} ± {speed_sem_drug:.4f}, '
    f'median {speed_med_drug:.4f} [{speed_p25_drug:.4f}, {speed_p75_drug:.4f}]'
)
print(f'Speed Wilcoxon p = {p_speed:.4g}')

lp, = ax.plot(XAXIS, ms_baseline, color='grey')
ax.fill_between(XAXIS, ms_baseline + ss_baseline, ms_baseline - ss_baseline,
                color='grey', alpha=.15, edgecolor='none')

ld, = ax.plot(XAXIS, ms_drug, color='darkgreen')
ax.fill_between(XAXIS, ms_drug + ss_drug, ms_drug - ss_drug,
                color='darkgreen', alpha=.15, edgecolor='none')

ax.legend([lp, ld], ['Baseline', 'Prazosin'], frameon=False)
ax.set(xlim=(0, 180), ylim=(0, 75),
       xlabel='Distance (cm)', ylabel='Speed (cm/s)')

ax.text(
    0.02, 0.98,
    f'baseline: {speed_med_baseline:.2f} '
    f'[{speed_p25_baseline:.2f}, {speed_p75_baseline:.2f}]\n'
    f'drug: {speed_med_drug:.2f} '
    f'[{speed_p25_drug:.2f}, {speed_p75_drug:.2f}]\n'
    f'wilcoxon p={p_speed:.3g}',
    transform=ax.transAxes, va='top', ha='left', fontsize=7
)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

for ext in ['.png', '.pdf']:
    fig.savefig(
        pharmacology_stem / 'prazosin' / f'speed_profile{ext}',
        dpi=300, bbox_inches='tight'
    )


#%% LICK PLOT
fig, ax = plt.subplots(figsize=(2, 1.7))

ml_baseline = np.mean(mean_licks_baseline, axis=0) / 10
ml_drug     = np.mean(mean_licks_drug, axis=0) / 10

sl_baseline = sem(mean_licks_baseline, axis=0) / 10
sl_drug     = sem(mean_licks_drug, axis=0) / 10

lick_means_baseline = np.mean(mean_licks_baseline[:, 1200:1800], axis=1) / 10
lick_means_drug     = np.mean(mean_licks_drug[:, 1200:1800], axis=1) / 10

lick_mean_baseline = np.mean(lick_means_baseline)
lick_sem_baseline  = sem(lick_means_baseline)
lick_mean_drug     = np.mean(lick_means_drug)
lick_sem_drug      = sem(lick_means_drug)

lick_med_baseline = np.median(lick_means_baseline)
lick_p25_baseline = np.percentile(lick_means_baseline, 25)
lick_p75_baseline = np.percentile(lick_means_baseline, 75)

lick_med_drug = np.median(lick_means_drug)
lick_p25_drug = np.percentile(lick_means_drug, 25)
lick_p75_drug = np.percentile(lick_means_drug, 75)

_, p_lick = wilcoxon(lick_means_baseline, lick_means_drug)

print(
    f'Lick baseline: mean {lick_mean_baseline:.4f} ± {lick_sem_baseline:.4f}, '
    f'median {lick_med_baseline:.4f} [{lick_p25_baseline:.4f}, {lick_p75_baseline:.4f}]'
)
print(
    f'Lick drug    : mean {lick_mean_drug:.4f} ± {lick_sem_drug:.4f}, '
    f'median {lick_med_drug:.4f} [{lick_p25_drug:.4f}, {lick_p75_drug:.4f}]'
)
print(f'Lick Wilcoxon p = {p_lick:.4g}')

lp, = ax.plot(XAXIS, ml_baseline, color='grey')
ax.fill_between(XAXIS, ml_baseline + sl_baseline, ml_baseline - sl_baseline,
                color='grey', alpha=.15, edgecolor='none')

ld, = ax.plot(XAXIS, ml_drug, color='darkgreen')
ax.fill_between(XAXIS, ml_drug + sl_drug, ml_drug - sl_drug,
                color='darkgreen', alpha=.15, edgecolor='none')

ax.set(xlim=(30, 219), ylim=(0, 0.6),
       xlabel='Distance (cm)', ylabel='Lick density (count/cm)',
       yticks=[0, 0.3, 0.6])

ax.legend([lp, ld], ['Baseline', 'Prazosin'], frameon=False)

ax.text(
    0.02, 0.98,
    f'baseline: {lick_med_baseline:.3f} '
    f'[{lick_p25_baseline:.3f}, {lick_p75_baseline:.3f}]\n'
    f'drug: {lick_med_drug:.3f} '
    f'[{lick_p25_drug:.3f}, {lick_p75_drug:.3f}]\n'
    f'wilcoxon p={p_lick:.3g}',
    transform=ax.transAxes, va='top', ha='left', fontsize=7
)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

for ext in ['.png', '.pdf']:
    fig.savefig(
        pharmacology_stem / 'prazosin' / f'lick_profile{ext}',
        dpi=300, bbox_inches='tight'
    )


#%% REWARD PERCENTAGE
plot_violin_with_scatter(reward_percentages_baseline, reward_percentages_drug, 
                         'grey', 'darkgreen',
                         figsize=(1.8, 1.8),
                         ylim=(-0.1, 1.05),
                         print_statistics=True,
                         save=True,
                         savepath=pharmacology_stem / 'prazosin' / 'reward_percentage')


#%% lick comparison: 30–100 cm
lick_window = (XAXIS >= 30) & (XAXIS <= 100)

lick_window_baseline = np.mean(mean_licks_baseline[:, lick_window], axis=1) / 10
lick_window_drug     = np.mean(mean_licks_drug[:, lick_window], axis=1) / 10

lw_med_baseline = np.median(lick_window_baseline)
lw_p25_baseline = np.percentile(lick_window_baseline, 25)
lw_p75_baseline = np.percentile(lick_window_baseline, 75)

lw_med_drug = np.median(lick_window_drug)
lw_p25_drug = np.percentile(lick_window_drug, 25)
lw_p75_drug = np.percentile(lick_window_drug, 75)

_, p_lw = wilcoxon(lick_window_baseline, lick_window_drug)

print(
    f'licks 30–100 cm baseline: median {lw_med_baseline:.4g} '
    f'[{lw_p25_baseline:.4g}, {lw_p75_baseline:.4g}]'
)
print(
    f'licks 30–100 cm drug    : median {lw_med_drug:.4g} '
    f'[{lw_p25_drug:.4g}, {lw_p75_drug:.4g}]'
)
print(f'licks 30–100 cm Wilcoxon p = {p_lw:.4g}')