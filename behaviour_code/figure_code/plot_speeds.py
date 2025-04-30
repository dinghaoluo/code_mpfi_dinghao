# -*- coding: utf-8 -*-
"""
Created on Mon 28 Apr 16:37:41 2025

plot the speed profiles 

@author: Dinghao Luo
"""


#%% imports 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys 
import numpy as np 
from scipy.stats import sem

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% load data 
exp_name = 'LC'  # HPCLC, HPCLCterm, LC, HPCGRABNE, LCHPCGCaMP
df = pd.read_pickle(
    rf'Z:/Dinghao/code_dinghao/behaviour/all_{exp_name}_sessions.pkl'
    )


#%% load behav data 
run_onsets = df['run_onsets']
speed_times = df['speed_times_aligned']
speed_distances = df['speed_distances']

# stack all speed profiles (ignore time; assume uniform sampling)
trial_run_onsets = [
    trial
    for session in run_onsets
    for trial in session
    ]
trial_speeds = [
    np.array([s for t, s in trial])
    for session in speed_times
    for trial in session
    ]
trial_speeds_distances = [
    np.array([s for s in trial])
    for session in speed_distances
    for trial in session
    ]

# filtering 
trial_speeds = [s for i, s in enumerate(trial_speeds) 
                if trial_run_onsets[i]!=-1]
trial_speeds_distances = [s for i, s in enumerate(trial_speeds_distances)
                          if trial_run_onsets[i]!=-1]

# determine max trial length
max_len = max(len(speeds) for speeds in trial_speeds)

# initialise with nan
all_speeds = np.full((len(trial_speeds), max_len), np.nan)

# fill in speed values
for i, speeds in enumerate(trial_speeds):
    all_speeds[i, :len(speeds)] = speeds

# compute mean ignoring nans
mean_speed = np.nanmean(all_speeds, axis=0)
sem_speed = sem(all_speeds, axis=0, nan_policy='omit')

mean_speed_distances = np.nanmean(trial_speeds_distances, axis=0)
sem_speed_distances = sem(trial_speeds_distances, axis=0, nan_policy='omit')

# time axis based on sampling rate
sampling_rate = 50  # Hz
dt = 1 / sampling_rate
time_axis = np.arange(max_len) * dt

distance_axis = np.arange(220)


#%% main 
fig, ax = plt.subplots(figsize=(1.9,1.7))

ax.plot(time_axis, mean_speed, color='navy', lw=1)
ax.fill_between(time_axis, mean_speed+sem_speed,
                           mean_speed-sem_speed,
                color='navy', alpha=.2)

ax.set(xlabel='time from run-onset (s)', xlim=(0, 4), xticks=[0,2,4],
       ylabel='speed (cm/s)', ylim=(0, 70))
for s in ['top','right']: 
    ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()
for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\behaviour\speed_profiles\speed_profile_time_{exp_name}{ext}',
        dpi=300,
        bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(1.9,1.7))

ax.plot(mean_speed_distances, color='navy', lw=1)
ax.fill_between(distance_axis, mean_speed_distances+sem_speed_distances,
                               mean_speed_distances-sem_speed_distances,
                color='navy', alpha=.2)

ax.set(xlim=(0, 200), xlabel='dist. from run-onset (cm)', 
       ylabel='speed (cm/s)')
for s in ['top','right']: 
    ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()
for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\behaviour\speed_profiles\speed_profile_distance_{exp_name}{ext}',
        dpi=300,
        bbox_inches='tight')
plt.close(fig)