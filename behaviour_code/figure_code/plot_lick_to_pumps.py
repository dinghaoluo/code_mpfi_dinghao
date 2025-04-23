# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:46:46 2024
Modified on Fri  Sept 20 15:15:12 2024 to plot lick to pump distribution 
Modified on Tue 10 Dec 2024 to accommodate all recording lists

plot the lick-to-pump profiles (both time and distance)

@author: Dinghao Luo
"""


#%% imports 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys 
import numpy as np 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% load data 
exp_name = 'LC'  # HPCLC, HPCLCterm, LC, HPCGRABNE, HPCLCGCaMP
df = pd.read_pickle(r'Z:/Dinghao/code_dinghao/behaviour/all_{}_sessions.pkl'.format(
    exp_name))


#%% load behav data 
speed_times = df['speed_times']
speed_distance = df['speed_distance']

# stack all speed profiles (ignore time; assume uniform sampling)
trial_speeds = [
    np.array([s for t, s in trial])
    for session in speed_times
    for trial in session
    ]
trial_speeds_distance = [
    np.array([s for s in trial])
    for session in speed_distance
    for trial in session
    ]

# determine max trial length
max_len = max(len(speeds) for speeds in trial_speeds)

# initialise with nan
all_speeds = np.full((len(trial_speeds), max_len), np.nan)

# fill in speed values
for i, speeds in enumerate(trial_speeds):
    all_speeds[i, :len(speeds)] = speeds

# compute mean ignoring nans
mean_speed = np.nanmean(all_speeds, axis=0)
mean_speed[0] = 15

mean_speed_distance = np.nanmean(trial_speeds_distance, axis=0)

# time axis based on sampling rate
sampling_rate = 50  # Hz
dt = 1 / sampling_rate
time_axis = np.arange(max_len) * dt


#%% main 
fig, ax = plt.subplots(figsize=(1.9,1.7))

ax.plot(time_axis, mean_speed, color='k')

ax.set(xlim=(0, 4), xlabel='time from run-onset (s)', xticks=[0,2,4],
       ylabel='speed (cm/s)', yticks=[10, 30, 50])
for s in ['top','right']: ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()
for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\behaviour\speed_profiles\speed_profile_time_{exp_name}{ext}',
        dpi=300,
        bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(1.9,1.7))

ax.plot(mean_speed_distance, color='k')

ax.set(xlim=(0, 200), xlabel='dist. from run-onset (cm)', 
       ylabel='speed (cm/s)')
for s in ['top','right']: ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()
for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\behaviour\speed_profiles\speed_profile_distance_{exp_name}{ext}',
        dpi=300,
        bbox_inches='tight')
plt.close(fig)