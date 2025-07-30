# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 19:33:12 2025

plot single EXAMPLE session speed and lick profiles 
code extracted from full plot_single_session.py

@author: Dinghao Luo 
"""

#%% imports 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import sys

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\behaviour_code\utils')
import behaviour_functions as bf

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, replace_outlier, smooth_convolve
mpl_formatting()


#%% txt path
# txt_path = r'Z:\Dinghao\MiceExp\ANMD029\AD29-20220616-01T.txt'
txt_path = r'Z:\Dinghao\MiceExp\ANMD029r\A029r-20220627\A029r-20220627-02\A29r-20220627-02T.txt'


#%% main
recname = txt_path.split('\\')[-1].split('T')[0]

beh = bf.process_behavioural_data(txt_path)

# === Speed (time) ===
speed_times_aligned = beh['speed_times_aligned']
speed_aligned = [replace_outlier(np.array([s[1] for s in trial])) 
                 for trial in speed_times_aligned if trial]
speed_arr = np.zeros((len(speed_aligned), max(len(t) for t in speed_aligned)))
for i, s in enumerate(speed_aligned):
    speed_arr[i, :len(s)] = s
mean_speed_times = np.nanmean(speed_arr, axis=0)[:5000]
sem_speed_times = sem(speed_arr, axis=0, nan_policy='omit')[:5000]
speed_time_axis = np.arange(5000) / 1000  # seconds

# === Speed (space) ===
speed_distances = np.array([replace_outlier(np.array(trial)) 
                            for trial in beh['speed_distances_aligned'] if len(trial)>0])
mean_speeds_distances = np.nanmean(speed_distances, axis=0)
sem_speeds_distances = sem(speed_distances, axis=0, nan_policy='omit')
speed_distance_axis = np.arange(2200) / 10  # cm

# === Licks (space) ===
lick_maps = np.array([smooth_convolve(np.array(trial), sigma=10) * 10
                      for trial in beh['lick_maps'] if len(trial)>0])
mean_lick_maps = np.nanmean(lick_maps, axis=0)
sem_lick_maps = sem(lick_maps, axis=0, nan_policy='omit')
lick_distance_axis = np.arange(2200) / 10

# === Licks (time) ===
lick_times = beh['lick_times_aligned']
lick_times_map = np.zeros((len(lick_times), 5000))
for trial, licks in enumerate(lick_times):
    if isinstance(licks, list):
        for lick in licks:
            if lick < 5000:
                lick = int(lick)
                lick_times_map[trial, lick] += 1000  # Hz
        lick_times_map[trial, :] = smooth_convolve(lick_times_map[trial, :], sigma=10)
mean_lick_times_maps = np.nanmean(lick_times_map, axis=0)
sem_lick_times_maps = sem(lick_times_map, axis=0, nan_policy='omit')
lick_times_axis = np.arange(5000) / 1000  # seconds

# === Plot: overlay (spatial) ===
fig, ax = plt.subplots(figsize=(3, 2.3))
ax.plot(speed_distance_axis, mean_speeds_distances, c='navy')
ax.fill_between(speed_distance_axis, mean_speeds_distances + sem_speeds_distances,
                                     mean_speeds_distances - sem_speeds_distances,
                color='navy', alpha=.2)
axt = ax.twinx()
axt.plot(lick_distance_axis, mean_lick_maps*10, c='orchid')
axt.fill_between(lick_distance_axis, mean_lick_maps*10 + sem_lick_maps*10,
                                     mean_lick_maps*10 - sem_lick_maps*10,
                 color='orchid', alpha=.2)
ax.axvspan(179.5, 220, facecolor='darkgreen', alpha=.15)
ax.set(xlabel='distance (cm)', xlim=(0, 220),
       ylabel='speed (cm/s)', ylim=(0, 55))
axt.set(ylabel='licks', ylim=(0, 6.5))
plt.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\behaviour\example_figures\{recname}_spatial{ext}',
                dpi=300,
                bbox_inches='tight')


# === Plot: overlay (temporal) ===
fig, ax = plt.subplots(figsize=(3, 2.3))
ax.plot(speed_time_axis, mean_speed_times, c='navy')
ax.fill_between(speed_time_axis, mean_speed_times + sem_speed_times,
                               mean_speed_times - sem_speed_times,
                color='navy', alpha=.2)
axt = ax.twinx()
axt.plot(lick_times_axis, mean_lick_times_maps, c='orchid')
axt.fill_between(lick_times_axis, mean_lick_times_maps + sem_lick_times_maps,
                                    mean_lick_times_maps - sem_lick_times_maps,
                 color='orchid', alpha=.2)
ax.set(xlabel='time from run-onset (s)', xlim=(0, 5), ylim=(0,50), ylabel='speed (cm/s)')
axt.set(ylabel='lick rate (Hz)', ylim=(0, 13))
plt.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\behaviour\example_figures\{recname}_temporal{ext}',
                dpi=300,
                bbox_inches='tight')