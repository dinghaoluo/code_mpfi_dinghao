# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:03:16 2025

plot single session speed and lick profiles 

@author: Dinghao Luo 
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import sem 
import sys 
import os 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, replace_outlier
mpl_formatting()


#%% main 
exp_name_list = ['HPCLC', 'HPCLCterm', 'LC', 'HPCGRABNE', 'LCHPCGCaMP']

for exp_name in exp_name_list:
    save_path_stem = rf'Z:\Dinghao\code_dinghao\behaviour\session_profiles\{exp_name}'
    os.makedirs(save_path_stem, exist_ok=True)
    
    df = pd.read_pickle(
        rf'Z:/Dinghao/code_dinghao/behaviour/all_{exp_name}_sessions.pkl'
        )
    
    for recname, sess in df.iterrows():        
        speed_times = sess['speed_times_aligned']
        speeds = [replace_outlier(np.array([s[1] for s in trial])) 
                  for i, trial in enumerate(speed_times)]
        max_length_speeds = np.max([len(trial) for trial in speeds])
        speeds_arr = np.zeros((len(speeds), max_length_speeds))
        for i in range(len(speeds)):
            speeds_arr[i, :len(speeds[i])] = speeds[i]
            
        speed_distances = np.array(
            [replace_outlier(np.array(trial))
            for i, trial in enumerate(sess['speed_distances'])]
            )
        
        lick_maps = np.array(
            [np.array(trial[30:])
            for i, trial in enumerate(sess['lick_maps'])]
            )
            
        mean_speeds = np.nanmean(speeds_arr, axis=0)[:5*50]
        sem_speeds = sem(speeds_arr, axis=0, nan_policy='omit')[:5*50]
        speed_time_axis = np.arange(5 * 50) / 50  # 50 Hz
        
        mean_speeds_distances = np.nanmean(speed_distances, axis=0)
        sem_speeds_distances = sem(speed_distances, axis=0, nan_policy='omit')
        speed_distance_axis = np.arange(220)
        
        mean_lick_maps = np.nanmean(lick_maps, axis=0)
        sem_lick_maps = sem(lick_maps, axis=0, nan_policy='omit')
        lick_distance_axis = np.arange(30, 220)
        
        
        ## plotting 
        fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4))
        
        axs[0].plot(speed_time_axis, mean_speeds, 
                    c='navy', 
                    zorder=10)
        axs[0].fill_between(speed_time_axis, mean_speeds+sem_speeds, 
                                             mean_speeds-sem_speeds,
                            color='navy', alpha=.2,
                            zorder=10)
        axs[0].plot(speed_time_axis, 
                    speeds_arr[:, :5 * 50].T,
                    c='grey', lw=.5, alpha=.05,
                    zorder=1)
        axs[0].set(xlabel='time from run-onset (s)', 
                   xticks=[0, 2, 4],
                   ylabel='speed (cm/s)',
                   ylim=(0, np.max(speeds_arr[:, :5*50])+1))
        
        axs[1].plot(speed_distance_axis, mean_speeds_distances, 
                    c='navy',
                    zorder=10)
        axs[1].fill_between(speed_distance_axis, mean_speeds_distances+sem_speeds_distances, 
                                                 mean_speeds_distances-sem_speeds_distances,
                            color='navy', alpha=.2,
                            zorder=10)
        axs[1].plot(speed_distance_axis, 
                    speed_distances.T,
                    c='grey', lw=.5, alpha=.05,
                    zorder=1)
        axs[1].set(xlabel='distance (cm)', 
                   xticks=[0, 90, 180],
                   ylabel='speed (cm/s)',
                   ylim=(0, np.max(speed_distances[:, :5*50])+1))
        
        axs[2].plot(lick_distance_axis, mean_lick_maps, 
                    c='orchid')
        axs[2].fill_between(lick_distance_axis, mean_lick_maps+sem_lick_maps, 
                                                mean_lick_maps-sem_lick_maps,
                            color='orchid', alpha=.2)
        axs[2].axvspan(179.5, 220, 
                       edgecolor='none', facecolor='darkgreen', alpha=.15, 
                       zorder=10)
        ymax = max(mean_lick_maps+sem_lick_maps)*1.01
        if np.isnan(ymax): ymax=0  # for some reason one recording has nan licks...
        axs[2].set(xlabel='distance (cm)', 
                   xticks=[90, 180],
                   ylabel='lick rate (Hz)',
                   ylim=(0, ymax))
        for i in range(3):
            for s in ['top', 'right']:
                axs[i].spines[s].set_visible(False)
        
        fig.suptitle(recname)
        
        fig.tight_layout()
        plt.show()
        
        for ext in ['.png', '.pdf']:
            fig.savefig(rf'{save_path_stem}\{recname}{ext}',
                        dpi=300,
                        bbox_inches='tight')
            
        plt.close(fig)