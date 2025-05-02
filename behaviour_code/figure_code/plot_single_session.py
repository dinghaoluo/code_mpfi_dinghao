# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:03:16 2025

plot single session speed and lick profiles 

@author: Dinghao Luo 
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
from scipy.stats import sem 
import sys 
import os 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, replace_outlier, smooth_convolve
mpl_formatting()


#%% path list 
sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list


#%% main 
for exp_name in ['HPCLC', 'HPCLCterm', 'LC', 'HPCGRABNE', 'LCHPCGCaMP']:
    datapath_stem = os.path.join(
        r'Z:\Dinghao\code_dinghao\behaviour\all_experiments', exp_name
        )
    if exp_name == 'HPCLC':
        paths = rec_list.pathHPCLCopt
    elif exp_name == 'HPCLCterm':
        paths = rec_list.pathHPCLCtermopt
    elif exp_name == 'LC':
        paths = rec_list.pathLC
    elif exp_name == 'HPCGRABNE':
        paths = rec_list.pathHPCGRABNE
    elif exp_name == 'LCHPCGCaMP':
        paths = rec_list.pathLCHPCGCaMP
        
    os.makedirs(
        rf'Z:\Dinghao\code_dinghao\behaviour\session_profiles\{exp_name}', 
        exist_ok=True
        )
    
    for path in paths:
        recname = path[-17:]
        print(f'\n{recname}')
        
        with open(os.path.join(datapath_stem, f'{recname}.pkl'), 'rb') as f:
            try:
                data = pickle.load(f)
            except EOFError:
                print(f'[ERROR] Could not load {recname} — file may be corrupted.')
                continue
        
        output_path = os.path.join(
            r'Z:\Dinghao\code_dinghao\behaviour\session_profiles', 
            exp_name, recname
            )
        
        # speed (temporal)
        speed_times_aligned = data['speed_times_aligned']
        speed_aligned = [replace_outlier(np.array([s[1] for s in trial])) 
                         for i, trial in enumerate(speed_times_aligned)
                         if trial]
        max_length_speeds = np.max([len(trial) for trial in speed_times_aligned])
        speed_arr = np.zeros((len(speed_aligned), max_length_speeds))
        for i in range(len(speed_aligned)):
            speed_arr[i, :len(speed_aligned[i])] = speed_aligned[i]
        
        # speed (spatial)
        speed_distances = np.array(
            [replace_outlier(np.array(trial))
            for i, trial in enumerate(data['speed_distances_aligned'])
            if len(trial)>0]
            )
        
        # licks (spatial)
        lick_maps = np.array(
            [smooth_convolve(np.array(trial), sigma=10) * 10  # convert from mm to cm 
            for i, trial in enumerate(data['lick_maps'])
            if len(trial)>0]
            )
            
        mean_speeds = np.nanmean(speed_arr, axis=0)[:5*1000]  # 5 s 
        sem_speeds = sem(speed_arr, axis=0, nan_policy='omit')[:5*1000]
        speed_time_axis = np.arange(5 * 1000) / 1000  # 50 Hz
        
        mean_speeds_distances = np.nanmean(speed_distances, axis=0)
        sem_speeds_distances = sem(speed_distances, axis=0, nan_policy='omit')
        speed_distance_axis = np.arange(2200) / 10 
        
        mean_lick_maps = np.nanmean(lick_maps, axis=0)
        sem_lick_maps = sem(lick_maps, axis=0, nan_policy='omit')
        lick_distance_axis = np.arange(2200) / 10
        
        
        ## plotting 
        fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4))
        
        axs[0].plot(speed_time_axis, mean_speeds, 
                    c='navy', 
                    zorder=10)
        axs[0].fill_between(speed_time_axis, mean_speeds+sem_speeds, 
                                             mean_speeds-sem_speeds,
                            color='navy', alpha=.2, edgecolor='none',
                            zorder=10)
        axs[0].plot(speed_time_axis, 
                    speed_arr[:, :5 * 1000].T,
                    c='grey', lw=.5, alpha=.05,
                    zorder=1)
        axs[0].set(xlabel='time from run-onset (s)', 
                   xticks=[0, 2, 4],
                   ylabel='speed (cm/s)',
                   ylim=(0, np.max(speed_arr[:, :5*1000])+1))
        
        axs[1].plot(speed_distance_axis, mean_speeds_distances, 
                    c='navy',
                    zorder=10)
        axs[1].fill_between(speed_distance_axis, mean_speeds_distances+sem_speeds_distances, 
                                                 mean_speeds_distances-sem_speeds_distances,
                            color='navy', alpha=.2, edgecolor='none',
                            zorder=10)
        axs[1].plot(speed_distance_axis, 
                    speed_distances.T,
                    c='grey', lw=.5, alpha=.05,
                    zorder=1)
        axs[1].set(xlabel='distance (cm)', 
                   xticks=[0, 90, 180],
                   ylabel='speed (cm/s)',
                   ylim=(0, np.max(speed_distances[:, :5*1000])+1))
        
        axs[2].plot(lick_distance_axis, mean_lick_maps, 
                    c='orchid', lw=1)
        axs[2].fill_between(lick_distance_axis, mean_lick_maps+sem_lick_maps, 
                                                mean_lick_maps-sem_lick_maps,
                            color='orchid', alpha=.2, edgecolor='none')
        axs[2].axvspan(179.5, 220, 
                       edgecolor='none', facecolor='darkgreen', alpha=.15, 
                       zorder=10)
        ymax = max(mean_lick_maps+sem_lick_maps)*1.01
        if np.isnan(ymax): ymax=0  # for some reason one recording has nan licks...
        axs[2].set(xlabel='distance (cm)', 
                   xticks=[90, 180],
                   ylabel='licks',
                   ylim=(0, ymax))
        for i in range(3):
            for s in ['top', 'right']:
                axs[i].spines[s].set_visible(False)
        
        fig.suptitle(recname)
        
        fig.tight_layout()
        plt.show()
        
        for ext in ['.png', '.pdf']:
            fig.savefig(rf'{output_path}{ext}',
                        dpi=300,
                        bbox_inches='tight')
            
        plt.close(fig)
        
        
        ## overlay plots 
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        
        ax.plot(speed_distance_axis, mean_speeds_distances, 
                    c='navy',
                    zorder=10)
        ax.fill_between(speed_distance_axis, mean_speeds_distances+sem_speeds_distances, 
                                                 mean_speeds_distances-sem_speeds_distances,
                            color='navy', alpha=.2, edgecolor='none')
        
        axt = ax.twinx()
        axt.plot(lick_distance_axis, mean_lick_maps, 
                  c='orchid')
        axt.fill_between(lick_distance_axis, mean_lick_maps+sem_lick_maps, 
                                              mean_lick_maps-sem_lick_maps,
                          color='orchid', alpha=.2, edgecolor='none')
        
        ax.axvspan(179.5, 220, 
                   edgecolor='none', facecolor='darkgreen', alpha=.15)
        
        ax.spines['top'].set_visible(False)
        axt.spines['top'].set_visible(False)
        
        ax.set(xlabel='distance (cm)',
               xlim=(0, 220),
               ylabel='speed (cm/s)',
               ylim=(0, max(mean_speeds_distances)*1.1))
        axt.set(xlim=(0, 220),
                ylabel='licks',
                ylim=(0, max(mean_lick_maps)*1.1))
        
        fig.tight_layout()
        plt.show()
        
        for ext in ['.png', '.pdf']:
            fig.savefig(rf'{output_path}_overlay_dist{ext}',
                        dpi=300,
                        bbox_inches='tight')
            
        plt.close(fig)