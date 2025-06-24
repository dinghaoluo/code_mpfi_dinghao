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

remove_recs = [
    'A060r-20230602-01',
    'A062r-20230626-01',
    
    'A063r-20230708-01',
    'A063r-20230708-02',
    
    'A069r-20230905-01',
    'A069r-20230905-02',
    
    'A070r-20231109-01',
    'A070r-20231110-01',
    'A070r-20231115-01',
    'A070r-20231116-01',
    'A070r-20231117-01',
    
    'A078r-20240124-01',
    'A078r-20240125-01',
    'A078r-20240129-01',
    'A078r-20240130-01',
    'A078r-20240131-01',
    'A078r-20240201-01',
    'A078r-20240202-01',
    
    'A093i-20240620-01',
    'A093i-20240621-01',
    'A093i-20240625-01',
    'A093i-20240626-01',
    'A093i-20240708-01',
    'A093i-20240708-02',
    
    'A094i-20240701-01',
    'A094i-20240705-01',
    'A094i-20240705-02',
    'A094i-20240709-01',
    'A094i-20240711-01',
    'A094i-20240712-01',
    'A094i-20240717-01',
    'A094i-20240718-01',
    'A094i-20240718-02',
    'A094i-20240719-01',
    'A094i-20240719-02',
    'A094i-20240807-01',
    
    'A097i-20240826-01',
    'A097i-20240826-02',
    'A097i-20240827-01',
    'A097i-20240827-02',
    'A097i-20240829-01',
    'A097i-20240829-02',
    
    'A098i-20240923-01',
    'A098i-20240924-01',
    'A098i-20240926-01',
    'A098i-20240927-01',
    'A098i-20240927-02',
    'A098i-20241002-01',
    'A098i-20241004-01',
    'A098i-20241007-01',
    'A098i-20241023-01',
    
    'A101i-20241029-02',
    'A101i-20241031-01',
    'A101i-20241031-02',
    'A101i-20241101-01',
    'A101i-20241105-02',
    'A101i-20241106-01',
    'A101i-20241107-01',
    'A101i-20241107-02',
    'A101i-20241107-03',
    
    'A106i-20250128-01',
    'A106i-20250128-02'
    ]


#%% main 
all_speed_time_curves = []
all_lick_time_curves = []

for exp_name in ['HPCLC', 
                 'HPCLCterm', 
                 'LC', 
                 'HPCGRABNE', 
                 'LCHPCGCaMP',
                 'HPCdLightLCOpto']:
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
    elif exp_name == 'HPCdLightLCOpto':
        paths = rec_list.pathdLightLCOpto
        
    os.makedirs(
        rf'Z:\Dinghao\code_dinghao\behaviour\session_profiles\{exp_name}', 
        exist_ok=True
        )
    
    for path in paths:
        recname = path[-17:]
        
        if recname in remove_recs:
            print(f'\nskipping {recname}')
            continue 
        
        print(f'\n{recname}')
        
        with open(os.path.join(datapath_stem, f'{recname}.pkl'), 'rb') as f:
            try:
                beh = pickle.load(f)
            except EOFError:
                print(f'[ERROR] Could not load {recname}; file may be corrupted')
                continue
        if not beh['run_onsets']:
            print(f'{recname} is an immobile session; skipped')
            continue 
        
        output_path = os.path.join(
            r'Z:\Dinghao\code_dinghao\behaviour\session_profiles', 
            exp_name, recname
            )
        
        # speed (temporal)
        speed_times_aligned = beh['speed_times_aligned']
        speed_aligned = [replace_outlier(np.array([s[1] for s in trial])) 
                         for i, trial in enumerate(speed_times_aligned)
                         if trial]
        max_length_speeds = np.max([len(trial) for trial in speed_times_aligned])
        speed_arr = np.zeros((len(speed_aligned), max_length_speeds))
        for i in range(len(speed_aligned)):
            speed_arr[i, :len(speed_aligned[i])] = speed_aligned[i]
        mean_speed_times = np.nanmean(speed_arr, axis=0)
        sem_speed_times = sem(speed_arr, axis=0)

        # speed (spatial)
        speed_distances = np.array(
            [replace_outlier(np.array(trial))
            for i, trial in enumerate(beh['speed_distances_aligned'])
            if len(trial)>0]
            )
        
        # licks (spatial)
        lick_maps = np.array(
            [smooth_convolve(np.array(trial), sigma=10) * 10  # convert from mm to cm 
            for i, trial in enumerate(beh['lick_maps'])
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
        
        # licks (temporal)
        lick_times = beh['lick_times_aligned']
        tot_trials = sum([1 for trial in lick_times if isinstance(trial, list)])
        
        lick_times_map = np.zeros((tot_trials, 5000))
        
        for trial in range(tot_trials):
            for lick in lick_times[trial]:
                if lick < 5000:
                    lick = int(lick)
                    lick_times_map[trial, lick] += 1000  # 1000 due to 1000 Hz sampling rate 
                lick_times_map[trial, :] = smooth_convolve(lick_times_map[trial, :],
                                                           sigma=10)
                
        mean_lick_times_maps = np.nanmean(lick_times_map, axis=0)
        sem_lick_times_maps = sem(lick_times_map, axis=0)
        lick_times_axis = np.arange(5000) / 1000
        
        # store for global average (clip to 5000 samples = 5 s)
        all_speed_time_curves.append(mean_speed_times[:5000])
        all_lick_time_curves.append(mean_lick_times_maps[:5000])
        
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
            fig.savefig(rf'{output_path}_dist{ext}',
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
        
        
        ## overlay plots (temp)
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        
        ax.plot(speed_time_axis, mean_speed_times[:5000], 
                c='navy',
                zorder=10)
        ax.fill_between(speed_time_axis, mean_speed_times[:5000]+sem_speed_times[:5000], 
                                         mean_speed_times[:5000]-sem_speed_times[:5000], 
                        color='navy', alpha=.2, edgecolor='none')
        
        axt = ax.twinx()
        axt.plot(lick_times_axis, mean_lick_times_maps, 
                  c='orchid')
        axt.fill_between(lick_times_axis, mean_lick_times_maps+sem_lick_times_maps, 
                                          mean_lick_times_maps-sem_lick_times_maps,
                         color='orchid', alpha=.2, edgecolor='none')
        
        ax.spines['top'].set_visible(False)
        axt.spines['top'].set_visible(False)
        
        ax.set(xlabel='time from run-onset (5)',
               xlim=(0, 5),
               ylabel='speed (cm/s)',
               ylim=(0, max(mean_speed_times)*1.1))
        axt.set(xlim=(0, 5),
                ylabel='lick rate (Hz)',
                ylim=(0, max(mean_lick_times_maps)*1.1))
        
        fig.tight_layout()
        plt.show()
        
        for ext in ['.png', '.pdf']:
            fig.savefig(rf'{output_path}_overlay_time{ext}',
                        dpi=300,
                        bbox_inches='tight')
            
        plt.close(fig)


#%% all 
all_speed_time_curves = np.array(all_speed_time_curves)
all_lick_time_curves = np.array(all_lick_time_curves)

mean_speed_group = np.nanmean(all_speed_time_curves, axis=0)
sem_speed_group = sem(all_speed_time_curves, axis=0, nan_policy='omit')
mean_lick_group = np.nanmean(all_lick_time_curves, axis=0)
sem_lick_group = sem(all_lick_time_curves, axis=0, nan_policy='omit')

fig, ax = plt.subplots(figsize=(2.8, 2.3))

ax.plot(speed_time_axis, mean_speed_group,
        c='navy', zorder=10)
ax.fill_between(speed_time_axis, mean_speed_group + sem_speed_group,
                                 mean_speed_group - sem_speed_group,
                color='navy', alpha=.2, edgecolor='none')

axt = ax.twinx()
axt.plot(lick_times_axis, mean_lick_group, c='orchid')
axt.fill_between(lick_times_axis, mean_lick_group + sem_lick_group,
                                     mean_lick_group - sem_lick_group,
                 color='orchid', alpha=.2, edgecolor='none')

ax.spines['top'].set_visible(False)
axt.spines['top'].set_visible(False)

ax.set(xlabel='time from run-onset (s)',
       xlim=(0, 5),
       ylabel='speed (cm/s)',
       ylim=(0, np.nanmax(mean_speed_group)*1.1))
axt.set(xlim=(0, 5),
        ylabel='lick rate (Hz)',
        ylim=(0, np.nanmax(mean_lick_group)*1.1))

fig.suptitle('all sessions')
fig.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\behaviour\session_profiles\mean_overlay_time{ext}',
        dpi=300, bbox_inches='tight'
    )
plt.close(fig)