# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:03:16 2025

plot single session speed and lick profiles 

@author: Dinghao Luo 
"""

#%% imports 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
from scipy.stats import sem, ttest_1samp, wilcoxon

from common import mpl_formatting, replace_outlier, smooth_convolve
mpl_formatting()

import rec_list


#%% paths and parameters 
all_beh_stem  = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments')
all_sess_stem = Path('Z:/Dinghao/code_dinghao/behaviour/session_profiles')

N_SHUF  = 500
RAMP_T0 = 2.0
RAMP_T1 = 4.0
i0 = int(RAMP_T0 * 1000)
i1 = int(RAMP_T1 * 1000)


#%% bad behaviour exclusion 
remove_recnames = [
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

# predictive licking containers 
all_real_slopes = []
all_shuf_means  = []
all_shuf_stds   = []

for exp_name in ['HPCLC', 
                 'HPCLCterm', 
                 'LC', 
                 'HPCGRABNE', 
                 'LCHPCGCaMP',
                 'HPCdLightLCOpto']:
    beh_stem = all_beh_stem / exp_name
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
        
    exp_stem = all_sess_stem / exp_name
    
    for path in paths:
        recname = Path(path).name
        
        if recname in remove_recnames:
            print(f'\n{recname}\nSkipped')
            continue 
        
        print(f'\n{recname}')
        
        beh_path = beh_stem / f'{recname}.pkl'
        with open(beh_path, 'rb') as f:
            try:
                beh = pickle.load(f)
            except EOFError:
                print(f'[ERROR] Could not load {recname}; file may be corrupted')
                continue
        if not beh['run_onsets']:
            print(f'{recname} is an immobile session; skipped')
            continue 
                
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
        
        # -------------------------------
        # predictive ramp quantification 
        # -------------------------------
        x = lick_times_axis[i0:i1]
        y = mean_lick_times_maps[i0:i1]
        
        if np.all(np.isnan(y)):
            real_slope = np.nan
        else:
            real_slope = np.polyfit(x, y, 1)[0]
            
        shuf_slopes = []
        for _ in range(N_SHUF):
        
            # circularly shift EACH TRIAL'S temporal lick trace
            shuf = np.array([
                np.roll(tr, np.random.randint(tr.size))
                for tr in lick_times_map
            ])
        
            # average across trials
            shuf_mean = np.nanmean(shuf, axis=0)
            y_shuf = shuf_mean[i0:i1]
        
            if np.all(np.isnan(y_shuf)):
                slope_shuf = np.nan
            else:
                slope_shuf = np.polyfit(x, y_shuf, 1)[0]
        
            shuf_slopes.append(slope_shuf)
        
        shuf_slopes = np.array(shuf_slopes)
        shuf_slopes = shuf_slopes[~np.isnan(shuf_slopes)]
        
        if len(shuf_slopes) > 2:
            shuf_mean = np.mean(shuf_slopes)
            shuf_std  = np.std(shuf_slopes)
        else:
            shuf_mean = np.nan
            shuf_std  = np.nan
        
        # store
        all_real_slopes.append(real_slope)
        all_shuf_means.append(shuf_mean)
        all_shuf_stds.append(shuf_std)
        # ------------------------------------
        # predictive ramp quantification ends 
        # ------------------------------------
        
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
        axs[2].set(xlabel='Distance (cm)', 
                   xticks=[90, 180],
                   ylabel='Licks',
                   ylim=(0, ymax))
        for i in range(3):
            for s in ['top', 'right']:
                axs[i].spines[s].set_visible(False)
        
        fig.suptitle(recname)
        
        fig.tight_layout()
        plt.show()
        
        for ext in ['.png', '.pdf']:
            fig.savefig(exp_stem / f'{recname}_dist{ext}',
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
        
        ax.set(xlabel='Distance (cm)',
               xlim=(0, 220),
               ylabel='Speed (cm/s)',
               ylim=(0, max(mean_speeds_distances)*1.1))
        axt.set(xlim=(0, 220),
                ylabel='Licks',
                ylim=(0, max(mean_lick_maps)*1.1))
        
        fig.tight_layout()
        plt.show()
        
        for ext in ['.png', '.pdf']:
            fig.savefig(exp_stem / f'{recname}_overlay_dist{ext}',
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
        
        ax.set(xlabel='Time from run-onset (s)',
               xlim=(0, 5),
               ylabel='Speed (cm/s)',
               ylim=(0, max(mean_speed_times)*1.1))
        axt.set(xlim=(0, 5),
                ylabel='Lick rate (Hz)',
                ylim=(0, max(mean_lick_times_maps)*1.1))
        
        fig.tight_layout()
        plt.show()
        
        for ext in ['.png', '.pdf']:
            fig.savefig(exp_stem / f'{recname}_overlay_time{ext}',
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

ax.set(xlabel='Time from run-onset (s)',
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
        all_sess_stem  / f'mean_overlay_time{ext}',
        dpi=300, bbox_inches='tight'
    )
plt.close(fig)


#%% predictive temporal lick ramp group statistics
tval, p_t = ttest_1samp(all_real_slopes, 0)
wstat, p_w = wilcoxon(all_real_slopes)

fig, ax = plt.subplots(figsize=(1.6, 2.2))

# violin
parts = ax.violinplot(all_real_slopes, positions=[1],
                      showmeans=False, showmedians=True, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('orchid')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)
parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

# scatter individual r's
ax.scatter(np.ones(len(all_real_slopes)), all_real_slopes,
           color='orchid', ec='none', s=10, alpha=0.5, zorder=3)

# shuffle + CI
shuf_mean = np.nanmean(all_shuf_means)
shuf_std  = np.nanmean(all_shuf_stds)

lower_95 = shuf_mean - 1.96 * shuf_std
upper_95 = shuf_mean + 1.96 * shuf_std

ax.axhline(shuf_mean, color='gray', lw=1, ls='--')

ax.fill_between(
    [0, 2],
    lower_95, upper_95,
    color='gray', alpha=0.2, edgecolor='none', zorder=0
)

mean_r, sem_r = np.nanmean(all_real_slopes), sem(all_real_slopes)
ymax = np.max(all_real_slopes)
ax.text(1, ymax + 0.05*(ymax - np.min(all_real_slopes)),
        f'{mean_r:.2f} ± {sem_r:.2f}',
        ha='center', va='bottom', fontsize=7, color='orchid')

ax.text(1, np.min(all_real_slopes) - 0.10*(ymax - np.min(all_real_slopes)),
        f't(1-samp)={tval:.2f}, p={p_t:.2e}\n'
        f'Wilcoxon={wstat:.2f}, p={p_w:.2e}',
        ha='center', va='top', fontsize=6.5, color='black')

# formatting
ax.set(xlim=(0.5, 1.5), xticks=[],
       ylabel='Predictive lick slope (Hz/s)',
       title='Predictive lick slope')
ax.spines[['top', 'right', 'bottom']].set_visible(False)

plt.tight_layout()
plt.show()

for ext in ['.pdf', '.png']:
    fig.savefig(
        all_sess_stem / f'predictive_lick_slope_2to4_violinplot{ext}',
        dpi=300,
        bbox_inches='tight'
        )