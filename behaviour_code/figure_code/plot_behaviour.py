# -*- coding: utf-8 -*-
"""
Created on Fri 6 Dec 17:45:35 2024
Modified on Wed 30 Apr 18:48:23 2025

plot continuous trial traces 
modified to read the newly formatted dictionaries instead of the pd dataframes

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import pickle
import matplotlib.pyplot as plt 
import os 
import sys 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import replace_outlier, mpl_formatting
mpl_formatting()


#%% paths 
sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list


#%% parameters 
save_every = 50  # save every x seconds 


#%% load data 
for exp_name in ['HPCLC', 'HPCLCterm', 'LC', 'HPCGRABNE', 'LCHPCGCaMP']:
# for exp_name in ['LCHPCGCaMP']:
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
        rf'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{exp_name}', 
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
        
        output_dir = os.path.join(
            r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles', 
            exp_name, recname
            )
        os.makedirs(output_dir, exist_ok=True)
        
        times_ms = data['upsampled_timestamps_ms']
        speeds = data['upsampled_speed_cm_s']
        licks = [t for trial in data['lick_times'] for (t, _) in trial]
        rewards = [r[0] if isinstance(r, list) and r else r 
                   for r in data['reward_times'] if not np.isnan(r)]
        trial_ends = [float(s[1]) for s in data['new_trial_statements']]
        run_onsets = [r for r in data['run_onsets']]

        # parameters
        window_size = save_every * 1000  # 30 seconds in ms
        n_windows = (len(times_ms) + window_size - 1) // window_size

        for window_idx in range(n_windows):
            # indices to locate the window in the timestamp series
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, len(times_ms)-1)
            
            # these are the actual timestamps for slicing into the window 
            start_time = times_ms[start_idx]
            end_time = times_ms[end_idx]
            
            # plot devices (nice)
            times_plot = np.array(times_ms[start_idx:end_idx]) / 1000
            speeds_plot = speeds[start_idx:end_idx]
            speed_max = np.max(speeds_plot)
            speed_min = np.min(speeds_plot)

            fig, ax = plt.subplots(figsize=(4, 1))
            ax.plot(times_plot, 
                    replace_outlier(speeds_plot), 
                    color='black', 
                    linewidth=1)

            # mark run onsets
            for r in run_onsets:
                if start_time <= r < end_time:
                    ax.axvline(x=r / 1000, color='red', linewidth=1)

            # mark licks
            for l in licks:
                if start_time <= l < end_time:
                    ax.vlines(x=l / 1000, ymin=speed_max + speed_max * .01, 
                              ymax=speed_max + speed_max * .1, 
                              color='orchid', linewidth=0.3)

            # mark rewards
            for r in rewards:
                if start_time <= r < end_time:
                    ax.vlines(x=r / 1000, ymin=speed_max, 
                              ymax=speed_max + speed_max * .11,
                              color='darkgreen', linewidth=1)
                    
            # mark trial_ends for debugging 
            for e in trial_ends:
                if start_time <= e < end_time:
                    ax.vlines(x=e / 1000, ymin=speed_max, 
                              ymax=speed_max + speed_max * .11,
                              color='royalblue', linewidth=1)

            ax.set(
                xlabel='time (s)',
                ylabel='speed (cm/s)',
                ylim=(0, speed_max + speed_max * .12),
                title=f'{recname} {window_idx*save_every}-{(window_idx+1)*save_every}s'
                )
            for s in ['top', 'right']:
                ax.spines[s].set_visible(False)

            for ext in ['.png', '.pdf']:
                fig.savefig(os.path.join(
                    output_dir, f'{window_idx*save_every}_{(window_idx+1)*save_every}s{ext}'
                    ), 
                    dpi=300, 
                    bbox_inches='tight')
            plt.close(fig)

        
#%% if we want to plot example trials
use_sess = 'A067r-20230821-02'
start_trial = 43
end_trial = 46

def plot_example_trials(use_sess, start_trial, end_trial):
    os.makedirs(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\example_trials\{}'.format(use_sess), 
                exist_ok=True)
    
    row = df.loc[use_sess]
    speeds_rec = row['speed_times']
    
    fig, ax = plt.subplots(figsize=(1.5,.5))
    speed_max = 0
    all_speeds = []  # combine speeds across trials to circumvent the 'missing connection' problem
    holdout_licks = []; holdout_rews = []; holdout_starts = []
    for trial in range(start_trial, end_trial+1):
        # combine speeds
        curr_speeds = np.asarray(speeds_rec[trial])
        if len(all_speeds) > 0:  # add a gap of zeros between the last and current trial
            # determine the time gap
            last_time = all_speeds[-1][-1][0]
            first_time = curr_speeds[0, 0]
            gap_times = np.arange(last_time + 1, first_time, 20)  # 1 ms bins in gap
            gap_speeds = np.zeros_like(gap_times)
            gap_data = np.column_stack((gap_times, gap_speeds))
            all_speeds.append(gap_data)
        all_speeds.append(curr_speeds)
            
        if row['run_onsets'][trial]:
            curr_start = row['run_onsets'][trial]/1000
            holdout_starts.append(curr_start)
            
        if row['lick_times'][trial]:  # if the current trial has any lick 
            curr_licks = np.asarray(row['lick_times'][trial])/1000
            holdout_licks = np.concatenate((holdout_licks, curr_licks))
        
        if row['reward_times'][trial]:  # if the current trial has a reward 
            curr_rew = row['reward_times'][trial][0]/1000
            holdout_rews.append(curr_rew)
            
    all_speeds = np.vstack(all_speeds)
    times = all_speeds[:,0]/1000; start_time = times[0]; times-=start_time  # align everything back to t0
    speeds = replace_outlier(all_speeds[:,1])
    speed_min = min(speeds)
    speed_max = max(speed_max, max(speeds))
    ax.plot(times, speeds, linewidth=1, color='k')
    
    for start in holdout_starts:
        ax.axvline(x=start-start_time, linewidth=1, color='red')
    for lick in holdout_licks:
        ax.vlines(x=lick-start_time, ymin=speed_max+speed_max*.01, ymax=speed_max+speed_max*.1,
                  linewidth=.3, color='orchid')
    for rew in holdout_rews:
        ax.vlines(x=rew-start_time, ymin=speed_max, ymax=speed_max+speed_max*.11, 
                  linewidth=1, color='darkgreen')
    
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.set(xlabel='time (s)', xlim=(0, 11.5),
           ylabel='speed (cm/s)', ylim=(max(speed_min-1, 0), speed_max+speed_max*.11),
           title='{}_{}'.format(start_trial, end_trial))
    for ext in ['.png', '.pdf']:
        fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\example_trials\{}\{}{}'.format(
            use_sess, f'{start_trial}_{end_trial}', ext
            ), 
            dpi=300,
            bbox_inches='tight')
    plt.close(fig)