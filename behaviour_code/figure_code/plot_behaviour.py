# -*- coding: utf-8 -*-
"""
Created on Fri 6 Dec 17:45:35 2024

plot continuous trial traces 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os 
import sys 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import replace_outlier, mpl_formatting
mpl_formatting()


#%% parameters 
save_every = 5  # save every n trials


#%% load data 
exp_name = 'HPCLCterm'  # HPCLC, HPCLCterm, LC, HPCGRABNE, HPCLCGCaMP
df = pd.read_pickle(r'Z:/Dinghao/code_dinghao/behaviour/all_{}_sessions.pkl'.format(
    exp_name))


#%% main 
os.makedirs(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{}'.format(exp_name), 
            exist_ok=True)

for recname, row in df.iterrows():
    print(recname)
    os.makedirs(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{}\{}'.format(exp_name, recname), 
                exist_ok=True)
    
    speeds_rec = row['speed_times']
    tot_trial = len(speeds_rec)
    
    fig, ax = plt.subplots(figsize=(4, .75))
    speed_max = 0
    all_speeds = []  # combine speeds across trials with interpolated gaps
    holdout_licks = []
    holdout_rews = []
    for trial in range(tot_trial):
        if np.mod(trial, save_every) == 0 and trial != 0:  # plot, save and close every `save_every` trials
            all_speeds = np.vstack(all_speeds)
            times = all_speeds[:, 0] / 1000
            speeds = replace_outlier(all_speeds[:, 1])
            speed_min = min(speeds)
            speed_max = max(speed_max, max(speeds))
            ax.plot(times, speeds, linewidth=1, color='k')
        
            for lick in holdout_licks:
                ax.vlines(x=lick, ymin=speed_max + speed_max * 0.01, ymax=speed_max + speed_max * 0.1,
                          linewidth=.3, color='orchid')
            for rew in holdout_rews:
                ax.vlines(x=rew, ymin=speed_max, ymax=speed_max + speed_max * 0.11, 
                          linewidth=1, color='darkgreen')
            
            for s in ['top', 'right']: ax.spines[s].set_visible(False)
            ax.set(xlabel='time (s)',
                   ylabel='speed (cm/s)', ylim=(max(speed_min - 1, 0), speed_max + speed_max * 0.11),
                   title='{}_{}'.format(trial - (save_every - 1), trial))
            for ext in ['.png', '.pdf']:
                fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{}\{}\{}{}'.format(
                    exp_name, recname, f'{trial - (save_every - 1)}_{trial}', ext
                    ), 
                    dpi=300,
                    bbox_inches='tight')
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(4, .75))
            speed_max = 0
            all_speeds = []
            holdout_licks = []
            holdout_rews = []
                
        # combine speeds
        curr_speeds = np.asarray(speeds_rec[trial])
        if len(all_speeds) > 0:  # add a gap of zeros between the last and current trial
            last_time = all_speeds[-1][-1][0]
            first_time = curr_speeds[0, 0]
            gap_times = np.arange(last_time + 1, first_time, 1)  # 1 ms bins in gap
            gap_speeds = np.zeros_like(gap_times)
            gap_data = np.column_stack((gap_times, gap_speeds))
            all_speeds.append(gap_data)
        
        all_speeds.append(curr_speeds)
        
        curr_start = row['run_onsets'][trial] / 1000
        ax.axvline(x=curr_start, linewidth=1, color='red', alpha=.8)
        
        if row['lick_times'][trial]:  # if the current trial has any lick 
            curr_licks = np.asarray(row['lick_times'][trial]) / 1000
            holdout_licks = np.concatenate((holdout_licks, curr_licks))
        
        if row['reward_times'][trial]:  # if the current trial has a reward 
            curr_rew = row['reward_times'][trial][0] / 1000
            holdout_rews.append(curr_rew)

            
            
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