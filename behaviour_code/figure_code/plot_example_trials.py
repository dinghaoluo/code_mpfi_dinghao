# -*- coding: utf-8 -*-
"""
Created on Fri 6 Dec 17:45:35 2024
Modified on Wed 30 Apr 18:48:23 2025
Updated on Wed 10 Sept 2025 

plot continuous trial traces 
modified to read the newly formatted dictionaries instead of the pd dataframes

updated 10 Sept 2025:
    - modernisation 
    - focusses on example trial plotting 

@author: Dinghao Luo
"""


#%% imports 
from pathlib import Path 

import numpy as np 
import pickle
import matplotlib.pyplot as plt 

from common import replace_outlier, mpl_formatting, smooth_convolve
mpl_formatting()


#%% parameters 
use_sess = 'A067r-20230821-02'
start_trial = 43
end_trial = 46
 
example_stem = Path(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\example_trials')

        
#%% main
(example_stem / use_sess).mkdir(exist_ok=True)

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