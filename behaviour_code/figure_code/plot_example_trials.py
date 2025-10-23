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


#%% path stems 
example_stem = Path('Z:/Dinghao/code_dinghao/behaviour/trial_profiles/example_trials')
experiment_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments')


#%% parameters 
experiment  = 'LC'
use_sess    = 'A067r-20230821-02'
start_trial = 43
end_trial   = 46

        
#%% main
beh_path = experiment_stem / experiment / f'{use_sess}.pkl'
try: 
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)
        (example_stem / use_sess).mkdir(exist_ok=True)
        print('behaviour file loaded')
except Exception:
    print('behaviour file not found; please choose a different session')
    import sys 
    sys.exit()  # halt 

speed_times = beh['speed_times_aligned']

fig, ax = plt.subplots(figsize=(2.1,.55))
speed_max = 0
all_speeds = []
holdout_licks = []; holdout_rews = []; holdout_starts = []; holdout_cues = []
for trial in range(start_trial, end_trial+1):
    # combine speeds
    curr_speeds = np.asarray(speed_times[trial])
    if len(all_speeds) > 0:  # add a gap of zeros between the last and current trial
        # determine the time gap
        last_time = all_speeds[-1][-1][0]
        first_time = curr_speeds[0, 0]
        gap_times = np.arange(last_time + 1, first_time, 20)  # 1 ms bins in gap
        gap_speeds = np.zeros_like(gap_times)
        gap_data = np.column_stack((gap_times, gap_speeds))
        all_speeds.append(gap_data)
    all_speeds.append(curr_speeds)
    
    # run onset
    run_onset = beh['run_onsets'][trial]
    if not np.isnan(run_onset):
        curr_start = run_onset / 1000.0
        holdout_starts.append(curr_start)
    
    # licks 
    licks = [t[0] for t in beh['lick_times'][trial]]  # extract timestamps 
    if licks:  # if the current trial has any lick 
        curr_licks = np.asarray(licks) / 1000.0
        holdout_licks = np.concatenate((holdout_licks, curr_licks))
    
    # reward 
    reward = beh['reward_times'][trial]
    if not np.isnan(reward):  # if the current trial has a reward 
        curr_rew = reward / 1000.0
        holdout_rews.append(curr_rew)
    
    # cue
    cue = beh['start_cue_times'][trial]
    if not np.isnan(cue):
        curr_cue = cue / 1000.0
        holdout_cues.append(curr_cue)
        
all_speeds = np.vstack(all_speeds)
times = all_speeds[:,0] / 1000.0
start_time = times[0]
times -= start_time  # align everything back to t0

speeds = replace_outlier(all_speeds[:,1])  # filter 

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
    
for cue in holdout_cues:
    ax.axvline(x=cue-start_time, linewidth=1, color='grey')

for s in ['top','right']: ax.spines[s].set_visible(False)

ax.set(xlabel='Time (s)', xlim=(-1, 11.5), xticks=[-1, 4, 9], xticklabels=[0, 5, 10],
       ylabel='Speed (cm/s)', ylim=(max(speed_min-1, 0), speed_max+speed_max*.11), yticks=(0, 60),
       title=f'{use_sess}\n{start_trial}-{end_trial}')

plt.tight_layout()
plt.show()

# save 
for ext in ['.png', '.pdf']:
    fig.savefig(example_stem / f'{start_trial}_{end_trial}{ext}', 
        dpi=300,
        bbox_inches='tight')