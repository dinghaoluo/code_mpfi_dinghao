# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:32:20 2025

simple script to quickly check immobile training progress

@author: Dinghao 
"""


#%% what to process 
animal_id = 132
date = 20250716
sess = 1


#%% imports 
import sys 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\behaviour_code\utils')
from behaviour_functions import process_behavioural_data_immobile


#%% parameters
pre_cue_time = 4  # in seconds, for plotting 


#%% main 
# unpack the file
animal = f'ANMD{animal_id}'
recname = f'A{animal_id}-{date}-0{sess}'

file = process_behavioural_data_immobile(f'Z:/Dinghao/MiceExp/{animal}/{recname}T.txt')

lick_times = file['lick_times']
start_cue_times = file['start_cue_times']
reward_times = file['reward_times']
trial_statements = file['trial_statements']
tot_trials = len(lick_times)

# here we use the last trial statement because sometimes we forget to enter the correct parameters before starting 
start_cue_duration = float(trial_statements[-1][2])/1000/1000  # to get the duraction in seconds 
delay_duration = float(trial_statements[-1][3])/1000/1000  # same as above 
reward_duration = float(trial_statements[-1][4])/1000/1000  # same as above 

# replacing blackout_duration with blackout_dur_list since using variable blackouts, 30 June 2025 
blackout_dur_list = [float(trial[5])/1000/1000 for trial in trial_statements] 
trial_dur_list = [start_cue_duration + delay_duration + blackout_dur_list[i]
                  for i in range(tot_trials)]
max_trial_time = max(trial_dur_list)

# read reward times for more accurate reward labelling, 30 June 2025
reward_times = file['reward_times']

# pre cue 
total_dur_list = [trial_dur_list[i] + pre_cue_time
                  for i in range(tot_trials)]

# histogram for calculating mean profile 
bin_size = .1  # in seconds 
bins = np.arange(-pre_cue_time, max_trial_time+bin_size, bin_size)
sum_licks = np.zeros(len(bins)-1)

# raster
fig, axs = plt.subplots(1, 2, figsize=(6.5,3))

for trial in range(tot_trials):
    trial_start_time = float(trial_statements[trial][1])
    
    rewarded = len(reward_times[trial])>0  # flag for future use
    if rewarded:
        curr_reward = (reward_times[trial][0] - trial_start_time) / 1000
        
        axs[0].scatter(curr_reward, trial, 
                       color='darkgreen', edgecolor='none',
                       s=.5, alpha=1, zorder=10)
    
    pre_licks = [(l - trial_start_time) / 1000 for l in lick_times[trial-1] 
                 if trial>0 and (l-trial_start_time)/1000 > -pre_cue_time]
    curr_licks = [(l - trial_start_time) / 1000 for l in lick_times[trial]]
    
    all_licks = pre_licks + curr_licks
    hist, _ = np.histogram(all_licks, bins=bins)
    sum_licks += hist
    axs[0].scatter(all_licks, [(trial+1)]*len(all_licks),
                   s=1, c='magenta', ec='none')
    
    # blackout (last)
    if trial > 0:
        prev_blackout = blackout_dur_list[trial-1]
        axs[0].scatter(-prev_blackout, trial,
                       color='black', edgecolor='none',
                       s=.5, alpha=1, zorder=10)
    
axs[0].axvspan(0, start_cue_duration, 
               facecolor='red', edgecolor='none', alpha=.15)
axs[0].axvspan(start_cue_duration, start_cue_duration+delay_duration,
               facecolor='grey', edgecolor='none', alpha=.15)
axs[0].set(xlim=(-pre_cue_time, max_trial_time), xlabel='time from cue (s)',
           ylim=(0, tot_trials), ylabel='trial #')

# mean profile 
mean_lick_profile = (sum_licks / tot_trials) / bin_size

axs[1].plot(bins[:-1] + bin_size/2, mean_lick_profile, color='magenta', linewidth=1)
axs[1].axvspan(0, start_cue_duration, 
               facecolor='red', edgecolor='none', alpha=.15)
axs[1].axvspan(start_cue_duration, start_cue_duration+delay_duration,
               facecolor='grey', edgecolor='none', alpha=.15)
axs[1].axvspan(start_cue_duration+delay_duration, start_cue_duration+delay_duration+reward_duration,
               facecolor='darkgreen', edgecolor='none', alpha=.15)
axs[1].set(xlim=(-pre_cue_time, max_trial_time), xlabel='time from cue (s)',
           ylim=(0, max(mean_lick_profile)), ylabel='lick rate (Hz)')

for s in ['top', 'right']:
    axs[1].spines[s].set_visible(False)

# ramp index
delay_start = start_cue_duration
delay_end = start_cue_duration + delay_duration

# get time points and corresponding lick rates during delay period
time_points = bins[:-1] + bin_size/2
in_delay = (time_points >= delay_start) & (time_points <= delay_end)
delay_times = time_points[in_delay] - delay_start  # make time relative to delay start
delay_rates = mean_lick_profile[in_delay]

# calculate linear regression for ramp quantification
slope_delay, intercept_delay = np.polyfit(delay_times, delay_rates, 1)
r_squared_delay = np.corrcoef(delay_times, delay_rates)[0,1]**2

info = f'linear slope during delay: {slope_delay:.2f} Hz/s (R²={r_squared_delay:.2f})'

fig.suptitle(f'{recname}\n{info}')
fig.tight_layout()

outpath = f'Z:/Dinghao/MiceExp/{animal}/behaviour_plots'
os.makedirs(outpath, exist_ok=True)

fig.savefig(os.path.join(outpath, f'{recname}.png'),
            dpi=300,
            bbox_inches='tight')


#%% alignment test 
blackouts = []
first_lick_medians = []

last_lick_aligned = []
cue_aligned = []
last_reward_aligned = []

for trial in range(1, tot_trials):
    trial_start_time = float(trial_statements[trial][1])
    
    # blackout duration v licks 
    prev_blackout = blackout_dur_list[trial-1]
    
    curr_licks = [(l - float(trial_statements[trial][1])) / 1000
                  for l in lick_times[trial]]
    first_lick_median = np.median(curr_licks[:3]) if len(curr_licks) > 0 else None
    
    # last lick time
    prev_licks = [(l - trial_start_time) / 1000
                  for l in lick_times[trial-1]]
    last_lick_median = (
        np.median(prev_licks[-3:]) if len(prev_licks) >= 3 
        else np.median(prev_licks) if len(prev_licks) > 0 
        else None
        )
    
    # last reward time 
    if len(reward_times[trial-1]) > 0:
        last_reward_time = (reward_times[trial-1][0] - trial_start_time) / 1000
    else:
        last_reward_time = None
        
    # alignments
    if last_lick_median is not None and first_lick_median is not None:
        last_lick_aligned.append(first_lick_median - last_lick_median)
        
    if last_reward_time is not None and first_lick_median is not None:
        last_reward_aligned.append(first_lick_median - last_reward_time)
    
    if first_lick_median is not None:
        cue_aligned.append(first_lick_median)
        first_lick_medians.append(first_lick_median)
        blackouts.append(prev_blackout)
        
    
r, p = pearsonr(blackouts, first_lick_medians)

fig, ax = plt.subplots(figsize=(3,3))

ax.scatter(blackouts, first_lick_medians, color='magenta', s=10)
ax.set_xlabel('previous blackout duration (s)')
ax.set_ylabel('median of first licks (s)')
ax.set_title(f'r = {r:.2f}, p = {p:.4f}')
fig.tight_layout()

fig.savefig(os.path.join(outpath, f'{recname}_blackout_vs_firstlick.png'),
            dpi=300, bbox_inches='tight')


fig, axs = plt.subplots(1,3, figsize=(6,2.5))

axs[0].hist(last_lick_aligned, bins=30, color='magenta', alpha=0.7)
axs[0].set_title(f'align to last licks\nstd={np.std(last_lick_aligned):.2f}')

axs[1].hist(cue_aligned, bins=30, color='orange', alpha=0.7)
axs[1].set_title(f'align to cue\nstd={np.std(cue_aligned):.2f}')

axs[2].hist(last_reward_aligned, bins=30, color='green', alpha=0.7)
axs[2].set_title(f'align to last reward\nstd={np.std(last_reward_aligned):.2f}')

for i in range(3):
    axs[i].set(xlabel='aligned time (s)',
               ylabel='median 1st-lick (s)')

fig.tight_layout()
fig.savefig(os.path.join(outpath, f'{recname}_alignment_histograms.png'),
             dpi=300, bbox_inches='tight')


#%% plot example good trials
target_time = start_cue_duration + delay_duration
tolerance = 2  # seconds, adjust as needed

# collect example trials
example_trials = []
for trial in range(1, tot_trials):
    trial_start_time = float(trial_statements[trial][1])
    curr_licks = [(l - trial_start_time) / 1000 for l in lick_times[trial]]
    if len(curr_licks) == 0:
        continue
    first_lick_median = np.median(curr_licks[:3])
    if abs(first_lick_median - target_time) <= tolerance:
        example_trials.append((trial, curr_licks, reward_times[trial]))

fig, ax = plt.subplots(figsize=(2.8, len(example_trials)*.12))
for idx, (trial, curr_licks, rewards) in enumerate(example_trials):
    y = idx + 1  # simple sequential position
    ax.scatter(curr_licks, [y]*len(curr_licks), s=1.8, c='magenta', edgecolor='none')
    if len(rewards) > 0:
        reward_time = (rewards[0] - float(trial_statements[trial][1]))/1000
        ax.scatter([reward_time], [y], s=3, c='darkgreen', edgecolor='none')
ax.axvspan(0, start_cue_duration, facecolor='red', alpha=0.15)
ax.axvspan(start_cue_duration, start_cue_duration+delay_duration, facecolor='grey', alpha=0.15)
ax.axvspan(start_cue_duration+delay_duration, start_cue_duration+delay_duration+reward_duration,
           facecolor='darkgreen', alpha=0.15)

# set y-ticks to show actual trial indices
ax.tick_params(axis='y', labelsize=5)
ax.set_yticks(range(1, len(example_trials)+1))
ax.set_yticklabels([trial for trial,_,_ in example_trials])

ax.set(xlabel='time from cue (s)', ylabel='trial # (original index)',
       title=f'example good trials')
fig.tight_layout()

fig.savefig(os.path.join(outpath, f'{recname}_example_good_trials.png'),
            dpi=300, bbox_inches='tight')