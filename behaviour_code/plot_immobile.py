# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:32:20 2025

simple script to quickly check immobile training progress

@author: Dinghao 
"""


#%% imports 
import sys 
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from behaviour_functions import process_behavioural_data_immobile


#%% main 
# unpack the file
animal = 'ANMD108'
recname = f'A{animal[-3:]}-20250212-01'

file = process_behavioural_data_immobile(f'Z:/Dinghao/MiceExp/{animal}/{recname}T.txt')

lick_times = file['lick_times']
start_cue_times = file['start_cue_times']
reward_times = file['reward_times']
trial_statements = file['trial_statements']
tot_trials = len(lick_times)

# here we use the last trial statement because sometimes we forget to enter the correct parameters before starting 
start_cue_duration = float(trial_statements[-1][2])/1000/1000  # to get the duraction in seconds 
delay_duration = float(trial_statements[-1][3])/1000/1000  # same as above 
blackout_duration = float(trial_statements[-1][5])/1000/1000  # same as above 
trial_duration = start_cue_duration + delay_duration + blackout_duration

# pre cue 
pre_cue_time = 8  # in seconds 
total_duration = trial_duration + pre_cue_time

# histogram for calculating mean profile 
bin_size = .1  # in seconds 
bins = np.arange(-pre_cue_time, trial_duration+bin_size, bin_size)
sum_licks = np.zeros(len(bins)-1)

# raster
fig, axs = plt.subplots(1, 2, figsize=(6.5,3))

for trial in range(tot_trials):
    trial_start_time = float(trial_statements[trial][1])
    pre_licks = [(l - trial_start_time) / 1000 for l in lick_times[trial-1] 
                 if trial>0 and (l-trial_start_time)/1000 > -pre_cue_time]
    curr_licks = [(l - trial_start_time) / 1000 for l in lick_times[trial]]
    
    all_licks = pre_licks + curr_licks
    hist, _ = np.histogram(all_licks, bins=bins)
    sum_licks += hist
    axs[0].scatter(all_licks, [(trial+1)]*len(all_licks),
                   s=1.5, c='magenta', ec='none')
    
axs[0].axvspan(0, start_cue_duration, 
               color='red', edgecolor='none', alpha=.15)
axs[0].axvspan(start_cue_duration, start_cue_duration+delay_duration,
               color='grey', edgecolor='none', alpha=.15)
axs[0].axvspan(start_cue_duration+delay_duration, start_cue_duration+delay_duration+0.1,
               color='darkgreen', edgecolor='none', alpha=.15)
axs[0].set(xlim=(-pre_cue_time, trial_duration), xlabel='time from cue (s)',
           ylim=(0, tot_trials), ylabel='trial #')

# mean profile 
mean_lick_profile = (sum_licks / tot_trials) / bin_size

axs[1].plot(bins[:-1] + bin_size/2, mean_lick_profile, color='magenta', linewidth=1)
axs[1].axvspan(0, start_cue_duration, 
               color='red', edgecolor='none', alpha=.15)
axs[1].axvspan(start_cue_duration, start_cue_duration+delay_duration,
               color='grey', edgecolor='none', alpha=.15)
axs[1].axvspan(start_cue_duration+delay_duration, start_cue_duration+delay_duration+0.1,
               color='darkgreen', edgecolor='none', alpha=.15)
axs[1].set(xlim=(-pre_cue_time, trial_duration), xlabel='time from cue (s)',
           ylim=(0, max(mean_lick_profile)), ylabel='lick rate (Hz)')

for s in ['top', 'right']:
    axs[1].spines[s].set_visible(False)

# ramp index
delay_start = start_cue_duration
delay_end = start_cue_duration + delay_duration
blackout_start = delay_end + 2  # +2 to eliminate the consumption period
blackout_end = delay_end + blackout_duration

# get time points and corresponding lick rates during delay period
time_points = bins[:-1] + bin_size/2
in_delay = (time_points >= delay_start) & (time_points <= delay_end)
delay_times = time_points[in_delay] - delay_start  # make time relative to delay start
delay_rates = mean_lick_profile[in_delay]
in_blackout = (time_points >= blackout_start) & (time_points <= blackout_end)
blackout_times = time_points[in_blackout] - blackout_start
blackout_rates = mean_lick_profile[in_blackout]

# calculate linear regression for ramp quantification
slope_delay, intercept_delay = np.polyfit(delay_times, delay_rates, 1)
r_squared_delay = np.corrcoef(delay_times, delay_rates)[0,1]**2
slope_blackout, intercept_blackout = np.polyfit(blackout_times, blackout_rates, 1)
r_squared_blackout = np.corrcoef(blackout_times, blackout_rates)[0,1]**2

info = f'linear slope during delay: {slope_delay:.2f} Hz/s (R²={r_squared_delay:.2f})\
    \n linear slope during blackout: {slope_blackout:.2f} Hz/s (R²={r_squared_blackout:.2f})'

fig.suptitle(f'{recname}\n{info}')
fig.tight_layout()

plt.savefig(f'Z:/Dinghao/MiceExp/{animal}/behaviour_plots/{recname}.png',
            dpi=300,
            bbox_inches='tight')