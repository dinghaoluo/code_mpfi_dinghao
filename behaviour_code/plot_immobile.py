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
file = process_behavioural_data_immobile('Z:/Dinghao/MiceExp/ANMD107/A107-20250128-01T.txt')

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

# histogram for calculating mean profile 
bin_size = .1  # in seconds 
bins = np.arange(0, trial_duration+bin_size, bin_size)
sum_licks = np.zeros(len(bins)-1)

# raster
fig, axs = plt.subplots(1, 2, figsize=(7,3))

for trial in range(tot_trials):
    curr_licks = [(l-float(trial_statements[trial][1]))/1000 for l in lick_times[trial]]
    hist, _ = np.histogram(curr_licks, bins=bins)
    sum_licks += hist
    axs[0].scatter(curr_licks, [(trial+1)]*len(curr_licks),
                   s=1.5, c='magenta', ec='none')
    
axs[0].axvspan(0, start_cue_duration, 
               color='red', edgecolor='none', alpha=.15)
axs[0].axvspan(start_cue_duration, start_cue_duration+delay_duration,
               color='grey', edgecolor='none', alpha=.15)
axs[0].set(xlim=(0,trial_duration), xlabel='time from cue (s)',
           ylabel='trial #')

# mean profile 
mean_lick_profile = (sum_licks / tot_trials) / bin_size

axs[1].plot(bins[:-1] + bin_size/2, mean_lick_profile, color='magenta', linewidth=1)
axs[1].axvspan(0, start_cue_duration, 
               color='red', edgecolor='none', alpha=.15)
axs[1].axvspan(start_cue_duration, start_cue_duration+delay_duration,
               color='grey', edgecolor='none', alpha=.15)
axs[1].set(xlim=(0,trial_duration), xlabel='time from cue (s)',
           ylabel='lick rate (Hz)')

# ramp index
delay_start = start_cue_duration
delay_end = start_cue_duration + delay_duration

# get time points and corresponding lick rates during delay period
time_points = bins[:-1] + bin_size/2
in_delay = (time_points >= delay_start) & (time_points <= delay_end)
delay_times = time_points[in_delay] - delay_start  # make time relative to delay start
delay_rates = mean_lick_profile[in_delay]

# calculate linear regression for ramp quantification
slope, intercept = np.polyfit(delay_times, delay_rates, 1)
r_squared = np.corrcoef(delay_times, delay_rates)[0,1]**2

# alternative: compare first vs second half of delay period
half_point = delay_duration/2
first_half = delay_rates[delay_times <= half_point]
second_half = delay_rates[delay_times > half_point]
ramp_index = (np.nanmean(second_half) - np.nanmean(first_half))/delay_duration

info = f'linear slope during delay: {slope:.2f} Hz/s (R²={r_squared:.2f})'

fig.suptitle(info)
fig.tight_layout()