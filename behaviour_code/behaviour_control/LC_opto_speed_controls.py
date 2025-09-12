# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:51:52 2025

controls for LC run-onset peaks for opto 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from scipy.stats import sem  
import sys 

import rec_list
paths = rec_list.pathLCopt

from common import mpl_formatting, smooth_convolve
mpl_formatting()


#%% parameters 
XAXIS_DIST = np.arange(2200) / 10  # in cm 
XAXIS_TIME = np.arange(4000) / 1000  # in s


#%% main 
# containers 
all_mean_ctrl_speeds = []
all_mean_stim_speeds = []

all_mean_ctrl_speeds_time = []
all_mean_stim_speeds_time = []


for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    
    with open(
            rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\LC\{recname}.pkl',
            'rb'
            ) as f:
        beh = pickle.load(f)
    
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds)
                if cond!='0']
    ctrl_idx = [trial + 2 for trial in stim_idx]
    bad_idx = [trial for trial, bad in enumerate(beh['bad_trials'][1:])
               if bad]
    
    speed_trials = [[t for t in trial]
                    for trial in beh['speed_distances_aligned'][1:]]
    
    stim_speeds = [smooth_convolve(speeds, sigma=30) for trial, speeds 
                   in enumerate(speed_trials)
                   if trial in stim_idx and trial not in bad_idx]
    ctrl_speeds = [smooth_convolve(speeds, sigma=30) for trial, speeds 
                   in enumerate(speed_trials)
                   if trial in ctrl_idx and trial not in bad_idx]
    
    mean_stim_speeds = np.mean(stim_speeds, axis=0)
    sem_stim_speeds = sem(stim_speeds, axis=0)
    mean_ctrl_speeds = np.mean(ctrl_speeds, axis=0)
    sem_ctrl_speeds = sem(ctrl_speeds, axis=0)
    
    fig, ax = plt.subplots(figsize=(1.5,1.1))
    
    ax.plot(XAXIS_DIST, mean_ctrl_speeds, c='grey')
    ax.fill_between(XAXIS_DIST, mean_ctrl_speeds+sem_ctrl_speeds,
                                mean_ctrl_speeds-sem_ctrl_speeds,
                    color='grey', edgecolor='none', alpha=.25)
    
    ax.plot(XAXIS_DIST, mean_stim_speeds, c='royalblue')
    ax.fill_between(XAXIS_DIST, mean_stim_speeds+sem_stim_speeds,
                                mean_stim_speeds-sem_stim_speeds,
                    color='royalblue', edgecolor='none', alpha=.25)
    
    ax.set(title=recname,
           xlabel='distance (cm)', ylabel='velocity (cm·s$^{-1}$)',
           xlim=(0, 200),
           ylim=(0, max(max(mean_ctrl_speeds), max(mean_stim_speeds))+5))
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
        
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_controls\{recname}_speed_curves{ext}',
            dpi=300,
            bbox_inches='tight'
            )
        
    stim_means = [np.mean(trial[:1800]) for trial in stim_speeds]
    ctrl_means = [np.mean(trial[:1800]) for trial in ctrl_speeds]
    
    fig, ax = plt.subplots(figsize=(2,1))
    
    bp = ax.boxplot([ctrl_means, stim_means], vert=False, notch=True,
                    widths=0.8, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['grey', 'royalblue']):
        patch.set_facecolor(color)
        patch.set_edgecolor('k')
        patch.set_linewidth=1
        
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    ax.set(title=recname)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_controls\{recname}_mean_speeds{ext}',
            dpi=300,
            bbox_inches='tight'
            )
        
    speed_time_trials = [[t[1] for t in trial]
                         for trial in beh['speed_times_aligned'][1:]]
    
    stim_speeds_time = [smooth_convolve(speeds[:4000], sigma=30) for trial, speeds 
                        in enumerate(speed_time_trials)
                        if trial in stim_idx and trial not in bad_idx
                        and len(speeds)>=4000]
    ctrl_speeds_time = [smooth_convolve(speeds[:4000], sigma=30) for trial, speeds 
                        in enumerate(speed_time_trials)
                        if trial in ctrl_idx and trial not in bad_idx
                        and len(speeds)>=4000]
    
    mean_stim_speeds_time = np.mean(stim_speeds_time, axis=0)
    sem_stim_speeds_time = sem(stim_speeds_time, axis=0)
    mean_ctrl_speeds_time = np.mean(ctrl_speeds_time, axis=0)
    sem_ctrl_speeds_time = sem(ctrl_speeds_time, axis=0)
    
    fig, ax = plt.subplots(figsize=(1.5,1.1))
    
    ax.plot(XAXIS_TIME, mean_ctrl_speeds_time, c='grey')
    ax.fill_between(XAXIS_TIME, mean_ctrl_speeds_time+sem_ctrl_speeds_time,
                                mean_ctrl_speeds_time-sem_ctrl_speeds_time,
                    color='grey', edgecolor='none', alpha=.25)
    
    ax.plot(XAXIS_TIME, mean_stim_speeds_time, c='royalblue')
    ax.fill_between(XAXIS_TIME, mean_stim_speeds_time+sem_stim_speeds_time,
                                mean_stim_speeds_time-sem_stim_speeds_time,
                    color='royalblue', edgecolor='none', alpha=.25)
    
    ax.set(title=recname,
           xlabel='time (s)', ylabel='velocity (cm·s$^{-1}$)',
           xlim=(0, 4),
           ylim=(0, max(max(mean_ctrl_speeds_time), max(mean_stim_speeds_time))+5))
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
        
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_controls\{recname}_speed_curves_time{ext}',
            dpi=300,
            bbox_inches='tight'
            )
        
    # accumulate for across-session average (distance)
    all_mean_ctrl_speeds.append(mean_ctrl_speeds)
    all_mean_stim_speeds.append(mean_stim_speeds)
    
    # accumulate for across-session average (time)
    all_mean_ctrl_speeds_time.append(mean_ctrl_speeds_time)
    all_mean_stim_speeds_time.append(mean_stim_speeds_time)
        
        
#%% summary 
mean_ctrl = np.mean(all_mean_ctrl_speeds, axis=0)
sem_ctrl = sem(all_mean_ctrl_speeds, axis=0)
mean_stim = np.mean(all_mean_stim_speeds, axis=0)
sem_stim = sem(all_mean_stim_speeds, axis=0)

mean_ctrl_time = np.mean(all_mean_ctrl_speeds_time, axis=0)
sem_ctrl_time = sem(all_mean_ctrl_speeds_time, axis=0)
mean_stim_time = np.mean(all_mean_stim_speeds_time, axis=0)
sem_stim_time = sem(all_mean_stim_speeds_time, axis=0)


#%% plotting 
fig, ax = plt.subplots(figsize=(1.65,1.4))

ax.plot(XAXIS_DIST, mean_ctrl, c='grey', label='control')
ax.fill_between(XAXIS_DIST, mean_ctrl+sem_ctrl, mean_ctrl-sem_ctrl,
                color='grey', edgecolor='none', alpha=.25)
ax.plot(XAXIS_DIST, mean_stim, c='royalblue', label='stim')
ax.fill_between(XAXIS_DIST, mean_stim+sem_stim, mean_stim-sem_stim,
                color='royalblue', edgecolor='none', alpha=.25)

ax.set(xlabel='distance (cm)', ylabel='speed (cm·s$^{-1}$)',
       xlim=(0, 200),
       ylim=(0, max(np.max(mean_ctrl + sem_ctrl), np.max(mean_stim + sem_stim)) + 5),
       title='mean across sessions')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False)

for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_controls\mean_speed_curve_dist{ext}',
                dpi=300, bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(1.65,1.4))

ax.plot(XAXIS_TIME, mean_ctrl_time, c='grey', label='control')
ax.fill_between(XAXIS_TIME, mean_ctrl_time+sem_ctrl_time, mean_ctrl_time-sem_ctrl_time,
                color='grey', edgecolor='none', alpha=.25)
ax.plot(XAXIS_TIME, mean_stim_time, c='royalblue', label='stim')
ax.fill_between(XAXIS_TIME, mean_stim_time+sem_stim_time, mean_stim_time-sem_stim_time,
                color='royalblue', edgecolor='none', alpha=.25)

ax.set(xlabel='time from run onset (s)', ylabel='speed (cm·s$^{-1}$)',
       xlim=(0, 4),
       ylim=(0, max(np.max(mean_ctrl_time + sem_ctrl_time), np.max(mean_stim_time + sem_stim_time)) + 5),
       title='mean across sessions')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False)

for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_controls\mean_speed_curve_time{ext}',
                dpi=300, bbox_inches='tight')