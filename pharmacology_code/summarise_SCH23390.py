# -*- coding: utf-8 -*-
"""
Created on Wed 13 Nov 16:53:32 2024

summarise pharmacological experiments with SCH23390

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
from scipy.stats import sem, ttest_rel

from common import mpl_formatting, replace_outlier, smooth_convolve
mpl_formatting()

import behaviour_functions as bf

x_speed = np.arange(2200)/10
x_lick = np.arange(2200)/10


#%% recording list
import rec_list
pathSCH = rec_list.pathSCH
sessSCH = rec_list.sessSCH


#%% main
mean_speeds_baseline = [] 
mean_speeds_drug = []

mean_licks_baseline = []
mean_licks_drug = []

reward_percentages_baseline = []
reward_percentages_drug = []

for i, pathname in enumerate(pathSCH):    
    sessname = pathname[-13:]
    print(sessname)
    
    sesslist = sessSCH[i]
        
    for i, sess in enumerate(sesslist):
        recname = sessname+f'-0{sess}'
        
        txtpath = f'{pathname}\{recname}\{recname}T.txt'
            
        file = bf.process_behavioural_data(txtpath)
        
        # reward percentage 
        reward_times = file['reward_times'][1:-1]
        rewarded = [1 if not np.isnan(t) else 0 for t in reward_times]
        
        if i == 0:
            reward_percentages_baseline.append(sum(rewarded)/len(rewarded))
        elif i == 1:
            reward_percentages_drug.append(sum(rewarded)/len(rewarded))
        
        # speed (spatial)
        speed_dist = np.array(
            [replace_outlier(np.array(trial))
            for i, trial in enumerate(file['speed_distances_aligned'])
            if len(trial)>0]
            )
        
        if i == 0:
            mean_speeds_baseline.append(np.mean(speed_dist, axis=0))
        elif i == 1:
            mean_speeds_drug.append(np.mean(speed_dist, axis=0))
        
        # licks (spatial)
        lick_dist = np.array(
            [smooth_convolve(np.array(trial), sigma=10) * 10  # convert from mm to cm 
            for i, trial in enumerate(file['lick_maps'])
            if len(trial)>0]
            )
        mean_licks = np.mean(lick_dist, axis=0)
        
        if i == 0:
            mean_licks_baseline.append(np.mean(lick_dist, axis=0))
        elif i == 1:
            mean_licks_drug.append(np.mean(lick_dist, axis=0))
        

#%% speed plot  
fig, ax = plt.subplots(figsize=(2.3,1.7))

ms_baseline = np.mean(mean_speeds_baseline, axis=0)
ms_drug = np.mean(mean_speeds_drug, axis=0)
ss_baseline = sem(mean_speeds_baseline, axis=0)
ss_drug = sem(mean_speeds_drug, axis=0)

lp, = ax.plot(x_speed, ms_baseline, color='grey')
ax.fill_between(x_speed, ms_baseline+ss_baseline,
                         ms_baseline-ss_baseline,
                color='grey', edgecolor='none', alpha=.15)
ld, = ax.plot(x_speed, ms_drug, color='#004D80')
ax.fill_between(x_speed, ms_drug+ss_drug,
                         ms_drug-ss_drug,
                color='#004D80', edgecolor='none', alpha=.15)

plt.legend([lp, ld], ['baseline', 'exp.'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(0,180), xlabel='distance (cm)',
       ylabel='velocity (cm/s)', ylim=(0,75))

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\SCH23390\speed_profile{}'.format(ext),
                dpi=300, bbox_inches='tight')
    

#%% lick plot  
fig, ax = plt.subplots(figsize=(2.3,1.7))

ml_baseline = np.mean(mean_licks_baseline, axis=0)/10
ml_drug = np.mean(mean_licks_drug, axis=0)/10
sl_baseline = sem(mean_licks_baseline, axis=0)/10
sl_drug = sem(mean_licks_drug, axis=0)/10

lp, = ax.plot(x_lick, ml_baseline, color='grey')
ax.fill_between(x_lick, ml_baseline+sl_baseline,
                        ml_baseline-sl_baseline,
                color='grey', edgecolor='none', alpha=.15)
ld, = ax.plot(x_lick, ml_drug, color='#004D80')
ax.fill_between(x_lick, ml_drug+sl_drug,
                        ml_drug-sl_drug,
                color='#004D80', edgecolor='none', alpha=.15)

plt.legend([lp, ld], ['baseline', 'exp.'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(30,219), xlabel='distance (cm)',
       ylim=(0,0.2), ylabel='hist. licks', yticks=[0, 0.2])

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\SCH23390\lick_profile{}'.format(ext),
                dpi=300, bbox_inches='tight')
    
    
#%% bar plot for reward percentage with connected scatter + p-value
fig, ax = plt.subplots(figsize=(2.3,1.7))

# means and sems
mean_rp_baseline = np.mean(reward_percentages_baseline)
mean_rp_drug = np.mean(reward_percentages_drug)
sem_rp_baseline = sem(reward_percentages_baseline)
sem_rp_drug = sem(reward_percentages_drug)

# paired t-test
tstat, pval = ttest_rel(reward_percentages_baseline, reward_percentages_drug)
print(f'paired t-test p = {pval:.4f}')

# plot bars
positions = [1, 2]
bar_width = 0.5
ax.bar(positions[0], mean_rp_baseline, yerr=sem_rp_baseline, width=bar_width,
       color='grey', alpha=0.7, capsize=3)
ax.bar(positions[1], mean_rp_drug, yerr=sem_rp_drug, width=bar_width,
       color='#004D80', alpha=0.7, capsize=3)

# plot lines connecting individual animals
for b, d in zip(reward_percentages_baseline, reward_percentages_drug):
    ax.plot(positions, [b, d], color='black', alpha=0.4, linewidth=0.8, marker='o', markersize=3)

# add p-value text
ax.text(1.5, max(mean_rp_baseline, mean_rp_drug) * 1.05,
        f'p = {pval:.3f}', ha='center', va='bottom', fontsize=8)

# tidy up axes
ax.set(xticks=positions, xticklabels=['baseline', 'DA blocker'],
       ylabel='reward %')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set_xlim(.5, 2.5)
ax.set_ylim(0, 1.1)

# save
for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\SCH23390\reward_percentage{}'.format(ext),
                dpi=300, bbox_inches='tight')
    
    
#%% bar plot for mean speed with connected scatter + p-value
fig, ax = plt.subplots(figsize=(2.3,1.7))

# means and sems
mean_ms_baseline = np.mean(mean_speeds_baseline, axis=1)
mean_ms_drug = np.mean(mean_speeds_drug, axis=1)

mean_speed_baseline = np.mean(mean_ms_baseline)
mean_speed_drug = np.mean(mean_ms_drug)
sem_speed_baseline = sem(mean_ms_baseline)
sem_speed_drug = sem(mean_ms_drug)

# paired t-test
tstat, pval = ttest_rel(mean_ms_baseline, mean_ms_drug)
print(f'paired t-test p = {pval:.4f}')

# plot bars
positions = [1, 2]
bar_width = 0.5
ax.bar(positions[0], mean_speed_baseline, yerr=sem_speed_baseline, width=bar_width,
       color='grey', alpha=0.7, capsize=3)
ax.bar(positions[1], mean_speed_drug, yerr=sem_speed_drug, width=bar_width,
       color='#004D80', alpha=0.7, capsize=3)

# plot lines connecting individual animals
for b, d in zip(mean_ms_baseline, mean_ms_drug):
    ax.plot(positions, [b, d], color='black', alpha=0.4, linewidth=0.8, marker='o', markersize=3)

# add p-value text
ax.text(1.5, max(mean_speed_baseline, mean_speed_drug) * 1.05,
        f'p = {pval:.3f}', ha='center', va='bottom', fontsize=8)

# tidy up axes
ax.set(xticks=positions, xticklabels=['baseline', 'NE blocker'],
       ylabel='mean speed (cm/s)')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set_xlim(.5, 2.5)

# save
for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\SCH23390\mean_speed{}'.format(ext),
                dpi=300, bbox_inches='tight')