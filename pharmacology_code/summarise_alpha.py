# -*- coding: utf-8 -*-
"""
Created on Wed 13 Nov 16:53:32 2024

summarise pharmacological experiments with SCH23390

@author: Dinghao Luo
"""


#%% imports 
import os 
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import sem

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, smooth_convolve, replace_outlier
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\behaviour_code\utils')
import behaviour_functions as bf

x_speed = np.arange(2200)/10
x_lick = np.arange(2200)/10


#%% recording list
sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathNEblocker = rec_list.pathAlphaBlocker
sessNEblocker = rec_list.sessAlphaBlocker


#%% main
mean_speeds_baseline = [] 
mean_speeds_drug = []

mean_licks_baseline = []
mean_licks_drug = []

for i, pathname in enumerate(pathNEblocker):    
    sessname = pathname[-13:]
    print(sessname)
    
    sesslist = sessNEblocker[i]
        
    for i, sess in enumerate(sesslist):
        recname = sessname+f'-0{sess}'
        
        if pathname in rec_list.pathXiaoliangBetaBlocker:
            txtpath = os.path.join(pathname,
                                   recname,
                                   f'{recname}T.txt')
        else:
            suffix = f'-0{sess}'
            txtpath = f'{pathname}{suffix}T.txt'
            
        file = bf.process_behavioural_data(txtpath)
        
        # speed (spatial)
        speed_dist = np.array(
            [replace_outlier(np.array(trial))
            for i, trial in enumerate(file['speed_distances_aligned'])
            if len(trial)>0]
            )
        
        if i == 0:
            mean_speeds_baseline.append(np.mean(speed_dist, axis=0))
        elif i >= 1:
            mean_speeds_drug.append(np.mean(speed_dist, axis=0))
        
        # licks (spatial)
        lick_dist = np.array(
            [smooth_convolve(np.array(trial), sigma=10) * 10  # convert from mm to cm 
            for i, trial in enumerate(file['lick_maps'])
            if len(trial)>0]
            )

        if i == 0:
            mean_licks_baseline.append(np.mean(lick_dist, axis=0))
        elif i >= 1:
            mean_licks_drug.append(np.mean(lick_dist, axis=0))
        

#%% speed plot  
fig, ax = plt.subplots(figsize=(2.3,1.7))

ms_baseline = np.mean(mean_speeds_baseline, axis=0)/10
ms_drug = np.mean(mean_speeds_drug, axis=0)/10
ss_baseline = sem(mean_speeds_baseline, axis=0)/10
ss_drug = sem(mean_speeds_drug, axis=0)/10

lp, = ax.plot(x_speed, ms_baseline, color='grey')
ax.fill_between(x_speed, ms_baseline+ss_baseline,
                         ms_baseline-ss_baseline,
                color='grey', edgecolor='none', alpha=.15)
ld, = ax.plot(x_speed, ms_drug, color='darkgreen')
ax.fill_between(x_speed, ms_drug+ss_drug,
                         ms_drug-ss_drug,
                color='darkgreen', edgecolor='none', alpha=.15)

plt.legend([lp, ld], ['baseline', 'exp.'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(0,180), xlabel='distance (cm)',
       ylabel='velocity (cm/s)')

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\alpha\speed_profile{}'.format(ext),
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
ld, = ax.plot(x_lick, ml_drug, color='darkgreen')
ax.fill_between(x_lick, ml_drug+sl_drug,
                        ml_drug-sl_drug,
                color='darkgreen', edgecolor='none', alpha=.15)

plt.legend([lp, ld], ['baseline', 'exp.'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(30,219), xlabel='distance (cm)',
       ylim=(0,0.4), ylabel='hist. licks', yticks=[0, 0.3])

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\alpha\lick_profile{}'.format(ext),
                dpi=300, bbox_inches='tight')