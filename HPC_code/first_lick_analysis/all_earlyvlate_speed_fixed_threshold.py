# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:37:16 2023
Modified on Tue 22 Apr 2025:
    - reworking the script to calculate multiple other factors other than 
        peak amplitude 
Modified on 16 July 2025:
    - reworked to look at the speed of the animals in early v late trials

@author: Dinghao Luo
"""


#%% imports
from pathlib import Path 
import pickle 
import numpy as np 
import matplotlib.pyplot as plt  
from scipy.stats import sem, ttest_rel

from common import mpl_formatting
from plotting_functions import plot_violin_with_scatter
mpl_formatting()

import rec_list
pathHPCLCopt = rec_list.pathHPCLCopt
pathHPCLCtermopt = rec_list.pathHPCLCtermopt
paths = pathHPCLCopt + pathHPCLCtermopt


#%% parameters 
XAXIS = np.arange(4000) / 1000


#%% path stems 
HPCLCstem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/HPCLC')
HPCLCtermstem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/HPCLCterm')


#%% main
early_profs = []
late_profs = []

early_means = []
late_means = []

early_acc = []
late_acc = []

for path in paths:
    recname = Path(path).name
    print(recname)
    
    beh_paths = [HPCLCstem / f'{recname}.pkl', HPCLCtermstem / f'{recname}.pkl']
    try:
        with open(beh_paths[0], 'rb') as f:
            beh = pickle.load(f)
    except FileNotFoundError:
        with open(beh_paths[1], 'rb') as f:
            beh = pickle.load(f)
    
    lick_times = beh['lick_times_aligned'][1:]
    tot_trials = len(lick_times)
    
    # get first-lick time
    first_licks = []
    for trial in range(tot_trials):
        if isinstance(lick_times[trial], list):
            lk = [l/1000 for l in lick_times[trial] 
                  if l/1000 > 0.5]  # only if the animal does not lick in the first half a second (carry-over licks)
        else:
            lk = []
        
        if len(lk)==0:  # no licks in the current trial
            first_licks.append(np.nan)
        else:  # if there are licks, append relative time of first lick
            first_licks.append(lk[0])
            
    speed_times = beh['speed_times_aligned'][1:]
    
    # compute mean speed and acceleration
    earlyp = []
    latep = []
    
    earlys = []
    lates = []
    
    earlya = []
    latea = []
    
    for trial in range(tot_trials):
        t = first_licks[trial]
        if np.isnan(t):
            continue
        
        speeds = [pt[1] for pt in speed_times[trial]]
        if len(speeds) < 1000:
            continue  # skip too-short trials (which have likely been cut off)
        
        mean_speed = np.mean(speeds)
        accel = speeds[999] - speeds[0]  # Δv over 1s

        if t < 2.5:
            if len(speeds)>4000:
                earlyp.append(speeds[:4000])
            else:
                earlyp.append(np.pad(speeds, (0, 4000-len(speeds)), constant_values=0))
            earlys.append(mean_speed)
            earlya.append(accel)
        if 2.5 < t < 3.5:
            if len(speeds)>4000:
                latep.append(speeds[:4000])
            else:
                latep.append(np.pad(speeds, (0, 4000-len(speeds)), constant_values=0))
            lates.append(mean_speed)
            latea.append(accel)
    
    if len(earlya) > 10:
        early_profs.append(np.mean(earlyp, axis=0))
        early_means.append(np.mean(earlys))
        early_acc.append(np.mean(earlya))
        
    if len(latea) > 10:
        late_profs.append(np.mean(latep, axis=0))
        late_means.append(np.mean(lates))
        late_acc.append(np.mean(latea))
        

#%% filtering  
clean_early_means = []
clean_late_means = []

for em, lm in zip(early_means, late_means):
    if not np.isnan(em) and not np.isnan(lm):
        clean_early_means.append(em)
        clean_late_means.append(lm)

early_means = clean_early_means
late_means = clean_late_means

clean_early_acc = []
clean_late_acc = []

for em, lm in zip(early_acc, late_acc):
    if not np.isnan(em) and not np.isnan(lm):
        clean_early_acc.append(em)
        clean_late_acc.append(lm)

early_acc = clean_early_acc
late_acc = clean_late_acc


#%% plotting 
early_mean_prof = np.mean(np.array(early_profs), axis=0)
late_mean_prof = np.mean(np.array(late_profs), axis=0)

early_sem = sem(np.array(early_profs), axis=0)
late_sem = sem(np.array(late_profs), axis=0)


fig, ax = plt.subplots(figsize=(3, 3))

ax.plot(XAXIS, early_mean_prof, label='<2.5')
ax.fill_between(XAXIS, early_mean_prof - early_sem, early_mean_prof + early_sem, alpha=0.2)

ax.plot(XAXIS, late_mean_prof, label='2.5~3.5')
ax.fill_between(XAXIS, late_mean_prof - late_sem, late_mean_prof + late_sem, alpha=0.2)

ax.legend(frameon=False)

ax.set(xlabel='Time from run onset (s)', xlim=(0, 4),
       ylabel='Speed (cm/s)')


#%% stats 
plot_violin_with_scatter(early_means, late_means, 
                         colour0='grey', colour1=(0.20, 0.35, 0.65),
                         showscatter=False,
                         xlabel='Speed (cm/s)',
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\all_run_onset_speed_violin')

stat, p = ttest_rel(early_means, late_means)
print(
      f'early speed mean = {np.mean(early_means)}, sem = {sem(early_means)}\n'
      f'late speed mean = {np.mean(late_means)}, sem = {sem(late_means)}\n'
      f'p = {p}'
      )


plot_violin_with_scatter(early_acc, late_acc, 
                         colour0='grey', colour1=(0.20, 0.35, 0.65),
                         showscatter=False,
                         save=True,
                         xlabel='init. acc. (cm/s2)',
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\all_run_onset_acc_boxplot')

stat, p = ttest_rel(early_acc, late_acc)
print(
      f'early acc mean = {np.mean(early_means)}, sem = {sem(early_means)}\n'
      f'late acc mean = {np.mean(late_means)}, sem = {sem(late_means)}\n'
      f'p = {p}'
      )