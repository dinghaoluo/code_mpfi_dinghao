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
import sys
import os 
import pickle 
import numpy as np 
import matplotlib.pyplot as plt  
from scipy.stats import ranksums, sem

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
from plotting_functions import plot_box_with_scatter
mpl_formatting()

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC


#%% main
early_profs = []
late_profs = []

early_means = []
late_means = []

early_acc = []
late_acc = []

for path in paths:
    recname = path.split('\\')[-1]
    print(recname)
    
    beh_path = os.path.join(
        r'Z:\Dinghao\code_dinghao\behaviour\all_experiments\LC',
        f'{recname}.pkl'
        )
    with open(beh_path, 'rb') as f:
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
            continue  # skip too-short trials
        
        mean_speed = np.mean(speeds)
        accel = speeds[999] - speeds[0]  # Δv over 1s

        if .5 < t < 2.5:
            if len(speeds)>4500:
                earlyp.append(speeds[:4500])
            earlys.append(mean_speed)
            earlya.append(accel)
        elif t > 2.5:
            if len(speeds)>4500:
                latep.append(speeds[:4500])
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

for ea, la in zip(early_acc, late_acc):
    if not np.isnan(ea) and not np.isnan(la):
        clean_early_acc.append(ea)
        clean_late_acc.append(la)

early_acc = clean_early_acc
late_acc = clean_late_acc


#%% plotting 
early_mean_prof = np.mean(np.array(early_profs), axis=0)
late_mean_prof = np.mean(np.array(late_profs), axis=0)

fig, ax = plt.subplots(figsize=(3,3))

ax.plot(early_mean_prof)
ax.plot(late_mean_prof)


#%% stats 
plot_box_with_scatter(early_means, late_means, 
                      ctrl_color='grey', stim_color=(0.10, 0.25, 0.40),
                      show_scatter=False,                      
                      xlabel='speed (cm/s)',
                      savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\all_run_onset_speed_boxplot')

plot_box_with_scatter(early_acc, late_acc, 
                      ctrl_color='grey', stim_color=(0.10, 0.25, 0.40),
                      show_scatter=False,                      
                      xlabel='init. acc. (cm/s2)',
                      savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\all_run_onset_acc_boxplot')