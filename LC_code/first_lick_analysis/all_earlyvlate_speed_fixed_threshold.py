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


#%% parameters 
XAXIS = np.arange(4000) / 1000


#%% main
early_profs = []
mid_profs = []
late_profs = []
verylate_profs = []

early_means = []
mid_means = []
late_means = []
verylate_means = []

early_acc = []
mid_acc = []
late_acc = []
verylate_acc = []

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
    midp = []
    latep = []
    verylatep = []
    
    earlys = []
    mids = []
    lates = []
    verylates = []
    
    earlya = []
    mida = []
    latea = []
    verylatea = []
    
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
                midp.append(speeds[:4000])
            else:
                midp.append(np.pad(speeds, (0, 4000-len(speeds)), constant_values=0))
            mids.append(mean_speed)
            mida.append(accel)
        if 3.5 < t < 4.5:
            if len(speeds)>4000:
                latep.append(speeds[:4000])
            else:
                latep.append(np.pad(speeds, (0, 4000-len(speeds)), constant_values=0))
            lates.append(mean_speed)
            latea.append(accel)
        if t > 4.5:
            if len(speeds)>4000:
                verylatep.append(speeds[:4000])
            else:
                verylatep.append(np.pad(speeds, (0, 4000-len(speeds)), constant_values=0))
            verylates.append(mean_speed)
            verylatea.append(accel)
    
    if len(earlya) > 10:
        early_profs.append(np.mean(earlyp, axis=0))
        early_means.append(np.mean(earlys))
        early_acc.append(np.mean(earlya))
        
    if len(mida) > 10:
        mid_profs.append(np.mean(midp, axis=0))
        mid_means.append(np.mean(mids))
        mid_acc.append(np.mean(mida))
        
    if len(latea) > 10:
        late_profs.append(np.mean(latep, axis=0))
        late_means.append(np.mean(lates))
        late_acc.append(np.mean(latea))
        
    if len(verylatea) > 10:
        verylate_profs.append(np.mean(verylatep, axis=0))
        verylate_means.append(np.mean(verylates))
        verylate_acc.append(np.mean(verylatea))        
    

#%% filtering  
clean_early_means = []
clean_mid_means = []
clean_late_means = []
clean_verylate_means = []

for em, emm, mlm, lm in zip(early_means, mid_means, late_means, verylate_means):
    if not np.isnan(em) and not np.isnan(emm) and not np.isnan(mlm) and not np.isnan(lm):
        clean_early_means.append(em)
        clean_mid_means.append(emm)
        clean_late_means.append(mlm)
        clean_verylate_means.append(lm)

early_means = clean_early_means
mid_means = clean_mid_means
late_means = clean_late_means
verylate_means = clean_verylate_means


clean_early_acc = []
clean_mid_acc = []
clean_late_acc = []
clean_verylate_acc = []

for em, emm, mlm, lm in zip(early_acc, mid_acc, late_acc, verylate_acc):
    if not np.isnan(em) and not np.isnan(emm) and not np.isnan(mlm) and not np.isnan(lm):
        clean_early_acc.append(em)
        clean_mid_acc.append(emm)
        clean_late_acc.append(mlm)
        clean_verylate_acc.append(lm)

early_acc = clean_early_acc
mid_acc = clean_mid_acc
late_acc = clean_late_acc
verylate_acc = clean_verylate_acc


#%% plotting 
early_mean_prof = np.mean(np.array(early_profs), axis=0)
mid_mean_prof = np.mean(np.array(mid_profs), axis=0)
late_mean_prof = np.mean(np.array(late_profs), axis=0)
verylate_mean_prof = np.mean(np.array(verylate_profs), axis=0)

early_sem = sem(np.array(early_profs), axis=0)
mid_sem = sem(np.array(mid_profs), axis=0)
late_sem = sem(np.array(late_profs), axis=0)
verylate_sem = sem(np.array(verylate_profs), axis=0)


fig, ax = plt.subplots(figsize=(3, 3))

ax.plot(XAXIS, early_mean_prof, label='<2.5')
ax.fill_between(XAXIS, early_mean_prof - early_sem, early_mean_prof + early_sem, alpha=0.2)

ax.plot(XAXIS, mid_mean_prof, label='2.5~3.5')
ax.fill_between(XAXIS, mid_mean_prof - mid_sem, mid_mean_prof + mid_sem, alpha=0.2)

ax.plot(XAXIS, late_mean_prof, label='3.5~4.5')
ax.fill_between(XAXIS, late_mean_prof - late_sem, late_mean_prof + late_sem, alpha=0.2)

ax.plot(XAXIS, verylate_mean_prof, label='>4.5')
ax.fill_between(XAXIS, verylate_mean_prof - verylate_sem, verylate_mean_prof + verylate_sem, alpha=0.2)

ax.legend(frameon=False)

ax.set(xlabel='time from run onset (s)', xlim=(0, 4),
       ylabel='speed (cm/s)', ylim=(0, max(mid_mean_prof + mid_sem)+5))


#%% stats 
plot_box_with_scatter(early_means, verylate_means, 
                      ctrl_color='grey', stim_color=(0.10, 0.25, 0.40),
                      show_scatter=False,
                      xlabel='speed (cm/s)',
                      savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\all_run_onset_speed_boxplot')

plot_box_with_scatter(early_acc, verylate_acc, 
                      ctrl_color='grey', stim_color=(0.10, 0.25, 0.40),
                      show_scatter=False,
                      xlabel='init. acc. (cm/s2)',
                      savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis\all_run_onset_acc_boxplot')