# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:37:16 2023
Modified on Mon 10 Mar 17:45:12 2025:
    modified to work on HPC cells

loop over all cells for early v late trials

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd 
import sys 

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
# paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt
paths = rec_list.pathHPCLCopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from plotting_functions import plot_violin_with_scatter

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\HPC_code\decay_time')
from decay_time_analysis import detect_min_max, compute_tau, plot_fit_compare


#%% parameters 
SAMP_FREQ = 1250  # Hz
TIME = np.arange(-SAMP_FREQ, SAMP_FREQ*6)/SAMP_FREQ  # 7 seconds 

early_colour = (.804, .267, .267)  # early trials
late_colour = (.545, 0, 0)  # late trials


#%% load data 
print('loading dataframes...')

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
    )
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 

beh_df = pd.concat((
    pd.read_pickle(
        r'Z:/Dinghao/code_dinghao/behaviour/all_HPCLC_sessions.pkl'
        ),
    pd.read_pickle(
        r'Z:/Dinghao/code_dinghao/behaviour/all_HPCLCterm_sessions.pkl'
        )
    ))


#%% main loop 
early_ON_profs = []
late_ON_profs = []
early_OFF_profs = []
late_OFF_profs = []
early_mid_ON_profs = []
late_mid_ON_profs = []
early_mid_OFF_profs = []
late_mid_OFF_profs = []

tau_values_early = []
fit_results_early = []
tau_values_late = []
fit_results_late = []

for path in paths:
    recname = path[-17:]
    
    # get lick times 
    curr_beh_df = beh_df.loc[recname]  # subselect in read-only
    run_onsets = curr_beh_df['run_onsets'][1:]
    licks = [
        [(l-run_onset)/1000 for l in trial]  # convert from ms to s
        if len(trial)!=0 else np.nan
        for trial, run_onset in zip(
                curr_beh_df['lick_times'][1:],
                run_onsets
                )
        ]
    first_licks = np.asarray(
        [next((l for l in trial if l > 1), np.nan)  # >1 to prevent carry-over licks
        if isinstance(trial, list) else np.nan
        for trial in licks]
        )
    
    # get bad trials 
    behPar = sio.loadmat(
        rf'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}/{recname[:14]}/{recname}'
        rf'/{recname}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
        )
    bad_beh_ind = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    
    # get early and late lick trials (that are not stim. trials)
    stim_trials = np.where(
        np.asarray([
            trial[15] for trial
            in curr_beh_df['trial_statements']
            ])!='0'
        )[0]
    valid_trials = [i for i in range(len(first_licks)) 
                    if i not in stim_trials 
                    and i not in bad_beh_ind 
                    and not np.isnan(first_licks[i])]
    
    if len(valid_trials) < 50:
        continue 
    
    # load spike trains
    print(f'\n{recname}')
    trains = np.load(
        r'Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions'
        rf'/{recname}/{recname}_all_trains.npy',
        allow_pickle=True
        ).item()
        
    sorted_trials = sorted(valid_trials, 
                           key=lambda i: first_licks[i])[10:-10]  # avoid extremities 
    
    early_trials = sorted_trials[:10]
    early_mid_trials = sorted_trials[:int(len(sorted_trials)/2)]
    late_trials = sorted_trials[-10:]
    late_mid_trials = sorted_trials[int(len(sorted_trials)/2):]
    
    # get cell spiking data 
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    
    for cluname, session in curr_df_pyr.iterrows():
        train = trains[cluname]
        early_trains = [
            train[trial, :] for trial
            in early_trials
            ]
        late_trains = [
            train[trial, :] for trial
            in late_trials
            ]
        early_mid_trains = [
            train[trial, :] for trial
            in early_mid_trials
            ]
        late_mid_trains = [
            train[trial, :] for trial
            in late_mid_trials
            ]
        
        early_mean = np.mean(early_trains, axis=0)
        late_mean = np.mean(late_trains, axis=0)
        
        # ON and OFF only 
        if session['class']=='run-onset ON':
            early_ON_profs.append(early_mean)
            late_ON_profs.append(late_mean)
            early_mid_ON_profs.append(np.mean(early_mid_trains, axis=0))
            late_mid_ON_profs.append(np.mean(late_mid_trains, axis=0))
        elif session['class']=='run-onset OFF':
            early_OFF_profs.append(early_mean)
            late_OFF_profs.append(late_mean)
            early_mid_OFF_profs.append(np.mean(early_mid_trains, axis=0))
            late_mid_OFF_profs.append(np.mean(late_mid_trains, axis=0))
        
        # for decay time calculation 
        if session['class']!='run-onset unresponsive':
            mean_prof_early = early_mean[SAMP_FREQ*(3-1):SAMP_FREQ*(3+6)]
            mean_prof_late = late_mean[SAMP_FREQ*(3-1):SAMP_FREQ*(3+6)]
    
            peak_idx_early = detect_min_max(mean_prof_early, session['class'])
            peak_idx_late = detect_min_max(mean_prof_late, session['class'])
        
            tau_early, fit_params_early = compute_tau(
                TIME, mean_prof_early, peak_idx_early, session['class']
                )
            tau_late, fit_params_late = compute_tau(
                TIME, mean_prof_late, peak_idx_late, session['class']
                )
            
            # early and late plot 
            plot_fit_compare(TIME,
                             mean_prof_early, peak_idx_early, fit_params_early,
                             mean_prof_late, peak_idx_late, fit_params_late,
                             cluname, session['class'])
        
            tau_values_early.append(tau_early)
            fit_results_early.append(fit_params_early)
            tau_values_late.append(tau_late)
            fit_results_late.append(fit_params_late)


#%% save for further processing 
np.save(r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\early_ON_profs.npy',
        early_ON_profs,
        allow_pickle=True)
np.save(r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\early_OFF_profs.npy',
        early_OFF_profs,
        allow_pickle=True)
np.save(r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\late_ON_profs.npy',
        late_ON_profs,
        allow_pickle=True)
np.save(r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\late_OFF_profs.npy',
        late_OFF_profs,
        allow_pickle=True)
        

#%% compute mean and sem 
early_ON_profs_mean = np.mean(early_ON_profs, axis=0)
late_ON_profs_mean = np.mean(late_ON_profs, axis=0)
early_OFF_profs_mean = np.mean(early_OFF_profs, axis=0)
late_OFF_profs_mean = np.mean(late_OFF_profs, axis=0)
early_mid_ON_profs_mean = np.mean(early_mid_ON_profs, axis=0)
late_mid_ON_profs_mean = np.mean(late_mid_ON_profs, axis=0)
early_mid_OFF_profs_mean = np.mean(early_mid_OFF_profs, axis=0)
late_mid_OFF_profs_mean = np.mean(late_mid_OFF_profs, axis=0)

from scipy.stats import sem
early_ON_profs_sem = sem(early_ON_profs, axis=0)
late_ON_profs_sem = sem(late_ON_profs, axis=0)
early_OFF_profs_sem = sem(early_OFF_profs, axis=0)
late_OFF_profs_sem = sem(late_OFF_profs, axis=0)
early_mid_ON_profs_sem = sem(early_mid_ON_profs, axis=0)
late_mid_ON_profs_sem = sem(late_mid_ON_profs, axis=0)
early_mid_OFF_profs_sem = sem(early_mid_OFF_profs, axis=0)
late_mid_OFF_profs_sem = sem(late_mid_OFF_profs, axis=0)

early_ON_ratios = [np.nanmean(prof[3750-1250:3750])/np.nanmean(prof[3750:3750+1250])
                   if sum(prof[3750:3750+1250])>0 else 1
                   for prof in early_ON_profs]
late_ON_ratios = [np.nanmean(prof[3750-1250:3750])/np.nanmean(prof[3750:3750+1250])
                  if sum(prof[3750:3750+1250])>0 else 1
                  for prof in late_ON_profs]

outlier_mask = [i for i in range(len(early_ON_ratios))
                if early_ON_ratios[i] > 10 or late_ON_ratios[i] > 10]


early_ON_ratios = [v for i, v in enumerate(early_ON_ratios) 
                   if i not in outlier_mask]
late_ON_ratios = [v for i, v in enumerate(late_ON_ratios) 
                  if i not in outlier_mask]

early_mid_ON_ratios = [np.nanmean(prof[3750-1250:3750])/np.nanmean(prof[3750:3750+1250])
                       if sum(prof[3750:3750+1250])>0 else 1
                       for prof in early_mid_ON_profs]
late_mid_ON_ratios = [np.nanmean(prof[3750-1250:3750])/np.nanmean(prof[3750:3750+1250])
                      if sum(prof[3750:3750+1250])>0 else 1
                      for prof in late_mid_ON_profs]

outlier_mid_mask = [i for i in range(len(early_mid_ON_ratios))
                    if early_mid_ON_ratios[i] > 5 or late_mid_ON_ratios[i] > 5]

early_mid_ON_ratios = [v for i, v in enumerate(early_mid_ON_ratios) 
                       if i not in outlier_mid_mask]
late_mid_ON_ratios = [v for i, v in enumerate(late_mid_ON_ratios) 
                      if i not in outlier_mid_mask]

plot_violin_with_scatter(early_mid_ON_ratios, late_mid_ON_ratios, 'orange', 'darkred')


#%% decay time analysis 
tau_values_early, tau_values_late = zip(
    *[(x, y) for x, y in zip(tau_values_early, tau_values_late) 
      if x is not None and y is not None]
    )

tau_values_early_ON, tau_values_late_ON = zip(
    *[(x, y) for x, y in zip(tau_values_early, tau_values_late)
      if x > 0 and y > 0]
    )
tau_values_early_OFF, tau_values_late_OFF = zip(
    *[(x, y) for x, y in zip(tau_values_early, tau_values_late)
      if x < 0 and y < 0]
    )

plot_violin_with_scatter(tau_values_early_ON, tau_values_late_ON, 
                         'lightcoral', 'firebrick',
                         xticklabels=['early\n$1^{st}$-lick', 'late\n$1^{st}$-lick'],
                         ylabel='τ (s)',
                         title='run-onset ON',
                         showmeans=True,
                         showmedians=False,
                         showscatter=False,
                         ylim=(0,80),
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\ON_decay_constant')

plot_violin_with_scatter(tau_values_early_OFF, tau_values_late_OFF, 
                         'thistle', 'purple',
                         xticklabels=['early\n$1^{st}$-lick', 'late\n$1^{st}$-lick'],
                         ylabel='τ (s)',
                         title='run-onset OFF',
                         showmeans=True,
                         showmedians=False,
                         showscatter=False,
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\OFF_decay_constant')


#%% plotting 
xaxis = np.arange(-1250, 1250*4)/1250

fig, ax = plt.subplots(figsize=(3,2))

ax.plot(xaxis, 
        early_ON_profs_mean[3750-1250:3750+1250*4])
ax.plot(xaxis,
        late_ON_profs_mean[3750-1250:3750+1250*4], 
        color='red')
ax.fill_between(
    xaxis,
    early_ON_profs_mean[3750-1250:3750+1250*4]+early_ON_profs_sem[3750-1250:3750+1250*4],
    early_ON_profs_mean[3750-1250:3750+1250*4]-early_ON_profs_sem[3750-1250:3750+1250*4],
    alpha=.35
    )
ax.fill_between(
    xaxis,
    late_ON_profs_mean[3750-1250:3750+1250*4]+late_ON_profs_sem[3750-1250:3750+1250*4],
    late_ON_profs_mean[3750-1250:3750+1250*4]-late_ON_profs_sem[3750-1250:3750+1250*4],
    alpha=.35, color='red', edgecolor='none'
    )