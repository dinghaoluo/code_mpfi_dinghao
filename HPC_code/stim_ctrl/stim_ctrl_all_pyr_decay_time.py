# -*- coding: utf-8 -*-
"""
Created on Mon Mar  27 16:08:34 2025

analyse the decay time constants of pyramidal cells in stim. vs ctrl. trials 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import pandas as pd

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\HPC_code\decay_time')
from decay_time_analysis import detect_peak, compute_tau

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from plotting_functions import plot_violin_with_scatter, plot_ecdfs


#%% parameters 
SAMP_FREQ = 1250  # Hz
TIME = np.arange(-SAMP_FREQ*3, SAMP_FREQ*7)/SAMP_FREQ  # 10 seconds 

MAX_TIME = 10  # collect (for each trial) a maximum of 10 s of spiking-profile
MAX_SAMPLES = SAMP_FREQ * MAX_TIME


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


#%% main 
tau_values_ctrl_ON = []
tau_values_stim_ON = []
tau_values_ctrl_OFF = []
tau_values_stim_OFF = []

mean_prof_ctrl_ON = []
mean_prof_stim_ON = []
mean_prof_ctrl_OFF = []
mean_prof_stim_OFF = []

peak_ctrl_ON = []
peak_stim_ON = []
peak_ctrl_OFF = []
peak_stim_OFF = []

spike_rate_ON = []
spike_rate_OFF = []

for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    
    trains = np.load(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}\{recname}_all_trains.npy',
        allow_pickle=True
        ).item()
    curr_beh_df = beh_df.loc[recname]  # subselect in read-only
    stim_conds = [t[15] for t in curr_beh_df['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds)
                if cond!='0']
    ctrl_idx = [trial+2 for trial in stim_idx]
    baseline_idx = list(np.arange(stim_idx[0]))
    
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    
    for idx, session in curr_df_pyr.iterrows():
        
        cluname = idx
                    
        if session['pre_post']<=2/3:
            
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_ctrl_ON.append(mean_prof_ctrl)
            mean_prof_stim_ON.append(mean_prof_stim)
            spike_rate_ON.append(session['spike_rate'])
            
            peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                        session['class'],
                                        run_onset_bin=3750)
            tau_ctrl, fit_params_ctrl = compute_tau(
                TIME, mean_prof_ctrl, peak_idx_ctrl, session['class']
                )
        
            peak_idx_stim = detect_peak(mean_prof_stim, 
                                        session['class'],
                                        run_onset_bin=3750)
            tau_stim, fit_params_stim = compute_tau(
                TIME, mean_prof_stim, peak_idx_stim, session['class']
                )
            
            peak_ctrl_ON.append(peak_idx_ctrl)
            peak_stim_ON.append(peak_idx_stim)
            
            if (fit_params_ctrl['adj_r_squared']>0.6 and 
                fit_params_stim['adj_r_squared']>0.6):
                tau_values_ctrl_ON.append(tau_ctrl)
                tau_values_stim_ON.append(tau_stim)
                
        if session['pre_post']>=3/2:
            
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_ctrl_OFF.append(mean_prof_ctrl)
            mean_prof_stim_OFF.append(mean_prof_stim)
            spike_rate_OFF.append(session['spike_rate'])
            
            peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                        session['class'],
                                        run_onset_bin=3750)
        
            tau_ctrl, fit_params_ctrl = compute_tau(
                TIME, mean_prof_ctrl, peak_idx_ctrl, session['class']
                )
                        
            peak_idx_stim = detect_peak(mean_prof_stim, 
                                        session['class'],
                                        run_onset_bin=3750)
        
            tau_stim, fit_params_stim = compute_tau(
                TIME, mean_prof_stim, peak_idx_stim, session['class']
                )
            
            peak_ctrl_OFF.append(peak_idx_ctrl)
            peak_stim_OFF.append(peak_idx_stim)
            
            if (fit_params_ctrl['adj_r_squared']>0.6 and 
                fit_params_stim['adj_r_squared']>0.6):
                tau_values_ctrl_OFF.append(tau_ctrl)
                tau_values_stim_OFF.append(tau_stim)
                

#%% plotting 
plot_violin_with_scatter(tau_values_ctrl_ON, tau_values_stim_ON, 
                         'grey', 'royalblue',
                         xticklabels=['ctrl.', 'stim.'],
                         ylabel='τ (s)',
                         title='run-onset ON',
                         paired=True,
                         showmeans=True,
                         showmedians=False,
                         showscatter=False,
                         save=False,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\ON_decay_constant_ctrl_stim')


#%% cdfs 
plot_ecdfs(tau_values_ctrl_ON, tau_values_stim_ON,
           title='ECDF – run-onset ON',
           xlabel='τ (s)', 
           legend_labels=['ctrl.', 'stim.'],
           colours=['lightcoral', 'firebrick'],
           save=False,
           savepath='Z:/.../ecdf_run_onset_ON')


#%% mean profile 
mean_ctrl_ON = np.mean(mean_prof_ctrl_ON, axis=0)[2500:2500+5*1250]
mean_stim_ON = np.mean(mean_prof_stim_ON, axis=0)[2500:2500+5*1250]
mean_ctrl_OFF = np.mean(mean_prof_ctrl_OFF, axis=0)[2500:2500+5*1250]
mean_stim_OFF = np.mean(mean_prof_stim_OFF, axis=0)[2500:2500+5*1250]
xaxis = np.arange(-1*1250, 4*1250) / 1250

fig, ax = plt.subplots(figsize=(3.4,2.4))

ax.plot(xaxis, mean_ctrl_ON, label='mean_ctrl_ON', color='grey')
ax.plot(xaxis, mean_stim_ON, label='mean_stim_ON', color='royalblue')

# ax.set(xlabel='time from run-onset (s)',
#        ylabel='spike rate (Hz)')
# ax.legend()

# fig, ax = plt.subplots(figsize=(3,2))

ax.plot(xaxis, mean_ctrl_OFF, label='mean_ctrl_OFF', color='grey')
ax.plot(xaxis, mean_stim_OFF, label='mean_stim_OFF', color='royalblue')

ax.set(xlabel='time from run-onset (s)',
       ylabel='spike rate (Hz)')
ax.set_title('baseline-defined PyrUp (<2/3) and PyrDown (>3/2)', fontsize=10)
ax.legend(fontsize=6, frameon=False)


#%% peak 
plot_violin_with_scatter([s/1250-3 for s in peak_ctrl_ON], 
                         [s/1250-3 for s in peak_stim_ON], 
                         'grey', 'royalblue',
                         xticklabels=['ctrl.', 'stim.'],
                         ylabel='peak (s from run-onset)',
                         title='run-onset ON',
                         paired=True,
                         showline=False,
                         showmeans=True,
                         showmedians=False,
                         showscatter=True,
                         save=False,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\ON_decay_constant_ctrl_stim')

#%% plot rate 
fig, ax = plt.subplots(figsize=(3,3))

ax.hist(spike_rate_ON, bins=100, edgecolor='k', linewidth=.25)
ax.set(xlabel='spike rate (Hz)', ylabel='freq. of occur.',
       title='PyrUp spike rate')

fig, ax = plt.subplots(figsize=(3,3))

ax.hist(spike_rate_OFF, bins=100, edgecolor='k', linewidth=.25)
ax.set(xlabel='spike rate (Hz)', ylabel='freq. of occur.',
       title='PyrDown spike rate')