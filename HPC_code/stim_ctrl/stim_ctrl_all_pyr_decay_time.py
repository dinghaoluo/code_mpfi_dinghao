# -*- coding: utf-8 -*-
"""
Created on Mon Mar  27 16:08:34 2025

analyse the decay time constants of pyramidal cells in stim. vs ctrl. trials 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
import sys 
from scipy.stats import sem
import os 
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


#%% main 
tau_values_ctrl_ON = []
tau_values_stim_ON = []
tau_values_ctrl_OFF = []
tau_values_stim_OFF = []

tau_values_ctrl_new_ON = []
tau_values_stim_new_ON = []
tau_values_ctrl_new_OFF = []
tau_values_stim_new_OFF = []

tau_values_ctrl_only_ON = [] 
tau_values_stim_only_ON = []
tau_values_ctrl_only_OFF = []
tau_values_stim_only_OFF = []

mean_prof_ctrl_ON = []
mean_prof_stim_ON = []
mean_prof_ctrl_OFF = []
mean_prof_stim_OFF = []

mean_prof_ctrl_new_ON = []
mean_prof_stim_new_ON = []
mean_prof_ctrl_new_OFF = []
mean_prof_stim_new_OFF = []

mean_prof_ctrl_only_ON = []
mean_prof_stim_only_ON = []
mean_prof_ctrl_only_OFF = []
mean_prof_stim_only_OFF = []

peak_ctrl_ON = []
peak_stim_ON = []
peak_ctrl_OFF = []
peak_stim_OFF = []

peak_ctrl_new_ON = []
peak_stim_new_ON = []
peak_ctrl_new_OFF = []
peak_stim_new_OFF = []

peak_ctrl_only_ON = []
peak_stim_only_ON = []
peak_ctrl_only_OFF = []
peak_stim_only_OFF = []

for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    
    trains = np.load(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}\{recname}_all_trains.npy',
        allow_pickle=True
        ).item()
    
    if os.path.exists(
            rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl'
            ):
        with open(
                rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl',
                'rb'
                ) as f:
            beh = pickle.load(f)
    else:
        with open(
                rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLCterm\{recname}.pkl',
                'rb'
                ) as f:
            beh = pickle.load(f)
    
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds)
                if cond!='0']
    ctrl_idx = [trial+2 for trial in stim_idx]
    baseline_idx = list(np.arange(stim_idx[0]))
    
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    
    for idx, session in curr_df_pyr.iterrows():
        
        cluname = idx
                    
        if session['class']=='run-onset ON':
            
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_ctrl_ON.append(mean_prof_ctrl)
            mean_prof_stim_ON.append(mean_prof_stim)
            
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
        
        if session['class']!='run-onset ON' and session['class_stim']=='run-onset ON':
            
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_ctrl_new_ON.append(mean_prof_ctrl)
            mean_prof_stim_new_ON.append(mean_prof_stim)
            
            peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                        session['class_stim'],
                                        run_onset_bin=3750)
            tau_ctrl, fit_params_ctrl = compute_tau(
                TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_stim']
                )
                        
            peak_idx_stim = detect_peak(mean_prof_stim, 
                                        session['class_stim'],
                                        run_onset_bin=3750)

            tau_stim, fit_params_stim = compute_tau(
                TIME, mean_prof_stim, peak_idx_stim, session['class_stim']
                )
            
            peak_ctrl_new_ON.append(peak_idx_ctrl)
            peak_stim_new_ON.append(peak_idx_stim)
            
            if fit_params_stim['adj_r_squared']>0.6:
                tau_values_ctrl_new_ON.append(tau_ctrl)
                tau_values_stim_new_ON.append(tau_stim)
        
        if session['class']=='run-onset OFF':
            
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_ctrl_OFF.append(mean_prof_ctrl)
            mean_prof_stim_OFF.append(mean_prof_stim)
            
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
                
        if session['class']!='run-onset OFF' and session['class_stim']=='run-onset OFF':
            
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_ctrl_new_OFF.append(mean_prof_ctrl)
            mean_prof_stim_new_OFF.append(mean_prof_stim)
            
            peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                        session['class_stim'],
                                        run_onset_bin=3750)
            tau_ctrl, fit_params_ctrl = compute_tau(
                TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_stim']
                )
                        
            peak_idx_stim = detect_peak(mean_prof_stim, 
                                        session['class_stim'],
                                        run_onset_bin=3750)

            tau_stim, fit_params_stim = compute_tau(
                TIME, mean_prof_stim, peak_idx_stim, session['class_stim']
                )
            
            peak_ctrl_new_OFF.append(peak_idx_ctrl)
            peak_stim_new_OFF.append(peak_idx_stim)
            
            if fit_params_stim['adj_r_squared']>0.6:
                tau_values_ctrl_new_OFF.append(tau_ctrl)
                tau_values_stim_new_OFF.append(tau_stim)
                
                
        ## ctrl and stim only 
        if session['class_ctrl']=='run-onset ON':
            
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_ctrl_only_ON.append(mean_prof_ctrl)
            
            peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                        session['class_ctrl'],
                                        run_onset_bin=3750)
            tau_ctrl, fit_params_ctrl = compute_tau(
                TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_ctrl']
                )
            
            peak_ctrl_only_ON.append(peak_idx_ctrl)
            
            # if fit_params_ctrl['adj_r_squared']>0.6:
            tau_values_ctrl_only_ON.append(tau_ctrl)
                
        if session['class_stim']=='run-onset ON':
            
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_stim_only_ON.append(mean_prof_stim)
            
            peak_idx_stim = detect_peak(mean_prof_stim, 
                                        session['class_stim'],
                                        run_onset_bin=3750)
            tau_stim, fit_params_stim = compute_tau(
                TIME, mean_prof_stim, peak_idx_stim, session['class_stim']
                )
            
            peak_stim_only_ON.append(peak_idx_stim)
            
            # if fit_params_stim['adj_r_squared']>0.6:
            tau_values_stim_only_ON.append(tau_stim)
                
        if session['class_ctrl']=='run-onset OFF':
            
            mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
            mean_prof_ctrl_only_OFF.append(mean_prof_ctrl)
            
            peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                        session['class_ctrl'],
                                        run_onset_bin=3750)
            tau_ctrl, fit_params_ctrl = compute_tau(
                TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_ctrl']
                )
            
            peak_ctrl_only_OFF.append(peak_idx_ctrl)
            
            # if fit_params_ctrl['adj_r_squared']>0.6:
            tau_values_ctrl_only_OFF.append(tau_ctrl)
                
        if session['class_stim']=='run-onset OFF':
            
            mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
            mean_prof_stim_only_OFF.append(mean_prof_stim)
            
            peak_idx_stim = detect_peak(mean_prof_stim, 
                                        session['class_stim'],
                                        run_onset_bin=3750)
            tau_stim, fit_params_stim = compute_tau(
                TIME, mean_prof_stim, peak_idx_stim, session['class_stim']
                )
            
            peak_stim_only_OFF.append(peak_idx_stim)
            
            # if fit_params_stim['adj_r_squared']>0.6:
            tau_values_stim_only_OFF.append(tau_stim)
                

#%% plotting 
plot_violin_with_scatter(tau_values_ctrl_ON, tau_values_stim_new_ON, 
                         'lightcoral', 'darkorange',
                         xticklabels=['pers.\nON (ctrl.)', 'stim.-\ninduced ON'],
                         ylabel='τ (s)',
                         title='run-onset ON',
                         paired=False,
                         showmeans=True,
                         showmedians=False,
                         showscatter=True,
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\new_ON_decay_constant_ctrl_stim')

plot_violin_with_scatter(tau_values_stim_ON, tau_values_stim_new_ON, 
                         'firebrick', 'darkorange',
                         xticklabels=['pers.\nON (stim.)', 'stim.-\ninduced ON'],
                         ylabel='τ (s)',
                         title='run-onset ON',
                         paired=False,
                         showmeans=True,
                         showmedians=False,
                         showscatter=True,
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\new_ON_decay_constant_stim_stim')


#%% cdfs 
plot_ecdfs(tau_values_stim_ON, tau_values_stim_new_ON,
           title='ECDF – run-onset ON',
           xlabel='τ (s)', 
           legend_labels=['pers. ON', 'new ON'],
           colours=['firebrick', 'darkorange'],
           save=True,
           savepath=r'C:\Users\luod\OneDrive - Max Planck Florida Institute for Neuroscience\Desktop\persistentandNewON.png')


#%% mean profile 
# extract mean and SEM for all 8 traces
mean_ctrl_ON = np.mean(mean_prof_ctrl_ON, axis=0)[2500:2500+5*1250]
sem_ctrl_ON = sem(mean_prof_ctrl_ON, axis=0)[2500:2500+5*1250]

mean_stim_ON = np.mean(mean_prof_stim_ON, axis=0)[2500:2500+5*1250]
sem_stim_ON = sem(mean_prof_stim_ON, axis=0)[2500:2500+5*1250]

mean_ctrl_new_ON = np.mean(mean_prof_ctrl_new_ON, axis=0)[2500:2500+5*1250]
sem_ctrl_new_ON = sem(mean_prof_ctrl_new_ON, axis=0)[2500:2500+5*1250]

mean_stim_new_ON = np.mean(mean_prof_stim_new_ON, axis=0)[2500:2500+5*1250]
sem_stim_new_ON = sem(mean_prof_stim_new_ON, axis=0)[2500:2500+5*1250]

mean_ctrl_OFF = np.mean(mean_prof_ctrl_OFF, axis=0)[2500:2500+5*1250]
sem_ctrl_OFF = sem(mean_prof_ctrl_OFF, axis=0)[2500:2500+5*1250]

mean_stim_OFF = np.mean(mean_prof_stim_OFF, axis=0)[2500:2500+5*1250]
sem_stim_OFF = sem(mean_prof_stim_OFF, axis=0)[2500:2500+5*1250]

mean_ctrl_new_OFF = np.mean(mean_prof_ctrl_new_OFF, axis=0)[2500:2500+5*1250]
sem_ctrl_new_OFF = sem(mean_prof_ctrl_new_OFF, axis=0)[2500:2500+5*1250]

mean_stim_new_OFF = np.mean(mean_prof_stim_new_OFF, axis=0)[2500:2500+5*1250]
sem_stim_new_OFF = sem(mean_prof_stim_new_OFF, axis=0)[2500:2500+5*1250]

# time axis
xaxis = np.arange(-1*1250, 4*1250) / 1250

# plot
fig, ax = plt.subplots(figsize=(2.6,2))

# ax.plot(xaxis, mean_ctrl_ON, label='mean_ctrl_ON', color='lightcoral')
# ax.fill_between(xaxis, mean_ctrl_ON + sem_ctrl_ON, mean_ctrl_ON - sem_ctrl_ON,
#                 color='lightcoral', edgecolor='none', alpha=.15)

# ax.plot(xaxis, mean_stim_ON, label='mean_stim_ON', color='firebrick')
# ax.fill_between(xaxis, mean_stim_ON + sem_stim_ON, mean_stim_ON - sem_stim_ON,
#                 color='firebrick', edgecolor='none', alpha=.15)

# ax.plot(xaxis, mean_ctrl_new_ON, label='mean_ctrl_new_ON', color='moccasin')
# ax.fill_between(xaxis, mean_ctrl_new_ON + sem_ctrl_new_ON, mean_ctrl_new_ON - sem_ctrl_new_ON,
#                 color='moccasin', edgecolor='none', alpha=.15)

# ax.plot(xaxis, mean_stim_new_ON, label='mean_stim_new_ON', color='darkorange')
# ax.fill_between(xaxis, mean_stim_new_ON + sem_stim_new_ON, mean_stim_new_ON - sem_stim_new_ON,
#                 color='darkorange', edgecolor='none', alpha=.15)

ax.plot(xaxis, mean_ctrl_OFF, label='mean_ctrl_OFF', color='violet')
ax.fill_between(xaxis, mean_ctrl_OFF + sem_ctrl_OFF, mean_ctrl_OFF - sem_ctrl_OFF,
                color='violet', edgecolor='none', alpha=.15)

ax.plot(xaxis, mean_stim_OFF, label='mean_stim_OFF', color='purple')
ax.fill_between(xaxis, mean_stim_OFF + sem_stim_OFF, mean_stim_OFF - sem_stim_OFF,
                color='purple', edgecolor='none', alpha=.15)

ax.plot(xaxis, mean_ctrl_new_OFF, label='mean_ctrl_new_OFF', color='lightcyan')
ax.fill_between(xaxis, mean_ctrl_new_OFF + sem_ctrl_new_OFF, mean_ctrl_new_OFF - sem_ctrl_new_OFF,
                color='lightcyan', edgecolor='none', alpha=.15)

ax.plot(xaxis, mean_stim_new_OFF, label='mean_stim_new_OFF', color='darkblue')
ax.fill_between(xaxis, mean_stim_new_OFF + sem_stim_new_OFF, mean_stim_new_OFF - sem_stim_new_OFF,
                color='darkblue', edgecolor='none', alpha=.15)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
       ylabel='spike rate (Hz)', yticks=[1, 2, 3, 4], ylim=(.9, 4.1))
ax.set_title('baseline PyrUp (<2/3) and PyrDown (>3/2)', fontsize=10)
ax.legend(fontsize=4, frameon=False)

fig.savefig(r'C:\Users\luod\OneDrive - Max Planck Florida Institute for Neuroscience\Desktop\persistentandNewOFF.png',
            dpi=300, bbox_inches='tight')



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


#%% plot ctrl and stim only 
# extract mean and SEM
mean_ctrl_only_ON = np.mean(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
sem_ctrl_only_ON = sem(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]

mean_stim_only_ON = np.mean(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
sem_stim_only_ON = sem(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]

# plot
fig, ax = plt.subplots(figsize=(2.6,2))

ax.plot(xaxis, mean_ctrl_only_ON, label='mean_ctrl_ON', color='lightcoral')
ax.fill_between(xaxis, mean_ctrl_only_ON + sem_ctrl_only_ON, mean_ctrl_only_ON - sem_ctrl_only_ON,
                color='lightcoral', edgecolor='none', alpha=.15)

ax.plot(xaxis, mean_stim_only_ON, label='mean_stim_ON', color='firebrick')
ax.fill_between(xaxis, mean_stim_only_ON + sem_stim_only_ON, mean_stim_only_ON - sem_stim_only_ON,
                color='firebrick', edgecolor='none', alpha=.15)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
       ylabel='spike rate (Hz)', yticks=[1, 2, 3, 4], ylim=(.9, 4.1))
ax.set_title('baseline PyrUp (<2/3) and PyrDown (>3/2)', fontsize=10)
ax.legend(fontsize=4, frameon=False)

fig.savefig(r'C:\Users\luod\OneDrive - Max Planck Florida Institute for Neuroscience\Desktop\ctrl_stim_only_ON.png',
            dpi=300, bbox_inches='tight')


plot_ecdfs(tau_values_ctrl_only_ON, tau_values_stim_only_ON,
           title='ECDF – run-onset ON',
           xlabel='τ (s)', 
           legend_labels=['ctrl. ON', 'stim. ON'],
           colours=['lightcoral', 'firebrick'],
           save=True,
           savepath=r'C:\Users\luod\OneDrive - Max Planck Florida Institute for Neuroscience\Desktop\ctrl_stim_only_ON_ecdf.png')