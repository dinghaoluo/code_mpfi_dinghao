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
pathHPCLCopt = rec_list.pathHPCLCopt
pathHPCLCtermopt = rec_list.pathHPCLCtermopt

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
def main(paths, exp='HPCLC'):
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
    
    mean_prof_ctrl_only_ON_sess = []
    mean_prof_stim_only_ON_sess = []
    mean_prof_ctrl_only_OFF_sess = []
    mean_prof_stim_only_OFF_sess = []
    
    all_ctrl_stim_lick_time_delta = []
    all_ctrl_stim_lick_distance_delta = []
    
    all_amp_ON_delta_sum = []
    all_amp_ON_delta_mean = []
    
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
        
        first_lick_times = [t[0][0] - s for t, s
                            in zip(beh['lick_times'], beh['run_onsets'])
                            if t]
        ctrl_stim_lick_time_delta = np.mean(
            [t for i, t in enumerate(first_lick_times) if i in stim_idx]
            ) - np.mean(
                [t for i, t in enumerate(first_lick_times) if i in ctrl_idx]
                )
        all_ctrl_stim_lick_time_delta.append(ctrl_stim_lick_time_delta)
                
        first_lick_distances = [t[0]
                                if type(t)!=float and len(t)>0
                                else np.nan
                                for t 
                                in beh['lick_distances_aligned']
                                ][1:]
        ctrl_stim_lick_distance_delta = np.mean(
            [t for i, t in enumerate(first_lick_distances) if i in stim_idx]
            ) - np.mean(
                [t for i, t in enumerate(first_lick_distances) if i in ctrl_idx]
                )
        all_ctrl_stim_lick_distance_delta.append(ctrl_stim_lick_distance_delta)
        
        curr_df_pyr = df_pyr[df_pyr['recname']==recname]
        
        for idx, session in curr_df_pyr.iterrows():
            cluname = idx
    
            if session['class_ctrl']=='run-onset ON':
                
                mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
                mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
                mean_prof_ctrl_ON.append(mean_prof_ctrl)
                mean_prof_stim_ON.append(mean_prof_stim)
                
                peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                            session['class_ctrl'],
                                            run_onset_bin=3750)
                tau_ctrl, fit_params_ctrl = compute_tau(
                    TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_ctrl']
                    )
            
                peak_idx_stim = detect_peak(mean_prof_stim, 
                                            session['class_ctrl'],
                                            run_onset_bin=3750)
                tau_stim, fit_params_stim = compute_tau(
                    TIME, mean_prof_stim, peak_idx_stim, session['class_ctrl']
                    )
                
                peak_ctrl_ON.append(peak_idx_ctrl)
                peak_stim_ON.append(peak_idx_stim)
                
                if (fit_params_ctrl['adj_r_squared']>0.6 and 
                    fit_params_stim['adj_r_squared']>0.6):
                    tau_values_ctrl_ON.append(tau_ctrl)
                    tau_values_stim_ON.append(tau_stim)
            
            if session['class_ctrl']!='run-onset ON' and session['class_stim']=='run-onset ON':
                
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
            
            if session['class_ctrl']=='run-onset OFF':
                
                mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
                mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
                mean_prof_ctrl_OFF.append(mean_prof_ctrl)
                mean_prof_stim_OFF.append(mean_prof_stim)
                
                peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                            session['class_ctrl'],
                                            run_onset_bin=3750)
                tau_ctrl, fit_params_ctrl = compute_tau(
                    TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_ctrl']
                    )
                            
                peak_idx_stim = detect_peak(mean_prof_stim, 
                                            session['class_ctrl'],
                                            run_onset_bin=3750)
    
                tau_stim, fit_params_stim = compute_tau(
                    TIME, mean_prof_stim, peak_idx_stim, session['class_ctrl']
                    )
                
                peak_ctrl_OFF.append(peak_idx_ctrl)
                peak_stim_OFF.append(peak_idx_stim)
                
                if (fit_params_ctrl['adj_r_squared']>0.6 and 
                    fit_params_stim['adj_r_squared']>0.6):
                    tau_values_ctrl_OFF.append(tau_ctrl)
                    tau_values_stim_OFF.append(tau_stim)
                    
            if session['class_ctrl']!='run-onset OFF' and session['class_stim']=='run-onset OFF':
                
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
                
                if fit_params_ctrl['adj_r_squared']>0.6:
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
                
                if fit_params_stim['adj_r_squared']>0.6:
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
                
                if fit_params_ctrl['adj_r_squared']>0.6:
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
                
                if fit_params_stim['adj_r_squared']>0.6:
                    tau_values_stim_only_OFF.append(tau_stim)
                    
        # store per-session mean profile of ctrl-only and stim-only ON cells
        if len(mean_prof_ctrl_only_ON) > 0:
            sess_mean_ctrl = np.mean(mean_prof_ctrl_only_ON[-len(curr_df_pyr):], axis=0)
            mean_prof_ctrl_only_ON_sess.append(sess_mean_ctrl)
        
        if len(mean_prof_stim_only_ON) > 0:
            sess_mean_stim = np.mean(mean_prof_stim_only_ON[-len(curr_df_pyr):], axis=0)
            mean_prof_stim_only_ON_sess.append(sess_mean_stim)
            
        if len(mean_prof_ctrl_only_ON) > 0 and len(mean_prof_stim_only_ON) > 0:
            amp_ON_delta_sum = np.sum(sess_mean_stim[3750+625:3750+1825]) - np.mean(sess_mean_ctrl[3750+625:3750+1825])
            all_amp_ON_delta_sum.append(amp_ON_delta_sum)
            
            amp_ON_delta_mean = np.mean(sess_mean_stim[3750+625:3750+1825]) - np.mean(sess_mean_ctrl[3750+625:3750+1825])
            all_amp_ON_delta_mean.append(amp_ON_delta_mean)
    
    
    # plotting 
    plot_violin_with_scatter(tau_values_ctrl_ON, tau_values_stim_new_ON, 
                             'lightcoral', 'darkorange',
                             xticklabels=['pers.\nON (ctrl.)', 'stim.-\ninduced ON'],
                             ylabel='τ (s)',
                             title=rf'{exp}\nrun-onset ON',
                             paired=False,
                             showmeans=True,
                             showmedians=False,
                             showscatter=True,
                             save=True,
                             savepath=rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_new_ON_decay_constant_ctrl_stim')
    
    plot_violin_with_scatter(tau_values_stim_ON, tau_values_stim_new_ON, 
                             'firebrick', 'darkorange',
                             xticklabels=['pers.\nON (stim.)', 'stim.-\ninduced ON'],
                             ylabel='τ (s)',
                             title=rf'{exp}\nrun-onset ON',
                             paired=False,
                             showmeans=True,
                             showmedians=False,
                             showscatter=True,
                             save=True,
                             savepath=rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_new_ON_decay_constant_stim_stim')
    
    
    # cdfs 
    plot_ecdfs(tau_values_stim_ON, tau_values_stim_new_ON,
               title=rf'{exp}\nECDF – run-onset ON',
               xlabel='τ (s)', 
               legend_labels=['pers. ON', 'new ON'],
               colours=['firebrick', 'darkorange'],
               save=False,
               savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_new_ON_decay_constant_ecdf')
    
    
    # mean profile 
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
    
    ax.plot(xaxis, mean_ctrl_new_ON, label='mean_ctrl_new_ON', color='moccasin')
    ax.fill_between(xaxis, mean_ctrl_new_ON + sem_ctrl_new_ON, mean_ctrl_new_ON - sem_ctrl_new_ON,
                    color='moccasin', edgecolor='none', alpha=.15)
    
    ax.plot(xaxis, mean_stim_new_ON, label='mean_stim_new_ON', color='darkorange')
    ax.fill_between(xaxis, mean_stim_new_ON + sem_stim_new_ON, mean_stim_new_ON - sem_stim_new_ON,
                    color='darkorange', edgecolor='none', alpha=.15)
    
    # ax.plot(xaxis, mean_ctrl_OFF, label='mean_ctrl_OFF', color='violet')
    # ax.fill_between(xaxis, mean_ctrl_OFF + sem_ctrl_OFF, mean_ctrl_OFF - sem_ctrl_OFF,
    #                 color='violet', edgecolor='none', alpha=.15)
    
    # ax.plot(xaxis, mean_stim_OFF, label='mean_stim_OFF', color='purple')
    # ax.fill_between(xaxis, mean_stim_OFF + sem_stim_OFF, mean_stim_OFF - sem_stim_OFF,
    #                 color='purple', edgecolor='none', alpha=.15)
    
    # ax.plot(xaxis, mean_ctrl_new_OFF, label='mean_ctrl_new_OFF', color='lightcyan')
    # ax.fill_between(xaxis, mean_ctrl_new_OFF + sem_ctrl_new_OFF, mean_ctrl_new_OFF - sem_ctrl_new_OFF,
    #                 color='lightcyan', edgecolor='none', alpha=.15)
    
    # ax.plot(xaxis, mean_stim_new_OFF, label='mean_stim_new_OFF', color='darkblue')
    # ax.fill_between(xaxis, mean_stim_new_OFF + sem_stim_new_OFF, mean_stim_new_OFF - sem_stim_new_OFF,
    #                 color='darkblue', edgecolor='none', alpha=.15)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='spike rate (Hz)', yticks=[1, 2, 3, 4], ylim=(.9, 4.1))
    ax.set_title(rf'{exp}\nbaseline PyrUp (<2/3) and PyrDown (>3/2)', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_new_curves{ext}',
                    dpi=300, bbox_inches='tight')
    
    
    # plot ctrl and stim only 
    # extract mean and SEM
    mean_ctrl_only_ON = np.mean(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    sem_ctrl_only_ON = sem(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    
    mean_stim_only_ON = np.mean(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    sem_stim_only_ON = sem(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_only_OFF = np.mean(mean_prof_ctrl_only_OFF, axis=0)[2500:2500+5*1250]
    sem_ctrl_only_OFF = sem(mean_prof_ctrl_only_OFF, axis=0)[2500:2500+5*1250]
    
    mean_stim_only_OFF = np.mean(mean_prof_stim_only_OFF, axis=0)[2500:2500+5*1250]
    sem_stim_only_OFF = sem(mean_prof_stim_only_OFF, axis=0)[2500:2500+5*1250]
    
    
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
    ax.set_title(rf'{exp}\nbaseline PyrUp (<2/3) and PyrDown (>3/2)', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_only_ON.png',
                dpi=300, bbox_inches='tight')
    
    
    mean_ctrl_only_ON = np.mean(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    sem_ctrl_only_ON = sem(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    
    mean_stim_only_ON = np.mean(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    sem_stim_only_ON = sem(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    
    # plot
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(xaxis, mean_ctrl_only_OFF, label='mean_ctrl_OFF', color='violet')
    ax.fill_between(xaxis, mean_ctrl_only_OFF + sem_ctrl_only_OFF, mean_ctrl_only_OFF - sem_ctrl_only_OFF,
                    color='violet', edgecolor='none', alpha=.15)
    
    ax.plot(xaxis, mean_stim_only_OFF, label='mean_stim_OFF', color='purple')
    ax.fill_between(xaxis, mean_stim_only_OFF + sem_stim_only_OFF, mean_stim_only_OFF - sem_stim_only_OFF,
                    color='purple', edgecolor='none', alpha=.15)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='spike rate (Hz)', yticks=[1, 2, 3, 4], ylim=(.9, 4.1))
    ax.set_title(rf'{exp}\nbaseline PyrUp (<2/3) and PyrDown (>3/2)', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_only_OFF.png',
                dpi=300, bbox_inches='tight')
    
    
    # single session overlay
    save_dir = r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\single_sessions'
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(mean_prof_ctrl_only_ON_sess)):
        fig, ax = plt.subplots(figsize=(2.6,2))
    
        ctrl_trace = mean_prof_ctrl_only_ON_sess[i][2500:2500+5*1250]
        stim_trace = mean_prof_stim_only_ON_sess[i][2500:2500+5*1250]
    
        ax.plot(xaxis, ctrl_trace, label='ctrl', color='lightcoral')
        ax.plot(xaxis, stim_trace, label='stim', color='firebrick')
    
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
    
        ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
               ylabel='spike rate (Hz)')
        ax.set_title(f'{paths[i][-17:]} ON', fontsize=10)
        ax.legend(fontsize=5, frameon=False)
    
        fig.savefig(os.path.join(save_dir, f'{paths[i][-17:]}_ON.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        
        # fig, ax = plt.subplots(figsize=(2.6,2))
    
        # ctrl_trace = mean_prof_ctrl_only_OFF_sess[i][2500:2500+5*1250]
        # stim_trace = mean_prof_stim_only_OFF_sess[i][2500:2500+5*1250]
    
        # ax.plot(xaxis, ctrl_trace, label='ctrl', color='violet')
        # ax.plot(xaxis, stim_trace, label='stim', color='purple')
    
        # for s in ['top', 'right']:
        #     ax.spines[s].set_visible(False)
    
        # ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
        #        ylabel='spike rate (Hz)')
        # ax.set_title(f'{paths[i][-17:]} OFF', fontsize=10)
        # ax.legend(fontsize=5, frameon=False)
    
        # fig.savefig(os.path.join(save_dir, f'{paths[i][-17:]}_OFF.png'),
        #             dpi=300, bbox_inches='tight')
        # plt.close(fig)
        
        
    #
    mean_ctrl_only_ON = np.mean(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    sem_ctrl_only_ON = sem(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    
    mean_stim_only_ON = np.mean(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    sem_stim_only_ON = sem(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    
    # plot
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(xaxis, mean_ctrl_only_OFF, label='mean_ctrl_OFF', color='violet')
    ax.fill_between(xaxis, mean_ctrl_only_OFF + sem_ctrl_only_OFF, mean_ctrl_only_OFF - sem_ctrl_only_OFF,
                    color='violet', edgecolor='none', alpha=.15)
    
    ax.plot(xaxis, mean_stim_only_OFF, label='mean_stim_OFF', color='purple')
    ax.fill_between(xaxis, mean_stim_only_OFF + sem_stim_only_OFF, mean_stim_only_OFF - sem_stim_only_OFF,
                    color='purple', edgecolor='none', alpha=.15)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='spike rate (Hz)', yticks=[1, 2, 3, 4], ylim=(.9, 4.1))
    ax.set_title(rf'{exp}\nbaseline PyrUp (<2/3) and PyrDown (>3/2)', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_only_OFF{ext}',
            dpi=300, bbox_inches='tight'
            )
    

    # amplitude statistics
    amp_ctrl_only_ON = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_ctrl_only_ON
        ], 
        axis=1)
    amp_stim_only_ON = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_stim_only_ON
        ], 
        axis=1)
    
    plot_violin_with_scatter(amp_ctrl_only_ON, amp_stim_only_ON, 
                             'coral', 'firebrick',
                             xticklabels=['ctrl.', 'stim.'],
                             paired=False,
                             showscatter=True,
                             showmedians=True,
                             ylabel='spike rate (Hz)',
                             dpi=300,
                             save=True,
                             savepath=rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_only_stim_only_ON_amp')
    
    amp_ctrl_only_OFF = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_ctrl_only_OFF
        ], 
        axis=1)
    amp_stim_only_OFF = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_stim_only_OFF
        ], 
        axis=1)
    
    plot_violin_with_scatter(amp_ctrl_only_OFF, amp_stim_only_OFF, 
                             'violet', 'purple',
                             xticklabels=['ctrl.', 'stim.'],
                             paired=False,
                             ylabel='spike rate (Hz)',
                             dpi=300,
                             save=True,
                             savepath=rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_only_stim_only_OFF_amp')
    
    amp_stim_ON = [
        np.mean(prof[3750+625:3750+1875]) for prof
        in mean_prof_stim_ON
        ]
    amp_stim_new_ON = [
        np.mean(prof[3750+625:3750+1875]) for prof
        in mean_prof_stim_new_ON
        ]
    
    plot_violin_with_scatter(amp_stim_ON, amp_stim_new_ON, 
                             'firebrick', 'darkorange',
                             xticklabels=['stim.\n(ori.)', 'stim.\n(new)'],
                             paired=False,
                             showmedians=True,
                             ylabel='spike rate (Hz)',
                             dpi=300,
                             save=True,
                             savepath=rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_only_stim_new_ON_amp')
    
    
    # dist v amp_mean
    from scipy.stats import linregress
    
    # filter and clean data
    all_amp_ON_delta_mean_filt = [v for i, v in enumerate(all_amp_ON_delta_mean)
                             if not np.isnan(all_ctrl_stim_lick_distance_delta[i])]
    all_ctrl_stim_lick_distance_delta_filt = [v for i, v in enumerate(all_ctrl_stim_lick_distance_delta)
                                              if not np.isnan(all_ctrl_stim_lick_distance_delta[i])]
    
    # regression
    slope, intercept, r, p, _ = linregress(all_amp_ON_delta_mean_filt, all_ctrl_stim_lick_distance_delta_filt)
    
    # plot
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    
    # scatter
    ax.scatter(all_amp_ON_delta_mean_filt, all_ctrl_stim_lick_distance_delta_filt,
               color='teal', edgecolor='white', s=30, alpha=0.8)
    
    # regression line
    x_vals = np.linspace(min(all_amp_ON_delta_mean_filt), max(all_amp_ON_delta_mean_filt), 100)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='k', lw=1)
    
    # r and p text
    ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)
    
    # labels and style
    ax.set_xlabel('delta ON (Hz)', fontsize=10)
    ax.set_ylabel('delta lick dist. (stim − ctrl)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\population_ctrl_stim\{exp}_delta_amp_mean_dist{ext}',
                    dpi=300,
                    bbox_inches='tight')
        
    
    ## sum vs dist 
    all_amp_ON_delta_sum_filt = [v for i, v in enumerate(all_amp_ON_delta_sum)
                                 if not np.isnan(all_ctrl_stim_lick_distance_delta[i])]
    
    # regression
    slope, intercept, r, p, _ = linregress(all_amp_ON_delta_sum_filt, all_ctrl_stim_lick_distance_delta_filt)
    
    # plot
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    
    # scatter
    ax.scatter(all_amp_ON_delta_sum_filt, all_ctrl_stim_lick_distance_delta_filt,
               color='teal', edgecolor='white', s=30, alpha=0.8)
    
    # regression line
    x_vals = np.linspace(min(all_amp_ON_delta_sum_filt), max(all_amp_ON_delta_sum_filt), 100)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='k', lw=1)
    
    # r and p text
    ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)
    
    # labels and style
    ax.set_xlabel('delta ON (Hz)', fontsize=10)
    ax.set_ylabel('delta lick dist. (stim − ctrl)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\population_ctrl_stim\{exp}_delta_amp_sum_dist{ext}',
                    dpi=300,
                    bbox_inches='tight')
        

if __name__ == '__main__':
    main(pathHPCLCopt, 'HPCLC')
    main(pathHPCLCtermopt, 'HPCLCterm')