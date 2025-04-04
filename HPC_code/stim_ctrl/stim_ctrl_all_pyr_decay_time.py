# -*- coding: utf-8 -*-
"""
Created on Mon Mar  27 16:08:34 2025

analyse the decay time constants of pyramidal cells in stim. vs ctrl. trials 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 

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
TIME = np.arange(-SAMP_FREQ*1, SAMP_FREQ*6)/SAMP_FREQ  # 10 seconds 

MAX_TIME = 7  # collect (for each trial) a maximum of 10 s of spiking-profile
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
fit_results_ctrl_ON = []
fit_results_stim_ON = []
tau_values_ctrl_OFF = []
tau_values_stim_OFF = []
fit_results_ctrl_OFF = []
fit_results_stim_OFF = []

tau_values_converted_ON = []

for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    
    curr_beh_df = beh_df.loc[recname]  # subselect in read-only
    
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    
    for idx, session in curr_df_pyr.iterrows():
        # if session['class_ctrl']!='run-onset unresponsive':
        #     cluname = idx
            
        #     mean_prof_ctrl = session['prof_ctrl_mean']
            
        #     peak_idx_ctrl = detect_min_max(mean_prof_ctrl, 
        #                                    session['class_ctrl'],
        #                                    run_onset_bin=3750)
        
        #     tau_ctrl, fit_params_ctrl = compute_tau(
        #         TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_ctrl']
        #         )
            
        #     tau_values_ctrl.append(tau_ctrl)
        #     fit_results_ctrl.append(fit_params_ctrl)
            
        # if session['class_stim']!='run-onset unresponsive':
        #     cluname = idx
            
        #     mean_prof_stim = session['prof_stim_mean']
            
        #     peak_idx_stim = detect_min_max(mean_prof_stim, 
        #                                    session['class_stim'],
        #                                    run_onset_bin=3750)
        
        #     tau_stim, fit_params_stim = compute_tau(
        #         TIME, mean_prof_stim, peak_idx_stim, session['class_stim']
        #         )
            
        #     tau_values_stim.append(tau_stim)
        #     fit_results_stim.append(fit_params_stim)

        cluname = idx
                    
        if session['class_ctrl']=='run-onset ON' and session['class_stim']=='run-onset ON':

            mean_prof_ctrl = session['prof_ctrl_mean'][3750:]
            
            peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                        session['class_ctrl'],
                                        run_onset_bin=0)
        
            tau_ctrl, fit_params_ctrl = compute_tau(
                TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_ctrl']
                )
            
            mean_prof_stim = session['prof_stim_mean'][3750:]
            
            peak_idx_stim = detect_peak(mean_prof_stim, 
                                        session['class_stim'],
                                        run_onset_bin=0)
        
            tau_stim, fit_params_stim = compute_tau(
                TIME, mean_prof_stim, peak_idx_stim, session['class_stim']
                )
            
            if fit_params_ctrl['adj_r_squared']>0.7 and fit_params_stim['adj_r_squared']>0.7:
                tau_values_ctrl_ON.append(tau_ctrl)
                fit_results_ctrl_ON.append(fit_params_ctrl)
                tau_values_stim_ON.append(tau_stim)
                fit_results_stim_ON.append(fit_params_stim)
                
        if session['class_ctrl']!='run-onset ON' and session['class_stim']=='run-onset ON':
            mean_prof = session['prof_stim_mean'][3750:]
            
            peak_idx = detect_peak(mean_prof_stim,
                                   session['class_stim'],
                                   run_onset_bin=0)
        
            tau, fit_params = compute_tau(
                TIME, mean_prof, peak_idx, session['class_stim']
                )
            
            if fit_params['adj_r_squared']>0.7:
                tau_values_converted_ON.append(tau)
                
        if session['class_ctrl']=='run-onset OFF' and session['class_stim']=='run-onset OFF':
    
            mean_prof_ctrl = session['prof_ctrl_mean'][3750:]
            
            peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                        session['class_ctrl'],
                                        run_onset_bin=0)
        
            tau_ctrl, fit_params_ctrl = compute_tau(
                TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_ctrl']
                )
            
            mean_prof_stim = session['prof_stim_mean'][3750:]
            
            peak_idx_stim = detect_peak(mean_prof_stim, 
                                        session['class_stim'],
                                        run_onset_bin=0)
        
            tau_stim, fit_params_stim = compute_tau(
                TIME, mean_prof_stim, peak_idx_stim, session['class_stim']
                )
            
            if fit_params_ctrl['adj_r_squared']>0.7 and fit_params_stim['adj_r_squared']>0.7:
                tau_values_ctrl_OFF.append(tau_ctrl)
                fit_results_ctrl_OFF.append(fit_params_ctrl)
                tau_values_stim_OFF.append(tau_stim)
                fit_results_stim_OFF.append(fit_params_stim)
                

#%% plotting 
plot_violin_with_scatter(tau_values_ctrl_ON, tau_values_stim_ON, 
                         'lightcoral', 'firebrick',
                         xticklabels=['ctrl.', 'stim.'],
                         ylabel='τ (s)',
                         title='run-onset ON',
                         paired=True,
                         showmeans=True,
                         showmedians=False,
                         showscatter=False,
                         save=False,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\ON_decay_constant_ctrl_stim')

plot_violin_with_scatter(tau_values_ctrl_OFF, tau_values_stim_OFF, 
                         'thistle', 'purple',
                         xticklabels=['ctrl\n$1^{st}$-lick', 'stim\n$1^{st}$-lick'],
                         ylabel='τ (s)',
                         title='run-onset OFF',
                         showmeans=True,
                         showmedians=False,
                         showscatter=False,
                         save=False,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\OFF_decay_constant_ctrl_stim')


#%% cdfs 
plot_ecdfs(tau_values_ctrl_ON, tau_values_stim_ON,
           title='ECDF – run-onset ON',
           xlabel='τ (s)', 
           legend_labels=['ctrl.', 'stim.'],
           colours=['lightcoral', 'firebrick'],
           save=False,
           savepath='Z:/.../ecdf_run_onset_ON')

plot_ecdfs(tau_values_ctrl_ON, tau_values_converted_ON,
           title='ECDF – run-onset ON',
           xlabel='τ (s)', 
           legend_labels=['ctrl.', 'converted (new)'],
           colours=['lightcoral', 'firebrick'],
           save=False,
           savepath='Z:/.../ecdf_run_onset_ON')