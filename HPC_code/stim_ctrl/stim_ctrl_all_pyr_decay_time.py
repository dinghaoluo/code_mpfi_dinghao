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
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\HPC_code\decay_time')
from decay_time_analysis import detect_min_max, compute_tau

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from plotting_functions import plot_violin_with_scatter


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
tau_values_ctrl = []
tau_values_stim = []
fit_results_ctrl = []
fit_results_stim = []

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
        
        if session['class']!='run-onset unresponsive':
            cluname = idx
            
            mean_prof_ctrl = session['prof_ctrl_mean'][2500:2500+7*1250]
            
            peak_idx_ctrl = detect_min_max(mean_prof_ctrl, 
                                           session['class'],
                                           run_onset_bin=1250)
        
            tau_ctrl, fit_params_ctrl = compute_tau(
                TIME, mean_prof_ctrl, peak_idx_ctrl, session['class']
                )
            
            tau_values_ctrl.append(tau_ctrl)
            fit_results_ctrl.append(fit_params_ctrl)
            
            mean_prof_stim = session['prof_stim_mean'][2500:2500+7*1250]
            
            peak_idx_stim = detect_min_max(mean_prof_stim, 
                                           session['class'],
                                           run_onset_bin=1250)
        
            tau_stim, fit_params_stim = compute_tau(
                TIME, mean_prof_stim, peak_idx_stim, session['class']
                )
            
            tau_values_stim.append(tau_stim)
            fit_results_stim.append(fit_params_stim)
            

#%% plotting 
tau_values_ctrl, tau_values_stim = zip(
    *[(x, y) for x, y in zip(tau_values_ctrl, tau_values_stim) 
      if x is not None and y is not None]
    )

tau_values_ctrl_ON, tau_values_stim_ON = zip(
    *[(x, y) for x, y in zip(tau_values_ctrl, tau_values_stim)
      if 0 < x < 3 and 0 < y < 3]
    )
tau_values_ctrl_OFF, tau_values_stim_OFF = zip(
    *[(x, y) for x, y in zip(tau_values_ctrl, tau_values_stim)
      if x < 0 and y < 0]
    )

plot_violin_with_scatter(tau_values_ctrl_ON, tau_values_stim_ON, 
                         'lightcoral', 'firebrick',
                         xticklabels=['ctrl.', 'stim.'],
                         ylabel='τ (s)',
                         title='run-onset ON',
                         paired=True,
                         showmeans=True,
                         showmedians=False,
                         showscatter=False,
                         ylim=(0, 3),
                         save=False,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\ON_decay_constant')

plot_violin_with_scatter(tau_values_ctrl_OFF, tau_values_stim_OFF, 
                         'thistle', 'purple',
                         xticklabels=['ctrl\n$1^{st}$-lick', 'stim\n$1^{st}$-lick'],
                         ylabel='τ (s)',
                         title='run-onset OFF',
                         showmeans=True,
                         showmedians=False,
                         showscatter=False,
                         ylim=(-20, 0),
                         save=False,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\OFF_decay_constant')
