# -*- coding: utf-8 -*-
"""
Created on Mon Mar  27 16:08:34 2025

analyse the decay time constants of pyramidal cells in stim. vs ctrl. trials 

@author: Dinghao Luo
"""


#%% imports 
from pathlib import Path

import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
from scipy.stats import sem, linregress, ttest_rel
import pandas as pd 

from plotting_functions import plot_violin_with_scatter
from common import mpl_formatting
mpl_formatting()

import rec_list
pathHPCLCopt = rec_list.pathHPCLCopt
pathHPCLCtermopt = rec_list.pathHPCLCtermopt

from decay_time_analysis import detect_peak, compute_tau


#%% parameters 
SAMP_FREQ = 1250  # Hz
TIME = np.arange(-SAMP_FREQ*3, SAMP_FREQ*7)/SAMP_FREQ  # 10 seconds 

MAX_TIME = 10  # collect (for each trial) a maximum of 10 s of spiking-profile
MAX_SAMPLES = SAMP_FREQ * MAX_TIME

XAXIS = np.arange(-1*1250, 4*1250) / 1250

DELTA_THRES = 0.5  # Hz

bin_edges = np.arange(3750 + int(-0.5*1250), 3750 + int(3.5*1250) + 1, 1250)
bin_labels = ['-0.5-0.5', '0.5–1.5s', '1.5–2.5s', '2.5–3.5s']


#%% helper 
def _annotate_pvals(ax, pvals, y_level=4.05, star=True):
    """Annotate p-values or stars at bin midpoints."""
    mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    for mid, p in zip(mids, pvals):
        if star:
            if p < 0.001: text = '***'
            elif p < 0.01: text = '**'
            elif p < 0.05: text = round(p, 4)
            else: text = 'n.s.'
        else:
            text = f'{p:.3f}'
        ax.text((mid-3750)/1250, y_level, text,
                ha='center', va='bottom', fontsize=6, color='k')

def _binwise_test(ctrl_traces, stim_traces, label):
    """ctrl_traces and stim_traces are lists/arrays of shape (n_cells, n_timepoints)."""
    pvals = []
    print(f'\nBinwise stats for {label}:')
    for b in range(len(bin_edges)-1):
        start, end = bin_edges[b], bin_edges[b+1]
        ctrl_bin = [np.mean(tr[start:end]) for tr in ctrl_traces]
        stim_bin = [np.mean(tr[start:end]) for tr in stim_traces]
        stat, p = ttest_rel(ctrl_bin, stim_bin, nan_policy='omit')
        pvals.append(p)
        print(f'  {bin_labels[b]}: t = {stat:.2f}, p = {p:.3g}')
    return pvals


#%% path stems 
all_sess_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions')
all_beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments')
ctrl_stim_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/run_onset_response/ctrl_stim')


#%% load data 
print('loading dataframes...')

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
    )
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 


#%% main 
def main(paths, exp='HPCLC'):
    tau_values_ctrl_ON, tau_values_stim_ON   = [], []
    tau_values_ctrl_OFF, tau_values_stim_OFF = [], []
    
    # remainers, 18 Sept 2025
    tau_values_ctrl_remain_ON, tau_values_stim_remain_ON   = [], []
    tau_values_ctrl_remain_OFF, tau_values_stim_remain_OFF = [], []
    
    # leavers, 18 Sept 2025
    tau_values_ctrl_leave_ON, tau_values_stim_leave_ON   = [], []
    tau_values_ctrl_leave_OFF, tau_values_stim_leave_OFF = [], []
    
    tau_values_ctrl_new_ON, tau_values_stim_new_ON   = [], []
    tau_values_ctrl_new_OFF, tau_values_stim_new_OFF = [], []
    
    tau_values_ctrl_only_ON, tau_values_stim_only_ON   = [], []
    tau_values_ctrl_only_OFF, tau_values_stim_only_OFF = [], []
    
    mean_prof_ctrl_ON, mean_prof_stim_ON   = [], []
    mean_prof_ctrl_OFF, mean_prof_stim_OFF = [], []
    
    # remainers, 18 Sept 2025 
    mean_prof_ctrl_remain_ON, mean_prof_stim_remain_ON   = [], []
    mean_prof_ctrl_remain_OFF, mean_prof_stim_remain_OFF = [], []
    
    # leavers, 18 Sept 2025 
    mean_prof_ctrl_leave_ON, mean_prof_stim_leave_ON   = [], []
    mean_prof_ctrl_leave_OFF, mean_prof_stim_leave_OFF = [], []
    
    mean_prof_ctrl_new_ON, mean_prof_stim_new_ON   = [], []
    mean_prof_ctrl_new_OFF, mean_prof_stim_new_OFF = [], []
    
    mean_prof_ctrl_only_ON, mean_prof_stim_only_ON   = [], []
    mean_prof_ctrl_only_OFF, mean_prof_stim_only_OFF = [], []
    
    peak_ctrl_ON, peak_stim_ON   = [], []
    peak_ctrl_OFF, peak_stim_OFF = [], []
    
    # remainers, 18 Sept 2025
    peak_ctrl_remain_ON, peak_stim_remain_ON   = [], []
    peak_ctrl_remain_OFF, peak_stim_remain_OFF = [], []
    
    # leavers, 18 Sept 2025
    peak_ctrl_leave_ON, peak_stim_leave_ON   = [], []
    peak_ctrl_leave_OFF, peak_stim_leave_OFF = [], []
    
    peak_ctrl_new_ON, peak_stim_new_ON   = [], []
    peak_ctrl_new_OFF, peak_stim_new_OFF = [], []
    
    peak_ctrl_only_ON, peak_stim_only_ON   = [], []
    peak_ctrl_only_OFF, peak_stim_only_OFF = [], []
    
    mean_prof_ctrl_only_ON_sess, mean_prof_stim_only_ON_sess   = [], []
    mean_prof_ctrl_only_OFF_sess, mean_prof_stim_only_OFF_sess = [], []

    ## after rate-matching, 17 Sept 2025 Dinghao ##
    mean_prof_ctrl_only_ON_matched, mean_prof_stim_only_ON_matched   = [], []
    mean_prof_ctrl_only_OFF_matched, mean_prof_stim_only_OFF_matched = [], []
    
    ON_pairs_mean, ON_pairs_sum   = [], []
    OFF_pairs_mean, OFF_pairs_sum = [], []
    
    all_ctrl_stim_lick_time_delta     = []
    all_ctrl_stim_lick_distance_delta = []
    
    all_amp_ON_delta_mean  = []
    all_amp_OFF_delta_mean = []
    
    all_amp_remain_ON_delta_mean  = []
    all_amp_remain_OFF_delta_mean = []
    
    session_prop_remain_ON, session_prop_new_ON   = [], []
    session_prop_remain_OFF, session_prop_new_OFF = [], []
    
    for path in paths:
        recname = Path(path).name
        print(f'\n{recname}')
        
        train_path = all_sess_stem / recname / f'{recname}_all_trains.npy'
        trains = np.load(train_path, allow_pickle=True).item()
        
        
        if (all_beh_stem / 'HPCLC' / f'{recname}.pkl').exists():
            with open(all_beh_stem / 'HPCLC' / f'{recname}.pkl', 'rb') as f:
                beh = pickle.load(f)
        else:
            with open(all_beh_stem / 'HPCLCterm' / f'{recname}.pkl', 'rb') as f:
                beh = pickle.load(f)
        
        stim_conds = [t[15] for t in beh['trial_statements']][1:]
        stim_idx = [trial for trial, cond in enumerate(stim_conds)
                    if cond!='0']
        ctrl_idx = [trial+2 for trial in stim_idx]
        
        first_lick_times = [t[0][0] - s for t, s
                            in zip(beh['lick_times'], beh['run_onsets'])
                            if t]
        ctrl_stim_lick_time_delta = np.median(
            [t for i, t in enumerate(first_lick_times) if i in stim_idx]
            ) - np.median(
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
        
        
        ## STATE-TRANSITION ANALYSIS ##
        ctrl_ON_cells  = curr_df_pyr['class_ctrl_MATLAB'] == 'run-onset ON'
        ctrl_OFF_cells = curr_df_pyr['class_ctrl_MATLAB'] == 'run-onset OFF'
        stim_ON_cells  = curr_df_pyr['class_stim_MATLAB'] == 'run-onset ON'
        stim_OFF_cells = curr_df_pyr['class_stim_MATLAB'] == 'run-onset OFF'
        
        # remainers
        remain_ON  = np.sum(ctrl_ON_cells & stim_ON_cells)
        remain_OFF = np.sum(ctrl_OFF_cells & stim_OFF_cells)
        
        # new recruits
        new_ON  = np.sum(~ctrl_ON_cells & stim_ON_cells)
        new_OFF = np.sum(~ctrl_OFF_cells & stim_OFF_cells)
        
        # denominators
        n_stim_ON  = np.sum(stim_ON_cells)
        n_stim_OFF = np.sum(stim_OFF_cells)    
        
        # proportions (guard divide-by-zero)
        prop_remain_ON  = remain_ON / n_stim_ON  if n_stim_ON > 0 else np.nan
        prop_remain_OFF = remain_OFF / n_stim_OFF if n_stim_OFF > 0 else np.nan
        prop_new_ON  = new_ON / n_stim_ON
        prop_new_OFF = new_OFF / n_stim_OFF
        
        if not np.isnan(prop_remain_ON): 
            session_prop_remain_ON.append(prop_remain_ON)
            session_prop_new_ON.append(prop_new_ON)
        if not np.isnan(prop_remain_OFF):
            session_prop_remain_OFF.append(prop_remain_OFF)
            session_prop_new_OFF.append(prop_new_OFF)
        ## STATE-TRANSITION ANALYSIS ENDS ##
        
        for idx, session in curr_df_pyr.iterrows():
            cluname = idx
            
            ## CTRL ON ##
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
            
            ## CTRL ON, STIM ON ##
            if session['class_ctrl']=='run-onset ON' and session['class_stim']=='run-onset ON':
                
                mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
                mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
                mean_prof_ctrl_remain_ON.append(mean_prof_ctrl)
                mean_prof_stim_remain_ON.append(mean_prof_stim)
                
                peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                            session['class_ctrl'],
                                            run_onset_bin=3750)
                tau_ctrl, fit_params_ctrl = compute_tau(
                    TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_ctrl']
                    )
                            
                peak_idx_stim = detect_peak(mean_prof_stim, 
                                            session['class_stim'],
                                            run_onset_bin=3750)
                tau_stim, fit_params_stim = compute_tau(
                    TIME, mean_prof_stim, peak_idx_stim, session['class_stim']
                    )
                
                peak_ctrl_remain_ON.append(peak_idx_ctrl)
                peak_stim_remain_ON.append(peak_idx_stim)
                
                if fit_params_stim['adj_r_squared']>0.6:
                    tau_values_ctrl_remain_ON.append(tau_ctrl)
                    tau_values_stim_remain_ON.append(tau_stim)
                    
            ## CTRL ON, NOT STIM ON ##
            if session['class_ctrl']=='run-onset ON' and session['class_stim']!='run-onset ON':
                
                mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
                mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
                mean_prof_ctrl_leave_ON.append(mean_prof_ctrl)
                mean_prof_stim_leave_ON.append(mean_prof_stim)
                
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
                
                peak_ctrl_leave_ON.append(peak_idx_ctrl)
                peak_stim_leave_ON.append(peak_idx_stim)
                
                if fit_params_stim['adj_r_squared']>0.6:
                    tau_values_ctrl_leave_ON.append(tau_ctrl)
                    tau_values_stim_leave_ON.append(tau_stim)
            
            ## NOT CTRL ON, STIM ON ##
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
            
            ## CTRL OFF ##
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
                    
            ## CTRL OFF, STIM OFF ##
            if session['class_ctrl']=='run-onset OFF' and session['class_stim']=='run-onset OFF':
                
                mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
                mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
                mean_prof_ctrl_remain_OFF.append(mean_prof_ctrl)
                mean_prof_stim_remain_OFF.append(mean_prof_stim)
                
                peak_idx_ctrl = detect_peak(mean_prof_ctrl, 
                                            session['class_ctrl'],
                                            run_onset_bin=3750)
                tau_ctrl, fit_params_ctrl = compute_tau(
                    TIME, mean_prof_ctrl, peak_idx_ctrl, session['class_ctrl']
                    )
                            
                peak_idx_stim = detect_peak(mean_prof_stim, 
                                            session['class_stim'],
                                            run_onset_bin=3750)
                tau_stim, fit_params_stim = compute_tau(
                    TIME, mean_prof_stim, peak_idx_stim, session['class_stim']
                    )
                
                peak_ctrl_remain_OFF.append(peak_idx_ctrl)
                peak_stim_remain_OFF.append(peak_idx_stim)
                
                if fit_params_stim['adj_r_squared']>0.6:
                    tau_values_ctrl_remain_OFF.append(tau_ctrl)
                    tau_values_stim_remain_OFF.append(tau_stim)
            
            ## CTRL OFF, NOT STIM OFF ##
            if session['class_ctrl']=='run-onset OFF' and session['class_stim']!='run-onset OFF':
                
                mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
                mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
                mean_prof_ctrl_leave_OFF.append(mean_prof_ctrl)
                mean_prof_stim_leave_OFF.append(mean_prof_stim)
                
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
                
                peak_ctrl_leave_OFF.append(peak_idx_ctrl)
                peak_stim_leave_OFF.append(peak_idx_stim)
                
                if fit_params_stim['adj_r_squared']>0.6:
                    tau_values_ctrl_leave_OFF.append(tau_ctrl)
                    tau_values_stim_leave_OFF.append(tau_stim)
            
            ## NOT CTRL OFF, STIM OFF ##
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
                    
            
            ## CTRL AND STIM ONLY ##
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
                    
            ## CTRL AND STIM ONLY ENDS ##
            

        ## single-session firing rate matching, 17 Sept 2025 ##
        if len(mean_prof_ctrl_only_OFF) > 0 and len(mean_prof_stim_only_OFF) > 0:
            curr_ctrl_OFF = mean_prof_ctrl_only_OFF[-len(curr_df_pyr):]
            ctrl_OFF_baselines = [np.mean(train[2500:3750]) for train in curr_ctrl_OFF]
            ctrl_OFF_mean = np.mean(ctrl_OFF_baselines)
            ctrl_OFF_sem = sem(ctrl_OFF_baselines)
            
            curr_stim_OFF = mean_prof_stim_only_OFF[-len(curr_df_pyr):]
            stim_OFF_mean_baselines = [np.mean(train[2500:3750]) for train in curr_stim_OFF]
            
            # subselect stim. neurones that fall within the ctrl. range 
            ctrl_OFF_range = (ctrl_OFF_mean - ctrl_OFF_sem,
                              ctrl_OFF_mean + ctrl_OFF_sem)
            
            # pre-matched curves for visualisation
            fig, ax = plt.subplots(figsize=(3,3))
            # ax.plot(curr_ctrl_OFF, color='purple', alpha=.05)
            ax.plot(XAXIS, np.mean(curr_ctrl_OFF, axis=0)[2500:2500+1250*5], color='purple')
            # ax.plot(curr_stim_OFF, color='royalblue', alpha=.05)
            ax.plot(XAXIS, np.mean(curr_stim_OFF, axis=0)[2500:2500+1250*5], color='royalblue')
            fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\single_sessions_pre_matched\{recname}_OFF.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            matched_ctrl_OFF = [c for c in curr_ctrl_OFF]
            matched_stim_OFF = [s for s, sb in zip(curr_stim_OFF, stim_OFF_mean_baselines)
                                if ctrl_OFF_range[0] <= sb <= ctrl_OFF_range[1]]

            if len(matched_stim_OFF) > 0:  # only keep if non-empty stim.
                mean_prof_ctrl_only_OFF_matched.extend(matched_ctrl_OFF)
                mean_prof_stim_only_OFF_matched.extend(matched_stim_OFF)
        
                sess_mean_ctrl_OFF = np.mean(matched_ctrl_OFF, axis=0)
                sess_mean_stim_OFF = np.mean(matched_stim_OFF, axis=0)
                mean_prof_ctrl_only_OFF_sess.append(sess_mean_ctrl_OFF)
                mean_prof_stim_only_OFF_sess.append(sess_mean_stim_OFF)
                
                # delta firing rate vs dist/time
                amp_OFF_delta_mean = np.mean(sess_mean_stim_OFF[3750+625:3750+1825]) - np.mean(sess_mean_ctrl_OFF[3750+625:3750+1825])
                all_amp_OFF_delta_mean.append(amp_OFF_delta_mean)
                
                # pair with this session's distance (the last one you appended for this path)
                dist_delta = all_ctrl_stim_lick_distance_delta[-1]
                if not np.isnan(dist_delta):
                    OFF_pairs_mean.append((amp_OFF_delta_mean, dist_delta))
            
        # if len(mean_prof_ctrl_only_ON) > 0 and len(mean_prof_stim_only_ON) > 0:
        #     curr_ctrl_ON = mean_prof_ctrl_only_ON[-len(curr_df_pyr):]
        #     ON_mean_baseline = np.mean([np.mean(train[2500:3750]) for train in curr_ctrl_ON])
        #     ON_sem_baseline = sem([np.mean(train[2500:3750]) for train in curr_ctrl_ON])
        ## END of single-session firing rate matching ##
        
        
        # store per-session mean profile of ctrl-only and stim-only cells
        if len(mean_prof_ctrl_only_ON) > 0:
            sess_mean_ctrl = np.mean(
                mean_prof_ctrl_only_ON[-sum(curr_df_pyr['class_ctrl']=='run-onset ON'):], axis=0)  # corrected indexing, 19 Sept 2025
            mean_prof_ctrl_only_ON_sess.append(sess_mean_ctrl)
        if len(mean_prof_stim_only_ON) > 0:
            sess_mean_stim = np.mean(
                mean_prof_stim_only_ON[-sum(curr_df_pyr['class_stim']=='run-onset ON'):], axis=0)
            mean_prof_stim_only_ON_sess.append(sess_mean_stim)
        if len(mean_prof_ctrl_only_ON) > 0 and len(mean_prof_stim_only_ON) > 0:
            amp_ON_delta_mean = np.mean(sess_mean_stim[3750+625:3750+1825]) - np.mean(sess_mean_ctrl[3750+625:3750+1825])
            all_amp_ON_delta_mean.append(amp_ON_delta_mean)
            
            
        if len(mean_prof_ctrl_remain_ON) > 0:
            union = sum((curr_df_pyr['class_ctrl']=='run-onset ON') 
                        & (curr_df_pyr['class_stim']=='run-onset ON'))
            sess_mean_ctrl = np.mean(mean_prof_ctrl_remain_ON[-union:], axis=0)
            sess_mean_stim = np.mean(mean_prof_stim_remain_ON[-union:], axis=0)
            amp_remain_ON_delta_mean = np.mean(sess_mean_stim[3750+625:3750+1825]) - np.mean(sess_mean_ctrl[3750+625:3750+1825])
            all_amp_remain_ON_delta_mean.append(amp_remain_ON_delta_mean)
        if len(mean_prof_ctrl_remain_OFF) > 0:
            union = sum((curr_df_pyr['class_ctrl']=='run-onset OFF') 
                        & (curr_df_pyr['class_stim']=='run-onset OFF'))
            sess_mean_ctrl = np.mean(mean_prof_ctrl_remain_OFF[-union:], axis=0)
            sess_mean_stim = np.mean(mean_prof_stim_remain_OFF[-union:], axis=0)
            amp_remain_OFF_delta_mean = np.mean(sess_mean_stim[3750+625:3750+1825]) - np.mean(sess_mean_ctrl[3750+625:3750+1825])
            all_amp_remain_OFF_delta_mean.append(amp_remain_OFF_delta_mean)
    ## single-session processing ends ##
    
    
    ## STATE-TRANSITION CALCULATION ##
    labels = ['Remain', 'New']
    
    on_sizes  = [remain_ON, new_ON]
    off_sizes = [remain_OFF, new_OFF]
    
    colors_on  = ['firebrick', 'darkorange']
    colors_off = ['purple', 'darkblue']
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # ON pie
    axes[0].pie(on_sizes, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=colors_on)
    axes[0].set_title('Stim ON cells')
    
    # OFF pie
    axes[1].pie(off_sizes, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=colors_off)
    axes[1].set_title('Stim OFF cells')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(ctrl_stim_stem / f'{exp}_ON_OFF_proportions_pie.png', dpi=300, bbox_inches='tight')
    ## STATE-TRANSITION CALCULATION ENDS ##
    
     
    ## EXTRACT mean and SEM for all 8 traces ##
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
    ## EXTRACT mean and SEM for all 8 traces ENDS ##
    
    
    ## BIN-WISE TEST ##
    pvals_remain_ON  = _binwise_test(mean_prof_ctrl_remain_ON, mean_prof_stim_remain_ON, 'remain ON')
    pvals_remain_OFF = _binwise_test(mean_prof_ctrl_remain_OFF, mean_prof_stim_remain_OFF, 'remain OFF')
    pvals_leave_ON   = _binwise_test(mean_prof_ctrl_leave_ON, mean_prof_stim_leave_ON, 'leave ON')
    pvals_leave_OFF  = _binwise_test(mean_prof_ctrl_leave_OFF, mean_prof_stim_leave_OFF, 'leave OFF')
    pvals_new_ON     = _binwise_test(mean_prof_ctrl_new_ON, mean_prof_stim_new_ON, 'new ON')
    pvals_new_OFF    = _binwise_test(mean_prof_ctrl_new_OFF, mean_prof_stim_new_OFF, 'new OFF')
    ## BIN-WISE TEST ENDS ##
    
    
    ## PLOTTING ALL TRACES ##    
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_ON, label='mean_ctrl_ON', color='lightcoral')
    ax.fill_between(XAXIS, mean_ctrl_ON + sem_ctrl_ON, mean_ctrl_ON - sem_ctrl_ON,
                    color='lightcoral', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_ON, label='mean_stim_ON', color='firebrick')
    ax.fill_between(XAXIS, mean_stim_ON + sem_stim_ON, mean_stim_ON - sem_stim_ON,
                    color='firebrick', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_ctrl_new_ON, label='mean_ctrl_new_ON', color='moccasin')
    ax.fill_between(XAXIS, mean_ctrl_new_ON + sem_ctrl_new_ON, mean_ctrl_new_ON - sem_ctrl_new_ON,
                    color='moccasin', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_new_ON, label='mean_stim_new_ON', color='darkorange')
    ax.fill_between(XAXIS, mean_stim_new_ON + sem_stim_new_ON, mean_stim_new_ON - sem_stim_new_ON,
                    color='darkorange', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_ctrl_OFF, label='mean_ctrl_OFF', color='violet')
    ax.fill_between(XAXIS, mean_ctrl_OFF + sem_ctrl_OFF, mean_ctrl_OFF - sem_ctrl_OFF,
                    color='violet', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_OFF, label='mean_stim_OFF', color='purple')
    ax.fill_between(XAXIS, mean_stim_OFF + sem_stim_OFF, mean_stim_OFF - sem_stim_OFF,
                    color='purple', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_ctrl_new_OFF, label='mean_ctrl_new_OFF', color='lightcyan')
    ax.fill_between(XAXIS, mean_ctrl_new_OFF + sem_ctrl_new_OFF, mean_ctrl_new_OFF - sem_ctrl_new_OFF,
                    color='lightcyan', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_new_OFF, label='mean_stim_new_OFF', color='darkblue')
    ax.fill_between(XAXIS, mean_stim_new_OFF + sem_stim_new_OFF, mean_stim_new_OFF - sem_stim_new_OFF,
                    color='darkblue', edgecolor='none', alpha=.15)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nbaseline PyrUp (<2/3) and PyrDown (>3/2)', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(ctrl_stim_stem / f'{exp}_ctrl_stim_all_curves{ext}', dpi=300, bbox_inches='tight')
    ## ALL TRACES PLOTTED ##
    
    
    ## PLOTTING CTRL VS STIM (REMAINERS vs NEW) ##
    # extract mean and SEM
    mean_ctrl_remain_ON = np.mean(mean_prof_ctrl_remain_ON, axis=0)[2500:2500+5*1250]
    sem_ctrl_remain_ON = sem(mean_prof_ctrl_remain_ON, axis=0)[2500:2500+5*1250]
    
    mean_stim_remain_ON = np.mean(mean_prof_stim_remain_ON, axis=0)[2500:2500+5*1250]
    sem_stim_remain_ON = sem(mean_prof_stim_remain_ON, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_remain_OFF = np.mean(mean_prof_ctrl_remain_OFF, axis=0)[2500:2500+5*1250]
    sem_ctrl_remain_OFF = sem(mean_prof_ctrl_remain_OFF, axis=0)[2500:2500+5*1250]
    
    mean_stim_remain_OFF = np.mean(mean_prof_stim_remain_OFF, axis=0)[2500:2500+5*1250]
    sem_stim_remain_OFF = sem(mean_prof_stim_remain_OFF, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_leave_ON = np.mean(mean_prof_ctrl_leave_ON, axis=0)[2500:2500+5*1250]
    sem_ctrl_leave_ON = sem(mean_prof_ctrl_leave_ON, axis=0)[2500:2500+5*1250]
    
    mean_stim_leave_ON = np.mean(mean_prof_stim_leave_ON, axis=0)[2500:2500+5*1250]
    sem_stim_leave_ON = sem(mean_prof_stim_leave_ON, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_leave_OFF = np.mean(mean_prof_ctrl_leave_OFF, axis=0)[2500:2500+5*1250]
    sem_ctrl_leave_OFF = sem(mean_prof_ctrl_leave_OFF, axis=0)[2500:2500+5*1250]
    
    mean_stim_leave_OFF = np.mean(mean_prof_stim_leave_OFF, axis=0)[2500:2500+5*1250]
    sem_stim_leave_OFF = sem(mean_prof_stim_leave_OFF, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_new_ON = np.mean(mean_prof_ctrl_new_ON, axis=0)[2500:2500+5*1250]
    sem_ctrl_new_ON = sem(mean_prof_ctrl_new_ON, axis=0)[2500:2500+5*1250]
    
    mean_stim_new_ON = np.mean(mean_prof_stim_new_ON, axis=0)[2500:2500+5*1250]
    sem_stim_new_ON = sem(mean_prof_stim_new_ON, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_new_OFF = np.mean(mean_prof_ctrl_new_OFF, axis=0)[2500:2500+5*1250]
    sem_ctrl_new_OFF = sem(mean_prof_ctrl_new_OFF, axis=0)[2500:2500+5*1250]
    
    mean_stim_new_OFF = np.mean(mean_prof_stim_new_OFF, axis=0)[2500:2500+5*1250]
    sem_stim_new_OFF = sem(mean_prof_stim_new_OFF, axis=0)[2500:2500+5*1250]
    
    # PLOTTING ALL TRACES
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_remain_ON, label='ctrl. remain ON', color='firebrick')
    ax.fill_between(XAXIS, mean_ctrl_remain_ON + sem_ctrl_remain_ON, mean_ctrl_remain_ON - sem_ctrl_remain_ON,
                    color='firebrick', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_remain_ON, label='stim. remain ON', color=(87/255, 90/255, 187/255))
    ax.fill_between(XAXIS, mean_stim_remain_ON + sem_stim_remain_ON, mean_stim_remain_ON - sem_stim_remain_ON,
                    color=(87/255, 90/255, 187/255), edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_ctrl_remain_OFF, label='ctrl. remain OFF', color='purple')
    ax.fill_between(XAXIS, mean_ctrl_remain_OFF + sem_ctrl_remain_OFF, mean_ctrl_remain_OFF - sem_ctrl_remain_OFF,
                    color='purple', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_remain_OFF, label='stim. remain OFF', color=(78/255, 84/255, 206/255))
    ax.fill_between(XAXIS, mean_stim_remain_OFF + sem_stim_remain_OFF, mean_stim_remain_OFF - sem_stim_remain_OFF,
                    color=(78/255, 84/255, 206/255), edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_ctrl_new_ON, label='ctrl new ON', color='moccasin')
    ax.fill_between(XAXIS, mean_ctrl_new_ON + sem_ctrl_new_ON, mean_ctrl_new_ON - sem_ctrl_new_ON,
                    color='moccasin', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_new_ON, label='stim new ON', color='darkorange')
    ax.fill_between(XAXIS, mean_stim_new_ON + sem_stim_new_ON, mean_stim_new_ON - sem_stim_new_ON,
                    color='darkorange', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_ctrl_new_OFF, label='ctrl new OFF', color='lightcyan')
    ax.fill_between(XAXIS, mean_ctrl_new_OFF + sem_ctrl_new_OFF, mean_ctrl_new_OFF - sem_ctrl_new_OFF,
                    color='lightcyan', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_new_OFF, label='stim new OFF', color='darkblue')
    ax.fill_between(XAXIS, mean_stim_new_OFF + sem_stim_new_OFF, mean_stim_new_OFF - sem_stim_new_OFF,
                    color='darkblue', edgecolor='none', alpha=.15)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrUp (<2/3) and PyrDown (>3/2) all', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_remain_new{ext}',
                    dpi=300, bbox_inches='tight')
    
    ## PLOTTING, ONLY REMAINERS ##
    ## ON ##
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_remain_ON, label='ctrl. remain ON', color='firebrick')
    ax.fill_between(XAXIS, mean_ctrl_remain_ON + sem_ctrl_remain_ON, mean_ctrl_remain_ON - sem_ctrl_remain_ON,
                    color='firebrick', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_remain_ON, label='stim. remain ON', color=(87/255, 90/255, 187/255))
    ax.fill_between(XAXIS, mean_stim_remain_ON + sem_stim_remain_ON, mean_stim_remain_ON - sem_stim_remain_ON,
                    color=(87/255, 90/255, 187/255), edgecolor='none', alpha=.15)

    _annotate_pvals(ax, pvals_remain_ON, y_level=4.05, star=True)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrUp (<2/3) remainers', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_remain_ON{ext}',
                    dpi=300, bbox_inches='tight')
    
    ## OFF ##
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_remain_OFF, label='ctrl. remain OFF', color='purple')
    ax.fill_between(XAXIS, mean_ctrl_remain_OFF + sem_ctrl_remain_OFF, mean_ctrl_remain_OFF - sem_ctrl_remain_OFF,
                    color='purple', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_remain_OFF, label='stim. remain OFF', color=(78/255, 84/255, 206/255))
    ax.fill_between(XAXIS, mean_stim_remain_OFF + sem_stim_remain_OFF, mean_stim_remain_OFF - sem_stim_remain_OFF,
                    color=(78/255, 84/255, 206/255), edgecolor='none', alpha=.15)
    
    _annotate_pvals(ax, pvals_remain_OFF, y_level=4.05, star=True)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrDown (>3/2) remainers', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_remain_OFF{ext}',
                    dpi=300, bbox_inches='tight')
        
    
    # PLOTTING ONLY LEAVERS
    ## ON ##
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_leave_ON, label='ctrl. leave ON', color='firebrick')
    ax.fill_between(XAXIS, mean_ctrl_leave_ON + sem_ctrl_leave_ON, mean_ctrl_leave_ON - sem_ctrl_leave_ON,
                    color='firebrick', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_leave_ON, label='stim. leave ON', color=(87/255, 90/255, 187/255))
    ax.fill_between(XAXIS, mean_stim_leave_ON + sem_stim_leave_ON, mean_stim_leave_ON - sem_stim_leave_ON,
                    color=(87/255, 90/255, 187/255), edgecolor='none', alpha=.15)

    _annotate_pvals(ax, pvals_leave_ON, y_level=4.05, star=True)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrUp (<2/3) leavers', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_leave_ON{ext}',
                    dpi=300, bbox_inches='tight')
        
    ## OFF ##
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_leave_OFF, label='ctrl. leave OFF', color='purple')
    ax.fill_between(XAXIS, mean_ctrl_leave_OFF + sem_ctrl_leave_OFF, mean_ctrl_leave_OFF - sem_ctrl_leave_OFF,
                    color='purple', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_leave_OFF, label='stim. leave OFF', color=(78/255, 84/255, 206/255))
    ax.fill_between(XAXIS, mean_stim_leave_OFF + sem_stim_leave_OFF, mean_stim_leave_OFF - sem_stim_leave_OFF,
                    color=(78/255, 84/255, 206/255), edgecolor='none', alpha=.15)

    _annotate_pvals(ax, pvals_leave_OFF, y_level=4.05, star=True)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrDown (>3/2) leavers', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_leave_OFF{ext}',
                    dpi=300, bbox_inches='tight')
    
        
    ## PLOTTING NEW ##
    ## ON ##
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_new_ON, label='ctrl new ON', color='moccasin')
    ax.fill_between(XAXIS, mean_ctrl_new_ON + sem_ctrl_new_ON, mean_ctrl_new_ON - sem_ctrl_new_ON,
                    color='moccasin', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_new_ON, label='stim new ON', color='darkorange')
    ax.fill_between(XAXIS, mean_stim_new_ON + sem_stim_new_ON, mean_stim_new_ON - sem_stim_new_ON,
                    color='darkorange', edgecolor='none', alpha=.15)
    
    _annotate_pvals(ax, pvals_new_ON, y_level=4.05, star=True)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrUp (<2/3) new', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_new_ON{ext}',
                    dpi=300, bbox_inches='tight')
        
    ## OFF ##
    fig, ax = plt.subplots(figsize=(2.6,2))
        
    ax.plot(XAXIS, mean_ctrl_new_OFF, label='ctrl new OFF', color='lightcyan')
    ax.fill_between(XAXIS, mean_ctrl_new_OFF + sem_ctrl_new_OFF, mean_ctrl_new_OFF - sem_ctrl_new_OFF,
                    color='lightcyan', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_new_OFF, label='stim new OFF', color='darkblue')
    ax.fill_between(XAXIS, mean_stim_new_OFF + sem_stim_new_OFF, mean_stim_new_OFF - sem_stim_new_OFF,
                    color='darkblue', edgecolor='none', alpha=.15)
    
    _annotate_pvals(ax, pvals_new_OFF, y_level=4.05, star=True)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrDown (>3/2) new', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_new_OFF{ext}',
                    dpi=300, bbox_inches='tight')
    ## PLOTTING REMAIN LEAVE NEW DONE ##
    
    
    
    ## PLOTTING CTRL VS STIM (SEPARATE DETECTION) ##
    # extract mean and SEM
    mean_ctrl_only_ON = np.mean(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    sem_ctrl_only_ON = sem(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    
    mean_stim_only_ON = np.mean(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    sem_stim_only_ON = sem(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_only_OFF = np.mean(mean_prof_ctrl_only_OFF_matched, axis=0)[2500:2500+5*1250]
    sem_ctrl_only_OFF = sem(mean_prof_ctrl_only_OFF_matched, axis=0)[2500:2500+5*1250]
    
    mean_stim_only_OFF = np.mean(mean_prof_stim_only_OFF_matched, axis=0)[2500:2500+5*1250]
    sem_stim_only_OFF = sem(mean_prof_stim_only_OFF_matched, axis=0)[2500:2500+5*1250]
    
    # plotting
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_only_ON, label='mean_ctrl_ON', color='lightcoral')
    ax.fill_between(XAXIS, mean_ctrl_only_ON + sem_ctrl_only_ON, mean_ctrl_only_ON - sem_ctrl_only_ON,
                    color='lightcoral', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_only_ON, label='mean_stim_ON', color='firebrick')
    ax.fill_between(XAXIS, mean_stim_only_ON + sem_stim_only_ON, mean_stim_only_ON - sem_stim_only_ON,
                    color='firebrick', edgecolor='none', alpha=.15)
        
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrUp (<2/3)', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_only_ON{ext}',
                    dpi=300, bbox_inches='tight')
    
    
    mean_ctrl_only_ON = np.mean(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    sem_ctrl_only_ON = sem(mean_prof_ctrl_only_ON, axis=0)[2500:2500+5*1250]
    
    mean_stim_only_ON = np.mean(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    sem_stim_only_ON = sem(mean_prof_stim_only_ON, axis=0)[2500:2500+5*1250]
    
    # plot
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_only_OFF, label='mean_ctrl_OFF', color='violet')
    ax.fill_between(XAXIS, mean_ctrl_only_OFF + sem_ctrl_only_OFF, mean_ctrl_only_OFF - sem_ctrl_only_OFF,
                    color='violet', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_only_OFF, label='mean_stim_OFF', color='purple')
    ax.fill_between(XAXIS, mean_stim_only_OFF + sem_stim_only_OFF, mean_stim_only_OFF - sem_stim_only_OFF,
                    color='purple', edgecolor='none', alpha=.15)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrDown (>3/2)', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_stim_only_OFF{ext}',
                    dpi=300, bbox_inches='tight')
    
    
    # single session overlay
    save_path = ctrl_stim_stem / 'single_sessions'
    
    for i in range(len(mean_prof_ctrl_only_ON_sess)):
        fig, ax = plt.subplots(figsize=(2.6,2))
    
        ctrl_trace = mean_prof_ctrl_only_ON_sess[i][2500:2500+5*1250]
        stim_trace = mean_prof_stim_only_ON_sess[i][2500:2500+5*1250]
    
        ax.plot(XAXIS, ctrl_trace, label='ctrl.', color='firebrick')
        ax.plot(XAXIS, stim_trace, label='stim.', color='royalblue')
    
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
    
        ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
               ylabel='Firing rate (Hz)')
        ax.set_title(f'{paths[i][-17:]} ON', fontsize=10)
        ax.legend(fontsize=5, frameon=False)
    
        fig.savefig(save_path / f'{paths[i][-17:]}_ON.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    for i in range(len(mean_prof_ctrl_only_OFF_sess)):
        fig, ax = plt.subplots(figsize=(2.6,2))
    
        ctrl_trace = mean_prof_ctrl_only_OFF_sess[i][2500:2500+5*1250]
        stim_trace = mean_prof_stim_only_OFF_sess[i][2500:2500+5*1250]
    
        ax.plot(XAXIS, ctrl_trace, label='ctrl.', color='purple')
        ax.plot(XAXIS, stim_trace, label='stim.', color='royalblue')
    
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
    
        ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
               ylabel='Firing rate (Hz)')
        ax.set_title(f'{paths[i][-17:]} OFF', fontsize=10)
        ax.legend(fontsize=5, frameon=False)
    
        fig.savefig(save_path / f'{paths[i][-17:]}_OFF.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    

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
                             ylabel='Firing rate (Hz)',
                             dpi=300,
                             save=True,
                             savepath=rf'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\ctrl_stim\{exp}_ctrl_only_stim_only_ON_amp')
    
    amp_ctrl_only_OFF = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_ctrl_only_OFF_matched
        ], 
        axis=1)
    amp_stim_only_OFF = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_stim_only_OFF_matched
        ], 
        axis=1)
    
    plot_violin_with_scatter(amp_ctrl_only_OFF, amp_stim_only_OFF, 
                             'violet', 'purple',
                             xticklabels=['ctrl.', 'stim.'],
                             paired=False,
                             showscatter=True,
                             showmedians=True,
                             ylabel='Firing rate (Hz)',
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
    
    
    ## dist v amp_mean ##  
    # filter and clean data
    all_amp_remain_ON_delta_mean_filt_dist = []
    all_ctrl_stim_lick_distance_delta_filt = []
    for amp, dist in zip(all_amp_remain_ON_delta_mean, all_ctrl_stim_lick_distance_delta):
        if not np.isnan(amp) and not np.isnan(dist) and -DELTA_THRES < amp < 2:
            all_amp_remain_ON_delta_mean_filt_dist.append(amp)
            all_ctrl_stim_lick_distance_delta_filt.append(dist)
    
    all_amp_remain_ON_delta_mean_filt_time = []
    all_ctrl_stim_lick_time_delta_filt = []
    for amp, time in zip(all_amp_remain_ON_delta_mean, all_ctrl_stim_lick_time_delta):
        if not np.isnan(amp) and not np.isnan(time) and time < 800 and -DELTA_THRES < amp < 2:
            all_amp_remain_ON_delta_mean_filt_time.append(amp)
            all_ctrl_stim_lick_time_delta_filt.append(time)
    
    # regression
    slope, intercept, r, p, _ = linregress(all_amp_remain_ON_delta_mean_filt_dist, all_ctrl_stim_lick_distance_delta_filt)
    
    # plot
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    
    # scatter
    ax.scatter(all_amp_remain_ON_delta_mean_filt_dist, all_ctrl_stim_lick_distance_delta_filt,
               color='firebrick', edgecolor='none', s=30, alpha=0.8)
    
    # regression line
    x_vals = np.linspace(min(all_amp_remain_ON_delta_mean_filt_dist), max(all_amp_remain_ON_delta_mean_filt_dist), 100)
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
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\population_ctrl_stim\{exp}_remain_ON_delta_amp_mean_dist{ext}',
                    dpi=300,
                    bbox_inches='tight')
        
    # mean v time
    slope, intercept, r, p, _ = linregress(all_amp_remain_ON_delta_mean_filt_time, all_ctrl_stim_lick_time_delta_filt)
    
    # plot
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    
    # scatter
    ax.scatter(all_amp_remain_ON_delta_mean_filt_time, all_ctrl_stim_lick_time_delta_filt,
               color='firebrick', edgecolor='none', s=30, alpha=0.8)
    
    # regression line
    x_vals = np.linspace(min(all_amp_remain_ON_delta_mean_filt_time), max(all_amp_remain_ON_delta_mean_filt_time), 100)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='k', lw=1)
    
    # r and p text
    ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)
    
    # labels and style
    ax.set_xlabel('delta ON (Hz)', fontsize=10)
    ax.set_ylabel('delta lick time (stim − ctrl)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\population_ctrl_stim\{exp}_remain_ON_delta_amp_mean_time{ext}',
                    dpi=300,
                    bbox_inches='tight')
    
    
    ## OFF - the same 
    ## dist v amp_mean ##  
    # filter and clean data
    all_amp_OFF_delta_mean_filt_dist = []
    all_ctrl_stim_lick_distance_delta_filt = []
    for amp, dist in zip(all_amp_OFF_delta_mean, all_ctrl_stim_lick_distance_delta):
        if not np.isnan(amp) and not np.isnan(dist) and -2 < amp < DELTA_THRES:
            all_amp_OFF_delta_mean_filt_dist.append(amp)
            all_ctrl_stim_lick_distance_delta_filt.append(dist)
    
    all_amp_OFF_delta_mean_filt_time = []
    all_ctrl_stim_lick_time_delta_filt = []
    for amp, time in zip(all_amp_OFF_delta_mean, all_ctrl_stim_lick_time_delta):
        if not np.isnan(amp) and not np.isnan(time) and -400 < time < 1000 and -2 < amp < DELTA_THRES:
            all_amp_OFF_delta_mean_filt_time.append(amp)
            all_ctrl_stim_lick_time_delta_filt.append(time)
    
    # regression
    slope, intercept, r, p, _ = linregress(all_amp_OFF_delta_mean_filt_dist, all_ctrl_stim_lick_distance_delta_filt)
    
    # plot
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    
    # scatter
    ax.scatter(all_amp_OFF_delta_mean_filt_dist, all_ctrl_stim_lick_distance_delta_filt,
               color='purple', edgecolor='none', s=30, alpha=0.8)
    
    # regression line
    x_vals = np.linspace(min(all_amp_OFF_delta_mean_filt_dist), max(all_amp_OFF_delta_mean_filt_dist), 100)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='k', lw=1)
    
    # r and p text
    ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)
    
    # labels and style
    ax.set_xlabel('delta OFF (Hz)', fontsize=10)
    ax.set_ylabel('delta lick dist. (stim − ctrl)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\population_ctrl_stim\{exp}_remain_OFF_delta_amp_mean_dist{ext}',
                    dpi=300,
                    bbox_inches='tight')
        
    # mean v time
    slope, intercept, r, p, _ = linregress(all_amp_OFF_delta_mean_filt_time, all_ctrl_stim_lick_time_delta_filt)
    
    # plot
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    
    # scatter
    ax.scatter(all_amp_OFF_delta_mean_filt_time, all_ctrl_stim_lick_time_delta_filt,
               color='purple', edgecolor='none', s=30, alpha=0.8)
    
    # regression line
    x_vals = np.linspace(min(all_amp_OFF_delta_mean_filt_time), max(all_amp_OFF_delta_mean_filt_time), 100)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='k', lw=1)
    
    # r and p text
    ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)
    
    # labels and style
    ax.set_xlabel('delta OFF (Hz)', fontsize=10)
    ax.set_ylabel('delta lick time (stim − ctrl)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\population_ctrl_stim\{exp}_remain_OFF_delta_amp_mean_time{ext}',
                    dpi=300,
                    bbox_inches='tight')
    
    
#%% run 
if __name__ == '__main__':
    main(pathHPCLCopt, 'HPCLC')
    main(pathHPCLCtermopt, 'HPCLCterm')