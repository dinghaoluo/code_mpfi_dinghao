# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 16:25:22 2025

Modulation significance test against shuffled trial IDs

@author: Dinghao Luo
"""

#%% imports 
import os 
import sys 

import pandas as pd 
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt


#%% parameters
SAMP_FREQ = 1250  # Hz 
COMP_BINS = np.arange(int(SAMP_FREQ * 3.5), int(SAMP_FREQ * 7.5))
MIN_BINS = int(SAMP_FREQ * .3)


#%% load profiles 
print('loading dataframe...')
cell_profiles = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 


#%% main 
prop_activated = []
prop_inhibited = []

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
    
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    curr_df_pyr_ON = curr_df_pyr[curr_df_pyr['class_ctrl']=='run-onset ON']
    curr_df_pyr_OFF = curr_df_pyr[curr_df_pyr['class_ctrl']=='run-onset OFF']
    pyr_list = curr_df_pyr.index.tolist()
    pyr_ON_list = curr_df_pyr_ON.index.tolist()
    pyr_OFF_list = curr_df_pyr_OFF.index.tolist()
    
    n_cells = len(pyr_list)
    n_activated = 0
    n_inhibited = 0
    
    for cluname in pyr_list:
        if cluname in pyr_ON_list:
            clustr = f'{cluname} ON'
        elif cluname in pyr_OFF_list:
            clustr = f'{cluname} OFF'
        else:
            clustr = f'{cluname} other'
        
        train = trains[cluname]
        
        ctrl_arr = train[ctrl_idx][:, COMP_BINS]
        stim_arr = train[stim_idx][:, COMP_BINS]
        
        # paired t-test per bin
        t_stat, p_vals = ttest_rel(stim_arr, ctrl_arr, axis=0, nan_policy='omit')
        
        # significance masks
        act_bins = (p_vals < 0.05) & (stim_arr.mean(axis=0) > ctrl_arr.mean(axis=0))
        inh_bins = (p_vals < 0.05) & (stim_arr.mean(axis=0) < ctrl_arr.mean(axis=0))
        
        act_count = act_bins.sum()
        inh_count = inh_bins.sum()
        
        # classification by number of bins
        if act_count >= MIN_BINS and inh_count < MIN_BINS:
            modulation = 'activated'
            n_activated += 1
        elif inh_count >= MIN_BINS and act_count < MIN_BINS:
            modulation = 'inhibited'
            n_inhibited += 1
        elif act_count >= MIN_BINS and inh_count >= MIN_BINS:
            if act_count > inh_count:
                modulation = 'activated'
                n_activated += 1
            elif inh_count > act_count:
                modulation = 'inhibited'
                n_inhibited += 1
            else:
                modulation = 'ambiguous'
        else:
            modulation = 'ns'
        
        # plot stim & ctrl means
        fig, ax = plt.subplots(figsize=(4.5, 2.5))
        time_axis = np.arange(len(COMP_BINS)) / SAMP_FREQ
        
        stim_mean = stim_arr.mean(axis=0)
        ctrl_mean = ctrl_arr.mean(axis=0)
        
        ax.plot(time_axis, stim_mean, color='royalblue', lw=1, label='stim mean')
        ax.plot(time_axis, ctrl_mean, color='grey', lw=1, label='ctrl mean')
        
        # mark significant bins
        sig_y = max(stim_mean.max(), ctrl_mean.max()) + 0.05 * (stim_mean.max() - stim_mean.min())
        ax.scatter(time_axis[act_bins], np.full(act_bins.sum(), sig_y),
                   marker='^', color='red', s=10, label='activated bins')
        ax.scatter(time_axis[inh_bins], np.full(inh_bins.sum(), sig_y),
                   marker='v', color='blue', s=10, label='inhibited bins')
        
        ax.set_xlabel('time (s from start of COMP_BINS)')
        ax.set_ylabel('firing rate (Hz)')
        ax.set_title(f'{clustr}\nact={act_count} inh={inh_count} -> {modulation}')
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()
        
        for ext in ['.png', '.pdf']:
            fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_profiles\HPC_LC_pyr_mod_sig\{clustr}{ext}',
                        dpi=300,
                        bbox_inches='tight')
            
    # store per-session proportions
    prop_activated.append(n_activated / n_cells)
    prop_inhibited.append(n_inhibited / n_cells)
    
