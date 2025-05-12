# -*- coding: utf-8 -*-
"""
Created on Fri 20 Dec 17:30:12 2024
Modified on 10 May Sat 2025

plot rasters of HPC cells in ctrl and stim trials 
modified to label ON and OFF cells 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import pickle 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys 
import os 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLC = rec_list.pathHPCLCopt
pathHPCLCterm = rec_list.pathHPCLCtermopt
paths = pathHPCLC + pathHPCLCterm


#%% load dataframe 
print('loading dataframe...')
cell_profiles = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 


#%% parameters 
time_bef = 1  # second 
time_aft = 4
samp_freq = 1250  # Hz 


#%% main 
for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    
    rasters = np.load(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}\{recname}_all_rasters.npy',
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
    
    # extract stim times
    run_onsets = beh['run_onsets'][1:-1]
    pulse_times = beh['pulse_times'][1:]
    pulse_times_aligned = [p[0]-r for p, r in zip(pulse_times, run_onsets)
                           if not np.isnan(r) and p]
    
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    curr_df_pyr_ON = curr_df_pyr[curr_df_pyr['class_ctrl']=='run-onset ON']
    curr_df_pyr_OFF = curr_df_pyr[curr_df_pyr['class_ctrl']=='run-onset OFF']
    pyr_list = curr_df_pyr.index.tolist()
    pyr_ON_list = curr_df_pyr_ON.index.tolist()
    pyr_OFF_list = curr_df_pyr_OFF.index.tolist()
        
    tot_time = 1250 + 5000  # 1 s before, 4 s after 
    
    for cluname in pyr_list:
        if cluname in pyr_ON_list:
            clustr = f'{cluname} ON'
        elif cluname in pyr_OFF_list:
            clustr = f'{cluname} OFF'
        else:
            clustr = f'{cluname} other'
        
        raster = rasters[cluname]
    
        ctrl_matrix = raster[ctrl_idx]
        stim_matrix = raster[stim_idx]
        
        # plotting 
        fig, axs = plt.subplots(2, 1, figsize=(2.1,2.1))
        fig.suptitle(clustr, fontsize=10)
        
        for line in range(len(ctrl_idx)):
            axs[0].scatter(np.where(ctrl_matrix[line]==1)[0]/samp_freq-3,
                           [line+1]*int(sum(ctrl_matrix[line])),
                           c='grey', ec='none', s=2)
            axs[1].scatter(np.where(stim_matrix[line]==1)[0]/samp_freq-3,
                           [line+1]*int(sum(stim_matrix[line])),
                           c='royalblue', ec='none', s=2)
            try:
                axs[1].plot([pulse_times_aligned[line]/samp_freq, pulse_times_aligned[line]/samp_freq],
                            [line, line+1], c='red', lw=1)
            except IndexError:
                continue 
                        
        for i in range(2):
            axs[i].set(xticks=[0,2,4], xlim=(-1, 4),
                       ylabel='trial #')
            for p in ['top', 'right']:
                axs[i].spines[p].set_visible(False)
            
        # only set xlabel for ax 1
        axs[1].set(xlabel='time from run-onset (s)')

        fig.tight_layout()
        
        # save figure
        if path in pathHPCLC:
            fig.savefig(
                    rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_stim_ctrl_rasters\HPC_LC_pyr\{clustr}.png',
                    dpi=300,
                    bbox_inches='tight'
                    )
        if path in pathHPCLCterm:
            fig.savefig(
                    rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_stim_ctrl_rasters\HPC_LCterm_pyr\{clustr}.png',
                    dpi=300,
                    bbox_inches='tight'
                    )

        plt.close(fig)