# -*- coding: utf-8 -*-
"""
Created on Wed 27 Sept 14:44:27 2023
Modified on Fri 10 Nov 
Modified on Fri 20 Dec 2024:
    - merged everything together (HPCLC, HPCLCterm, pyr, int, you name it!)
    - use the HPC_all_profiles.pkl dataframe for info; discard everything else 

compare all HPC cell's spiking profile between baseline ctrl and stim 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
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


#%% parameters
run_onset_bin = 3750  # in samples
samp_freq = 1250  # in Hz
time_bef = 1  # in seconds 
time_aft = 4  # in seconds 
xaxis = np.arange(-samp_freq*time_bef, samp_freq*time_aft) / samp_freq
prof_window = (run_onset_bin-samp_freq*time_bef, run_onset_bin+samp_freq*time_aft)


#%% load dataframe 
print('loading dataframe...')
cell_profiles = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')


#%% main 
recname = cell_profiles.index[0]
for cluname in cell_profiles.index:
    if cluname[:17]!=recname:
        recname = cluname[:17]
        print(recname)
        
    clu_n = cluname.split(' ')[1]  # for plot titles 
    
    cell = cell_profiles.loc[cluname]
    
    cell_identity = cell['cell_identity']  # 'pyr' or 'int'
    ratiotype = cell['class_ctrl']  # run-onset activated etc.
    if ratiotype=='run-onset ON':
        rt = 'RO-ON'
    elif ratiotype=='run-onset OFF':
        rt = 'RO-OFF'
    else:
        rt = 'unresp.'
    
    ctrl_mean = cell['prof_ctrl_mean']
    ctrl_sem = cell['prof_ctrl_sem']
    stim_mean = cell['prof_stim_mean']
    stim_sem = cell['prof_stim_sem']
    
    # plotting 
    fig, ax = plt.subplots(figsize=(1.6,1.2))
        
    ctrlln, = ax.plot(
        xaxis, 
        ctrl_mean[prof_window[0]:prof_window[1]], 
        color='grey')
    ax.fill_between(
        xaxis, 
        ctrl_mean[prof_window[0]:prof_window[1]]+ctrl_sem[prof_window[0]:prof_window[1]],
        ctrl_mean[prof_window[0]:prof_window[1]]-ctrl_sem[prof_window[0]:prof_window[1]],
        alpha=.25, color='grey', edgecolor='none')
    stimln, = ax.plot(
        xaxis, 
        stim_mean[prof_window[0]:prof_window[1]], 
        color='royalblue')
    ax.fill_between(
        xaxis, 
        stim_mean[prof_window[0]:prof_window[1]]+stim_sem[prof_window[0]:prof_window[1]],
        stim_mean[prof_window[0]:prof_window[1]]-stim_sem[prof_window[0]:prof_window[1]],
        alpha=.25, color='royalblue', edgecolor='none')
    
    ax.legend(
        [ctrlln, stimln], ['ctrl.', 'stim.'], 
        frameon=False, fontsize=6)
    for p in ['top', 'right']:
        ax.spines[p].set_visible(False)
    ax.set(title=f'{recname}\n{clu_n} {rt}',
           xlabel='time from run-onset (s)', ylabel='spike rate (Hz)',
           xlim=(-time_bef, time_aft), xticks=(0,2,4))
    ax.title.set_fontsize(10)
    
    # pyr or int folder? 
    pyr_dir = rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}\profiles_ctrl_stim_pyr'
    int_dir = rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}\profiles_ctrl_stim_int'
    os.makedirs(pyr_dir, exist_ok=True)
    os.makedirs(int_dir, exist_ok=True)
    
    for ext in ['.png', '.pdf']:
        if cell_identity == 'pyr':
            fig.savefig(
                    rf'{pyr_dir}\{cluname}{ext}',
                    dpi=300,
                    bbox_inches='tight'
                    )
            if cell['rectype'] == 'HPCLC':
                fig.savefig(
                        rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_profiles\HPC_LC_pyr\{cluname}{ext}',
                        dpi=300,
                        bbox_inches='tight'
                        )
            if cell['rectype'] == 'HPCLCterm':
                fig.savefig(
                        rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_profiles\HPC_LCterm_pyr\{cluname}{ext}',
                        dpi=300,
                        bbox_inches='tight'
                        )
        else:
            fig.savefig(
                    rf'{int_dir}\{cluname}{ext}',
                    dpi=300,
                    bbox_inches='tight'
                    )
            if cell['rectype'] == 'HPCLC':
                fig.savefig(
                        rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_profiles\HPC_LC_int\{cluname}{ext}',
                        dpi=300,
                        bbox_inches='tight'
                        )
            if cell['rectype'] == 'HPCLCterm':
                fig.savefig(
                        rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_profiles\HPC_LCterm_int\{cluname}{ext}',
                        dpi=300,
                        bbox_inches='tight'
                        )

    plt.close(fig)