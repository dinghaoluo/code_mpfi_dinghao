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
for cluname in cell_profiles.index:
    recname = cluname[:17]
    print(recname)
    
    cell = cell_profiles.loc[cluname]
    
    cell_identity = cell['cell_identity']  # 'pyr' or 'int'
    ratiotype = cell['class']  # run-onset activated etc.
    
    ctrl_mean = cell['prof_ctrl_mean']
    ctrl_sem = cell['prof_ctrl_sem']
    stim_mean = cell['prof_stim_mean']
    stim_sem = cell['prof_stim_sem']
    
    # plotting 
    fig, ax = plt.subplots(figsize=(2,1.8))
        
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
    ax.set(title=f'{cluname}\n{ratiotype}',
           xlabel='time from run-onset (s)', ylabel='spike rate (Hz)',
           xlim=(-time_bef, time_aft), xticks=(0,2,4))
    ax.title.set_fontsize(10)
    
    fig.tight_layout()
    plt.show()
    
    # pyr or int folder? 
    pyr_or_int_folder = \
        r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{}\profiles_ctrl_stim_pyr'.format(recname) \
            if cell_identity=='pyr' else \
        r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{}\profiles_ctrl_stim_int'.format(recname)
    os.makedirs(pyr_or_int_folder, exist_ok=True)
    for ext in ['.png', '.pdf']:
        fig.savefig(
            f'{pyr_or_int_folder}\{cluname}{ext}',
            dpi=300,
            bbox_inches='tight')
     
    plt.close(fig)