# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:30:26 2025

plot ctrl vs stim profiles for LC cells 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import sys 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import mpl_formatting
mpl_formatting()


#%% main 
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )

samp_freq = 1250  # Hz
xaxis = np.arange(samp_freq*5) / samp_freq - 1  # -1~4
for clu in cell_profiles.itertuples():
    cluname = clu.Index
    
    stim_mean = clu.stim_mean
    if len(stim_mean)==0: continue  # empty list check
    stim_sem = clu.stim_sem
    ctrl_mean = clu.ctrl_mean
    ctrl_sem = clu.ctrl_sem
    
    fig, ax = plt.subplots(figsize=(2.55,1.8))
    ctrlln, = ax.plot(xaxis, ctrl_mean[3750-1250:3750+1250*4],
                      color='grey')
    ax.fill_between(xaxis, (ctrl_mean+ctrl_sem)[3750-1250:3750+1250*4],
                           (ctrl_mean-ctrl_sem)[3750-1250:3750+1250*4],
                    color='grey', edgecolor='none', alpha=.35)
    stimln, = ax.plot(xaxis, stim_mean[3750-1250:3750+1250*4],
                      color='royalblue')
    ax.fill_between(xaxis, (stim_mean+stim_sem)[3750-1250:3750+1250*4],
                           (stim_mean-stim_sem)[3750-1250:3750+1250*4],
                    color='royalblue', edgecolor='none', alpha=.35)
    
    ax.set(title=cluname,
           xlabel='time from run-onset (s)',
           ylabel='spike rate (Hz)')
    
    ax.legend((ctrlln, stimln), ('ctrl.', 'stim.'),
              frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\LC_ephys'
        rf'\single_cell_stim_response\{cluname}.png',
        dpi=300,
        bbox_inches='tight'
        )