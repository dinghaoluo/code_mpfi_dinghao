# -*- coding: utf-8 -*-
"""
Created on Thur 13 Feb 14:36:41 2025

plot stim responses for LC stim recordings 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import sys

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLCopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

behaviour = pd.read_pickle(
    r'Z:/Dinghao/code_dinghao/behaviour/all_LC_sessions.pkl'
    )
cell_profiles = pd.read_pickle(
    r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl'
    )


#%% create stim rasters 
for path in paths:
    recname = path[-17:]
    
    print(recname)
    
    beh = behaviour.loc[recname]
    stim_conds = [trial[15] for trial in beh['trial_statements']][1:]  # index 15 is the stim condition 
    
    rasters = np.load(
        r'Z:/Dinghao/code_dinghao/LC_ephys/all_sessions/{}/{}_all_rasters.npy'
        .format(recname, recname),
        allow_pickle=True).item()
    trains = np.load(
        r'Z:/Dinghao/code_dinghao/LC_ephys/all_sessions/{}/{}_all_trains.npy'
        .format(recname, recname),
        allow_pickle=True).item()
    
    for clu in list(rasters.keys()):
        if cell_profiles.loc[clu]['identity']=='tagged':
            curr_stim_spike_map = [spike_map for trial, spike_map 
                                   in enumerate(rasters[clu]) 
                                   if stim_conds[trial]!='0']
            curr_ctrl_spike_map = [spike_map for trial, spike_map
                                   in enumerate(rasters[clu][2:])
                                   if stim_conds[trial-2]!='0']
            
            curr_stim_trains_prof = np.mean([train for trial, train
                                             in enumerate(trains[clu])
                                             if stim_conds[trial]!='0'],
                                            axis=0)
            curr_ctrl_trains_prof = np.mean([train for trial, train
                                             in enumerate(trains[clu][2:])
                                             if stim_conds[trial-2]!='0'],
                                            axis=0)
            
            fig, axs = plt.subplots(2, 1, figsize=(2.4, 2.8))
            
            for idx in range(len(curr_stim_spike_map)):
                curr_stim_raster = [s/1250-3 for s, spike
                                    in enumerate(curr_stim_spike_map[idx])
                                    if spike]
                axs[0].scatter(curr_stim_raster,
                               [idx+1]*len(curr_stim_raster), 
                               color='lightsteelblue', s=1, ec='none')
                
            for idx in range(len(curr_ctrl_spike_map)):
                curr_ctrl_raster = [s/1250-3 for s, spike
                                    in enumerate(curr_ctrl_spike_map[idx])
                                    if spike]
                axs[1].scatter(curr_ctrl_raster,
                               [idx+1]*len(curr_ctrl_raster), 
                               color='grey', s=1, ec='none')
            
            axt0 = axs[0].twinx()
            axt0.plot(np.arange(-3750, 12500-3750)/1250,
                      curr_stim_trains_prof, 
                      color='royalblue')
            axt0.set(ylabel='spike rate\n(Hz)')
            axs[0].set(xlabel='time from run-onset (s)', xlim=(-1, 4), xticks=(0, 2, 4),
                     ylabel='trial #',
                     title='{}\nstim.'.format(clu))
            
            axt1 = axs[1].twinx()
            axt1.plot(np.arange(-3750, 12500-3750)/1250,
                      curr_ctrl_trains_prof, 
                      color='k')
            axt1.set(ylabel='spike rate\n(Hz)')
            axs[1].set(xlabel='time from run-onset (s)', xlim=(-1, 4), xticks=(0, 2, 4),
                     ylabel='trial #',
                     title='ctrl.')
            
            axs[0].spines['top'].set_visible(False)
            axt0.spines['top'].set_visible(False)
            
            axs[1].spines['top'].set_visible(False)
            axt1.spines['top'].set_visible(False)
            
            fig.tight_layout()
            
            for ext in ('.png', '.pdf'):
                fig.savefig(
                    r'Z:\Dinghao\code_dinghao\LC_ephys\tagged_stim_rasters'
                    rf'\{clu}{ext}',
                    dpi=300,
                    bbox_inches='tight')
