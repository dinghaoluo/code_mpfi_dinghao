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
    r'Z:/Dinghao/code_dinghao/behaviour/all_LC_sessions.pkl')
cell_profiles = pd.read_pickle(
    r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


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
            curr_stim_trains_prof = np.mean([train for trial, train
                                             in enumerate(trains[clu])
                                             if stim_conds[trial]!='0'],
                                            axis=0)
            
            fig, ax = plt.subplots(figsize=(2.3, 1.55))
            
            for idx in range(len(curr_stim_spike_map)):
                curr_stim_out_raster = [s/1250-3 for s, spike
                                        in enumerate(curr_stim_spike_map[idx])
                                        if spike and (s<3750 or s>3750+1250)]
                curr_stim_in_raster = [s/1250-3 for s, spike
                                       in enumerate(curr_stim_spike_map[idx])
                                       if spike and 3750<s<3750+1250]
                ax.scatter(curr_stim_out_raster,
                           [idx+1]*len(curr_stim_out_raster), 
                           color='grey', s=1, ec='none')
                ax.scatter(curr_stim_in_raster,
                           [idx+1]*len(curr_stim_in_raster), 
                           color='royalblue', s=1, ec='none')
            
            axt = ax.twinx()
            axt.plot(np.arange(-3750, 12500-3750)/1250,
                     curr_stim_trains_prof, 
                     color='royalblue')
            
            axt.set(ylabel='spike rate\n(Hz)')
            
            ax.set(xlabel='time from run-onset (s)', xlim=(-1, 4), xticks=(0, 2, 4),
                   ylabel='trial #',
                   title='{}\nrun-onset aligned'.format(clu))
            
            ax.spines['top'].set_visible(False)
            axt.spines['top'].set_visible(False)
            
            fig.tight_layout()
            
            for ext in ('.png', '.pdf'):
                fig.savefig(
                    r'Z:\Dinghao\code_dinghao\LC_ephys\tagged_stim_rasters\{}{}'
                    .format(clu, ext),
                    dpi=300,
                    bbox_inches='tight')