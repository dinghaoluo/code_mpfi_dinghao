# -*- coding: utf-8 -*-
"""
Created on Thur 13 Feb 14:36:41 2025

analyse and plot stim responses for LC stim recordings 

@author: Dinghao Luo
"""


#%% imports 
from pathlib import Path 

import numpy as np 
import pandas as pd
import pickle 
import matplotlib.pyplot as plt 

from plotting_functions import plot_violin_with_scatter
from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathLCopt


#%% paths and parameters 
beh_stem         = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')
all_session_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
stim_raster_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/stim_effects/single_cell_stim_rasters')

BEF = 3  # s
AFT = 7  # s 
SAMP_FREQ = 1250  # Hz
XAXIS = np.arange(-BEF * SAMP_FREQ, AFT * SAMP_FREQ) / SAMP_FREQ

AMP_WINDOW_LOW_S  = 0  # seconds; for calculating amplitude
AMP_WINDOW_HIGH_S = 1
AMP_WINDOW_LOW    = int((AMP_WINDOW_LOW_S + BEF) * SAMP_FREQ)
AMP_WINDOW_HIGH   = int((AMP_WINDOW_HIGH_S + BEF) * SAMP_FREQ)


#%% load data 
cell_prop = pd.read_pickle(r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')

# keys
clu_keys = list(cell_prop.index)
tagged_keys, putative_keys = [], []
tagged_RO_keys, putative_RO_keys = [], []
RO_keys = []
for clu in cell_prop.itertuples():
    if clu.identity == 'tagged':
        tagged_keys.append(clu.Index)
        if clu.run_onset_peak:
            tagged_RO_keys.append(clu.Index)
            RO_keys.append(clu.Index)
    if clu.identity == 'putative':
        putative_keys.append(clu.Index)
        if clu.run_onset_peak:
            putative_RO_keys.append(clu.Index)
            RO_keys.append(clu.Index)


#%% main loop
all_ctrl_amps = []
all_stim_amps = []

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    beh_path = beh_stem / f'{recname}.pkl'
    
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)
    
    stim_cds = [trial[15] for trial in beh['trial_statements']][1:]  # index 15 is the stim condition 
    
    run_raster_path = all_session_stem / recname / f'{recname}_all_rasters_run.npy'
    run_trains_path = all_session_stem / recname / f'{recname}_all_trains_run.npy'
    rasters = np.load(run_raster_path, allow_pickle=True).item()
    trains  = np.load(run_trains_path, allow_pickle=True).item()
    
    for clu in list(rasters.keys()):
        if clu in tagged_keys or clu in putative_keys:
            curr_stim_spike_map = [spike_map for trial, spike_map 
                                   in enumerate(rasters[clu]) 
                                   if stim_cds[trial]!='0']
            curr_ctrl_spike_map = [spike_map for trial, spike_map
                                   in enumerate(rasters[clu][2:])
                                   if stim_cds[trial-2]!='0']
            
            curr_stim_trains_prof = np.mean([train for trial, train
                                             in enumerate(trains[clu])
                                             if stim_cds[trial]!='0'],
                                            axis=0)
            curr_ctrl_trains_prof = np.mean([train for trial, train
                                             in enumerate(trains[clu][2:])
                                             if stim_cds[trial-2]!='0'],
                                            axis=0)
            
            # collect for summary stats
            all_ctrl_amps.append(np.nanmean(
                curr_ctrl_trains_prof[AMP_WINDOW_LOW : AMP_WINDOW_HIGH])
                )
            all_stim_amps.append(np.nanmean(
                curr_stim_trains_prof[AMP_WINDOW_LOW : AMP_WINDOW_HIGH])
                )
            
            # single-cell plotting 
            fig, axs = plt.subplots(2, 1, figsize=(2.4, 2.8))
            
            for idx in range(len(curr_stim_spike_map)):
                curr_stim_raster = [s/SAMP_FREQ - BEF for s, spike
                                    in enumerate(curr_stim_spike_map[idx])
                                    if spike]
                axs[0].scatter(curr_stim_raster,
                               [idx+1]*len(curr_stim_raster), 
                               color='lightsteelblue', s=1, ec='none')
                
            for idx in range(len(curr_ctrl_spike_map)):
                curr_ctrl_raster = [s/SAMP_FREQ - BEF for s, spike
                                    in enumerate(curr_ctrl_spike_map[idx])
                                    if spike]
                axs[1].scatter(curr_ctrl_raster,
                               [idx+1]*len(curr_ctrl_raster), 
                               color='grey', s=1, ec='none')
            
            axt0 = axs[0].twinx()
            axt0.plot(XAXIS,
                      curr_stim_trains_prof, 
                      color='royalblue')
            axt0.set(ylabel='Firing rate\n(Hz)')
            axs[0].set(xlabel='Time from run onset (s)', xlim=(-1, 4), xticks=(0, 2, 4),
                     ylabel='Trial #',
                     title='{}\nStim.'.format(clu))
            
            axt1 = axs[1].twinx()
            axt1.plot(XAXIS,
                      curr_ctrl_trains_prof, 
                      color='k')
            axt1.set(ylabel='Firing rate\n(Hz)')
            axs[1].set(xlabel='Time from run onset (s)', xlim=(-1, 4), xticks=(0, 2, 4),
                     ylabel='Trial #',
                     title='Ctrl.')
            
            axs[0].spines['top'].set_visible(False)
            axt0.spines['top'].set_visible(False)
            
            axs[1].spines['top'].set_visible(False)
            axt1.spines['top'].set_visible(False)
            
            fig.tight_layout()
            
            for ext in ['.png', '.pdf']:
                if clu in tagged_keys:
                    fig.savefig(
                        stim_raster_stem / f'{clu}_tagged{ext}',
                        dpi=300,
                        bbox_inches='tight')
                else:
                    fig.savefig(
                        stim_raster_stem / f'{clu}_putative{ext}',
                        dpi=300,
                        bbox_inches='tight')
                    
            plt.close(fig)


#%% summary stats 
plot_violin_with_scatter(all_ctrl_amps, all_stim_amps,
                         'grey', 'royalblue',
                         ylabel='Firing rate (Hz)',
                         xticklabels=['Ctrl.', 'Stim.'],
                         print_statistics=True,
                         save=True,
                         savepath='Z:/Dinghao/code_dinghao/LC_ephys/stim_effects/tagged_putative_ctrl_stim_violin')