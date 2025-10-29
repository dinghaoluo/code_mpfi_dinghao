# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 18:01:35 2025

Single-axon aligned to stim profiles

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path 

import numpy as np 
from scipy.stats import sem
import matplotlib.pyplot as plt 

import imaging_pipeline_functions as ipf
from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathdLightLCOpto


#%% path stems
all_sess_stem = Path('Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto/all_sessions')
mice_exp_stem = Path('Z:/Dinghao/MiceExp')


#%% parameters 
SAMP_FREQ = 30

BEF = 2
AFT = 10

PMT_BUFFER = 10  # frames

XAXIS = np.arange((BEF+AFT)*SAMP_FREQ) / SAMP_FREQ - BEF


#%% main
for path in paths:
    recname = Path(path).name
    sessname = recname.replace('i', '')
    print(f'\n{recname}')

    F_aligned_path  = all_sess_stem / recname / f'processed_data/{recname}_pixel_F_aligned.npy'
    F2_aligned_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_F2_aligned.npy'
    roi_dict_path   = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'
    txtpath         = mice_exp_stem / f'ANMD{recname[1:4]}' / f'{sessname}T.txt'
    
    if not F_aligned_path.exists() or not F2_aligned_path.exists() or not roi_dict_path.exists() or not txtpath.exists():
        print('missing arrays, skipped')
        continue
    
    F_aligned  = np.load(F_aligned_path, allow_pickle=True)
    F2_aligned = np.load(F2_aligned_path, allow_pickle=True)
    roi_dict   = np.load(roi_dict_path, allow_pickle=True).item()
    
    tot_stims = F_aligned.shape[0]
    
    # pulse frame masking 
    txt = ipf.process_txt_nobeh(txtpath)
    pulse_parameters = txt['pulse_parameters']
    pulse_width = float(pulse_parameters[-1][3])/1000000  # in s
    pulse_number = int(pulse_parameters[-1][4])
    pulse_duration = pulse_width * pulse_number
    pulse_frames = np.arange(
        BEF * SAMP_FREQ - 3,  # -3 as buffer 
        int(BEF * SAMP_FREQ + pulse_duration * SAMP_FREQ + PMT_BUFFER)
        )
    
    for roi_id, roi in roi_dict.items():
        # get mask 
        curr_y = roi['ypix']
        curr_x = roi['xpix']
        
        # get traces 
        curr_F_traces  = F_aligned[:, :, curr_y, curr_x]
        curr_F2_traces = F2_aligned[:, :, curr_y, curr_x]
        
        # get mean and sem 
        mean_F_traces  = np.mean(curr_F_traces, axis=(0,2))
        sem_F_traces   = sem(curr_F_traces, axis=(0,2))
        mean_F2_traces = np.mean(curr_F2_traces, axis=(0,2))
        sem_F2_traces  = sem(curr_F2_traces, axis=(0,2))
        
        # blank out the stim 
        mean_F_traces[pulse_frames]  = np.nan
        sem_F_traces[pulse_frames]   = np.nan
        mean_F2_traces[pulse_frames] = np.nan
        sem_F2_traces[pulse_frames]  = np.nan
        
        # plotting 
        fig, axs = plt.subplots(2, 1, figsize=(3,4), sharex=True)
        
        axs[0].plot(XAXIS, mean_F_traces, color='darkgreen', label='dLight')
        axs[0].fill_between(XAXIS, mean_F_traces+sem_F_traces,
                                   mean_F_traces-sem_F_traces,
                        color='darkgreen', edgecolor='none', alpha=.3)
        axs[1].plot(XAXIS, mean_F2_traces, color='darkred', label='tdTomato')
        axs[1].fill_between(XAXIS, mean_F2_traces+sem_F2_traces,
                                   mean_F2_traces-sem_F2_traces,
                        color='darkred', edgecolor='none', alpha=.3)
        axs[1].set(xlabel='Time from stim. (s)',
                   ylabel='F')
        axs[0].set(title=f'{recname} ROI {roi_id}')
        
        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)
            ax.set(xticks=(0, 5, 10))
        
        save_stem = all_sess_stem / recname / 'single_ROI_stim_aligned'
        save_stem.mkdir(exist_ok=True)
        fig.savefig(save_stem / f'ROI_{roi_id}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
