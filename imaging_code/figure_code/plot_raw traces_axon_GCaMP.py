# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:08:04 2025

plot the raw trace aligned with speeds for LC-HPC axon-GCaMP data, 
with distance resetting at trial start (180 cm per trial max).

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import sys 
import os
from pathlib import Path

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, smooth_convolve
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLCHPCGCaMP


#%% parameters 
ROI_size_threshold = 500  # pixels 
smoothing_sigma = 100  # in ms


#%% load data 
beh_df = pd.read_pickle(
    r'Z:/Dinghao/code_dinghao/behaviour/all_LCHPCGCaMP_sessions.pkl'
)


#%% main 
for path in paths:
    recname = Path(path).parts[-1]
    print(recname)
    
    F = np.load(
        rf'{path}/suite2p/plane0/F.npy'
    )
    
    valid_ROIs_coord_dict = np.load(
        rf'Z:/Dinghao/code_dinghao/LCHPC_axon_GCaMP/all_sessions/{recname}'
        r'/processed_data/valid_ROIs_coord_dict.npy',
        allow_pickle=True
    ).item()
    ROIs = [key for key, value in valid_ROIs_coord_dict.items() 
            if len(value[0]) > ROI_size_threshold]
    ROI_idx = [int(roi.split(' ')[-1]) for roi in ROIs]
    
    beh = beh_df.loc[recname]
    frame_times = beh['frame_times']
    first_frame_time = frame_times[0]
    last_frame_time = frame_times[-1]
    
    # speed interpolation
    times, speeds = zip(*[
        (time, int(speed)) 
        for time, speed in beh['speed_times_full'] 
        if first_frame_time < time < last_frame_time
    ])
    times_ms = [times[0] + i for i in np.arange(int(times[-1]-times[0]))]
    speeds_interp = np.interp(times_ms, times, speeds)
   
    # distance interpolation with resetting at trial starts
    run_onsets = beh['run_onsets']
    run_onsets = [onset for onset in run_onsets if first_frame_time < onset < last_frame_time]

    # generate distance array following run_onsets
    distance_interp = np.zeros_like(speeds_interp)
    run_onsets_idx = [np.searchsorted(times_ms, onset) for onset in run_onsets]
    
    trial_idx = 0
    accumulating = False
    distance = 0
    
    for idx in range(len(times_ms)):
        if trial_idx < len(run_onsets_idx) and idx == run_onsets_idx[trial_idx]:
            distance = 0
            accumulating = True
            trial_idx += 1
    
        if accumulating:
            distance += speeds_interp[idx] / 1000  # speeds in cm/ms
            if distance >= 180:  # 180 cm
                distance = 0
                accumulating = False
    
        distance_interp[idx] = distance
        
    speeds_interp_smoothed = smooth_convolve(
        speeds_interp, sigma=smoothing_sigma
        )
    
    # output folder 
    save_dir = (r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions'
                rf'\{recname}\ROI_distance_dFF_speed')
    os.makedirs(save_dir, exist_ok=True)
    
    ## plotting 
    window_size = 200 * 1000  # 100 s
    n_windows = (len(times_ms) + window_size - 1) // window_size
    for roi in ROI_idx:
        F_ROI = F[roi, :]
        F_ROI_interp = smooth_convolve(
            np.interp(times_ms, frame_times[:len(F_ROI)], F_ROI),
            sigma=smoothing_sigma
        )
    
        for fig_idx in range(0, n_windows, 4):  # 4 windows per figure
            # create more axes: 16 rows (4×(3+1)) = 4 windows × (3 panels + 1 blank)
            fig, axs = plt.subplots(16, 1, figsize=(14, 28), sharex=False)
            fig.subplots_adjust(hspace=0.4)  # smaller hspace globally
            
            for i in range(4):  # 4 windows per figure
                window_idx = fig_idx + i
                if window_idx >= n_windows:
                    break  # no more windows
            
                start_idx = window_idx * window_size
                end_idx = min((window_idx + 1) * window_size, len(times_ms))
            
                times_plot = np.array(times_ms[start_idx:end_idx]) / 1000  # ms to s
            
                base = i * 4  # shift every window block by 4 rows
            
                # distance
                axs[base + 0].plot(times_plot - times_plot[0], 
                                   distance_interp[start_idx:end_idx] / 100, 
                                   color='black')
                axs[base + 0].set_ylabel('distance (m)')
                axs[base + 0].set_title(f'ROI {roi} | Window {window_idx}')
            
                # fluorescence
                axs[base + 1].plot(times_plot - times_plot[0], 
                                   F_ROI_interp[start_idx:end_idx], 
                                   color='green')
                axs[base + 1].set_ylabel('dF/F')
            
                # speed
                axs[base + 2].plot(times_plot - times_plot[0], 
                                   speeds_interp_smoothed[start_idx:end_idx], 
                                   color='blue')
                axs[base + 2].set_ylabel('speed (cm/s)')
                axs[base + 2].set_xlabel('time (s)')
            
                # blank panel (axs[base + 3])
                axs[base + 3].axis('off')  # turn off the blank panel
    
            # save figure
            for ext in ['.png', '.pdf']:
                fig.savefig(
                    rf'{save_dir}\roi_{roi}_{int(fig_idx/4)+1}{ext}',
                    dpi=300,
                    bbox_inches='tight'
                    )
            plt.close(fig)
