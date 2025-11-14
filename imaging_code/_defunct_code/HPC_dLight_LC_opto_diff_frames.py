# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:47:50 2025

plot the movie frames during baseline and stim for <100% duty cycles

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathdLightLCOpto + rec_list.pathdLightLCOptoCtrl


#%% parameters 
SAMP_FREQ = 30  # Hz

FRAMES_PER_PULSE = 15 


#%% main 
for path in paths:
    recname = path.split('\\')[-1]
    print(f'\n{recname}')
    
    binpath = os.path.join(path, 'suite2p/plane0/data.bin')
    bin2path = os.path.join(path, 'suite2p/plane0/data_chan2.bin')
    opspath = os.path.join(path, 'suite2p/plane0/ops.npy')
    txtpath = os.path.join(r'Z:\Dinghao\MiceExp',
                           f'ANMD{recname[1:4]}',
                           f'{recname[:4]}{recname[5:]}T.txt')
    
    savepath = os.path.join(
        r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions',
        f'{recname}'
        )
    
    if os.path.exists(rf'{savepath}\{recname}_example_frames_diff.png'):
        print(f'{recname} processed... skipping')
        continue
    
    # load data 
    ops = np.load(opspath, allow_pickle=True).item()
    tot_frames = ops['nframes']
    shape = tot_frames, ops['Ly'], ops['Lx']
    
    print('loading movies...')
    mov = np.memmap(binpath, mode='r', dtype='int16', shape=shape).astype(np.float32)
    mov2 = np.memmap(bin2path, mode='r', dtype='int16', shape=shape).astype(np.float32)
    
    print('processing .txt file...')
    txt = ipf.process_txt_nobeh(txtpath)
    frame_times = txt['frame_times']
    pulse_times = txt['pulse_times']
    pulse_parameters = txt['pulse_parameters']
    
    # pulse parameters 
    print('extracting data...')
    
    pulse_width_ON = float(pulse_parameters[-1][2]) / 1000  # ms
    pulse_width = float(pulse_parameters[-1][3]) / 1000  # ms
    pulse_width_ON_s = float(pulse_parameters[-1][2]) / 1_000_000  # s
    pulse_width_s = float(pulse_parameters[-1][3]) / 1_000_000  # s
    pulse_number = int(pulse_parameters[-1][4])
    duty_cycle = f'{int(round(100 * pulse_width_ON / pulse_width, 0))}%'
        
    tot_pulses = int(len(pulse_times) / pulse_number)
    
    # extract pulse frame indices
    pulse_frames = [ipf.find_nearest(p, frame_times) for p in pulse_times]
    pulse_frames = [pulse_frames[p * pulse_number : p * pulse_number + pulse_number] 
                    for p in range(tot_pulses)]
    pulse_start_frames = [p[0] for p in pulse_frames]
    
    # baseline: immediate pre-train frames
    immediate_baseline_frames = []
    for i in range(tot_pulses):
        start_f = pulse_frames[i][0]
        baseline_chunk = np.arange(start_f - FRAMES_PER_PULSE, start_f)
        baseline_chunk = baseline_chunk[baseline_chunk >= 0]
        immediate_baseline_frames.extend(baseline_chunk)

    # stim: starting just after last pulse in each train
    true_stim_frames = []
    for i in range(tot_pulses):
        last_pulse_time = pulse_times[i * pulse_number + (pulse_number - 1)]
        last_pulse_end = last_pulse_time + pulse_width_ON

        post_idx = np.where(np.array(frame_times) > last_pulse_end)[0] + 1  # +1 as a buffer 
        if len(post_idx) == 0:
            continue
        start_f = post_idx[0]

        stim_chunk = np.arange(start_f, start_f + FRAMES_PER_PULSE)
        stim_chunk = stim_chunk[stim_chunk < tot_frames]
        true_stim_frames.extend(stim_chunk)

    # final sanity check
    min_len = min(len(true_stim_frames), len(immediate_baseline_frames))
    true_stim_frames = true_stim_frames[:min_len]
    immediate_baseline_frames = immediate_baseline_frames[:min_len]

    # compute mean images
    true_stim_mean = np.mean(mov[true_stim_frames, :, :], axis=0)
    immediate_baseline_mean = np.mean(mov[immediate_baseline_frames, :, :], axis=0)
    diff_map = true_stim_mean - immediate_baseline_mean

    # plot frames
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    
    axs[0].imshow(immediate_baseline_mean, 
                  aspect='auto', interpolation='none', cmap='gray')
    axs[0].set(title='mean baseline frame')
    
    axs[1].imshow(true_stim_mean, 
                  aspect='auto', interpolation='none', cmap='gray')
    axs[1].set(title='mean post-train frame')
    
    im = axs[2].imshow(diff_map, 
                       aspect='auto', interpolation='none', cmap='bwr',
                       vmin=-np.max(np.abs(diff_map)), vmax=np.max(np.abs(diff_map)))
    axs[2].set(title='stim − baseline')
    
    for ax in axs:
        ax.axis('off')
    
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'{savepath}\{recname}_example_frames_diff{ext}',
                    dpi=300,
                    bbox_inches='tight')