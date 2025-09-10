# -*- coding: utf-8 -*-
"""
Created originally by Yingxue Wang
Modified on 2 Sept 2025

using linear regression to demix motion artefacts from imaging recordings 
    (dual colour)
modified from Yingxue's original pipeline

@author: Dinghao Luo
"""


#%% imports
import gc
from pathlib import Path

import numpy as np


#%% paths and data
p_data = r'Z:\Dinghao\2p_recording\A126i\A126i-20250616\A126i-20250616-01\suite2p\plane0'
file_name  = r'\data.bin'        # green (dlight)
file_name2 = r'\data_chan2.bin'  # red (axon)

path_data_g = Path(p_data, file_name)
path_data_r = Path(p_data, file_name2)

path_result = Path(p_data) / 'result'
path_masks  = path_result / 'masks'
path_result.mkdir(parents=True, exist_ok=True)
path_masks.mkdir(parents=True, exist_ok=True)

beh_file = Path(r'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCdLightLCOpto\A126i-20250616-01.pkl')
beh_npz  = path_result / f'{beh_file.stem}_processed_behavior.npz'

# movie geometry
num_frames = 10000
height = 512
width = 512


#%% load movies via memmap (read-only)
mov  = np.memmap(path_data_g, dtype='float32', mode='r',
                 shape=(num_frames, height, width))  # green / dlight
movr = np.memmap(path_data_r, dtype='float32', mode='r',
                 shape=(num_frames, height, width))  # red / axon


#%% load behaviour event frames
print(f'loading data from: {beh_npz}')
ld = np.load(beh_npz)

print('\nvariables (keys) found in the file:')
print(ld.files)

speed_trace       = ld['speed']              if 'speed' in ld else None
lick_trace        = ld['licks']              if 'licks' in ld else None
run_onset_frames  = ld['run_onset_frames']   if 'run_onset_frames' in ld else None
reward_frames     = ld['reward_frames']      if 'reward_frames' in ld else None
start_cue_frames  = ld['start_cue_frames']   if 'start_cue_frames' in ld else None
first_lick_frames = ld['first_lick_frames']  if 'first_lick_frames' in ld else None
ld.close()

if run_onset_frames is None or reward_frames is None:
    raise RuntimeError('missing required event frames (run_onset_frames or reward_frames)')

if speed_trace is not None:
    print(f"\nloaded 'speed' trace with shape: {speed_trace.shape}")
if lick_trace is not None:
    print(f"loaded 'licks' trace with shape: {lick_trace.shape}")
print(f"loaded 'run_onset_frames' with {len(run_onset_frames)} events.")
print(f"loaded 'reward_frames' with {len(reward_frames)} events.")
if start_cue_frames is not None:
    print(f"loaded 'start_cue_frames' with {len(start_cue_frames)} events.")
if first_lick_frames is not None:
    print(f"loaded 'first_lick_frames' with {len(first_lick_frames)} events.")


#%% load masks
path_save_axon_mask                  = path_masks / 'global_axon_mask.npy'
path_save_dlight_mask                = path_masks / 'global_dlight_mask.npy'
path_save_axon_or_dlight_mask        = path_masks / 'global_axon_or_dlight_mask.npy'
path_save_dilated_axon_dlight_mask   = path_masks / 'dilated_global_axon_and_dlight_mask.npy'

global_axon_mask                 = np.load(path_save_axon_mask)
global_dlight_mask               = np.load(path_save_dlight_mask)
global_axon_or_dlight_mask       = np.load(path_save_axon_or_dlight_mask)
dilated_global_axon_dlight_mask  = np.load(path_save_dilated_axon_dlight_mask)


#%% regression using behaviour-averaged templates
import Regression_Red_From_Green_BehAvg_Template as rrfg

# single-block test; pick a block with enough axon & dlight active pixels
block_to_analyze = (20, 15)
grid_size = 16
pre = 90
post = 120

final_corrected_green_trace, original_axon_green_trace, estimated_full_artifact, \
estimated_red_kernel, estimated_red_kernel2 = rrfg.correct_green_trace_with_estimated_red_kernel(
    mov=mov,
    movr=movr,
    global_mask_green=dilated_global_axon_dlight_mask,
    global_mask_red=dilated_global_axon_dlight_mask,
    global_neuropil_rev=global_axon_or_dlight_mask,
    block_to_analyze=block_to_analyze,
    reward_frames=reward_frames,
    run_onset_frames=run_onset_frames,
    grid_size=grid_size,
    pre_frames=pre,
    post_frames=post,
    imaging_rate=30.0,
    baseline_period_s=(-3.0, -1.0)
)

# reward-aligned validation
rrfg.plot_correction_validation_suite_two_kernels(
    original_axon_green_trace,
    estimated_full_artifact,
    final_corrected_green_trace,
    estimated_red_kernel,
    estimated_red_kernel2,
    reward_frames,
    block_to_analyze,
    pre_frames=pre,
    post_frames=post,
    str_trial_type='Reward'
)

# run-onset-aligned validation
rrfg.plot_correction_validation_suite_two_kernels(
    original_axon_green_trace,
    estimated_full_artifact,
    final_corrected_green_trace,
    estimated_red_kernel,
    estimated_red_kernel2,
    run_onset_frames,
    block_to_analyze,
    pre_frames=pre,
    post_frames=post,
    str_trial_type='Run Onset'
)


#%% cleanup
del mov
del movr
gc.collect()
