# -*- coding: utf-8 -*-
"""
created originally by yingxue wang
modified on 2 sept 2025

event-locked grid/whole-fov analyses for dual-colour recordings
    (red = axon; green = dlight)

this version:
- uses numpy.memmap instead of FilterImage to reduce ram
- uses pathlib Path joins throughout
- loads behaviour frames from a processed .npz
- loads precomputed masks and runs fr-ed-per-trial routines

@author: dinghao luo
"""

#%% imports
import gc
from pathlib import Path

import numpy as np
import FRedPerTrial


#%% paths and basic settings
p_data = r'Z:\Dinghao\2p_recording\A126i\A126i-20250616\A126i-20250616-01\suite2p\plane0'
file_name = r'\data.bin'
file_name2 = r'\data_chan2.bin'

path_data_g = Path(p_data, file_name)    # green channel (dlight)
path_data_r = Path(p_data, file_name2)   # red channel (axon)

path_result = Path(p_data) / 'result'
path_masks  = path_result / 'masks'
path_result.mkdir(parents=True, exist_ok=True)
path_masks.mkdir(parents=True, exist_ok=True)  # ok if already exists

beh_file = Path(
    r'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCdLightLCOpto\A126i-20250616-01.pkl'
)
beh_npz = path_result / f'{beh_file.stem}_processed_behavior.npz'

# movie geometry
num_frames = 10000
height = 512
width = 512

# analysis parameters
grid_size = 16
pre = 90
post = 120
n_block_xy = 15
imaging_rate = 30.0


#%% load movies via memmap (read-only)
mov  = np.memmap(path_data_g, dtype='float32', mode='r',
                 shape=(num_frames, height, width))  # green / dlight
movr = np.memmap(path_data_r, dtype='float32', mode='r',
                 shape=(num_frames, height, width))  # red / axon


#%% load behavioural event frames
print(f'loading behaviour from: {beh_npz}')
with np.load(beh_npz) as ld:
    speed_trace        = ld['speed']               if 'speed' in ld else None
    lick_trace         = ld['licks']               if 'licks' in ld else None
    run_onset_frames   = ld['run_onset_frames']    if 'run_onset_frames' in ld else None
    reward_frames      = ld['reward_frames']       if 'reward_frames' in ld else None
    start_cue_frames   = ld['start_cue_frames']    if 'start_cue_frames' in ld else None
    first_lick_frames  = ld['first_lick_frames']   if 'first_lick_frames' in ld else None

if run_onset_frames is None or reward_frames is None:
    raise RuntimeError('missing required event frames (run_onset_frames or reward_frames) in behaviour npz')

print(f'run_onset_frames: {len(run_onset_frames)} events; reward_frames: {len(reward_frames)} events')


#%% load masks
path_save_axon_mask                 = path_masks / 'global_axon_mask.npy'
path_save_dlight_mask               = path_masks / 'global_dlight_mask.npy'
path_save_axon_dlight_mask          = path_masks / 'global_axon_and_dlight_mask.npy'
path_save_axon_or_dlight_mask       = path_masks / 'global_axon_or_dlight_mask.npy'
path_save_dilated_axon_dlight_mask  = path_masks / 'dilated_global_axon_and_dlight_mask.npy'

global_axon_mask                = np.load(path_save_axon_mask)
global_dlight_mask              = np.load(path_save_dlight_mask)
global_axon_dlight_mask         = np.load(path_save_axon_dlight_mask)
global_axon_or_dlight_mask      = np.load(path_save_axon_or_dlight_mask)
dilated_global_axon_dlight_mask = np.load(path_save_dilated_axon_dlight_mask)


#%% whole-fov traces (edge-trim), red and green, run onset
FRedPerTrial.run_whole_fov_analysis(
    movr,
    global_axon_mask,
    run_onset_frames,
    path_result,
    chan_name='red_axon',
    event_name='run_onset',
    mask_name='axon_mask',
    edge_pixels=grid_size,
    pre_frames=pre,
    post_frames=post,
    imaging_rate=imaging_rate
)

FRedPerTrial.run_whole_fov_analysis(
    mov,
    global_axon_mask,
    run_onset_frames,
    path_result,
    chan_name='dlight',
    event_name='run_onset',
    mask_name='axon_mask',
    edge_pixels=grid_size,
    pre_frames=pre,
    post_frames=post,
    imaging_rate=imaging_rate
)


#%% cleanup
del mov
del movr
gc.collect()