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
from pathlib import Path
import numpy as np
import Single_trial_dLight

#%% paths
p_data = r'Z:\Dinghao\2p_recording\A126i\A126i-20250616\A126i-20250616-01\suite2p\plane0'
path_result = Path(p_data) / 'result'
beh_file = Path(r'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCdLightLCOpto\A126i-20250616-01.pkl')

#%% load behaviour event frames
file_to_load = path_result / f'{beh_file.stem}_processed_behavior.npz'
print(f'Loading data from: {file_to_load}')
with np.load(file_to_load) as loaded_data:
    print('\nVariables (keys) found in the file:')
    print(loaded_data.files)

    speed_trace       = loaded_data['speed']              if 'speed' in loaded_data else None
    lick_trace        = loaded_data['licks']              if 'licks' in loaded_data else None
    run_onset_frames  = loaded_data['run_onset_frames']   if 'run_onset_frames' in loaded_data else None
    reward_frames     = loaded_data['reward_frames']      if 'reward_frames' in loaded_data else None
    start_cue_frames  = loaded_data['start_cue_frames']   if 'start_cue_frames' in loaded_data else None
    first_lick_frames = loaded_data['first_lick_frames']  if 'first_lick_frames' in loaded_data else None

if speed_trace is not None:
    print(f"\nLoaded 'speed' trace with shape: {speed_trace.shape}")
if lick_trace is not None:
    print(f"Loaded 'licks' trace with shape: {lick_trace.shape}")
if run_onset_frames is not None:
    print(f"Loaded 'run_onset_frames' with {len(run_onset_frames)} events.")
if reward_frames is not None:
    print(f"Loaded 'reward_frames' with {len(reward_frames)} events.")
if start_cue_frames is not None:
    print(f"Loaded 'start_cue_frames' with {len(start_cue_frames)} events.")
if first_lick_frames is not None:
    print(f"Loaded 'first_lick_frames' with {len(first_lick_frames)} events.")

#%% load masks (for plotting/weighting)
path_save_axon_mask = path_result / 'masks' / 'global_axon_mask.npy'
global_axon_mask = np.load(path_save_axon_mask)

#%% load corrected traces (template regression, axon∧dLight mask)
mask_str = 'G_axon_dlight_R_axon_dlight'
result_dir = path_result / 'regression_with_template' / mask_str
corrected_file_to_load = result_dir / 'all_corrected_traces_regress_with_template.npz'

print(f'Loading data from: {corrected_file_to_load}')
with np.load(corrected_file_to_load) as loaded_data:
    print('\nVariables (keys) found in the file:')
    print(loaded_data.files)

    print('\n--- Loading Data Arrays ---')
    original_dlight = loaded_data['original_dlight']
    corrected_dlight = loaded_data['corrected_dlight']
    red_trace = loaded_data['red_trace']
    global_mask_green_loaded = loaded_data['global_mask_green']

    pre = loaded_data['pre_frames'].item()
    post = loaded_data['post_frames'].item()

#%% single-grid trial inspection
block_to_analyze = (17, 10)
imaging_rate = 30.0

# high/low signal trials (ordered by change at event)
Single_trial_dLight.find_and_visualize_signal_trials_ordered_change_at_event(
    corrected_dlight, red_trace, speed_trace, lick_trace, run_onset_frames, pre, post,
    block_to_analyze, signal_z_threshold=1.0
)

# permutation-test-selected trials
Single_trial_dLight.find_signal_trials_with_permutation_test(
    corrected_dlight, red_trace, speed_trace, lick_trace, run_onset_frames, pre, post,
    block_to_analyze, imaging_rate, n_shuffles=1000, p_value_threshold=0.05
)

#%% population significance over all grids (axon∧dLight mask)
Single_trial_dLight.run_and_save_significance_analysis(
    corrected_traces_grid=corrected_dlight,
    red_traces_grid=red_trace,
    speed_trace=speed_trace,
    lick_trace=lick_trace,
    event_frames=run_onset_frames,
    mean_img=global_axon_mask,
    output_dir=result_dir,
    event_name='Run_onset',
    pre_frames=pre,
    post_frames=post,
    imaging_rate=30.0,
    p_value_threshold=0.05,
    n_shuffles=1000
)

#%% repeat with dilated axon∧dLight mask
mask_str = 'G_dilated_axon_dlight_R_dilated_axon_dlight'
result_dir = path_result / 'regression_with_template' / mask_str
corrected_file_to_load = result_dir / 'all_corrected_traces_regress_with_template.npz'

print(f'Loading data from: {corrected_file_to_load}')
with np.load(corrected_file_to_load) as loaded_data:
    print('\nVariables (keys) found in the file:')
    print(loaded_data.files)

    print('\n--- Loading Data Arrays ---')
    original_dlight = loaded_data['original_dlight']
    corrected_dlight = loaded_data['corrected_dlight']
    red_trace = loaded_data['red_trace']
    global_mask_green_loaded = loaded_data['global_mask_green']

    pre = loaded_data['pre_frames'].item()
    post = loaded_data['post_frames'].item()

Single_trial_dLight.run_and_save_significance_analysis(
    corrected_traces_grid=corrected_dlight,
    red_traces_grid=red_trace,
    speed_trace=speed_trace,
    lick_trace=lick_trace,
    event_frames=run_onset_frames,
    mean_img=global_axon_mask,
    output_dir=result_dir,
    event_name='Run_onset',
    pre_frames=pre,
    post_frames=post,
    imaging_rate=30.0,
    p_value_threshold=0.05,
    n_shuffles=1000
)