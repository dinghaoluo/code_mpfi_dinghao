# -*- coding: utf-8 -*-
"""
Created originally by Yingxue Wang
Modified on 2 Sept 2025

single-trial post-stimulus amplitude significance analysis on dlight signals 
    after regression with whole red trace
adapted to support both standard and dilated axon∧dlight masks

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path
import numpy as np
import Single_trial_dLight_post_amp_signif as stpost

#%% paths
p_data = r'Z:\Jingyu\2P_Recording\AC967\AC967-20250213\04\RegOnly\suite2p\plane0'
path_result = Path(p_data) / 'result'
beh_file = Path(r'Z:\Jingyu\2P_Recording\all_session_info\dlight_Ai14_Dbh\all_anm_behaviour\AC967-20250213-04.pkl')

#%% load behaviour event frames
file_to_load = path_result / f'{beh_file.stem}_processed_behavior.npz'
print(f'Loading data from: {file_to_load}')
with np.load(file_to_load) as ld:
    print('\nVariables (keys) found in the file:')
    print(ld.files)

    speed_trace       = ld['speed']              if 'speed' in ld else None
    lick_trace        = ld['licks']              if 'licks' in ld else None
    run_onset_frames  = ld['run_onset_frames']   if 'run_onset_frames' in ld else None
    reward_frames     = ld['reward_frames']      if 'reward_frames' in ld else None
    start_cue_frames  = ld['start_cue_frames']   if 'start_cue_frames' in ld else None
    first_lick_frames = ld['first_lick_frames']  if 'first_lick_frames' in ld else None

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

#%% load mask
path_save_axon_mask = path_result / 'masks' / 'global_axon_mask.npy'
global_axon_mask = np.load(path_save_axon_mask)

#%% helper function to load corrected traces
def load_corrected_traces(result_dir):
    corrected_file_to_load = result_dir / 'all_corrected_traces_regress_with_template.npz'
    print(f'Loading data from: {corrected_file_to_load}')
    with np.load(corrected_file_to_load) as ld:
        print('\nVariables (keys) found in the file:')
        print(ld.files)

        print('\n--- Loading Data Arrays ---')
        original_dlight = ld['original_dlight']
        corrected_dlight = ld['corrected_dlight']
        red_trace = ld['red_trace']
        global_mask_green_loaded = ld['global_mask_green']

        pre = ld['pre_frames'].item()
        post = ld['post_frames'].item()
    return original_dlight, corrected_dlight, red_trace, global_mask_green_loaded, pre, post

#%% analysis with axon∧dlight mask
mask_str = 'G_axon_dlight_R_axon_dlight'
result_dir = path_result / 'regression_whole_red_trace' / mask_str
original_dlight, corrected_dlight, red_trace, _, pre, post = load_corrected_traces(result_dir)

block_to_analyze = (17, 10)
post_time_window = [0.2, 1.0]  # seconds after event
p_value_threshold = 0.05

# trial variability dashboard
stpost.plot_trial_variability_dashboard_post_amp(
    corrected_dlight, red_trace, speed_trace, lick_trace, run_onset_frames, pre, post,
    block_to_analyze, signal_z_threshold=1.0, post_time_window=post_time_window
)

# permutation-test-selected trials
stpost.find_signal_trials_with_permutation_test_post_amp(
    corrected_dlight, red_trace, speed_trace, lick_trace, run_onset_frames, pre, post,
    block_to_analyze, p_value_threshold=p_value_threshold, post_time_window=post_time_window
)

# population significance across all blocks
stpost.run_and_save_significance_analysis(
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
    n_shuffles=1000,
    post_time_window=post_time_window
)

#%% analysis with dilated axon∧dlight mask
mask_str = 'G_dilated_axon_dlight_R_dilated_axon_dlight'
result_dir = path_result / 'regression_whole_red_trace' / mask_str
original_dlight, corrected_dlight, red_trace, _, pre, post = load_corrected_traces(result_dir)

stpost.run_and_save_significance_analysis(
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
    n_shuffles=1000,
    post_time_window=post_time_window
)
