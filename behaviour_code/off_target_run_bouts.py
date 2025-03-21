# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:39:49 2025

process off-target run-bout onsets

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pandas as pd 
from tqdm import tqdm


#%% load dataframe 
print('loading behaviour dataframe...')
df = pd.read_pickle(
    r'Z:/Dinghao/code_dinghao/behaviour/all_HPCLCGCaMP_sessions.pkl'
    )


#%% functions
def compute_speed_distances(speed_times):
    """
    computes cumulative distance from speed_times data.

    parameters:
    - speed_times: list of (timestamp, speed) tuples

    returns:
    - list of (timestamp, distance) tuples
    """
    if not speed_times:
        return []
    
    times, speeds = zip(*speed_times)
    times = np.array(times) / 1000  # convert ms to seconds
    speeds = np.array(speeds)  # speed is already in cm/s

    # compute time differences
    dt = np.diff(times, prepend=times[0])  # prepend first value to maintain size

    # compute cumulative distance
    distances = np.cumsum(speeds * dt)  # integration step

    return list(zip(times, distances))  # return timestamped distance list


def extract_run_bouts(df, 
                      MIN_RUN_TIME=0.3, 
                      SPEED_THRES=10, 
                      FRAME_RATE=30, 
                      DIST_THRES=50):
    """
    extracts running bouts from a behavioural dataframe.
    
    parameters:
    - df: pandas dataframe containing columns with speed, trial run-onsets, and imaging frames
    - min_run_time: minimum duration (seconds) for a valid run bout
    - speed_threshold: minimum speed (cm/s) for a valid run bout
    - imaging_fps: frame rate of the imaging system (Hz)
    - dist_thres: minimum distance (cm) required from the closest trial run-onset
    
    returns:
    - run_bout_df: pandas dataframe with extracted running bout details
    """
    
    run_bout_data = []
    
    for rec_name, row in tqdm(df.iterrows(), total=len(df)):
        speed_data = row['speed_times']  # list of lists of (time, speed)
        run_onsets = row['run_onsets']  # list of trial run-onset times
        frame_times = row['frame_times']  # list of imaging frame timestamps
        frame_numbers = np.arange(len(frame_times))  # corresponding frame indices
        
        if not speed_data or not frame_times:
            continue  # skip empty sessions
        
        for trial_idx, trial_speed in enumerate(speed_data):
            if not trial_speed:
                continue
            
            # compute timestamped distances for this trial
            trial_distance = compute_speed_distances(trial_speed)
            times, speeds = zip(*trial_speed)  # separate time and speed
            dist_times, distances = zip(*trial_distance)  # separate time and distance
            
            times = np.array(times) / 1000  # convert ms to seconds
            frame_times = np.array(row['frame_times']) / 1000  # convert ms to seconds
            speeds = np.array(speeds)
            distances = np.array(distances)
            
            # identify running periods above threshold
            running_mask = speeds > SPEED_THRES
            
            # find all start and end indices where running begins and ends
            run_change = np.diff(running_mask.astype(int))
            run_starts = np.where(run_change == 1)[0] + 1  # start when speed crosses threshold
            run_ends = np.where(run_change == -1)[0] + 1  # end when speed drops below threshold
            
            # ensure start/end pairs match correctly
            if running_mask[0]:  # starts running at the beginning
                run_starts = np.insert(run_starts, 0, 0)
            if running_mask[-1]:  # still running at the end
                run_ends = np.append(run_ends, len(speeds)-1)
            
            # ensure that each start has a corresponding end
            if len(run_starts) > len(run_ends):
                run_ends = np.append(run_ends, len(speeds)-1)
            
            for start, end in zip(run_starts, run_ends):
                run_duration = times[end] - times[start]
                if run_duration < MIN_RUN_TIME:
                    continue  # skip short runs
                
                mean_speed = speeds[start:end].mean()
                run_start_time = times[start]

                # interpolate distance at run start time
                run_start_distance = np.interp(run_start_time, dist_times, distances)

                # Exclude runs that are within 50 cm of the closest trial run-onset
                run_onset_distances = np.interp(run_onsets, dist_times, distances)  # Map onsets to distances
                closest_run_onset_dist = np.min(np.abs(run_start_distance - run_onset_distances)) if len(run_onset_distances) > 0 else np.inf
                if closest_run_onset_dist < DIST_THRES:
                    continue  # Skip runs too close to trial run-onsets
                
                # Find closest frame index instead of frame time
                frame_diffs = np.abs(frame_times - run_start_time)
                min_frame_idx = np.argmin(frame_diffs)
                run_start_frame = frame_numbers[min_frame_idx] if min_frame_idx < len(frame_numbers) else np.nan
                
                # compute preceding pause duration (if any)
                prev_end_idx = np.where(times < run_start_time)[0]
                precede_pause_length = (run_start_time - times[prev_end_idx[-1]]) if prev_end_idx.size > 0 else np.nan
                
                # ensure multiple running bouts within the same trial are all recorded
                run_bout_data.append({
                    'rec_name': rec_name,
                    'trial_idx': trial_idx,  # track trial index to differentiate multiple runs in one trial
                    'run_start_time': run_start_time,
                    'run_start_frame': run_start_frame,
                    'run_length_sec': run_duration,
                    'mean_speed_run': mean_speed,
                    'precede_pause_length_sec': precede_pause_length,
                    'run_start_distance': run_start_distance  # Store the distance at run onset
                })
    
    run_bout_df = pd.DataFrame(run_bout_data)
    return run_bout_df


#%% main
run_bout_df = extract_run_bouts(df[43:44])