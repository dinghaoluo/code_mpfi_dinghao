# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:11:42 2024

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import sys

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from txt_processing_functions import process_txt, correct_overflow


#%% functions 
def process_behavioural_data(txtfile, max_distance=220, distance_resolution=1, run_onset_initial=3.0, run_onset_sustained=10.0, run_onset_duration=300):
    '''
    Processes behavioural data from a txt file, aligning speed, lick, and 
        reward events to both time and distance bases.

    parameters:
    ----------
    txtfile : str
        path to the txt file containing behavioural data
    max_distance : int, optional
        maximum distance for aligning data across trials (default is 220 cm)
    distance_resolution : int, optional
        distance step size in cm for interpolated distance base (default is 1 cm)
    run_onset_initial : float, optional
        initial speed threshold for detecting run onset (default is 3.0 cm/s)
    run_onset_sustained : float, optional
        sustained speed threshold for detecting run onset (default is 10.0 cm/s)
    run_onset_duration : int, optional
        duration (in ms) for which the sustained speed threshold must be held to confirm run onset (default is 300 ms)

    returns:
    -------
    dict
        a dictionary containing aligned behavioural data across time and distance bases, including:
        - 'speed_times' : list of [timestamp, speed] for each trial
        - 'speed_distance' : list of speeds aligned to a common distance base across trials
        - 'lick_times' : list of lick event timestamps
        - 'lick_distance' : list of lick events aligned to the distance base
        - 'start_cue_times' : list of start cue times
        - 'reward_times' : list of reward delivery times
        - 'reward_distance' : list of reward events aligned to the distance base
        - 'trial_statement_times' : list of trial statement times
        - 'run_onsets' : list of detected run-onset timestamps for each trial
        - 'optogenetic_protocols' : list of optogenetic protocol numbers for each trial
    '''

    # load and parse the txt file
    data = process_txt(txtfile)  # uses user's custom `process_txt` function

    # correct for overflow in the data
    data['speed_times'] = correct_overflow(data['speed_times'], 'speed')
    data['lick_times'] = correct_overflow(data['lick_times'], 'lick')
    data['pump_times'] = correct_overflow(data['pump_times'], 'pump')
    data['movie_times'] = correct_overflow(data['movie_times'], 'movie')
    data['trial_statements'] = correct_overflow(data['trial_statements'], 'trial_statement')

    # define common time and distance bases
    common_distance_base = np.linspace(0, max_distance, int(max_distance / distance_resolution))
    
    # initialise lists for storing data across trials
    speed_times, speed_distance = [], []
    lick_times, lick_distance = [], []
    start_cue_times, reward_times, reward_distance = [], [], []
    trial_statement_times, run_onsets = [], []
    optogenetic_protocols = []

    # process each trial
    for trial_idx, (speed_trial, lick_trial, movie_trial, reward_trial) in enumerate(
        zip(data['speed_times'], data['lick_times'], data['movie_times'], data['pump_times'])
    ):
        # extract times and speeds
        times = [s[0] for s in speed_trial]
        speeds = [s[1] for s in speed_trial]
        formatted_speed_times = list(zip(times, speeds))
        speed_times.append(formatted_speed_times)

        if len(times) > 1:
            # calculate cumulative distance
            distances = np.cumsum(speeds) * (np.diff(times, prepend=times[0]) / 1000)  # cm
            distances -= distances[0]  # reset distance to 0

            # interpolate speeds onto the common distance base
            max_distance_trial = min(distances[-1], max_distance)
            valid_distance_base = common_distance_base[common_distance_base <= max_distance_trial]
            interpolated_speed = np.interp(valid_distance_base, distances, speeds)
            padded_speed = np.pad(interpolated_speed, (0, len(common_distance_base) - len(interpolated_speed)), 'constant')
            speed_distance.append(padded_speed)

            # run-onset detection
            run_onset = -1  # default to -1 if no valid onset found
            uni_time = np.linspace(times[0], times[-1], int(times[-1] - times[0]))
            uni_speed = np.interp(uni_time, times, speeds)

            sustained_threshold = run_onset_duration / 1000 * 1000  # convert 0.3 seconds to ms for sustained duration
            count = 0
            for i in range(len(uni_speed)):
                if uni_speed[i] > run_onset_sustained:
                    count += 1
                else:
                    count = 0
                if uni_speed[i] > run_onset_initial and count > sustained_threshold:
                    run_onset = uni_time[i] - sustained_threshold
                    break
            run_onsets.append(run_onset)
                
            # process lick data
            lick_times_trial = [event[0] for event in lick_trial]
            lick_distances_trial = np.interp(lick_times_trial, times, distances)
            lick_indices = np.searchsorted(common_distance_base, lick_distances_trial)
            lick_distance_trial = np.zeros(len(common_distance_base))
            valid_indices = lick_indices[lick_indices < len(lick_distance_trial)]
            lick_distance_trial[valid_indices] = 1
            lick_times.append(lick_times_trial)
            lick_distance.append(lick_distance_trial)

            # process reward data
            reward_times.append(reward_trial)
            reward_distance.append(np.interp(reward_trial, times, distances) if reward_trial else [])

            # extract start cue times
            start_cue_times.append([m[0] for m in movie_trial if m[1] == 2])

        # trial statement times and optogenetic protocol
        trial_statement = data['trial_statements'][trial_idx]
        trial_statement_times.append(trial_statement[1])  # second value
        optogenetic_protocols.append(trial_statement[-2])  # second to last value

    # structure the result
    return {
        'speed_times': speed_times,
        'speed_distance': speed_distance,
        'lick_times': lick_times,
        'lick_distance': lick_distance,
        'start_cue_times': start_cue_times,
        'reward_times': reward_times,
        'reward_distance': reward_distance,
        'trial_statement_times': trial_statement_times,
        'run_onsets': run_onsets,
        'optogenetic_protocols': optogenetic_protocols
    }
