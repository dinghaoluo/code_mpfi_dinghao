# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:13:00 2024

functions for processing behaviour text files 

@author: Dinghao Luo
@modifiers: Dinghao Luo, Jingyu Cao
"""


#%% imports 
import numpy as np 


#%% constants 
of_constant = (2**32-1)/1000  # overflow constant; this can only be hard-coded
early_late_lick_cutoff = 120  # cm; for lick-selectivity calculation


#%% main function 
def process_behavioural_data(
        txtfile: str, 
        max_distance=220, 
        distance_resolution=0.1,
        run_onset_initial=3.0, 
        run_onset_sustained=10.0, 
        run_onset_duration=300
        ) -> dict:
    '''
    processes behavioural data from a txt file, aligning speed, lick, and reward events 
    to both time and distance bases, while extracting metrics such as run onsets, 
    lick selectivity, trial quality, and full stop status.

    run-onset is detected using the interpolated wheel trace, matching the logic in the
    original MATLAB pipeline: first identifying sustained high-speed segments and then
    tracing back to the last sub-threshold frame.

    parameters:
    ----------
    txtfile : str
        path to the txt file containing behavioural data.
    max_distance : int, optional
        maximum distance for aligning data across trials (default is 220 cm).
    distance_resolution : int, optional
        distance step size in cm for interpolated distance base (default is 1 cm).
    run_onset_initial : float, optional
        initial speed threshold for detecting run onset (default is 3.0 cm/s).
    run_onset_sustained : float, optional
        sustained speed threshold for detecting run onset (default is 10.0 cm/s).
    run_onset_duration : int, optional
        duration (in ms) for which the sustained speed threshold must be held 
        to confirm run onset (default is 300 ms).

    returns:
    -------
    dict
        a dictionary containing aligned behavioural data across time and distance bases, including:
        - 'speed_times': list of lists, each containing [timestamp, speed] pairs for each trial.
        - 'speed_times_full': single list of all [timestamp, speed] pairs concatenated across trials.
        - 'speed_times_aligned': list of lists with speed traces aligned to run-onset (time = 0).
        - 'speed_distances': list of arrays, each containing speeds aligned to a common distance base.
        - 'lick_times': list of lists, each containing lick event timestamps for each trial.
        - 'lick_times_aligned': list of lists with lick timestamps aligned to run-onset.
        - 'lick_distances': list of arrays, each containing lick events aligned to the distance base.
        - 'lick_maps': list of arrays, each binary array representing lick events mapped onto the distance base.
        - 'start_cue_times': list of lists, each containing timestamps of start cues.
        - 'reward_times': list of lists, each containing timestamps of reward deliveries.
        - 'reward_distance': list of arrays, each containing reward events aligned to the distance base.
        - 'run_onsets': list of timestamps for detected run-onset events in each trial.
        - 'lick_selectivities': list of float values, one for each trial, representing lick selectivity indices.
        - 'trial_statements': list of lists containing trial-specific metadata and protocols.
        - 'full_stops': list of booleans indicating whether the animal fully stopped (speed < 10 cm/s) 
          between the previous trial's reward and the current trial's reward.
        - 'bad_trials': list of booleans indicating whether each trial was classified as "bad" based on:
            1. Presence of licks between 30 and 90 cm.
            2. Speeds below 10 cm/s for a cumulative duration >5 seconds between run-onset and reward.
            3. Lack of a detected run-onset or delivered reward.
        - 'frame_times': list of timestamps for each movie frame, corrected for overflow.
        - 'reward_omissions': raw omission info from txt parser.
    '''
    data = process_txt(txtfile)

    data['speed_times'] = correct_overflow(data['speed_times'], 'speed')
    data['lick_times'] = correct_overflow(data['lick_times'], 'lick')
    data['pump_times'] = correct_overflow(data['pump_times'], 'pump')
    data['movie_times'] = correct_overflow(data['movie_times'], 'movie')
    data['trial_statements'] = correct_overflow(
        data['trial_statements'], 'trial_statement'
        )  # equivalent to taskDescr in the MATLAB pipeline 
    data['new_trial_statements'] = correct_overflow(
        data['new_trial_statements'], 'new_trial_statement'
        )  # equivalent to trialDescr in the MATLAB pipeline 
    frame_times = correct_overflow(data['frame_times'], 'frame')

    # these are inputs to process_locomotion()
    wheel_tuples = [(t, v) for trial in data['speed_times'] for (t, _, v) in trial]
    trial_start_times = [float(x[1]) for x in data['trial_statements']]
    trial_end_times = [float(x[1]) for x in data['new_trial_statements']]

    (
        run_onsets, 
        upsampled_timestamps_ms, 
        upsampled_distance_cm, 
        smoothed_speed
    ) = process_locomotion(
        wheel_tuples,
        trial_start_times=trial_start_times,
        trial_end_times=trial_end_times,
        encoder_to_dist=0.04,  # encoder tick-to-cm constant 
        upsample_rate_hz=1000,  # upsample everything to 1 000 Hz 
        smooth_window_ms=100,  # for smoothing speeds
        min_speed1=run_onset_sustained,  # follows MATLAB naming conventions for easy debugging 
        min_speed=run_onset_initial,
        track_length_cm=180.0
    )
        
    # for distance accumulation 
    common_distance_base = np.linspace(
        0, max_distance, 
        int(max_distance / distance_resolution)
        )
    
    # now we start constructing aligned data structures
    speed_times_aligned = []
    speed_distances_aligned = []
    lick_times = []
    lick_distances_aligned = []
    lick_maps = []
    lick_selectivities = []
    start_cue_times = []
    reward_times = []
    reward_distances_aligned = []
    full_stops = []
    bad_trials = []

    # iterate through the trials to put together the data structures
    for trial_idx, onset in enumerate(run_onsets):
        if np.isnan(onset):
            speed_times_aligned.append([])  # append empty list if no onset 
            speed_distances_aligned.append([])  # same as above
            lick_distances_aligned.append([])
            lick_maps.append([])
            lick_selectivities.append(np.nan)
            start_cue_times.append(np.nan)
            reward_times.append(np.nan)
            reward_distances_aligned.append(np.nan)
            full_stops.append(np.nan)
            bad_trials.append(np.nan)
        else: 
            trial_end = (trial_end_times[trial_idx] 
                         if trial_idx < len(trial_end_times) 
                         else upsampled_timestamps_ms[-1])
            
            # slice into the trial window
            onset_idx = np.searchsorted(upsampled_timestamps_ms, onset)
            trial_end_idx = np.searchsorted(upsampled_timestamps_ms, trial_end)
            aligned_times = upsampled_timestamps_ms[onset_idx:trial_end_idx]
            aligned_speeds = smoothed_speed[onset_idx:trial_end_idx]
            
            # append aligned speed-time trace
            speed_times_aligned.append(list(zip(aligned_times, aligned_speeds)))
    
            # integrate speed to get distance (in cm), using fixed dt = 0.001 s
            distances = np.cumsum(aligned_speeds) * 0.001  # cm/s × s = cm
            
            # edge case
            if distances.size == 0:
                speed_times_aligned.append([])
                speed_distances_aligned.append([])
                lick_distances_aligned.append([])
                lick_maps.append([])
                lick_selectivities.append(np.nan)
                start_cue_times.append(np.nan)
                reward_times.append(np.nan)
                reward_distances_aligned.append(np.nan)
                full_stops.append(False)
                bad_trials.append(True)
                continue  # skip to next trial
            else:
                distances -= distances[0]  # shift to start from 0
            
            # interpolate onto distance base
            valid_base = common_distance_base[common_distance_base <= distances[-1]]
            interp_speeds = np.interp(valid_base, distances, aligned_speeds)
            padded_speeds = np.pad(
                interp_speeds, 
                (0, len(common_distance_base) - len(interp_speeds)), 
                constant_values=0
            )
            speed_distances_aligned.append(padded_speeds)
            
            # extract lick timestamps that occurred after run-onset
            try:
                lick_times_raw = [t for t, _ in data['lick_times'][trial_idx]]
                lick_times_post_onset = [t for t in lick_times_raw if t >= onset]
    
                # map these lick timestamps to distances within the trial
                lick_dists = np.interp(
                    lick_times_post_onset,
                    aligned_times,  # already absolute
                    distances,
                    left=np.nan,
                    right=np.nan
                )
    
                # clean up invalid mappings
                lick_dists = lick_dists[~np.isnan(lick_dists)]
                lick_distances_aligned.append(lick_dists)
    
                # construct binary lick map over distance base
                lick_map = np.zeros_like(common_distance_base)
                lick_bin_indices = np.searchsorted(common_distance_base, lick_dists)
                valid = lick_bin_indices < len(lick_map)
                lick_map[lick_bin_indices[valid]] = 1
                lick_maps.append(lick_map)
                
                # compute lick selectivity 
                lick_selectivities.append(lick_index(lick_map))
                
                # extract start cue times
                start_cue_times.append(data['movie_times'][trial_idx][0][0])
                
                # extract reward time and distance
                reward_time = (data['pump_times'][trial_idx][-1] 
                               if data['pump_times'][trial_idx] 
                               else np.nan)
                reward_times.append(reward_time)
                
                if not np.isnan(reward_time):
                    reward_dist = np.interp(
                        reward_time,
                        aligned_times,
                        distances,
                        left=np.nan,
                        right=np.nan
                    )
                    reward_distances_aligned.append(reward_dist)
                else:
                    reward_distances_aligned.append(np.nan)
                    
                # full stop detection
                prev_reward_time = data['pump_times'][trial_idx - 1][-1] if data['pump_times'][trial_idx - 1] else None
                curr_reward_time = reward_time if reward_time else None
                if prev_reward_time and curr_reward_time:
                    mask = (upsampled_timestamps_ms >= prev_reward_time) & (upsampled_timestamps_ms < curr_reward_time)
                    full_stop = np.any(smoothed_speed[mask] < 10)
                else:
                    full_stop = False
                full_stops.append(full_stop)
        
                is_bad_lick = np.any(lick_dists <= early_late_lick_cutoff)
                if not np.isnan(onset) and not np.isnan(reward_time):
                    mask = (upsampled_timestamps_ms >= onset) & (upsampled_timestamps_ms <= reward_time)
                    low_speed_duration = np.sum(smoothed_speed[mask] < 10) * 0.001
                    is_bad_speed = low_speed_duration > 5
                else:
                    is_bad_speed = True
                is_bad_reward = np.isnan(reward_time)
                bad_trials.append(is_bad_lick or is_bad_speed or is_bad_reward)
                
            except IndexError:  # the last trial's data are not logged due to incompletion
                lick_times.append(np.nan)
                lick_distances_aligned.append(np.nan)
                lick_selectivities.append(np.nan)
                start_cue_times.append(np.nan)
                reward_times.append(np.nan)
                reward_distances_aligned.append(np.nan)
                full_stops.append(np.nan)
                bad_trials.append(np.nan)

    return {
        'speed_times_aligned': speed_times_aligned,
        'speed_distances_aligned': speed_distances_aligned,
        'lick_times': data['lick_times'],
        'lick_distances_aligned': lick_distances_aligned,
        'lick_maps': lick_maps,
        'start_cue_times': start_cue_times,
        'reward_times': reward_times,
        'reward_distances_aligned': reward_distances_aligned,
        'run_onsets': run_onsets,
        'lick_selectivities': lick_selectivities,
        'trial_statements': data['trial_statements'],
        'new_trial_statements': data['new_trial_statements'],
        'full_stops': full_stops,
        'bad_trials': bad_trials,
        'frame_times': frame_times,
        'reward_omissions': data['reward_omissions'],
        'upsampled_timestamps_ms': upsampled_timestamps_ms,
        'upsampled_distance_cm': upsampled_distance_cm,
        'upsampled_speed_cm_s': smoothed_speed
    }


#%% for different types of recording
def process_behavioural_data_imaging(txtfile: str, 
                                     max_distance=220, 
                                     distance_resolution=1, 
                                     run_onset_initial=3.0, 
                                     run_onset_sustained=10.0, 
                                     run_onset_duration=300, 
                                     frame_threshold_ms=50) -> dict:
    """
    processes behavioural and imaging data, aligning wheel-derived events with imaging frames.

    parameters:
    - txtfile: path to behaviour .txt file
    - max_distance: max distance in cm for common distance base
    - distance_resolution: spacing of distance bins in cm
    - run_onset_initial: pre-run threshold in cm/s
    - run_onset_sustained: high-speed threshold in cm/s
    - run_onset_duration: sustained threshold duration in ms
    - frame_threshold_ms: max allowed gap between frame timestamps before interpolation

    returns:
    - dict: containing behavioural and imaging-aligned outputs, including:
        - speed/lick/reward aligned to time and distance
        - run_onset_frames: frame index of each run-onset
        - reward_frames: frame index of each pump time
        - start_cue_frames: frame index of each start cue
    """
    # process behavioural data
    behavioural_data = process_behavioural_data(txtfile, max_distance, distance_resolution, 
                                                run_onset_initial, run_onset_sustained, 
                                                run_onset_duration)

    # extract frame times
    frame_times = behavioural_data.get('frame_times', [])
    if not frame_times:
        raise ValueError('Frame times missing or empty in the txt file.')

    # correct for dropped frames (interpolate)
    for i in range(len(frame_times) - 1):
        if frame_times[i + 1] - frame_times[i] > frame_threshold_ms:
            interp_fm = (frame_times[i + 1] + frame_times[i]) / 2
            frame_times.insert(i + 1, interp_fm)

    # compute frame indices for behavioural events
    run_onset_frames = []
    for onset in behavioural_data['run_onsets']:
        if onset < frame_times[0] or onset > frame_times[-1]:
            run_onset_frames.append(-1)  # out of frame range
        elif onset != -1:
            run_onset_frames.append(find_nearest(onset, frame_times))
        else:
            run_onset_frames.append(-1)

    pump_frames = []
    for pump in behavioural_data['reward_times']:
        if np.isnan(pump) or pump < frame_times[0] or pump > frame_times[-1]:
            pump_frames.append(-1)  # out of frame range or no reward
        else:
            pump_frames.append(find_nearest(pump, frame_times))

    cue_frames = []
    for cue in behavioural_data['start_cue_times']:
        if np.isnan(cue) or cue < frame_times[0] or cue > frame_times[-1]:
            cue_frames.append(-1)  # out of frame range or no cue
        else:
            cue_frames.append(find_nearest(cue, frame_times))

    # add imaging data to output
    behavioural_data.update({
        'run_onset_frames': run_onset_frames,
        'reward_frames': pump_frames,
        'start_cue_frames': cue_frames
    })

    return behavioural_data

def process_behavioural_data_immobile(txtfile: str) -> dict: #Jingyu, 4/17/2025
    """
    processes behavioural data for immobile experiments.

    parameters:
    - txtfile: path to the behaviour .txt file

    returns:
    - dict: trial-wise parsed outputs including:
        - lick_times: lick timestamps
        - reward_times: pump delivery times
        - start_cue_times: trial start cue times
        - trial_statements: metadata and protocol for each trial
        - frame_times: raw imaging frame timestamps (if available)

    notes:
    - assumes animal is immobile, and wheel/speed data are irrelevant.
    """
    # load and parse the txt file
    data = process_txt_immobile(txtfile)  # uses user's custom `process_txt` function
    
    # correct for overflow in the data
    data['lick_times'] = correct_overflow(data['lick_times'], 'lick')
    data['pump_times'] = correct_overflow(data['pump_times'], 'pump')
    data['movie_times'] = correct_overflow(data['movie_times'], 'movie')
    data['trial_statements'] = correct_overflow(data['trial_statements'], 'trial_statement')
    if len(data['frame_times'])>0:
        frame_times = correct_overflow(data['frame_times'], 'frame')
    else:
        frame_times = []
    
    # initialise lists for storing data across trials
    lick_times = []
    start_cue_times, reward_times = [], []
    trial_statements = []
    
    # process each trial
    for trial_idx, (lick_trial, movie_trial, reward_trial) in enumerate(
        zip(data['lick_times'], data['movie_times'], data['pump_times'])
    ):
        # process licks
        lick_times_trial = [event[0] for event in lick_trial]
        
        # process lick data
        lick_times_trial = [event[0] for event in lick_trial]
        lick_times.append(lick_times_trial)
    
        # process reward data
        reward_times.append(reward_trial)
    
        # extract start cue times
        start_cue_times.append([m[0] for m in movie_trial if m[1] == 2])
    
        # trial statement times and optogenetic protocol
        trial_statement = data['trial_statements'][trial_idx]
        trial_statements.append(trial_statement)
    
    # structure the result
    return {
        'lick_times': lick_times,
        'start_cue_times': start_cue_times,
        'reward_times': reward_times,
        'trial_statements': trial_statements,
        'frame_times': frame_times,  # this was actually added just to prevent np.array(... type='object) from automatically producing a 2D array, 6 Dec 2024
        # Jingyu, 4/17/2025
    }

def process_behavioural_data_immobile_imaging(txtfile, frame_threshold_ms=50):  # Jingyu 4/17/2025
    """
    adds frame-aligned indices to immobile behavioural data.

    parameters:
    - txtfile: path to the behavioural .txt file
    - frame_threshold_ms: max gap allowed between frame timestamps

    returns:
    - dict: everything from `process_behavioural_data_immobile`, plus:
        - reward_frames: frame index of each pump event
        - start_cue_frames: frame index of each start cue event
    """    
    # process behavioural data
    behavioural_data = process_behavioural_data_immobile(txtfile)

    # extract frame times
    frame_times = behavioural_data.get('frame_times', [])
    if not frame_times:
        raise ValueError('Frame times missing or empty in the txt file.')

    # correct for dropped frames (interpolate)
    for i in range(len(frame_times) - 1):
        if frame_times[i + 1] - frame_times[i] > frame_threshold_ms:
            interp_fm = (frame_times[i + 1] + frame_times[i]) / 2
            frame_times.insert(i + 1, interp_fm)

    # compute frame indices for behavioural events

    pump_frames = []
    for pump in behavioural_data['reward_times']:
        if not pump or pump[0] < frame_times[0] or pump[0] > frame_times[-1]:
            pump_frames.append(-1)  # out of frame range or no reward
        else:
            pump_frames.append(find_nearest(pump[0], frame_times))

    cue_frames = []
    for cue in behavioural_data['start_cue_times']:
        if not cue or cue[0] < frame_times[0] or cue[0] > frame_times[-1]:
            cue_frames.append(-1)  # out of frame range or no cue
        else:
            cue_frames.append(find_nearest(cue[0], frame_times))

    # add imaging data to output
    behavioural_data.update({
        'reward_frames': pump_frames,
        'start_cue_frames': cue_frames
    })
    
    return behavioural_data


#%% utils 
def find_nearest(value, arr):
    '''
    find the index of the nearest value in a list.

    parameters:
    - value: the value to match.
    - arr: list of float values to search from.

    returns:
    - int: index of the nearest value in arr.
    '''
    nearest_value = min(arr, key=lambda x: abs(x-value))
    nearest_value_index = arr.index(nearest_value)
    return nearest_value_index

def lick_index(lick_map):
    '''
    compute post-vs-pre lick selectivity from a binary lick map.
    
    parameters:
    - lick_map: binary 1d array indicating presence of licks at each distance bin.
    
    returns:
    - float: selectivity index (nan if no licks present).
    '''
    midpoint = len(lick_map) // 2
    sum_pre = np.sum(lick_map[:midpoint])
    sum_post = np.sum(lick_map[midpoint:])
    denom = sum_pre + sum_post
    if denom == 0:
        return np.nan
    return sum_post / denom

def process_txt(txtfile):
    '''
    parse a behavioural text file and extract trial-resolved events and metadata.

    parameters:
    - txtfile: path to the behavioural .txt file.

    returns:
    - dict: contains per-trial structured data for speed, licks, reward, cues, frames, and trial metadata.
    '''
    curr_logfile = {} 
    file = open(txtfile, 'r')
    
    line = ['']
    while line[0] != '$TR':
        line = get_next_line(file)
        
    lick_times = []
    pump_times = []
    movie_times = []
    speed_times = []
    # motor_times = []
    pulse_times = []
    frame_times = []
    
    # mt_trial = []
    wt_trial = []
    lt_trial = []
    pt_trial = []
    mv_trial = []
    pc_trial = []
    pulse_command_list = []
    current_pulse_command = []
    
    trial_statements = []  # start of trial 
    new_trial_statements = []  # end of trial 
    reward_omissions = []
    
    while line[0].find('$') == 0:
        if line[0] == '$TR': # need to update motor_times here - but ignore motors before first trial started. 
            # motor_times.append(mt_trial)
            # mt_trial = []
            trial_statements.append(line)
        if line[0] == '$MV':
            mv_trial.append([float(line[1]), float(line[2])])
        if line[0] == '$WE':
            wt_trial.append(
                [float(line[1]), 
                 float(line[2])*.04*50,  # the number of clicks per 20 ms, and each click corresponds to .04 cm, Dinghao, 20240625
                 float(line[3])]  # distance accumulation
                )  
        if line[0] == '$LE' and line[3] == '1':
            lt_trial.append([float(line[1]), float(line[2])]) 
        if line[0] == '$PE' and line[3] == '1':
            pt_trial.append(float(line[1]))
        # if line[0] == '$MT':
            # mt_trial.append([float(line[1]), float(line[2])])
        if line[0] == '$PC':
            pc_trial.append(float(line[1]))
        if line[0] == '$PP':
            current_pulse_command = line       
        if line[0] == '$NT':
            new_trial_statements.append(line)
            
            lick_times.append(lt_trial)
            movie_times.append(mv_trial)
            pump_times.append(pt_trial)
            speed_times.append(wt_trial)
            pulse_times.append(pc_trial)
            pulse_command_list.append(current_pulse_command)
            if len(line) == 5: # $NT line has reward omission label, Jingyu, 8/14/2024
                reward_omissions.append(line[-1]) #Jingyu, 8/14/2024
            lt_trial = []
            mv_trial = []
            pt_trial = []
            wt_trial = []
            pc_trial = []
        if line[0] == '$FM' and line[2] == '0':
            frame_times.append(float(line[1]))
        line = get_next_line(file)
        
    curr_logfile['speed_times'] = speed_times
    curr_logfile['movie_times'] = movie_times
    curr_logfile['lick_times'] = lick_times
    curr_logfile['pump_times'] = pump_times
    # curr_logfile['motor_times'] = motor_times
    curr_logfile['pulse_times'] = pulse_times
    curr_logfile['frame_times'] = frame_times
    curr_logfile['trial_statements'] = trial_statements
    curr_logfile['new_trial_statements'] = new_trial_statements
    curr_logfile['pulse_descriptions'] = pulse_command_list
    curr_logfile['reward_omissions'] = reward_omissions  # jingyu, 8/14/2024
    return curr_logfile

def process_txt_immobile(txtfile): # Jingyu 4/17/2025
    '''
    parse a behavioural .txt file for immobile experiments.

    parameters:
    - txtfile: path to the immobile session behaviour log.

    returns:
    - dict: trial-wise data including licks, rewards, start cues, and optional frame times.
    '''
    curr_logfile = {} 
    file = open(txtfile, 'r')
    
    line = ['']
    while line[0] != '$TR':
        line = get_next_line(file)
        
    lick_times = []
    pump_times = []
    movie_times = []
    frame_times = [] # Jingyu, 4/17/2025
    
    lt_trial = []
    pt_trial = []
    mv_trial = []
    
    trial_statements = []
    
    while line[0].find('$') == 0:
        if line[0] == '$TR': 
            trial_statements.append(line)
        if line[0] == '$MV':
            mv_trial.append([float(line[1]), float(line[2])])
        if line[0] == '$LE':
            if len(line)<4: print(line)
        if line[0] == '$LE' and line[3] == '1':
            lt_trial.append([float(line[1]), float(line[2])]) 
        if line[0] == '$PE' and line[3] == '1':
            pt_trial.append(float(line[1]))    
        if line[0] == '$NT':
            lick_times.append(lt_trial)
            movie_times.append(mv_trial)
            pump_times.append(pt_trial)
            lt_trial = []
            mv_trial = []
            pt_trial = []
        if line[0] == '$FM' and line[2] == '0': # Jingyu, 4/17/2025
            frame_times.append(float(line[1]))  
        line = get_next_line(file)
        
    curr_logfile['movie_times'] = movie_times
    curr_logfile['lick_times'] = lick_times
    curr_logfile['pump_times'] = pump_times
    curr_logfile['trial_statements'] = trial_statements
    curr_logfile['frame_times'] = frame_times # Jingyu, 4/17/2025
    return curr_logfile

def get_next_line(file):
    '''
    read and return the next non-empty, comma-split line from a file.

    parameters:
    - file: open file handle.

    returns:
    - list: line split by commas.
    '''
    line = file.readline().rstrip('\n').split(',')
    if len(line) == 1: # read an empty line
        line = file.readline().rstrip('\n').split(',')
    return line

def correct_overflow(data, label):
    '''
    adjust timestamp-based trial data for hardware overflow events.

    parameters:
    - data: list of per-trial timestamp events (e.g., speed, lick).
    - label: type of data ('speed', 'lick', 'movie', etc.).

    returns:
    - list: overflow-corrected trial-wise event data.
    '''
    tot_trial = len(data)
    new_data = []
    
    if label=='speed':
        curr_time = data[0][0][0]
        for t in range(tot_trial):
            if data[t][-1][0]-curr_time>=0:  # if the last speed cell is within overflow, then simply append
                new_data.append(data[t])
                curr_time = data[t][-1][0]
            else:  # once overflow is detected, do not update curr_time
                new_trial = []
                curr_trial = data[t]
                curr_length = len(curr_trial)
                for s in range(curr_length):
                    if curr_trial[s][0]-curr_time>0:
                        new_trial.append(curr_trial[s])
                    else:
                        new_trial.append([curr_trial[s][0]+of_constant, curr_trial[s][1], curr_trial[s][2]])
                new_data.append(new_trial)              
    if label == 'lick':  # added by Jingyu 6/22/2024
        first_trial_with_licks = next(x for x in data if len(x)!=0)  # in case the first trial has no licks, Dinghao, 20240626
        curr_time = first_trial_with_licks[0][0]
        for t in range(tot_trial):
            if len(data[t])==0:  # if there is no lick, append an empty list
                new_data.append([])
            elif data[t][-1][0]-curr_time>=0:  # if the last lick cell is within overflow, then simply append
                new_data.append(data[t])
                curr_time = data[t][-1][0]
            else:  # once overflow is detected, do not update curr_time
                new_trial = []
                curr_trial = data[t]
                curr_length = len(curr_trial)
                for s in range(curr_length):
                    if curr_trial[s][0]-curr_time>0:
                        new_trial.append(curr_trial[s])
                    else:
                        new_trial.append([curr_trial[s][0]+of_constant, curr_trial[s][1]])
                new_data.append(new_trial)
    if label=='pulse':
        try:
            first_trial_with_pulse = next(x for x in data if len(x)!=0)  # first trial with pulse, Jingyu, 20240926
            curr_time = first_trial_with_pulse[0]
            for t in range(tot_trial):
                if len(data[t])==0:  # if there is no pulse, append an empty list
                    new_data.append([])
                elif data[t][-1]-curr_time>=0:  # if the last pulse cell is within overflow, then simply append
                    new_data.append(data[t])
                    curr_time = data[t][-1]
                else:  # once overflow is detected, do not update curr_time
                    new_trial = []
                    curr_trial = data[t]
                    curr_length = len(curr_trial)
                    for s in range(curr_length):
                        if curr_trial[s]-curr_time>0:
                            new_trial.append(curr_trial[s])
                        else:
                            new_trial.append(curr_trial[s]+of_constant)
                    new_data.append(new_trial)
        except StopIteration:  # if no pulses in this session 
            new_data = data
    if label=='pump':
        first_trial_with_pump = next(x for x in data if len(x)!=0)  # in case the first trial has no pump, Dinghao, 20240704
        curr_time = first_trial_with_pump[0]
        for t in range(tot_trial):
            if len(data[t])==0: # if there is no reward, append 0, Jingyu, 20270730
                new_data.append([])
            elif len(data[t])!=0:
                if data[t][0]-curr_time>=0:
                    new_data.append([data[t][0]])
                    curr_time = data[t][0]
                else:  # once overflow is detected, do not update curr_time
                    new_data.append([data[t][0]+of_constant])
    if label=='movie':
        curr_time = data[0][0][0]
        for t in range(tot_trial):
            if data[t][-1][0]-curr_time>=0:
                new_data.append(data[t])
                curr_time = data[t][-1][0]
            else:  # once overflow is detected, do not update curr_time
                new_trial = []
                curr_trial = data[t]
                curr_length = len(curr_trial)
                for s in range(curr_length):
                    if curr_trial[s][0]-curr_time>=0: #Jingyu 8/18/24
                        new_trial.append(curr_trial[s])
                    else:
                        new_trial.append([curr_trial[s][0]+of_constant, curr_trial[s][1]])
                new_data.append(new_trial)
    if label=='frame':
        if not data:
            new_data.append([])
        else:
            curr_time = data[0]
            for f in data:
                if f-curr_time>=0:
                    new_data.append(f)
                    curr_time = f
                else:  # once overflow is detected, do not update curr_time
                    new_data.append(f+of_constant)
    if label=='trial_statement':
        curr_time = float(data[0][1])
        for t in range(tot_trial):
            if float(data[t][1])-curr_time>=0:
                new_data.append(data[t])
                curr_time = float(data[t][1])
            else:  # once overflow is detected, do not update curr_time
                new_trial = data[t]
                new_time = float(new_trial[1])+of_constant
                new_trial[1] = str(new_time)
                new_data.append(new_trial)
    if label=='new_trial_statement':
        curr_time = float(data[0][1])
        for t in range(tot_trial):
            if float(data[t][1])-curr_time>=0:
                new_data.append(data[t])
                curr_time = float(data[t][1])
            else:  # once overflow is detected, do not update curr_time
                new_trial = data[t]
                new_time = float(new_trial[1])+of_constant
                new_trial[1] = str(new_time)
                new_data.append(new_trial)
        
    return new_data


#%% run-onset detection 
from scipy.ndimage import uniform_filter1d

def process_locomotion(wheel_tuples,
                       trial_start_times,
                       trial_end_times,
                       encoder_to_dist=0.04,
                       upsample_rate_hz=1000,
                       smooth_window_ms=100,
                       sample_freq_hz=1000,
                       min_speed1=10.0,
                       min_speed=1.0,
                       track_length_cm=180.0):
    '''
    process raw wheel encoder ticks to produce upsampled distance, smoothed speed,
    and run-onset timestamps replicating MATLAB logic.

    parameters:
    -----------
    wheel_tuples : list of tuples
        list of (timestamp in ms, tick count) pairs sorted in time.
    trial_start_times : list
        list of trial start timestamps (in ms).
    trial_end_times : list
        list of trial end timestamps (in ms); may be one shorter than start times.
    encoder_to_dist : float
        conversion factor from encoder ticks to cm (default: 0.04).
    upsample_rate_hz : int
        output resolution (default: 1000 Hz = 1 ms).
    smooth_window_ms : int
        speed smoothing window size (default: 100 ms).
    sample_freq_hz : int
        sampling frequency of interpolated data (default: 1000 Hz).
    min_speed1 : float
        high-speed threshold for detecting run (default: 10 cm/s).
    min_speed : float
        low-speed threshold for identifying stillness (default: 1 cm/s).
    track_length_cm : float
        track length for reset detection (default: 180 cm).

    returns:
    --------
    run_onset_times : list
        list of detected run-onset times in ms (first trial is skipped).
    upsampled_timestamps : np.ndarray
        array of uniformly sampled timepoints in ms.
    upsampled_distance_cm : np.ndarray
        distance trace including resets (in cm).
    speed_smoothed_cm_s : np.ndarray
        smoothed speed trace (in cm/s).
    '''
    if not wheel_tuples:
        return [], np.array([]), np.array([]), np.array([])

    # unpack raw wheel ticks
    timestamps, ticks = zip(*wheel_tuples)
    timestamps = np.array(timestamps)
    ticks = np.array(ticks)

    # compute cumulative distance with reset
    raw_distances = []
    acc = 0
    for i in range(len(ticks)):
        if i > 0 and (ticks[i] - ticks[i - 1]) < -10:
            acc = 0
        else:
            acc += ticks[i] - ticks[i - 1] if i > 0 else 0
        raw_distances.append(acc * encoder_to_dist)

    raw_distances = np.array(raw_distances)
    raw_distances_reset = raw_distances.copy()

    # make continuous (remove resets)
    resets = np.where(np.diff(raw_distances) < 0)[0] + 1
    resets = np.concatenate([resets, [len(raw_distances)]])
    for i in range(len(resets) - 1):
        start = resets[i]
        end = resets[i+1]
        offset = raw_distances[start - 1]
        raw_distances[start:end] += offset

    continuous_distance_cm = raw_distances - raw_distances[0]

    # upsample to uniform timestamps
    start_time = timestamps[0]
    end_time = timestamps[-1]
    upsampled_timestamps = np.arange(start_time, end_time + 1, 1000 / upsample_rate_hz)
    upsampled_distance_cm = np.interp(upsampled_timestamps, timestamps, raw_distances_reset)
    upsampled_distance_cm_all = np.interp(upsampled_timestamps, timestamps, continuous_distance_cm)

    # compute smoothed speed
    dt = 1 / upsample_rate_hz
    speed = np.gradient(upsampled_distance_cm_all, dt)
    window_samples = max(int(smooth_window_ms / 1000 * upsample_rate_hz), 1)
    speed_smoothed = uniform_filter1d(speed, size=window_samples)

    # detect run onsets
    run_onset_times = [np.nan]  # first trial skipped

    for i in range(1, len(trial_start_times)):
        t_prev = trial_start_times[i-1]
        t0     = trial_start_times[i]
        t1     = trial_end_times[i] if i < len(trial_end_times) else upsampled_timestamps[-1]

        # trial boundaries in the upsampled array
        idx_prev_start = np.searchsorted(upsampled_timestamps, t_prev)
        idx_curr_start = np.searchsorted(upsampled_timestamps, t0)
        idx_curr_end   = np.searchsorted(upsampled_timestamps, t1)

        # — reset detection on raw distances (same as before) —
        trace_prev = upsampled_distance_cm[idx_prev_start:idx_curr_start]
        diffs      = np.diff(trace_prev)
        resets     = np.where(diffs < -track_length_cm/2)[0]

        if resets.size > 0:
            last_reset = resets[-1] + 1
            post       = np.where(trace_prev[last_reset:] > track_length_cm)[0]
            if post.size:
                idx_exit = idx_prev_start + last_reset + post[0]
            else:
                overall = np.where(trace_prev > track_length_cm)[0]
                idx_exit = (idx_prev_start + overall[0]) if overall.size else idx_curr_start
        else:
            overall = np.where(trace_prev > track_length_cm)[0]
            idx_exit = (idx_prev_start + overall[0]) if overall.size else idx_curr_start

        idx_start = idx_exit
        idx_end   = idx_curr_end

        # define segments
        len_diff   = idx_curr_start - idx_start
        speed_all  = speed_smoothed[idx_start:idx_end]
        speed_cur  = speed_smoothed[idx_curr_start:idx_curr_end]

        # compute run/stop lengths for the current-trial speed
        is_run = speed_cur > min_speed1
        run_lengths, stop_lengths = [], []
        cnt = 0
        for v in is_run:
            if v:
                if cnt < 0:
                    stop_lengths.append(-cnt)
                    cnt = 1
                else:
                    cnt += 1
            else:
                if cnt > 0:
                    run_lengths.append(cnt)
                    cnt = -1
                else:
                    cnt -= 1
        if cnt > 0:
            run_lengths.append(cnt)
        elif cnt < 0:
            stop_lengths.append(-cnt)

        # first running sample in trial
        ind_first = np.where(is_run)[0]
        ind_first = ind_first[0] if ind_first.size else None

        # first continuous-run block >0.3 s
        ind_conti = next((j for j, r in enumerate(run_lengths) if r > 0.3 * sample_freq_hz), None)

        # compute ind_start (0-based into speed_all)
        if ind_first is not None and ind_first > 0:
            if ind_conti is not None and ind_conti > 0:
                offset = sum(run_lengths[:ind_conti]) + sum(stop_lengths[:ind_conti])
                ind_start = offset + len_diff
            else:
                ind_start = ind_first + len_diff
        else:
            if ind_conti is not None and ind_conti > 0:
                offset = sum(run_lengths[:ind_conti]) + sum(stop_lengths[:ind_conti])
                ind_start = offset + len_diff
            else:
                # no adequate run in trial: find last full stop in pre-trial segment
                pre_mask = speed_all[:len_diff] > min_speed1
                zeros    = np.where(~pre_mask)[0]
                ind_start = zeros[-1] if zeros.size else -2

        # now back-search for the last moment ≤ min_speed before ind_start
        if ind_start > 0:
            segment = speed_all[:ind_start]
            lows    = np.where(segment <= min_speed)[0]
            if lows.size:
                onset_local = lows[-1]
            else:
                # fallback: pick lowest-speed sample
                onset_local = int(np.argmin(segment))
        else:
            # no full stop at all: pick min-speed in pre-trial window
            seg_pre = speed_all[:len_diff]
            onset_local = int(np.argmin(seg_pre)) if seg_pre.size else np.nan

        true_idx = idx_start + onset_local
        if np.isnan(true_idx):
            run_onset_times.append(np.nan)
        else:
            run_onset_times.append(upsampled_timestamps[int(true_idx)])

    return run_onset_times, upsampled_timestamps, upsampled_distance_cm, speed_smoothed