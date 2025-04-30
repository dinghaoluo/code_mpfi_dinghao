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
        distance_resolution=1, 
        run_onset_initial=3.0, 
        run_onset_sustained=10.0, 
        run_onset_duration=300
        ) -> dict:
    '''
    processes behavioural data from a txt file, aligning speed, lick, and reward events 
    to both time and distance bases, while extracting metrics such as run onsets, 
    lick selectivity, trial quality, and full stop status.

    gaps between trials are filled with 0 cm/s data sampled at 50 Hz to ensure continuous 
    speed trace for stitching. run-onset is detected by first identifying a sustained high-speed 
    segment, then tracing back to the last sub-threshold speed before it.

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
    data['trial_statements'] = correct_overflow(data['trial_statements'], 'trial_statement')
    frame_times = correct_overflow(data['frame_times'], 'frame')

    common_distance_base = np.linspace(0, max_distance, int(max_distance / distance_resolution))

    speed_times = []
    lick_times = []
    lick_maps = []
    start_cue_times = []
    reward_times = []
    reward_distances = []
    run_onsets = []
    trial_statements = []
    lick_selectivities = []
    full_stops = []
    bad_trials = []

    speed_times_full = []
    speed_times_aligned, lick_times_aligned = [], []
    speed_distances, lick_distances = [], []

    previous_times, previous_speeds = [], []
    previous_reward = []

    for trial_idx, (speed_trial, lick_trial, movie_trial, reward_trial) in enumerate(
        zip(data['speed_times'], data['lick_times'], data['movie_times'], data['pump_times'])
    ):
        times = [s[0] for s in speed_trial]
        speeds = [s[1] for s in speed_trial]
        formatted_speed_times = list(zip(times, speeds))
        speed_times.append(formatted_speed_times)
        speed_times_full.extend(formatted_speed_times)

        distances = np.cumsum(speeds) * (np.diff(times, prepend=times[0]) / 1000)
        distances -= distances[0]

        run_onset = np.nan

        # fill the gap between trials with speed=0 if necessary
        if previous_times and previous_speeds:
            gap_start = previous_times[-1]
            gap_end = times[0]
            if gap_end - gap_start > 20:
                gap_times = np.arange(gap_start + 20, gap_end, 20)
                gap_speeds = np.zeros_like(gap_times)
                previous_times = previous_times + list(gap_times)
                previous_speeds = previous_speeds + list(gap_speeds)

        # stitch previous and current trial
        concat_times = previous_times + times
        concat_speeds = previous_speeds + speeds
        concat_times = np.array(concat_times)
        concat_speeds = np.array(concat_speeds)

        # compute uniform time base for putative onset detection
        uni_time = np.linspace(times[0], times[-1], int(times[-1] - times[0]))
        uni_speed = np.interp(uni_time, times, speeds)

        # find sustained high-speed segment as putative onset
        putative_onset = np.nan
        count = 0
        for i in range(len(uni_speed)):
            if uni_speed[i] > run_onset_sustained:
                count += 1
            else:
                count = 0
            if uni_speed[i] > run_onset_initial and count > run_onset_duration:
                putative_onset = uni_time[i] - run_onset_duration
                break

        # if found, search back to last sub-threshold dip
        if not np.isnan(putative_onset):
            prior_idx = np.where(concat_times <= putative_onset)[0]
            prior_speeds = concat_speeds[prior_idx]
            prior_times = concat_times[prior_idx]
            below_idx = np.where(prior_speeds < run_onset_sustained)[0]
            if len(below_idx) > 0 and (trial_idx == 0 or prior_times[below_idx[-1]] > previous_reward):
                run_onset = prior_times[below_idx[-1]]
            else:
                run_onset = putative_onset
        else:
            # fallback: no sustained segment found, trace back from start of trial
            prior_idx = np.where(concat_times <= times[0])[0]
            prior_speeds = concat_speeds[prior_idx]
            prior_times = concat_times[prior_idx]
            below_idx = np.where(prior_speeds < run_onset_sustained)[0]
            if len(below_idx) > 0 and (trial_idx == 0 or prior_times[below_idx[-1]] > previous_reward[-1]):
                run_onset = prior_times[below_idx[-1]]
            else:
                run_onset = np.nan

        run_onsets.append(run_onset)

        if not np.isnan(run_onset):
            aligned_times = [t - run_onset for t in concat_times if t > run_onset]
            aligned_speeds = [s for t, s in zip(concat_times, concat_speeds) if t > run_onset]
            speed_times_aligned.append(list(zip(aligned_times, aligned_speeds)))
        else:
            speed_times_aligned.append([])

        lick_times_trial = [event[0] for event in lick_trial]
        lick_times.append(lick_times_trial)

        if not np.isnan(run_onset):
            lick_times_aligned.append([t - run_onset for t in lick_times_trial if t > run_onset])
        else:
            lick_times_aligned.append([])

        lick_distances_trial = np.interp(lick_times_trial, times, distances)

        if not np.isnan(run_onset):
            run_distance = np.interp(run_onset, times, distances)
            lick_distances.append(np.array([d - run_distance for t, d in zip(lick_times_trial, lick_distances_trial) if t > run_onset]))

            distances_shifted = distances - run_distance
            valid_distance_base = common_distance_base[common_distance_base <= (distances_shifted[-1] if len(distances_shifted) > 0 else 0)]
            interpolated_speed_shifted = np.interp(valid_distance_base, distances_shifted, speeds, left=0, right=0)
            padded_speed_shifted = np.pad(interpolated_speed_shifted, (0, len(common_distance_base) - len(interpolated_speed_shifted)), 'constant')
            speed_distances.append(padded_speed_shifted)
        else:
            lick_distances.append([])
            speed_distances.append(np.zeros_like(common_distance_base))

        lick_indices = np.searchsorted(common_distance_base, lick_distances_trial)
        lick_map_trial = np.zeros(len(common_distance_base))
        valid_indices = lick_indices[lick_indices < len(lick_map_trial)]
        lick_map_trial[valid_indices] = 1
        lick_maps.append(lick_map_trial)

        reward_times.append(reward_trial)
        reward_distances.append(np.interp(reward_trial, times, distances) if reward_trial else [])

        start_cue_times.append([m[0] for m in movie_trial if m[1] == 2])
        trial_statements.append(data['trial_statements'][trial_idx])

        lick_selectivities.append(lick_index(lick_map_trial))

        if trial_idx > 0:
            previous_reward_time = reward_times[trial_idx - 1][-1] if reward_times[trial_idx - 1] else None
            current_reward_time = reward_trial[-1] if reward_trial else None
            if previous_reward_time is not None and current_reward_time is not None:
                time_mask = [(t > previous_reward_time and t < current_reward_time) for t in times]
                speeds_during_pause = np.array(speeds)[time_mask]
                full_stop = np.any(speeds_during_pause < 10)
            else:
                full_stop = False
        else:
            full_stop = False
        full_stops.append(full_stop)

        is_bad_trial_lick = np.any((lick_distances_trial >= 30) & (lick_distances_trial <= 90))
        if not np.isnan(run_onset) and reward_trial:
            reward_time = reward_trial[-1]
            speeds_between_onset_reward = [speed for t, speed in formatted_speed_times if run_onset <= t <= reward_time]
            total_low_speed_duration = np.sum(np.array(speeds_between_onset_reward) < 10) * (times[1] - times[0]) / 1000
            is_bad_trial_speed = total_low_speed_duration > 5
        else:
            is_bad_trial_speed = True
        is_bad_trial_reward = len(reward_trial) == 0
        bad_trials.append(is_bad_trial_lick or is_bad_trial_speed or is_bad_trial_reward)

        previous_times = times
        previous_speeds = speeds
        previous_reward = reward_trial

    return {
        'speed_times': speed_times,
        'speed_distances': speed_distances,
        'lick_times': lick_times,
        'lick_distances': lick_distances,
        'lick_maps': lick_maps,
        'start_cue_times': start_cue_times,
        'reward_times': reward_times,
        'reward_distances': reward_distances,
        'run_onsets': run_onsets,
        'lick_selectivities': lick_selectivities,
        'trial_statements': trial_statements,
        'full_stops': full_stops,
        'bad_trials': bad_trials,
        'frame_times': frame_times,
        'reward_omissions': data['reward_omissions'],
        'speed_times_full': speed_times_full,
        'speed_times_aligned': speed_times_aligned,
        'lick_times_aligned': lick_times_aligned
    }

def process_behavioural_data_imaging(txtfile: str, 
                                     max_distance=220, 
                                     distance_resolution=1, 
                                     run_onset_initial=3.0, 
                                     run_onset_sustained=10.0, 
                                     run_onset_duration=300, 
                                     frame_threshold_ms=50) -> dict:
    '''
    processes behavioural and imaging data, aligning speed, lick, reward events, 
    and imaging frames to both time and distance bases.

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
        duration (in ms) for which the sustained speed threshold must be held to confirm run onset (default is 300 ms).
    frame_threshold_ms : int, optional
        maximum allowed gap (in ms) between consecutive frame times before interpolation (default is 50 ms).

    returns:
    -------
    dict
        a dictionary containing aligned behavioural and imaging data, including:
        - 'speed_times', 'speed_distance', 'lick_times', 'lick_distance', etc. 
          (from process_behavioural_data).
        - 'frame_times': list of interpolated imaging frame timestamps.
        - 'run_onset_frames': list of frame indices closest to run-onset times.
        - 'pump_frames': list of frame indices closest to pump times.
        - 'cue_frames': list of frame indices closest to cue times.
    '''
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
        'run_onset_frames': run_onset_frames,
        'reward_frames': pump_frames,
        'start_cue_frames': cue_frames
    })

    return behavioural_data

def process_behavioural_data_immobile(txtfile: str) -> dict: #Jingyu, 4/17/2025
    '''
    processes behavioural data for immobile experiments, aligning lick, reward events, and start cue times.

    parameters:
    - txtfile: path to the behaviour text file to process.
    
    returns:
    - dict: a dictionary containing trial-wise data for:
        - 'lick_times': list of lists, each containing lick event timestamps for each trial.
        - 'start_cue_times': list of lists, each containing timestamps of start cues.
        - 'reward_times': list of lists, each containing reward delivery timestamps.
        - 'trial_statements': list of trial-specific metadata and protocols.
    
    notes:
    - assumes the experiment involves immobile animals.
    - file format must adhere to specific `$TR`, `$NT`-type headers.
    '''
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

def process_behavioural_data_immobile_imaging(txtfile, frame_threshold_ms=50): # Jingyu 4/17/2025
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
    finds the nearest value in a list and returns its index and value.

    parameters:
    ----------
    value : float
        the target value to find the nearest match for.
    arr : list of float
        the list to search for the nearest value.

    returns:
    -------
    int
        - the index of the nearest value in the list
    '''
    nearest_value = min(arr, key=lambda x: abs(x-value))
    nearest_value_index = arr.index(nearest_value)
    return nearest_value_index

def lick_index(lick_vector):
    '''
    calculates the lick-selectivity index (LI) for a trial.

    parameters:
    ----------
    lick_vector : list of float
        a list containing lick distances a trial.

    returns:
    -------
    float
        LI for the current trial. 
        LI is the proportion of licks in post-120cm (or any defined) locations 
        relative to the total licks.

    notes:
    -----
    - LI is nan if there are no licks in any trial.
    '''
    sum_pre = sum(lick_vector[:early_late_lick_cutoff])
    sum_post = sum(lick_vector[early_late_lick_cutoff:])
    return sum_post/(sum_pre+sum_post)

def process_txt(txtfile):
    '''
    parses a behaviour text file and extracts trial-related data.
    
    parameters:
    ----------
    txtfile : str
    path to the behaviour text file to process.
    
    returns:
    -------
    dict
    a dictionary containing trial-wise data for:
    - speed_times: trial-wise speed timestamps and speeds
    - movie_times: timestamps and events from movie sequences
    - lick_times: lick event timestamps
    - pump_times: reward delivery timestamps
    - motor_times: motor control events (if relevant)
    - pulse_times: optogenetic pulse timestamps
    - frame_times: timestamps of individual movie frames
    - trial_statements: trial-specific metadata and protocols
    - pulse_descriptions: descriptions of optogenetic pulses
    - reward_omissions: whether rewards were omitted in each trial
    
    notes:
    -----
    - each trial's data is reset and processed sequentially.
    - file format must adhere to specific `$TR`, `$NT`-type headers.
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
    
    trial_statements = []
    reward_omissions = []
    
    while line[0].find('$') == 0:
        if line[0] == '$TR': # need to update motor_times here - but ignore motors before first trial started. 
            # motor_times.append(mt_trial)
            # mt_trial = []
            trial_statements.append(line)
        if line[0] == '$MV':
            mv_trial.append([float(line[1]), float(line[2])])
        if line[0] == '$WE':
            wt_trial.append([float(line[1]), float(line[2])*.04*50, float(line[3])])  # 2nd value in each line is the number of clicks per 20 ms, and each click corresponds to .04 cm, Dinghao, 20240625
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
    curr_logfile['pulse_descriptions'] = pulse_command_list
    curr_logfile['reward_omissions'] = reward_omissions  # jingyu, 8/14/2024
    return curr_logfile

def process_txt_immobile(txtfile): # Jingyu 4/17/2025
    '''
    parses a behaviour text file and extracts trial-related data for immobile experiments.

    parameters:
    - txtfile: path to the behaviour text file to process.
    
    returns:
    - dict: a dictionary containing trial-wise data for:
        - 'movie_times': timestamps and events from movie sequences.
        - 'lick_times': lick event timestamps.
        - 'pump_times': reward delivery timestamps.
        - 'trial_statements': trial-specific metadata and protocols.
    
    notes:
    - each trial's data is reset and processed sequentially.
    - file format must adhere to specific `$TR`, `$NT`-type headers.
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
    line = file.readline().rstrip('\n').split(',')
    if len(line) == 1: # read an empty line
        line = file.readline().rstrip('\n').split(',')
    return line

def correct_overflow(data, label):
    '''
    adjusts timestamps or trial events to correct for hardware overflow.

    parameters:
    ----------
    data : list
        list of trial-related timestamps or events to correct.
    label : str
        the label indicating the type of data (e.g., 'speed', 'lick', 'movie').

    returns:
    -------
    list
        a list of data corrected for overflow, preserving trial structure.

    notes:
    -----
    - overflow is assumed to occur when timestamps reset unexpectedly.
    - uses `of_constant` to adjust for the overflow period.
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
        
    return new_data