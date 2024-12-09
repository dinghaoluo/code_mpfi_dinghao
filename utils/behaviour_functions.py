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
def process_behavioural_data(txtfile: str, 
                             max_distance=220, 
                             distance_resolution=1, 
                             run_onset_initial=3.0, 
                             run_onset_sustained=10.0, 
                             run_onset_duration=300) -> dict:
    '''
    processes behavioural data from a txt file, aligning speed, lick, and reward events 
    to both time and distance bases, while extracting metrics such as run onsets, 
    lick selectivity, trial quality, and full stop status.

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
        - 'speed_distance': list of arrays, each containing speeds aligned to a common distance base.
        - 'lick_times': list of lists, each containing lick event timestamps for each trial.
        - 'lick_distance': list of arrays, each containing lick events aligned to the distance base.
        - 'start_cue_times': list of lists, each containing timestamps of start cues.
        - 'reward_times': list of lists, each containing timestamps of reward deliveries.
        - 'reward_distance': list of arrays, each containing reward events aligned to the distance base.
        - 'run_onsets': list of timestamps for detected run-onset events in each trial.
        - 'lick_selectivities': list of float values, one for each trial, representing lick selectivity indices.
        - 'trial_statements': list of lists containing trial-specific metadata and protocols.
        - 'full_stops': list of booleans indicating whether the animal fully stopped (speed < 10 cm/s) 
          between the previous trial's reward and the current trial's reward.
        - 'trial_quality': list of strings ('good' or 'bad') classifying each trial based on the following criteria:
            - 'bad' trials meet at least one of the following:
                1. licks occur between 30 cm and 90 cm.
                2. speed drops below 10 cm/s for a total duration exceeding 5 seconds between run-onset and reward.
                3. no full stop is observed before the run-onset.
                4. no reward is delivered in the trial.
            - 'good' trials meet none of these criteria.
    '''
    # load and parse the txt file
    data = process_txt(txtfile)  # uses user's custom `process_txt` function

    # correct for overflow in the data
    data['speed_times'] = correct_overflow(data['speed_times'], 'speed')
    data['lick_times'] = correct_overflow(data['lick_times'], 'lick')
    data['pump_times'] = correct_overflow(data['pump_times'], 'pump')
    data['movie_times'] = correct_overflow(data['movie_times'], 'movie')
    data['trial_statements'] = correct_overflow(data['trial_statements'], 'trial_statement')
    frame_times = correct_overflow(data['frame_times'], 'frame')

    # define common time and distance bases
    common_distance_base = np.linspace(0, max_distance, int(max_distance / distance_resolution))
    
    # initialise lists for storing data across trials
    speed_times, speed_distance = [], []
    lick_times, lick_distance = [], []
    start_cue_times, reward_times, reward_distance = [], [], []
    run_onsets = []
    trial_statements = []
    lick_selectivities = []
    full_stops = []
    bad_trials = []

    # process each trial
    for trial_idx, (speed_trial, lick_trial, movie_trial, reward_trial) in enumerate(
        zip(data['speed_times'], data['lick_times'], data['movie_times'], data['pump_times'])
    ):
        # extract times and speeds
        times = [s[0] for s in speed_trial]
        speeds = [s[1] for s in speed_trial]
        formatted_speed_times = list(zip(times, speeds))
        speed_times.append(formatted_speed_times)
        
        # calculate full stop
        if trial_idx > 0:  # no full stop check for the first trial
            previous_reward_time = reward_times[trial_idx - 1][-1] if reward_times[trial_idx - 1] else None
            current_reward_time = reward_trial[-1] if reward_trial else None
            if previous_reward_time is not None and current_reward_time is not None:
                # filter speed data within the range
                time_mask = [(t > previous_reward_time and t < current_reward_time) for t in times]
                speeds_during_pause = np.array(speeds)[time_mask]
                full_stop = np.any(speeds_during_pause < 10)  # true if speed drops below 10 cm/s
            else:
                full_stop = False
        else:
            full_stop = False  # first trial can't have a full stop
        full_stops.append(full_stop)
            
        # process licks
        lick_times_trial = [event[0] for event in lick_trial]
        lick_distances_trial = np.interp(lick_times_trial, times, np.cumsum(speeds) * (np.diff(times, prepend=times[0]) / 1000))

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
        
        # bad trial metric #1: is there any lick between 30 and 90 cm?
        is_bad_trial_lick = np.any((lick_distances_trial >= 30) & (lick_distances_trial <= 90))
        
        # bad trial metrics #2, #3 and #4: 
        #     is there not a valid run-onset?
        #     did the animal stop for more than 5 seconds in the trial?
        #     did the animal not get a reward (implicit)
        if run_onset != -1 and reward_trial:
            reward_time = reward_trial[-1]
            speeds_between_onset_reward = [speed for t, speed in formatted_speed_times if run_onset <= t <= reward_time]
            total_low_speed_duration = np.sum(np.array(speeds_between_onset_reward) < 10) * (times[1] - times[0]) / 1000
            is_bad_trial_speed = total_low_speed_duration > 5  # bad if more than 5 seconds of stoptime
        else:
            is_bad_trial_speed = True  # bad if no valid run-onset or no reward
            
        # bad trial metric #4: is there not a reward?
        is_bad_trial_reward = len(reward_trial) == 0
        
        # final disjunction for bad/good trial
        bad_trials.append(is_bad_trial_lick or is_bad_trial_speed or is_bad_trial_reward)
        
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
        trial_statements.append(trial_statement)
        
        # lick selectivity
        lick_selectivities.append(lick_index(lick_distance_trial))

    # structure the result
    return {
        'speed_times': speed_times,
        'speed_distance': speed_distance,
        'lick_times': lick_times,
        'lick_distance': lick_distance,
        'start_cue_times': start_cue_times,
        'reward_times': reward_times,
        'reward_distance': reward_distance,
        'run_onsets': run_onsets,
        'lick_selectivities': lick_selectivities,
        'trial_statements': trial_statements,
        'full_stops': full_stops,
        'bad_trials': bad_trials,
        'frame_times': frame_times  # this was actually added just to prevent np.array(... type='object) from automatically producing a 2D array, 6 Dec 2024
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