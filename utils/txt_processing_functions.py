# -*- coding: utf-8 -*-
"""
Created on Fri 12 July 17:38:13 2024

functions for processing behaviour log .txt files

@author: Dinghao Luo
"""


#%% constants 
of_constant = (2**32-1)/1000  # overflow constant 


#%% behaviour file processing
def process_txt(txtfile):
    curr_logfile = {} 
    file = open(txtfile, 'r')
    
    line = ['']
    while line[0] != '$TR':
        line = get_next_line(file)
        
    lick_times = []
    pump_times = []
    movie_times = []
    speed_times = []
    motor_times = []
    pulse_times = []
    frame_times = []
    
    mt_trial = []
    wt_trial = []
    lt_trial = []
    pt_trial = []
    mv_trial = []
    pc_trial = []
    pulse_command_list = []
    current_pulse_command = []
    
    trial_statements = []
    
    while line[0].find('$') == 0:
        if line[0] == '$TR' and len(speed_times)>0: # need to update motor_times here - but ignore motors before first trial started. 
            motor_times.append(mt_trial)
            mt_trial = []
            trial_statements.append(line)
        if line[0] == '$MV':
            mv_trial.append([float(line[1]), float(line[2])])
        if line[0] == '$WE':
            wt_trial.append([float(line[1]), float(line[2])*.04*50, float(line[3])])  # 2nd value in each line is the number of clicks per 20 ms, and each click corresponds to .04 cm, Dinghao, 20240625
            # wt_trial.append([float(line[1]), float(line[2])*.04*50, float(line[3])])  # old way of doing this, reading in the 2nd value directly, but it is not the true speed
        if line[0] == '$LE' and line[3] == '1':
            lt_trial.append([float(line[1]), float(line[2])]) 
        if line[0] == '$PE' and line[3] == '1':
            pt_trial.append(float(line[1]))
        if line[0] == '$MT':
            mt_trial.append([float(line[1]), float(line[2])])
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
    curr_logfile['motor_times'] = motor_times
    curr_logfile['pulse_times'] = pulse_times
    curr_logfile['frame_times'] = frame_times
    curr_logfile['trial_statements'] = trial_statements
    curr_logfile['pulse_descriptions'] = pulse_command_list
    
    return curr_logfile
    

def get_next_line(file):
    line = file.readline().rstrip('\n').split(',')
    if len(line) == 1: # read an empty line
        line = file.readline().rstrip('\n').split(',')
    return line


def correct_overflow(data, label):
    """
    Parameters
    ----------
    data : list
        speed_times, pump_times, frame_times, movie_times etc.
    label : str
        the label of the data array (eg. 'speed').

    Returns
    -------
    new_data : list
        data corrected for overflow.
    """
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
            first_trial_with_pulse = next(x for x in data if len(x)!=0)  # in case the first trial has no licks, Dinghao, 20240626
            curr_time = first_trial_with_pulse[0][0]
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
        except StopIteration:  # if no pulses in this session 
            new_data = data
    if label=='pump':
        first_trial_with_pump = next(x for x in data if len(x)!=0)  # in case the first trial has no pump, Dinghao, 20240704
        curr_time = first_trial_with_pump[0]
        for t in range(tot_trial):
            if len(data[t])==0: # if there is no reward, append 0, Jingyu, 20270730
                new_data.append(0)
            elif len(data[t])!=0:
                if data[t][0]-curr_time>=0:
                    new_data.append(data[t][0])
                    curr_time = data[t][0]
                else:  # once overflow is detected, do not update curr_time
                    new_data.append(data[t][0]+of_constant)
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
                    if curr_trial[s][0]<of_constant:
                        new_trial.append(curr_trial[s])
                    else:
                        new_trial.append([curr_trial[s][0]+of_constant, curr_trial[s][1]])
                new_data.append(new_trial)
    if label=='frame':
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


def get_onset(uni_speeds, uni_times, threshold=0.3):  # 0.3 seconds
    count = 0
    for i in range(len(uni_speeds)):
        count = fast_in_a_row(uni_speeds[i], count, 10)
        if count>threshold*1000:
            index = uni_times[i]-threshold*1000
            break
    if count<threshold*1000:
        index = -1  # cannot find clear run-onsets
    return index

def fast_in_a_row(speed_value, count, threshold):
    if speed_value > threshold:
        count-=-1
    else:
        count=0
    return count

def find_nearest(value, arr):
    # return value and index of nearest value in arr to input value
    nearest_value = min(arr, key=lambda x: abs(x-value))
    nearest_value_index = arr.index(nearest_value)
    return nearest_value_index