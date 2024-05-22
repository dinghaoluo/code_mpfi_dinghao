# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:50:53 2024

functions for grid-ROI processing

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 


#%% main 
def sum_mat(matrix):
    return sum(map(sum, matrix))

def make_grid(stride=8, dim=512):
    """
    Parameters
    ----------
    stride : int, default=8
        how many pixels per grid.
    dim : int, default=512
        x/y dimension; either should do since we are imaging squared images.

    Returns
    -------
    a list of grid points.
    """
    return list(np.arange(0, dim, stride))

def run_grid(frame, grids, tot_grid, stride=8):
    """
    Parameters
    ----------
    frame : array
        current frame as an array (default dim.=512x512).
    grid_list : list 
        a list of grid points.
    tot_grid : int
        total number of grids.
    stride : int, default=8
        how many pixels per grid.

    Returns
    -------
    gridded : array
        3-dimensional array at tot_grid x stride x stride.
    """
    gridded = np.zeros((tot_grid, stride, stride))
    
    grid_count = 0
    for hgp in grids:
        for vgp in grids:
            gridded[grid_count,:,:] = frame[hgp:hgp+stride, vgp:vgp+stride]
            grid_count+=1
            
    return gridded

def plot_reference(mov, grids, stride):
    # plot the mean image (Z-projection)
    ref_im = np.mean(mov, axis=0)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(ref_im, aspect='auto', cmap='gist_gray', interpolation='none',
              extent=[0, 32, 32, 0])
    for g in grids:
        ax.plot([0,32], [g/stride,g/stride], color='white', linewidth=.2, alpha=.2)
        ax.plot([g/stride,g/stride], [0,32], color='white', linewidth=.2, alpha=.2)
    ax.set(xlim=(0,32), ylim=(0,32))
    plt.show()
    
def find_nearest(value, arr):
    # return value and index of nearest value in arr to input value
    nearest_value = min(arr, key=lambda x: abs(x-value))
    nearest_value_index = arr.index(nearest_value)
    return nearest_value_index
    
    
#%% text file processing
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
            wt_trial.append([float(line[1]), float(line[2]), float(line[3])])
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

def correct_overflow(speed_times, lick_times = [], movie_times = [], pump_times = [], trial_times = []):
    times = np.array([s[0] for s in speed_times])
    index_before_overflow = np.where(np.diff(times) < 0)[0]
    
    new_speeds = []
    new_trial_times = []
    if len(index_before_overflow) == 0: # no overflow during this trial
        return [speed_times, lick_times, movie_times,pump_times, trial_times]
    else:
        time_before_overflow = times[index_before_overflow[0]]
        # print('overflow detected')
    
    for i in range(len(speed_times)):
        entry = speed_times[i]      
        if entry[0] < times[0]:
            entry[0] = entry[0] + time_before_overflow
        new_speeds.append(entry)
            
    for j in range(len(lick_times)):
        if lick_times[j] < times[0]:
            lick_times[j] = lick_times[j] + time_before_overflow
    for k in range(len(movie_times)):
        if movie_times[k][0] < times[0]:
            movie_times[k][0] = movie_times[k][0] + time_before_overflow
    for l in range(len(pump_times)):
        if pump_times[l] < times[0]:
            pump_times[l] = pump_times[l] + time_before_overflow
    for m in range(len(trial_times)):
        time = trial_times[m]
        if time < trial_times[0]:
            time += time_before_overflow
        new_trial_times.append(time)
    
    return new_speeds

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