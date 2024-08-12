# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:13:00 2024

functions for processing behaviour 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import pandas as pd
import math

def find_nearest(value, arr):
    # return value and index of nearest value in arr to input value
    nearest_value = min(arr, key=lambda x: abs(x-value))
    nearest_value_index = arr.index(nearest_value)
    return nearest_value_index


#%% main 
def get_lick_locs(speeds, licks):
    """
    Parameters
    ----------
    speeds : array
    licks : array

    Returns
    -------
    lick_locs : list of lists
        where each lick is in terms of distance, segmented into trials.
    """
    tot_trial = len(speeds)
    
    lick_locs = []
    for t in range(tot_trial):
        curr_speeds = speeds[t]
        temp_locs = np.array([point[2] for point in curr_speeds])
        start = np.argmax(temp_locs<2500)  # find the first instance of true start (distance reset)
        
        time_p = [point[0] for point in speeds[t][start:]]
        speed_p = [point[1] for point in speeds[t][start:]]
        loc_p = [point[2] for point in speeds[t][start:]]
        
        # interpolation 
        uni_times = list(np.linspace(time_p[0], time_p[-1], int((time_p[-1]-time_p[0]))))
        uni_speeds = np.interp(uni_times, time_p, speed_p)
        uni_locs = np.interp(uni_times, time_p, loc_p)/25  # convert to cm
        
        curr_licks = licks[t]
        l_locs = []  # locations of the licks 
        for l in curr_licks:
            ind = find_nearest(l[0], uni_times)
            l_locs.append(uni_locs[ind])
    
        lick_locs.append(l_locs)

    return lick_locs


#%% functions 
def lick_index(lick_lists):
    """
    Parameters
    ----------
    lick_lists : list of lists for lick locations 
        lick distances in a session 

    Returns
    -------
    LI : float 
        lick-selectivity index
    """
    tot_trial = len(lick_lists)
    lick_index = []
    for t in range(tot_trial):
        curr_lick_locs = lick_lists[t]
        if len(curr_lick_locs)>0:  # only execute if there are licks in the current trial
            sum_pre = sum(i<120 for i in curr_lick_locs)
            sum_post = sum(i>=120 for i in curr_lick_locs)
            lick_index.append(sum_post/(sum_pre+sum_post))
    
    return np.nanmean(lick_index, axis=0)