# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:24:46 2024

utility functions for statistical analyses (imaging data)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 


#%% functions 
def circ_shuffle(arr, alpha=.01, num_shuf=5000):
    """
    Parameters
    ----------
    arr : array
        dFF array.
    alpha : float, optional
        significance threshold. The default is 0.01.
    num_shuf : int, optional
        how many times to shuffle. The default is 5000.

    Returns
    -------
    list of mean, alpha and 1-alpha shuf.
    """
    
    vector = False  # default to 2D array 
    
    sig_perc = (1-alpha)*100  # significance for percentile
    try:
        tot_trial, tot_time = arr.shape
    except ValueError:  # if input is 1D
        tot_time = arr.shape[0]
        vector = True
    
    shuf_mean_array = np.zeros([num_shuf, tot_time])
    
    for i in range(num_shuf):
        if vector:
            rand_shift = np.random.randint(1, tot_time)
            shuf_mean_array[i,:]+=np.roll(arr, -rand_shift)
        else:
            for t in range(tot_trial):
                rand_shift = np.random.randint(1, tot_time)
                shuf_mean_array[i,:]+=np.roll(arr[t,:], -rand_shift)
    if not vector:
        shuf_mean_array/=num_shuf

    return [np.mean(shuf_mean_array, axis=0), 
            np.percentile(shuf_mean_array, sig_perc, axis=0, method='midpoint'),
            np.percentile(shuf_mean_array, 100-sig_perc, axis=0, method='midpoint')]