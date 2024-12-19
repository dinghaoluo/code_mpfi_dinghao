# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:19:22 2024

functions for alignment testing 

@author: Dinghao Luo
"""


#%% definitions
def cir_shuf(conv_aligned_spike_arr, 
             length=6*1250, 
             GPU_AVAILABLE=False):
    """
    Parameters
    ----------
    conv_aligned_spike_array : numpy array
        smoothed spike array of one clu in this sessios, aligned to 1st-licks.
    length : int, optional
        the number of time bins in the input array per row.

    Returns
    -------
        the flattened array containing every trial in this session each shuffled once.
    """
    if GPU_AVAILABLE: 
        import cupy as xp
    else:
        import numpy as xp
    
    tot_trial = conv_aligned_spike_arr.shape[0]
    trial_shuf_array = xp.zeros([tot_trial, length])
    for trial in range(tot_trial):
        rand_shift = xp.random.randint(1, length/2)
        trial_shuf_array[trial,:] = xp.roll(conv_aligned_spike_arr[trial], -rand_shift)
    return xp.mean(trial_shuf_array, axis=0)

def bootstrap_ratio(spike_arr, 
                    bootstrap=500, 
                    samp_freq=1250,
                    length=6*1250, 
                    GPU_AVAILABLE=False):
    """
    Parameters
    ----------
    spike_arr : numpy array
        smoothed spike array of one clu in this sessios, aligned to 1st-licks.
    bootstrap : int, optional
        the number of times we want to run the bootstrapping. The default is 500.
    length : int, optional
        the number of time bins in the input array per row.

    Returns
    -------
        the percentage thresholds for the bootstrapping result.
    """    
    from tqdm import tqdm
    if GPU_AVAILABLE:
        import cupy as xp
        device = 'GPU'
    else:
        import numpy as xp
        device = 'CPU'
    
    tot_trials = spike_arr.shape[0]
    shuf_ratio = xp.zeros(bootstrap)
    
    # pre-compute random shifts
    indices = xp.arange(length)

    # shuffle in parallel 
    for shuf in tqdm(range(bootstrap), desc=f'lick sensitivity ({device})'):
        rand_shifts = xp.random.randint(1, length, tot_trials)
        shifted_indices = (indices[None, :] - rand_shifts[:, None]) % length
        shuf_arr = spike_arr[xp.arange(tot_trials)[:,None], shifted_indices]
        shuf_result = xp.mean(shuf_arr, axis=0)
        
        shuf_ratio[shuf] = xp.sum(shuf_result[length//2:length//2+samp_freq])/xp.sum(shuf_result[length//2-samp_freq:length//2])
        
    return xp.percentile(shuf_ratio, [99.9, 99, 95, 50, 5, 1, .1], axis=0)