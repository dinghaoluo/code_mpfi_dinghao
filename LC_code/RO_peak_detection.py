# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:39:04 2023

run-onset peak detection

@author: Dinghao Luo
"""


#%% imports 
import sys
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial' 

if ('Z:\Dinghao\code_dinghao\LC_code' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_code')
import paramC


#%% functions
def neu_shuffle(spikeArr, alpha, peak_width=1, num_shuf=5000):
    """
    Parameters
    ----------
    spikeArr : numpy array
        Average spike profile (pre-convolution) of a single cell.
    alpha : float     
        Significance threshold.
    peak_width : int, OPTIONAL
        Expected width of peaks at the run-onset. Set to 1 s by default.
    num_shuf : int, OPTIONAL
        Number of shuffles to do.

    Returns
    -------
    list
        DESCRIPTION.

    """
    sig_perc = (1-alpha)*100  # significance for percentile
    perimeter = 6  # 6 s around run-onset for shuffling
    
    # tot_trials = spikeArr.shape[0]
    shuf_mean_array = np.zeros([num_shuf, 1250*perimeter])
    
    # for i in range(num_shuf):
    #     # for every shuffle, shuffle every single trial
    #     shuf_array = np.zeros([tot_trials, 1250*perimeter])
    #     rand_shift = np.random.randint(1, 1250*perimeter, tot_trials)
    #     for j in range(tot_trials):
    #         shuf_array[j,:] = np.roll(spikeArr[j][:1250*perimeter], -rand_shift[j])
    #     shuf_mean_array[i,:] = np.mean(shuf_array, axis=0)
    
    for i in range(num_shuf):
        rand_shift = np.random.randint(1, 1250*perimeter)
        shuf_mean_array[i,:] = np.roll(spikeArr[:1250*perimeter], -rand_shift)
    
    return [np.mean(shuf_mean_array, axis=0), 
            np.percentile(shuf_mean_array, sig_perc, axis=0, interpolation='midpoint'),
            shuf_mean_array]
           # i.e. avg_shuf and sig_shuf, packaged into a list
 

# this shuffles EVERY SINGLE TRIAL 
def neu_shuffle_single(spikeArr, alpha, peak_width=1, num_shuf=1000):
    """
    Parameters
    ----------
    spikeArr : numpy array
        Smoothed spike profile of a single cell.
    alpha : float     
        Significance threshold.
    peak_width : int, OPTIONAL
        Expected width of peaks at the run-onset. Set to 1 s by default.
    num_shuf : int, OPTIONAL
        Number of shuffles to do.

    Returns
    -------
    list
        DESCRIPTION.

    """
    sig_perc = (1-alpha)*100  # significance for percentile
    tot_trials = spikeArr.shape[0]
    perimeter = 6  # 6 s around run-onset for shuffling
    
    # tot_trials = spikeArr.shape[0]
    shuf_mean_array = np.zeros([num_shuf, 1250*perimeter])
    
    for i in range(num_shuf):
        # for every shuffle, shuffle every single trial
        shuf_array = np.zeros([tot_trials, 1250*perimeter])
        rand_shift = np.random.randint(1, 1250*perimeter, tot_trials)
        for j in range(tot_trials):
            shuf_array[j,:] = np.roll(spikeArr[j][:1250*perimeter], -rand_shift[j])
        shuf_mean_array[i,:] = np.mean(shuf_array, axis=0)
    
    return [np.mean(shuf_mean_array, axis=0), 
            np.percentile(shuf_mean_array, sig_perc, axis=0, interpolation='midpoint'),
            shuf_mean_array]
           # i.e. avg_shuf and sig_shuf, packaged into a list
        

def RO_peak_detection(spikeArr, first_stim=-1, peak_width=1, min_peak=.2, alpha=.001):
    """
    Parameters
    ----------
    spikeArr : numpy array, trial x time bins
        Raw spike array (raster) of a single cell.
    first_stim : int
        Index of first stim trial.
    peak_width : int, OPTIONAL
        Expected width of peaks at the run-onset. Set to 1 s by default.
    min_peak : float, OPTIONAL
        Expected minimum length of an RO peak.
    alpha : float, OPTIONAL     
        Significance threshold.
    
    Returns
    -------
    list : 
        0: a Boolean value indicative of peakness.
        1: average spiking profile around run-onset.
        2: significance threshold for spiking profile peaks.
    """
    
    # METHOD 1: single trial shuffle
    tot_trials = spikeArr.shape[0]
    trial_length = spikeArr[0].shape[0]
    conv_profile = np.zeros((tot_trials, trial_length))
    for trial in range(tot_trials):
        conv_profile[trial,:] = np.convolve(spikeArr[trial], paramC.gaus_spike, mode='same')
    [avg_shuf, sig_shuf, shuf_mean] = neu_shuffle_single(conv_profile, alpha)
    
    avg_profile = np.mean(spikeArr[:first_stim], axis=0)  # only baseline trials
    avg_profile = np.convolve(avg_profile, paramC.gaus_spike, mode='same')
    
    
    # METHOD 2: average and then shuffle 
    # avg_profile = np.mean(spikeArr[:first_stim], axis=0)  # only baseline trials
    # avg_profile = np.convolve(avg_profile, paramC.gaus_spike, mode='same')
    
    # [avg_shuf, sig_shuf, shuf_mean] = neu_shuffle(avg_profile, alpha)


    # comparison
    peak_window = [int(3750-1250*(peak_width/2)),
                   int(3750+1250*(peak_width/2))]
    
    # sig_shuf = np.convolve(sig_shuf, paramC.gaus_spike, mode='same')
    sig_shuf = sig_shuf[peak_window[0]:peak_window[1]] * 1250
    avg_shuf = avg_shuf[peak_window[0]:peak_window[1]] * 1250
    
    avg_profile = avg_profile[peak_window[0]:peak_window[1]] * 1250
    
    diff_avg_shuf = avg_profile - sig_shuf
    ind_diff = [diff>0 for diff in diff_avg_shuf]

    # detect consecutive ones 
    pre_groups = groupby(ind_diff, lambda x: x)
    tot_groups = len(list(pre_groups))
    
    groups = groupby(ind_diff, lambda x: x)
    max_trues = 0
    group_count = 0
    for key, group in groups:
        consecutive_true = sum(list(group))
        if group_count!=0 and group_count!=tot_groups-1 and consecutive_true>max_trues:
            max_trues = consecutive_true
        group_count+=1
    
    return [max_trues>int(min_peak*1250),
            avg_profile,
            sig_shuf]


def plot_RO_peak(cluname, avg_profile, sig_shuf):
    print('plotting {}...'.format(cluname))
    
    fig, ax = plt.subplots()
    
    maxpt = max(max(avg_profile), max(sig_shuf))
    minpt = min(min(avg_profile), min(sig_shuf))
    ax.set(title=cluname,
           xlim=(-.5, .5),
           ylim=(minpt*.9, maxpt*1.2),
           xlabel='time (s)',
           ylabel='spike rate (Hz)')
    
    xaxis = np.arange(-625, 625)/1250
    avg, = ax.plot(xaxis, avg_profile)
    sigshuf, = ax.plot(xaxis, sig_shuf, color='grey')
    
    ax.legend([avg, sigshuf],
              ['avg.', 'sig. shuffle'])
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_RO_peak\{}.png'.format(cluname),
                dpi=300,
                bbox_inches='tight',
                transparent=False)
    
    plt.close(fig)


#%% main calls for testing 
# clukeys = list(rasters.keys())

# peaks = []
# for cluname in clukeys[2:4]:
#     print(cluname)
#     peaks.append(RO_peak_detection(rasters[cluname]))