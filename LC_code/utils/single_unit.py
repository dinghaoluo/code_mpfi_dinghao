# -*- coding: utf-8 -*-
"""
basic functions for single-unit analysis
Created on Tue Dec 20 16:02:38 2022

@author: Dinghao Luo
"""

import numpy as np
import matplotlib.pyplot as plt
import time as tm


def spike_rate(spike_train_array):
    '''
    calculate spike rate of a neurone rounded to 2 prec.
    
    takes: single neurone spike array (filtered), trialx x bins
    returns: single float64
    '''
    trial_means = [np.mean(trial) for trial in spike_train_array]
    return np.round(np.mean(trial_means), 2)


def neu_shuffle(norm_spike_array, num_shufs, wind_width): 
    '''
    sub-function of neu_peak_detection()
    shuffle neuronal activity within the burst window
    
    takes: single neurone spike array (filtered), trials x bins
           number of shuffles 
           burst detection window width
    '''
    burst_win = wind_width*2*1250
    
    trials = norm_spike_array.shape[0]
    spike_shuf_array = np.zeros([num_shufs, burst_win])
    for i in range(num_shufs):
        shuf_array = np.zeros([trials, burst_win])
        for j in range(trials):
            rand_shift = np.random.randint(1, burst_win+1)
            shift_tmp = np.roll(norm_spike_array[j][:burst_win], -rand_shift)
            shuf_array[j,:] = shift_tmp
        spike_shuf_array[i,:] = np.mean(shuf_array, axis=0)
    avg_shuf = np.mean(spike_shuf_array, axis=0)
    sig_shuf = np.percentile(spike_shuf_array, .05, axis=0)
    
    # return the shuf_array, averaged trial (limited by burst_win) and signif.
    return spike_shuf_array, avg_shuf, sig_shuf



def neu_shuffle_avg(norm_spike_avg, num_shufs, wind_width): 
    '''
    sub-function of neu_peak_detection()
    shuffle neuronal activity within the burst window
    
    takes: single neurone spike array (filtered), trials x bins
           number of shuffles 
           burst detection window width
    '''
    burst_win = int(wind_width*2*1250)
    
    spike_shuf_array = np.zeros([num_shufs, burst_win])
    for i in range(num_shufs):
        rand_shift = np.random.randint(1, burst_win+1)
        shift_tmp = np.roll(norm_spike_avg[:burst_win], -rand_shift)
        spike_shuf_array[i,:] = shift_tmp
    avg_shuf = np.mean(spike_shuf_array, axis=0)
    sig_shuf = np.percentile(spike_shuf_array, .05, axis=0)
    
    # return the shuf_array, averaged trial (limited by burst_win) and signif.
    return spike_shuf_array, avg_shuf, sig_shuf


def neu_peak_detection(norm_spike_avg,
                       wind_width=1, burst_width=.35, num_shufs=100):  
    '''
    shuffle single neuron spikes within 3 seconds of run-onset of each trial
    subtract shuffled rate from real rate to identify run-onset bursts
    window is determined based on real data
    
    takes: single neurone spike avg (filtered), bins 
           neurone number
           burst detection window width 
           burst width 
           number of shuffles
    returns: bool neu_peak, avg spike rate and avg shuf rate around run-onset
    '''
    burst_win = int(wind_width*1250)
    avg_profile = norm_spike_avg[3750-burst_win:3750+burst_win]
    shuf, avg_shuf, sig_shuf = neu_shuffle_avg(avg_profile, 
                                               num_shufs, wind_width)
    win_min = burst_win-(0.5*1250); win_max = burst_win+(0.5*1250)
    burst_crit = burst_width*1250  # how long the burst has to be 
    
    diff_avg_shuf = avg_profile - avg_shuf
    diff_burst_win = diff_avg_shuf[int(win_min):int(win_max)]
    all_diff = diff_burst_win[diff_burst_win>0]
    if len(all_diff) > burst_crit:
        neu_peak = True
    else:
        neu_peak = False
    return neu_peak


def neu_peak_detection_plot(norm_spike_avg,
                            wind_width=1, burst_width=.35, num_shufs=1000):  
    '''
    shuffle single neuron spikes within 3 seconds of run-onset of each trial
    subtract shuffled rate from real rate to identify run-onset bursts
    window is determined based on real data
    
    takes: single neurone spike avg (filtered), bins 
           neurone number
           burst detection window width 
           burst width 
           number of shuffles
    returns: bool neu_peak, avg spike rate and avg shuf rate around run-onset
    '''
    burst_win = int(wind_width*1250)
    avg_profile = norm_spike_avg[3750-burst_win:3750+burst_win]
    shuf, avg_shuf, sig_shuf = neu_shuffle_avg(avg_profile, 
                                               num_shufs, wind_width)
    win_min = burst_win-(0.5*1250); win_max = burst_win+(0.5*1250)
    burst_crit = burst_width*1250  # how long the burst has to be 
    
    diff_avg_shuf = avg_profile - avg_shuf
    diff_burst_win = diff_avg_shuf[int(win_min):int(win_max)]
    all_diff = diff_burst_win[diff_burst_win>0]
    
    fig, ax = plt.subplots()
    taxis = np.linspace(-1, 1, 2500)
    data, = ax.plot(taxis, avg_profile*1250, color='coral')
    shuffle, = ax.plot(taxis, avg_shuf*1250, color='grey')
    ax.fill_between(taxis, 0, avg_shuf*1250, color='grey', alpha=.2)
    # shuf_mean = [np.mean(avg_shuf*1250)] * 2500
    # shuffle, = ax.plot(taxis, shuf_mean, color='grey')
    # ax.fill_between(taxis, 0, shuf_mean, color='grey', alpha=.2)
    if len(all_diff) > burst_crit:
        ax.set(title='data vs shuffled (burst)')
    else:
        ax.set(title='data vs shuffled (no burst)')
    ax.set(xlabel='time (s)',
           ylabel='spike rate (Hz)',
           xlim=(-1,1),
           ylim=(.5, 2.25))
    ax.legend([data, shuffle], ['data', 'mean shuffled'])
    
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_peak_det_eg.png')