# -*- coding: utf-8 -*-
"""
functions for analysing recording sessions
Created on Tue Dec 20 16:22:43 2022

@author: Dinghao Luo
"""

import sys
import scipy.io as sio
import numpy as np
import h5py
import time as tm

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


def prep_speed_and_spike(pathname):
    '''
    load speed and spike files and process them into:
        spike train 
        spike train convolved with gaussian (stored in final dict)
        speed
        speed convolved with a different gaussian (stored as well)
        norm_speed
        norm_spike
        etc.
    
    takes: pathname (without file name per se)
    returns: NA
    '''
    print(pathname[-17:])
    print('compiling speed and spike info...')
    
    start_time = tm.time()
    
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    
    speed_time_file = sio.loadmat(filename + '_alignRun_msess1.mat')

    spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']

    time_bef = spike_time_file['TimeBef']; time = spike_time_file['Time']
    tot_clu = time.shape[1]
    tot_trial = time.shape[0]  # trial 1 is empty but tot_trial includes it for now
    
    samp_freq = 1250  # Hz
    gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
    sigma_speed = 12.5
    gx_spike = np.arange(-500, 500, 1)
    sigma_spike = 125

    # speed of all trials
    speed_time_bef = speed_time_file['trialsRun'][0]['speed_MMsecBef'][0][0][1:]
    speed_time = speed_time_file['trialsRun'][0]['speed_MMsec'][0][0][1:]
    gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
                  np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]
    # concatenate bef and after running onset, and convolve with gaus_speed
    speed_time_all = np.empty(shape=speed_time.shape[0], dtype='object')
    for i in range(speed_time.shape[0]):
        bef = speed_time_bef[i]; aft =speed_time[i]
        speed_time_all[i] = np.concatenate([bef, aft])
        speed_time_all[i][speed_time_all[i]<0] = 0
    speed_time_conv = [np.convolve(np.squeeze(single), gaus_speed)[50:-49] 
                       for single in speed_time_all]
    norm_speed = [normalise(s) for s in speed_time_conv]

    # trial length for equal length deployment (use speed trial length)
    trial_length = [trial.shape[0] for trial in speed_time_conv]

    # spike reading
    spike_time = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    spike_time_bef = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    for i in range(tot_clu):
        for j in range(1, tot_trial):  # trial 0 is an empty trial
            spike_time[i,j-1] = spike_time_file[time[j,i]][0]
            spike_time_bef[i,j-1] = spike_time_file[time_bef[j,i]][0]
    spike_train_all = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    spike_train_conv = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    norm_spike = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    gaus_spike = [1 / (sigma_spike*np.sqrt(2*np.pi)) * 
                  np.exp(-x**2/(2*sigma_spike**2)) for x in gx_spike]
    for clu in range(tot_clu):
        for trial in range(tot_trial-1):
            spikes = np.concatenate([spike_time_bef[clu][trial].reshape(-1),
                                     spike_time[clu][trial].reshape(-1)])
            spikes = [int(s+3750) for s in spikes]
            spike_train_trial = np.zeros(trial_length[trial])
            spike_train_trial[spikes] = 1
            spike_train_all[clu][trial] = spike_train_trial
            spike_train_conv[clu][trial] = np.convolve(spike_train_trial, 
                                                       gaus_spike, mode='same')
            norm_spike[clu][trial] = normalise(spike_train_conv[clu][trial])
    
    dict_save = {'norm_spike': norm_spike,
                 'norm_speed': norm_speed,
                 'gaus_spike': gaus_spike,
                 'sigma_gaus_spike': sigma_spike,
                 'gaus_speed': gaus_speed,
                 'sigma_gaus_speed': sigma_speed,
                 'samp_freq': samp_freq,
                 'speed_time_conv': speed_time_conv,
                 'spike_train_conv': spike_train_conv,
                 'speed_time_all': speed_time_all,
                 'spike_train_all': spike_train_all,
                 'tot_trial': tot_trial,
                 'tot_clu': tot_clu,
                 'trial_length': trial_length}
    
    np.save(r'Z:\Dinghao\code_dinghao\all_session_prep'+pathname[-18:]+'_prep_dict',
            dict_save)
    
    print('compiling complete, saved as {} (execution time: {}s)\n'.
          format(pathname[-17:]+'_prep_dict.npy', 
                 np.round(tm.time()-start_time, 2)))