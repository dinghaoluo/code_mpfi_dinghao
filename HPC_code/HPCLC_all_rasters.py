# -*- coding: utf-8 -*-
"""
Created on Wed Aug 2 16:17:23 2023

save all HPC LC rasters into .npy files (one file for one session)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import scipy.io as sio
import h5py
import mat73
import sys

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% create stim rasters 
for pathname in pathHPC[26:]:
    curr_rasters = {}; curr_rasters_simp = {}  # reset dictionaries
    
    print(pathname[-17:])
    
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    BehavLFP = mat73.loadmat('{}.mat'.format(pathname+pathname[-18:]+'_BehavElectrDataLFP'))
    Clu = BehavLFP['Clu']
    shank = Clu['shank']
    localClu = Clu['localClu']
    
    # check if this is an opto-stim session
    beh_info_file = sio.loadmat(filename+'_Info.mat')['beh']
    stim_trial = np.squeeze(np.where(beh_info_file['pulseMethod'][0][0][0]!=0))
    if stim_trial.size!=0:
        stimtype = beh_info_file['pulseMethod'][0][0][0][stim_trial[0]]
        stimwind = [stim_trial[0], stim_trial[-1]]
    else: 
        stimtype = 'NA'
        stimwind = ['NA', 'NA']
        
    spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run1.mat')['trialsRunSpikes']
    
    time_bef = spike_time_file['TimeBef']; time = spike_time_file['Time']
    tot_clu = time.shape[1]
    tot_trial = time.shape[0]  # trial 1 is empty but tot_trial includes it for now
    all_id = list(np.arange(2, tot_clu+2))
    
    # spike reading
    spike_time = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    spike_time_bef = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    for i in range(tot_clu):
        i = int(i)-2
        for j in range(1, tot_trial):  # trial 0 is an empty trial
            spike_time[i,j-1] = spike_time_file[time[j,i]][0]
            spike_time_bef[i,j-1] = spike_time_file[time_bef[j,i]][0]
    
    max_length = 16250  # 3 s before, 10 s after
    spike_train_all = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    for i in range(tot_clu):
        for trial in range(tot_trial-1):
            clu_id = int(all_id[i])-2
            spikes = np.concatenate([spike_time_bef[clu_id][trial].reshape(-1),
                                     spike_time[clu_id][trial].reshape(-1)])
            spikes = [int(s+3750) for s in spikes if s<-1 or s>1]
            spikes = [s for s in spikes if s<max_length]
            spike_train_trial = np.zeros(max_length)
            spike_train_trial[spikes] = 1
            spike_train_all[i][trial] = spike_train_trial
    
    # save into all_rasters
    i = 0
    for clu in all_id:
        cluname = '{} clu{} {} {} {} {} {}'.format(pathname[-17:], clu, int(shank[clu-2]), int(localClu[clu-2]), int(stimtype), int(stimwind[0]), int(stimwind[1]))
        simp_cluname = '{} clu{} {} {}'.format(pathname[-17:], clu, int(shank[clu-2]), int(localClu[clu-2]))
        curr_rasters[cluname] = spike_train_all[i]
        curr_rasters_simp[simp_cluname] = spike_train_all[i]
        i+=1
        
    np.save('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy\{}.npy'.format(pathname[-17:]), 
            curr_rasters)
    np.save('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(pathname[-17:]), 
            curr_rasters_simp)


#%%
# PLEASE DON'T PLOT THIS THING. IT IS TOO BIG TO PLOT AND THERE'S NO POINT IN 
# DOING SO ANYWAYS. BE STUBBORN AT YOUR OWN RISK.