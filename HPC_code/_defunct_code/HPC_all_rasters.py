# -*- coding: utf-8 -*-
"""
Created on Wed Aug 2 16:17:23 2023
Modified on 20 Dec 2024:
    - merged functionalities from similar scripts 

save all HPC rasters into .npy files (one file for one session)

@author: Dinghao Luo
"""


#%% imports 
import os 
import h5py
import mat73
import sys

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLC = rec_list.pathHPCLCopt
pathHPCLCterm = rec_list.pathHPCLCtermopt
paths = pathHPCLC + pathHPCLCterm


#%% GPU acceleration
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
except ModuleNotFoundError:
    print('CuPy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
except Exception as e:
    # catch any other unexpected errors and print a general message
    print(f'An error occurred: {e}')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    # we are assuming that there is only 1 GPU device
    xp = cp
    import numpy as np  # needed for the empty arrays... CuPy empty arrays are not the same as NumPy ones
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print(f'GPU-acceleration with {str(name)} and cupy')
else:
    import numpy as np 
    xp = np
    print('GPU-acceleartion unavailable')


#%% create stim rasters 
for pathname in paths[:1]:
    curr_rasters = {}
    
    recname = pathname[-17:]
    print(recname)
    
    filename = os.path.join(pathname, f'{pathname[-17:]}_DataStructure_mazeSection1_TrialType1')
    BehavLFP = mat73.loadmat('{}.mat'.format(
        os.path.join(pathname, f'{pathname[-17:]}_BehavElectrDataLFP')))
    Clu = BehavLFP['Clu']
    shank = Clu['shank']
    localClu = Clu['localClu']
        
    spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run1.mat')['trialsRunSpikes']
    
    time_bef = spike_time_file['TimeBef']
    time_aft = spike_time_file['Time']
    tot_clu = time_aft.shape[1]
    tot_trial = time_aft.shape[0]  # trial 1 is empty but tot_trial includes it for now
    
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