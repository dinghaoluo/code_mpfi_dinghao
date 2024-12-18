# -*- coding: utf-8 -*-
"""
Created on Mon 10 July 10:02:32 2023

pool all cells from all recording sessions
modified 11 Dec 2024 to process with all trials (not skipping trial 0) and 
    added GPU support

*contains interneurones*

@author: Dinghao Luo
"""


#%% imports 
import sys
import h5py
import mat73
import os
from tqdm import tqdm
from time import time
from datetime import timedelta

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import gaussian_kernel_unity


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


#%% main
for pathname in paths:
    all_info = {}
    
    t0 = time()
    
    recname = pathname[-17:]
    print(recname)  # recnames are as such: 'Axxxr-202xxxxx-0x'
    
    filename = os.path.join(pathname, f'{pathname[-17:]}_DataStructure_mazeSection1_TrialType1')
    BehavLFP = mat73.loadmat('{}.mat'.format(
        os.path.join(pathname, f'{pathname[-17:]}_BehavElectrDataLFP')))
    Clu = BehavLFP['Clu']
    shank = Clu['shank']
    localClu = Clu['localClu']
    
    spike_time_file = h5py.File(f'{filename}_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']
    
    time_bef = spike_time_file['TimeBef']; time_aft = spike_time_file['Time']
    tot_clu = time_aft.shape[1]
    tot_trial = time_aft.shape[0]  # trial 1 is empty but tot_trial includes it for now
    
    samp_freq = 1250  # Hz
    sigma_spike = samp_freq/10
    
    gaus_spike = gaussian_kernel_unity(sigma_spike, GPU_AVAILABLE)
        
    # spike reading
    spike_time_aft = np.empty(shape=(tot_clu, tot_trial), dtype='object')
    spike_time_bef = np.empty(shape=(tot_clu, tot_trial), dtype='object')
    for clu in tqdm(range(tot_clu), desc='reading spike trains'):
        for trial in range(1, tot_trial):
            if isinstance(spike_time_file[time_aft[trial,clu]][0], np.uint64):
                spike_time_aft[clu,trial] = []
            else:
                spike_time_aft[clu,trial] = [t for t in spike_time_file[time_aft[trial,clu]][0]]
            if isinstance(spike_time_file[time_bef[trial,clu]][0], np.uint64):
                spike_time_bef[clu,trial] = []
            else:
                spike_time_bef[clu,trial] = [t for t in spike_time_file[time_bef[trial,clu]][0]]
                
    # the convolution part is the only part where GPU acceleration is used (xp instead of np)
    conv_spike = np.empty(shape=(tot_clu, tot_trial), dtype='object')  # contain the convolved spike trains of all neurones
    for clu in tqdm(range(tot_clu), desc='convolving spike trains'):
        for trial in range(tot_trial):
            
            # pre-post concatenation logic 
            if spike_time_bef[clu][trial]!=None and spike_time_aft[clu][trial]!=None:
                spikes = np.concatenate([spike_time_bef[clu][trial],
                                         spike_time_aft[clu][trial]])
            elif spike_time_bef[clu][trial]!=None:
                spikes = np.asarray(spike_time_bef[clu][trial])
            elif spike_time_aft[clu][trial]!=None:
                spikes = np.asarray(spike_time_aft[clu][trial])
            else:
                conv_spike[clu][trial] = np.zeros(12500)  # default to 10 s of emptiness if no spikes in this trial
                continue
            
            spikes = [int(s+3750) for s in spikes]  # aligning the spikes correctly, otherwise this 'spike_train_trial[spikes] = 1' makes no sense
            
            if len(spikes)>0:
                spike_train_trial = xp.zeros(spikes[-1]+1)  # +1 to ensure inclusion of the last spike 
                spike_train_trial[spikes] = 1
                spike_train_conv = xp.convolve(spike_train_trial, gaus_spike, mode='same')
                conv_spike[clu][trial] = spike_train_conv.get() if GPU_AVAILABLE else spike_train_conv
    
    for clu in range(tot_clu):
        cluname = f'{pathname[-17:]} clu{clu+2} {int(shank[clu])} {int(localClu[clu])}'
        clu_conv_spike = conv_spike[clu]
        all_info[cluname] = clu_conv_spike
    
    print('done; saving...')
    outdirroot = r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{}'.format(recname)
    os.makedirs(outdirroot, exist_ok=True)
    xp.save(r'{}\HPC_all_info_{}.npy'.format(outdirroot, recname), 
            all_info)
    print(f'saved to {outdirroot} ({str(timedelta(seconds=int(time() - t0)))})\n')
