# -*- coding: utf-8 -*-
"""
Created on Mon 10 July 10:02:32 2023
Modified 13 Feb 12:47 2025 to work on LC recordings

pool all cells from all recording sessions
modified 11 Dec 2024 to process with all trials (not skipping trial 0) and 
    added GPU support
    - memory leakage problems on GPU, 20 Dec 2024 
    - issue persisted, but I think the problem is that previously we used the 
        same variable names for rasters, trains in VRAM and in RAM; I changed 
        the GPU versions to rasters_gpu and trains_gpu, 26 Dec 2024 
    - we may be able to circumvent this problem completely by wrapping the 
        processing steps in a function since variable references are destroyed 
        after function executations, 26 Dec 2024

@author: Dinghao Luo
"""


#%% imports
import gc 
import sys
import h5py
import os
from tqdm import tqdm
from time import time
from datetime import timedelta

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC

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
    import cupyx.scipy.signal as cpss # for GPU
    import numpy as np 
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print(f'GPU-acceleration with {str(name)} and cupy\n')
else:
    import numpy as np 
    xp = np
    from scipy.signal import fftconvolve  # for CPU
    print('GPU-acceleartion unavailable\n')


#%% main 
def main():

    # parameters 
    samp_freq = 1250  # Hz
    sigma_spike = samp_freq/10
    max_length = 12500  # samples 
    
    gaus_spike = gaussian_kernel_unity(sigma_spike, GPU_AVAILABLE)
    
    for pathname in paths:
        all_trains = {}
        all_rasters = {}
        
        recname = pathname[-17:]
        print(recname)
            
        filename = os.path.join(pathname, f'{pathname[-17:]}_DataStructure_mazeSection1_TrialType1')
        
        spike_time_file = h5py.File(f'{filename}_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']
        
        time_bef = spike_time_file['TimeBef']; time_aft = spike_time_file['Time']
        tot_clu = time_aft.shape[1]
        tot_trial = time_aft.shape[0]-1  # trial 1 is empty
        
        # spike reading
        spike_time = np.empty((tot_clu, tot_trial), dtype='object')
        for clu in tqdm(range(tot_clu), desc='reading spike trains'):
            # process all trials at once using vectorised operations
            combined_spike_time = np.array([
                # concatenate spike times from before and after
                np.concatenate((
                    spike_time_file[time_bef[trial, clu]][0]
                    if not isinstance(spike_time_file[time_bef[trial, clu]][0], np.uint64) else [],
                    spike_time_file[time_aft[trial, clu]][0]
                    if not isinstance(spike_time_file[time_aft[trial, clu]][0], np.uint64) else []
                ))
                for trial in range(1, tot_trial+1)  # trial 1 is empty 
            ], dtype='object')
        
            # store the combined spike times
            spike_time[clu, :] = combined_spike_time
        
        # initialisation
        ''' 
        this has undergone some revamping--empty object arrays were initially used 
        when initialising all_trains and all_rasters, but the outdated array type
        defies vectorisation; considering that one never needed the arrays to be of
        variable-length, I now initialise them to be fixed-length, numeric arrays 
        for better performance and less memory usage
        20 Dec 2024 Dinghao 
        '''
        if GPU_AVAILABLE:    
            rasters_gpu = xp.zeros((tot_clu, tot_trial, max_length), dtype=xp.uint16)
            trains_gpu = xp.zeros_like(rasters_gpu)
            for clu in tqdm(range(tot_clu), desc='generating spike array (GPU)'):
                for trial in range(tot_trial):
                    # adjust spike index alignment (no negative indices)
                    spikes = xp.array(spike_time[clu, trial], dtype=xp.int32) + 3 * samp_freq
                    spikes = spikes[spikes < max_length]  # clip spikes beyond max_length
                    rasters_gpu[clu, trial, spikes] = 1  # set spike times to 1
        else:
            rasters = xp.zeros((tot_clu, tot_trial, max_length), dtype=xp.uint16)
            trains = xp.zeros_like(rasters)
            for clu in tqdm(range(tot_clu), desc='generating spike array (GPU)'):
                for trial in range(tot_trial):
                    # adjust spike index alignment (no negative indices)
                    spikes = xp.array(spike_time[clu, trial], dtype=xp.int32) + 3 * samp_freq
                    spikes = spikes[spikes < max_length]  # clip spikes beyond max_length
                    rasters[clu, trial, spikes] = 1  # set spike times to 1
        
        t0 = time()
        if GPU_AVAILABLE:
            # GPU-accelerated convolution using CuPy
            trains_gpu = cpss.fftconvolve(
                rasters_gpu, gaus_spike[None, None, :], 
                mode='same'
                ) * samp_freq
            trains = trains_gpu.get()
            rasters = rasters_gpu.get()
            print(
                'convolution on GPU done in '
                f'{str(timedelta(seconds=int(time() - t0)))} s')
        else:
            # CPU convolution using SciPy's FFT-based convolution for better performance
            trains = fftconvolve(
                rasters, gaus_spike[None, None, :], 
                mode='same'
                )
            print(
                'convolution on CPU done in '
                f'{str(timedelta(seconds=int(time() - t0)))} s')
        
        for clu in range(tot_clu):
            cluname = f'{recname} clu{clu+2}'
            all_trains[cluname] = trains[clu]
            all_rasters[cluname] = rasters[clu]
        
        print('done; saving...')
        sess_folder = rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}'
        np.save(
            rf'{sess_folder}\{recname}_all_trains.npy',
            all_trains
            )
        np.save(
            rf'{sess_folder}\{recname}_all_rasters.npy',
            all_rasters
            )
        print(f'saved to {sess_folder}'
              f'({str(timedelta(seconds=int(time() - t0)))})\n')
        
        # free memory pool if GPU
        if GPU_AVAILABLE:
            del rasters_gpu, trains_gpu
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        del rasters, trains, spike_time, all_trains, all_rasters
        # spike_time_file.close()  # Close HDF5 file
        gc.collect()
        
if __name__ == '__main__':
    main()