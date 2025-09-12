# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 13:38:49 2025

extract LC spike trains aligned to all behavioural landmarks 

@author: Dinghao Luo
"""

#%% imports 
import h5py
import os
import scipy.io as sio 
from tqdm import tqdm
from time import time
from datetime import timedelta

from common import gaussian_kernel_unity

import rec_list
paths = rec_list.pathLC


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


#%% parameters 
SAMP_FREQ = 1250  # Hz
SIGMA_SPIKE = int(SAMP_FREQ * 0.05)  # 50 ms
GAUS_SPIKE = gaussian_kernel_unity(SIGMA_SPIKE, GPU_AVAILABLE)

MAX_LENGTH = 12500  # samples ♣

BEF = 3  # seconds before 
AFT = 7  # seconds after 


#%% main 
def main(path):
    recname = path[-17:]
    
    # aligned behavioural landmarks 
    aligned_run_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}',  # numbers + r
                                    recname[:14],  # till end of date
                                    recname,
                                    f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat')
    aligned_cue_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}',
                                    recname[:14],
                                    recname,
                                    f'{recname}_DataStructure_mazeSection1_TrialType1_alignCue_msess1.mat')
    aligned_rew_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}',
                                    recname[:14],
                                    recname,
                                    f'{recname}_DataStructure_mazeSection1_TrialType1_alignRew_msess1.mat')
    
    # spike file paths 
    clu_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}', 
                            recname[:14],
                            recname,
                            f'{recname}.clu.1')
    res_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}', 
                            recname[:14],
                            recname,
                            f'{recname}.res.1')
    
    # check
    if (not os.path.exists(aligned_run_path)
        or not os.path.exists(aligned_cue_path)
        or not os.path.exists(aligned_rew_path)
        or not os.path.exists(clu_path)
        or not os.path.exists(res_path)):
        print(f'\nmissing data for {recname}; skipped')
    else:
        print(f'\n{recname}')
    
    # alignment timepoints 
    aligned_run = sio.loadmat(aligned_run_path)['trialsRun'][0][0]
    run_onsets = aligned_run['startLfpInd'][0][1:]  # discard the first trial which is empty 
    
    aligned_cue = sio.loadmat(aligned_cue_path)['trialsCue'][0][0]
    cue_onsets = aligned_cue['startLfpInd'][0][1:]  # similar to above 
    
    aligned_rew = sio.loadmat(aligned_rew_path)['trialsRew'][0][0]
    rew_onsets = aligned_rew['startLfpInd'][0][1:]  # this marks the last trial's reward 
    
    tot_trials = len(run_onsets)
    if len(rew_onsets) != tot_trials or len(cue_onsets) != tot_trials:
        print('WARNING: onsets of different lengths')
        
    # spike reading
    clusters = np.loadtxt(clu_path, dtype=int, skiprows=1)  # first line = number of clusters
    spike_times = np.loadtxt(res_path, dtype=int) / (20_000 / SAMP_FREQ)  # convert to behavioural time scale
    spike_times = spike_times.astype(int)  # ensure integer indices for indexing
    
    unique_clus = [clu for clu in np.unique(clusters) if clu not in [0, 1]]
    tot_clus = len(unique_clus)
    clu_to_row = {clu: i for i, clu in enumerate(unique_clus)}  # map cluster ID to row index
    
    max_time = spike_times.max() + 1  # +1 to make sure last time index is included
    spike_map = np.zeros((len(unique_clus), max_time), dtype=int)
    spike_array = np.zeros((len(unique_clus), max_time))
    
    for t, clu in zip(spike_times, clusters):
        if clu in [0, 1]:
            continue  # skip noise
        row = clu_to_row[clu]
        spike_map[row, t] = 1  # set spike bin to 1
    
    ''' now everything should be in milliseconds '''
    ''' and we start alignment '''
    
    all_trains_run = {}
    all_trains_cue = {}
    all_trains_rew = {}
    all_rasters_run = {}
    all_rasters_cue = {}
    all_rasters_rew = {}
    
    filename = os.path.join(path, f'{recname}_DataStructure_mazeSection1_TrialType1')
    
    spike_time_file = h5py.File(f'{filename}_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']
    
    time_bef = spike_time_file['TimeBef']; time_aft = spike_time_file['Time']
    tot_clu = time_aft.shape[1]
    tot_trial = time_aft.shape[0]-1  # trial 1 is empty
    
    if GPU_AVAILABLE:    
        rasters_run_gpu = xp.zeros((tot_clu, tot_trial, MAX_LENGTH), dtype=xp.uint16)
        rasters_rew_gpu = xp.zeros_like(rasters_run_gpu)
        rasters_cue_gpu = xp.zeros_like(rasters_run_gpu)
        
        for clu in tqdm(range(tot_clu), desc='generating spike array (GPU)'):
            curr_clu_run = np.zeros((tot_trials, MAX_LENGTH))
            curr_clu_rew = np.zeros((tot_trials, MAX_LENGTH))
            curr_clu_cue = np.zeros((tot_trials, MAX_LENGTH))
            for trial in range(tot_trials):
                run_x0 = max(run_onsets[trial] - BEF*SAMP_FREQ, 0)
                run_x1 = min(run_onsets[trial] + AFT*SAMP_FREQ, max_time)
                rasters_run_gpu[clu, trial, : run_x1-run_x0] = cp.asarray(
                    spike_map[clu, run_x0 : run_x1])
                
                rew_x0 = max(rew_onsets[trial] - BEF*SAMP_FREQ, 0)
                rew_x1 = min(rew_onsets[trial] + AFT*SAMP_FREQ, max_time)
                rasters_rew_gpu[clu, trial, : rew_x1-rew_x0] = cp.asarray(
                    spike_map[clu, rew_x0 : rew_x1])
                
                cue_x0 = max(cue_onsets[trial] - BEF*SAMP_FREQ, 0)
                cue_x1 = min(cue_onsets[trial] + AFT*SAMP_FREQ, max_time)
                rasters_cue_gpu[clu, trial, : cue_x1-cue_x0] = cp.asarray(
                    spike_map[clu, cue_x0 : cue_x1])
        
    else:
        rasters_run = xp.zeros((tot_clu, tot_trial, MAX_LENGTH), dtype=xp.uint16)
        rasters_rew = xp.zeros_like(rasters_run)
        rasters_cue = xp.zeros_like(rasters_run)
        
        for clu in tqdm(range(tot_clu), desc='generating spike array (GPU)'):
            curr_clu_run = np.zeros((tot_trials, MAX_LENGTH))
            curr_clu_rew = np.zeros((tot_trials, MAX_LENGTH))
            curr_clu_cue = np.zeros((tot_trials, MAX_LENGTH))
            for trial in range(tot_trials):
                run_x0 = max(run_onsets[trial] - BEF*SAMP_FREQ, 0)
                run_x1 = min(run_onsets[trial] + AFT*SAMP_FREQ, max_time)
                rasters_run[clu, trial, : run_x1-run_x0] = spike_map[clu, run_x0 : run_x1] 
                
                rew_x0 = max(rew_onsets[trial] - BEF*SAMP_FREQ, 0)
                rew_x1 = min(rew_onsets[trial] + AFT*SAMP_FREQ, max_time)
                rasters_rew[clu, trial, : rew_x1-rew_x0] = spike_map[clu, rew_x0 : rew_x1] 
                
                cue_x0 = max(cue_onsets[trial] - BEF*SAMP_FREQ, 0)
                cue_x1 = min(cue_onsets[trial] + AFT*SAMP_FREQ, max_time)
                rasters_cue[clu, trial, : cue_x1-cue_x0] = spike_map[clu, cue_x0 : cue_x1] 
    
    t0 = time()
    if GPU_AVAILABLE:
        # GPU-accelerated convolution using CuPy
        trains_run = (cpss.fftconvolve(
            rasters_run_gpu, GAUS_SPIKE[None, None, :], 
            mode='same'
            ) * SAMP_FREQ).get()
        trains_rew = (cpss.fftconvolve(
            rasters_rew_gpu, GAUS_SPIKE[None, None, :], 
            mode='same'
            ) * SAMP_FREQ).get()
        trains_cue = (cpss.fftconvolve(
            rasters_cue_gpu, GAUS_SPIKE[None, None, :], 
            mode='same'
            ) * SAMP_FREQ).get()
        
        rasters_run = rasters_run_gpu.get()
        rasters_rew = rasters_rew_gpu.get()
        rasters_cue = rasters_cue_gpu.get()
        
        print(
            'convolution on GPU done in '
            f'{str(timedelta(seconds=int(time() - t0)))} s')
    else:
        # CPU convolution using SciPy's FFT-based convolution for better performance
        trains_run = fftconvolve(
            rasters_run, GAUS_SPIKE[None, None, :], 
            mode='same'
            ) * SAMP_FREQ
        trains_rew = fftconvolve(
            rasters_rew, GAUS_SPIKE[None, None, :], 
            mode='same'
            ) * SAMP_FREQ
        trains_cue = fftconvolve(
            rasters_cue, GAUS_SPIKE[None, None, :], 
            mode='same'
            ) * SAMP_FREQ
        
        print(
            'convolution on CPU done in '
            f'{str(timedelta(seconds=int(time() - t0)))} s')
    
    for clu in range(tot_clu):
        cluname = f'{recname} clu{clu+2}'

        all_trains_run[cluname] = trains_run[clu]
        all_rasters_run[cluname] = rasters_run[clu]
        
        all_trains_rew[cluname] = trains_rew[clu]
        all_rasters_rew[cluname] = rasters_rew[clu]
        
        all_trains_cue[cluname] = trains_cue[clu]
        all_rasters_cue[cluname] = rasters_cue[clu]
    
    print('done; saving...')
    sess_folder = rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}'
    np.save(
        rf'{sess_folder}\{recname}_all_trains_run.npy',
        all_trains_run
        )
    np.save(
        rf'{sess_folder}\{recname}_all_rasters_run.npy',
        all_rasters_run
        )
    np.save(
        rf'{sess_folder}\{recname}_all_trains_rew.npy',
        all_trains_rew
        )
    np.save(
        rf'{sess_folder}\{recname}_all_rasters_rew.npy',
        all_rasters_rew
        )
    np.save(
        rf'{sess_folder}\{recname}_all_trains_cue.npy',
        all_trains_cue
        )
    np.save(
        rf'{sess_folder}\{recname}_all_rasters_cue.npy',
        all_rasters_cue
        )
    print(f'saved to {sess_folder}'
          f'({str(timedelta(seconds=int(time() - t0)))})\n')
        
if __name__ == '__main__':
    for path in paths:
        main(path)