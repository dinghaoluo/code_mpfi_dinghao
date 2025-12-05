# -*- coding: utf-8 -*-
"""
Created on Mon 10 July 10:02:32 2023
Modified on 24 Nov 2025 to mirror the LC processing pipeline 

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
from pathlib import Path

import h5py
import scipy.io as sio 
import mat73
from tqdm import tqdm
from time import time
from datetime import timedelta

from common import gaussian_kernel_unity

import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt


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

MAX_LENGTH = 12500  # samples

BEF = 3  # seconds before 
AFT = 7  # seconds after 


#%% main 
def main(path):
    recname = Path(path).name
    
    rec_stem = Path(f'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}') / recname[:14] / recname
    
    # aligned behavioural landmarks 
    aligned_run_path = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
    aligned_cue_path = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignCue_msess1.mat'
    aligned_rew_path = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRew_msess1.mat'
    
    # spike file paths 
    clu_paths = [rec_stem / f'{recname}.clu.{probe}' for probe in range(1,7)]
    res_paths = [rec_stem / f'{recname}.res.{probe}' for probe in range(1,7)]
    
    # get cluname 
    filename = Path(path) / f'{recname}_BehavElectrDataLFP.mat'
    BehavLFP = mat73.loadmat(filename)
    Clu = BehavLFP['Clu']
    shank = Clu['shank']
    localClu = Clu['localClu']
    
    # check
    if (not aligned_run_path.exists()
        or not aligned_cue_path.exists()
        or not aligned_rew_path.exists()):
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
        
    ## ---- spike reading ---- ##
    clusters    = np.loadtxt(clu_paths[0], dtype=int, skiprows=1)  # initiate 
    spike_times = np.loadtxt(res_paths[0], dtype=int) / (20_000 / SAMP_FREQ)
    valid = (clusters != 0) & (clusters != 1)  # filter out the MUA and noise 
    clusters    = clusters[valid]
    clusters    = np.array([clu - 1 for clu in clusters])
    spike_times = spike_times[valid]
    spike_times = spike_times.astype(int)
    
    for probe in range(1, 6):
        try:
            last_cluster = clusters.max()
        except ValueError:
            last_cluster = 2
        
        new_clusters    = np.loadtxt(clu_paths[probe], dtype=int, skiprows=1)
        new_spike_times = np.loadtxt(res_paths[probe], dtype=int) / (20_000 / SAMP_FREQ)
        valid = (new_clusters != 0) & (new_clusters != 1)
        new_clusters    = new_clusters[valid]
        new_clusters    = [clu - 1 for clu in new_clusters]
        new_clusters    = np.array([clu + last_cluster for clu in new_clusters])
        new_spike_times = new_spike_times[valid]
        new_spike_times = new_spike_times.astype(int)
        
        clusters    = np.concatenate((clusters, new_clusters))
        spike_times = np.concatenate((spike_times, new_spike_times))
    ## ---- spike reading ends ---- ##
    
    unique_clus = [clu for clu in np.unique(clusters)]
    clu_to_row = {clu: i for i, clu in enumerate(unique_clus)}  # map cluster ID to row index
    
    max_time = spike_times.max() + 1  # +1 to make sure last time index is included
    spike_map = np.zeros((len(unique_clus), max_time), dtype=int)
    
    for t, clu in zip(spike_times, clusters):
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
    
    spikefile_name = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignedSpikesPerNPerT_msess1_Run0.mat'
    spike_time_file = h5py.File(spikefile_name)['trialsRunSpikes']
    
    time_aft = spike_time_file['Time']
    tot_clu = time_aft.shape[1]
    tot_trial = time_aft.shape[0]-1  # trial 1 is empty
    
    if GPU_AVAILABLE:    
        rasters_run_gpu = xp.zeros((tot_clu, tot_trial, MAX_LENGTH), dtype=xp.uint16)
        rasters_rew_gpu = xp.zeros_like(rasters_run_gpu)
        rasters_cue_gpu = xp.zeros_like(rasters_run_gpu)
        
        for clu in tqdm(range(tot_clu), desc='Generating spike array (GPU)'):
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
        
        for clu in tqdm(range(tot_clu), desc='Generating spike array (GPU)'):
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
            'Convolution on GPU done in '
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
            'Convolution on CPU done in '
            f'{str(timedelta(seconds=int(time() - t0)))} s')
    
    for clu in range(tot_clu):
        cluname = f'{recname} clu{clu+2} {int(shank[clu])} {int(localClu[clu])}'

        all_trains_run[cluname] = trains_run[clu]
        all_rasters_run[cluname] = rasters_run[clu]
        
        all_trains_rew[cluname] = trains_rew[clu]
        all_rasters_rew[cluname] = rasters_rew[clu]
        
        all_trains_cue[cluname] = trains_cue[clu]
        all_rasters_cue[cluname] = rasters_cue[clu]
        
    # smooth entire spike map for all clusters
    print('Smoothing full spike maps...')
    if GPU_AVAILABLE:
        spike_map_gpu = cp.asarray(spike_map)
        smoothed_spike_map = (cpss.fftconvolve(
            spike_map_gpu, GAUS_SPIKE[None, :], mode='same'
            ) * SAMP_FREQ).get()
        del spike_map_gpu
    else:
        smoothed_spike_map = fftconvolve(
            spike_map, GAUS_SPIKE[None, :], mode='same'
            ) * SAMP_FREQ
    
    print('Done; saving...')
    sess_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions') / recname
    sess_stem.mkdir(exist_ok=True)
    np.save(
        sess_stem / f'{recname}_all_trains_run.npy',
        all_trains_run
        )
    np.save(
        sess_stem / f'{recname}_all_rasters_run.npy',
        all_rasters_run
        )
    np.save(
        sess_stem / f'{recname}_all_trains_rew.npy',
        all_trains_rew
        )
    np.save(
        sess_stem / f'{recname}_all_rasters_rew.npy',
        all_rasters_rew
        )
    np.save(
        sess_stem / f'{recname}_all_trains_cue.npy',
        all_trains_cue
        )
    np.save(
        sess_stem / f'{recname}_all_rasters_cue.npy',
        all_rasters_cue
        )
    np.save(
        sess_stem / f'{recname}_smoothed_spike_map.npy',
        smoothed_spike_map
        )
    print(f'saved to {str(sess_stem)}'
          f'({str(timedelta(seconds=int(time() - t0)))})\n')
    
if __name__ == '__main__':
    for path in paths:
        main(path)