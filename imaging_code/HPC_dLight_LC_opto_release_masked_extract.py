# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:49:48 2025

extract dLight traces based on releasing ROIs

@author: Dinghao Luo
"""

#%% imports 
import os
import sys
from time import time
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from tqdm import tqdm
import pickle

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list

paths = rec_list.pathdLightLCOpto


#%% parameters 
SAMP_FREQ = 30

BEF = 1
AFT = 4

XAXIS = np.arange((BEF + AFT) * SAMP_FREQ) / SAMP_FREQ - BEF


#%% GPU acceleration
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
    if GPU_AVAILABLE:
        print(
            'using GPU-acceleration with '
            f'{str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode("UTF-8"))} '
            'and CuPy')
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
    else:
        print('GPU acceleration unavailable')
        Exception('please do NOT attempt to run this on CPU; aborting')
except ModuleNotFoundError:
    print('CuPy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
    Exception('please do NOT attempt to run this on CPU; aborting')
except Exception as e:
    # catch any other unexpected errors and print a general message
    print(f'an error occurred: {e}')
    GPU_AVAILABLE = False
    Exception('please do NOT attempt to run this on CPU; aborting')


#%% main 
for path in paths:
    recname = os.path.basename(path)
    print(f'\n{recname}...')
    
    # results directory
    sess_dir = os.path.join(
        r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions',
        recname
        )
    proc_dir = os.path.join(
        sess_dir,
        'processed_data'
        )
    
    # releasing roi dict check 
    releasing_roi_dict_path = os.path.join(
        proc_dir, f'{recname}_releasing_ROI_dict.npy'
        )
    if not (os.path.exists(releasing_roi_dict_path)):
        print(f'missing releasing ROI dict for {recname}; skipped')
        continue
    
    # behaviour data check  
    if os.path.exists(
            rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCdLightLCOpto\{recname}.pkl'
            ):
        with open(
                rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCdLightLCOpto\{recname}.pkl',
                'rb'
                ) as f:
            beh = pickle.load(f)
    else:
        print(f'missing behaviour file for {recname}; skipped')
        continue
    
    if not beh['run_onsets']:
        print(f'{recname} is an immobile recording; skipped')
        continue
    
    # if all these data are ready, then proceed
    binpath = os.path.join(path, 'suite2p/plane0/data.bin')
    bin2path = os.path.join(path, 'suite2p/plane0/data_chan2.bin')
    opspath = os.path.join(path, 'suite2p/plane0/ops.npy')
    txtpath = os.path.join(r'Z:\Dinghao\MiceExp',
                           f'ANMD{recname[1:4]}',
                           f'{recname[:4]}{recname[5:]}T.txt')
    
    # load data 
    ops = np.load(opspath, allow_pickle=True).item()
    tot_frames = ops['nframes']
    Ly, Lx = ops['Ly'], ops['Lx']
    
    print('loading movies...')
    t0 = time()
    mov = np.memmap(
        binpath, mode='r', dtype='int16', shape=(tot_frames, Ly, Lx)
        ).astype(np.float32)
    mov2 = np.memmap(
        bin2path, mode='r', dtype='int16', shape=(tot_frames, Ly, Lx)
        ).astype(np.float32)
    print(f'movies loaded ({str(timedelta(seconds=int(time()-t0)))})')
    
    # reshape the movies to use advanced indexing (for mean later in the loop)
    mov = mov.reshape(tot_frames, -1)
    mov2 = mov2.reshape(tot_frames, -1)
    
    tot_frames = mov.shape[0]  # once loaded, update tot_frames to be the max frame number, 16 June 2025

    # run-onset aligned directory 
    run_aligned_dir = os.path.join(
        sess_dir,
        'run_aligned'
        )
    os.makedirs(run_aligned_dir, exist_ok=True)
    
    # release mask (this needs to be computed in correlation analysis)
    releasing_roi_dict = np.load(
        releasing_roi_dict_path, allow_pickle=True
        ).item()
    
    # extract fluorescence traces 
    roi_ids = list(releasing_roi_dict.keys())
    roi_traces = np.zeros((len(roi_ids), tot_frames), dtype=np.float32)
    roi_traces_ch2 = np.zeros_like(roi_traces)
    
    for i, roi_id in tqdm(enumerate(roi_ids),
                          total=len(roi_ids),
                          desc='building trace arrays'):
        roi = releasing_roi_dict[roi_id]
        pix_idx = np.ravel_multi_index(
            (roi['ypix'], roi['xpix']),
            dims=(Ly, Lx)
        )
        roi_traces[i] = np.mean(mov[:, pix_idx], axis=1)
        roi_traces_ch2[i] = np.mean(mov2[:, pix_idx], axis=1)
            
    # calculate all the ROIs at the same time 
    print('computing dFFs for both channels...')
    dFF_all = ipf.calculate_dFF(roi_traces, GPU_AVAILABLE=GPU_AVAILABLE)
    dFF_all_ch2 = ipf.calculate_dFF(roi_traces_ch2, GPU_AVAILABLE=GPU_AVAILABLE)

    releasing_roi_traces = {roi_id: dFF_all[i] for i, roi_id in enumerate(roi_ids)}
    releasing_roi_traces_ch2 = {roi_id: dFF_all_ch2[i] for i, roi_id in enumerate(roi_ids)}
        
    np.save(os.path.join(proc_dir, f'{recname}_releasing_traces_dFF.npy'),
            releasing_roi_traces)
    np.save(os.path.join(proc_dir, f'{recname}_releasing_traces_dFF_ch2.npy'),
            releasing_roi_traces_ch2)
    
    # align to behaviour
    print('aligning to behaviour...')
    
    run_onsets = beh['run_onsets']
    run_onset_frames = beh['run_onset_frames']
    stim_idx = [trial for trial, trial_statement 
                in enumerate(beh['trial_statements'])
                if trial_statement[15] != '0']
    ctrl_idx = [trial for trial in np.arange(len(run_onsets))
                if trial not in stim_idx]
    
    run_aligned_traces = {}
    run_aligned_traces_ch2 = {}
    for roi_id in tqdm(releasing_roi_dict.keys(),
                       total=len(releasing_roi_dict)):
        run_aligned = []
        run_aligned_ch2 = []
        for trial in ctrl_idx:  # only trials with no stim 
            if (
                    not np.isnan(run_onsets[trial]) and 
                    run_onset_frames[trial] > BEF * SAMP_FREQ and 
                    run_onset_frames[trial] < tot_frames - AFT * SAMP_FREQ
                ):
                run_onset_frame = run_onset_frames[trial]
                run_aligned.append(
                    releasing_roi_traces[roi_id][
                        run_onset_frame-BEF*SAMP_FREQ : run_onset_frame+AFT*SAMP_FREQ
                        ]
                    )
                run_aligned_ch2.append(
                    releasing_roi_traces_ch2[roi_id][
                        run_onset_frame-BEF*SAMP_FREQ : run_onset_frame+AFT*SAMP_FREQ
                        ]
                    )
        run_aligned = np.array(run_aligned)
        run_aligned_ch2 = np.array(run_aligned_ch2)
        
        run_aligned_traces[roi_id] = run_aligned
        run_aligned_traces_ch2[roi_id] = run_aligned_ch2
        
        # plotting 
        run_aligned_mean = np.nanmean(run_aligned, axis=0)
        run_aligned_sem = sem(run_aligned, axis=0)
        run_aligned_ch2_mean = np.nanmean(run_aligned_ch2, axis=0)
        run_aligned_ch2_sem = sem(run_aligned_ch2, axis=0)
        
        fig, ax = plt.subplots(figsize=(4,3))
        
        ax.plot(XAXIS, run_aligned_ch2_mean,
                c='indianred')
        ax.fill_between(XAXIS, run_aligned_ch2_mean+run_aligned_ch2_sem,
                               run_aligned_ch2_mean-run_aligned_ch2_sem,
                        color='indianred', edgecolor='none', alpha=.25)
        ax.plot(XAXIS, run_aligned_mean,
                c='darkgreen')
        ax.fill_between(XAXIS, run_aligned_mean+run_aligned_sem,
                               run_aligned_mean-run_aligned_sem,
                        color='darkgreen', edgecolor='none', alpha=.25)
        
        ax.set(xlabel='time from run-onset (s)', xticks=[0, 2, 4],
               ylabel='dF/F',
               title=f'{recname}\n{roi_id}')
        
        fig.tight_layout()
        fig.savefig(os.path.join(run_aligned_dir, str(roi_id)),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
        
    np.save(os.path.join(sess_dir, 
                         f'{recname}_releasing_traces_dFF_run_aligned.npy'),
            run_aligned_traces)
    np.save(os.path.join(sess_dir, 
                         f'{recname}_releasing_traces_dFF_run_aligned_ch2.npy'),
            run_aligned_traces_ch2)