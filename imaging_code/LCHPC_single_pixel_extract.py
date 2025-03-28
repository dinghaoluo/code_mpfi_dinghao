# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:46:39 2025

Extract single-pixel fluorescence traces after spatial filtering

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import os 
from tqdm import tqdm
from time import time 
from datetime import timedelta 
import sys 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf
import support_LCHPC_axon as support

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLCHPCGCaMP


#%% GPU acceleration
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
    if GPU_AVAILABLE:
        print(
            'using GPU-acceleration with '
            f'{str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode("UTF-8"))} '
            'and CuPy'
            )
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
    else:
        print('GPU acceleration unavailable')
except ModuleNotFoundError:
    print('CuPy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
except Exception as e:
    # catch any other unexpected errors and print a general message
    print(f'an error occurred: {e}')
    GPU_AVAILABLE = False
    
    
#%% main 
for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    t0 = time()
    
    ops_path = path+r'/suite2p/plane0/ops.npy'
    bin_path = path+r'/suite2p/plane0/data.bin'
    bin2_path = path+r'/suite2p/plane0/data_chan2.bin'
    stat_path = path+r'/suite2p/plane0/stat.npy'
    
    # folder to put processed data and single-session plots 
    proc_path = (r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP'
                 rf'\all_sessions\{recname}')
    os.makedirs(proc_path, exist_ok=True)
    
    proc_data_path = (r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP'
                      rf'\all_sessions\{recname}\processed_data')
    os.makedirs(proc_data_path, exist_ok=True)
    
    # load files 
    stat = np.load(stat_path, allow_pickle=True)
    if 'inmerge' not in stat[0]: 
        sys.exit('halting executation: no merging detected')  # detect merging
    
    # get roi idx
    valid_rois_dict = support.filter_valid_rois(stat)
    valid_rois = [*valid_rois_dict]  # using lists because of the later serial comprehension 
    
    # get x and y pix idx and put in a dict
    roi_coords_dict = {
        f'ROI {roi}': [stat[roi]['xpix'], stat[roi]['ypix']]
        for roi in valid_rois
        }
    
    # read in bin files 
    ops = np.load(ops_path, allow_pickle=True).item()
    tot_frames = ops['nframes']
    shape = (tot_frames, ops['Ly'], ops['Lx'])
    mov = np.memmap(bin_path, mode='r', dtype='int16', shape=shape)
    mov2 = np.memmap(bin2_path, mode='r', dtype='int16', shape=shape)
    
    # median-filter spatially after loading movies
    print('channel 1')
    mov_filtered = support.spatial_median_filter(
        mov,
        size=5, 
        GPU_AVAILABLE=GPU_AVAILABLE
        )
    print('channel 2')
    mov2_filtered = support.spatial_median_filter(
        mov2,
        size=5, 
        GPU_AVAILABLE=GPU_AVAILABLE
        )
    
    roi_pixels_dict = {}
    print('extracting pixel traces...')
    for roi in roi_coords_dict:
        t1 = time()
        pixx = roi_coords_dict[roi][0]
        pixy = roi_coords_dict[roi][1]
        
        pixels_traces = mov_filtered[:, pixx, pixy]
        pixels_dFF = ipf.calculate_dFF(pixels_traces,
                                       t_axis=0,
                                       CHUNK=False,
                                       GPU_AVAILABLE=True)
        
        curr_roi_dict = {
            'coord': list(zip(pixx, pixy)),
            'dFF': pixels_dFF
            }
        roi_pixels_dict[roi] = curr_roi_dict
        
        print(f'{roi} done ({timedelta(seconds=int(time()-t1))})')
    
    np.save(
        rf'{proc_data_path}/roi_pixel_dFF.npy', 
        roi_pixels_dict
        )
    print(f'{recname} done ({timedelta(seconds=int(time()-t0))})')