# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:52:50 2024
Modified on Thur 24 Apr 2025 to add a size filter 

plot the heatmap of pooled ROI activity of axon-GCaMP animals 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt
import sys
import pandas as pd 
from tqdm import tqdm

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, smooth_convolve, normalise
mpl_formatting()

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLCHPCGCaMP


#%% parameters 
ROI_size_threshold = 500  # pixel count 


#%% load data 
df = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP'
    r'\LCHPC_axon_GCaMP_all_profiles.pkl'
    )

proc_path = r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions'
temp_dict = np.load(
    rf'{proc_path}\{paths[0][-17:]}\processed_data\RO_aligned_mean_dict.npy',
    allow_pickle=True
    ).item()
temp_coord_dict = np.load(
    rf'{proc_path}\{paths[0][-17:]}\processed_data\valid_rois_coord_dict.npy',
    allow_pickle=True
    ).item()
pooled_ROIs = np.row_stack(
    [temp_dict[key] 
     for key in temp_dict
     if len(temp_coord_dict[key][0]) > ROI_size_threshold]
    )

for rec_path in tqdm(paths, desc='loading sessions'):
    temp_dict = np.load(
        rf'{proc_path}\{rec_path[-17:]}\processed_data\RO_aligned_mean_dict.npy',
        allow_pickle=True
        ).item()
    temp_coord_dict = np.load(
        rf'{proc_path}\{rec_path[-17:]}\processed_data\valid_rois_coord_dict.npy',
        allow_pickle=True
        ).item()
    temp_array = np.row_stack(
        [temp_dict[key] 
         for key in temp_dict
         if len(temp_coord_dict[key][0]) > ROI_size_threshold]
        )
    pooled_ROIs = np.vstack((pooled_ROIs, temp_array))
                            
tot_rois = pooled_ROIs.shape[0]
pooled_ROIs = normalise(smooth_convolve(pooled_ROIs))


#%% plotting 
keys = np.argsort([np.argmax(pooled_ROIs[roi, :]) for roi in range(tot_rois)])
im_matrix = pooled_ROIs[keys, :]

fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from run-onset (s)',
       ylabel='ROI #')
ax.set_aspect('equal')
fig.suptitle('LC-CA1 GCaMP')

im_ordered = ax.imshow(im_matrix, 
                       cmap='viridis', aspect='auto', extent=(-1, 4, 0, tot_rois))
plt.colorbar(im_ordered, shrink=.5, ticks=[0,1], label='norm. dF/F')

for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP'
        rf'\pooled_ordered_heatmap_RO_aligned{ext}',
        dpi=300,
        bbox_inches='tight'
        )