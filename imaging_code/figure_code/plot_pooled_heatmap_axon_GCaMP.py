# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:52:50 2024
Modified on Thur 24 Apr 2025 to add a size filter 

plot the heatmap of pooled ROI activity of axon-GCaMP animals 

@author: Dinghao Luo
"""


#%% imports 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm

from common import mpl_formatting, smooth_convolve, normalise
mpl_formatting()

import rec_list
paths = rec_list.pathLCHPCGCaMP


#%% paths and parameters
axon_GCaMP_stem = Path('Z:/Dinghao/code_dinghao/LCHPC_axon_GCaMP')
 
ROI_size_threshold = 500  # pixel count 


#%% load data 
df = pd.read_pickle(axon_GCaMP_stem / 'LCHPC_axon_GCaMP_all_profiles.pkl')

proc_path = axon_GCaMP_stem / 'all_sessions'

# initialise with the first session
temp_dict = np.load(
    proc_path / paths[0][-17:] / 'processed_data' / 'RO_aligned_mean_dict.npy',
    allow_pickle=True
    ).item()
temp_coord_dict = np.load(
    proc_path / paths[0][-17:] / 'processed_data' / 'valid_rois_coord_dict.npy',
    allow_pickle=True
    ).item()
pooled_ROIs = np.row_stack(
    [temp_dict[key][60:60+30*5]  # -1 ~ 4 s
     for key in temp_dict
     if len(temp_coord_dict[key][0]) > ROI_size_threshold]
    )

for rec_path in tqdm(paths[1:], desc='loading sessions'):
    temp_dict = np.load(
        proc_path / rec_path[-17:] / 'processed_data' / 'RO_aligned_mean_dict.npy',
        allow_pickle=True
        ).item()
    temp_coord_dict = np.load(
        proc_path / rec_path[-17:] / 'processed_data' / 'valid_rois_coord_dict.npy',
        allow_pickle=True
        ).item()
    temp_array = np.row_stack(
        [temp_dict[key][60:60+30*5]  # -1 ~ 4 s 
         for key in temp_dict
         if len(temp_coord_dict[key][0]) > ROI_size_threshold]
        )
    
    # stack to previously saved array
    pooled_ROIs = np.vstack((pooled_ROIs, temp_array))
                            
tot_rois = pooled_ROIs.shape[0]
pooled_ROIs = normalise(smooth_convolve(pooled_ROIs))


#%% plotting 
keys = np.argsort([np.argmax(pooled_ROIs[roi, :]) for roi in range(tot_rois)])
im_matrix = pooled_ROIs[keys, :]

fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from run onset (s)',
       ylabel='ROI #')
ax.set_aspect('equal')
fig.suptitle('LC-CA1 GCaMP')

im_ordered = ax.imshow(im_matrix, 
                       cmap='viridis', aspect='auto', extent=(-1, 4, 0, tot_rois))
plt.colorbar(im_ordered, shrink=.5, ticks=[0,1], label='norm. dF/F')

for ext in ['.png', '.pdf']:
    fig.savefig(
        axon_GCaMP_stem / f'pooled_ordered_heatmap_RO_aligned{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    

#%% run onset peaks?
run_onset_peaks = df['run_onset_peak']

print(f'Percentage of run-onset-peaking axon ROIs: {sum(run_onset_peaks) / len(run_onset_peaks) * 100}%')