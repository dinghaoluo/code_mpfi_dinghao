# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:52:50 2024

plot the heatmap of pooled ROI activity of axon-GCaMP animals 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, smooth_convolve, normalise
mpl_formatting()

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLCGCaMP = rec_list.pathHPCLCGCaMP


#%% load data 
proc_path = r'Z:\Dinghao\code_dinghao\axon_GCaMP\single_sessions'
temp_dict = np.load(r'{}\{}\RO_aligned_dict.npy'.format(proc_path, pathHPCLCGCaMP[0][-17:]),
                      allow_pickle=True).item()
pooled_ROIs = np.row_stack([temp_dict[key] for key in temp_dict])
for rec_path in tqdm(pathHPCLCGCaMP, desc='loading sessions'):
    temp_dict = np.load(r'{}\{}\RO_aligned_dict.npy'
                        .format(proc_path, rec_path[-17:]),
                        allow_pickle=True).item()
    temp_array = np.row_stack([temp_dict[key] for key in temp_dict])
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

im_ordered = ax.imshow(im_matrix, cmap='viridis', aspect='auto', extent=(-1, 4, 0, tot_rois))
plt.colorbar(im_ordered, shrink=.5, ticks=[0,1], label='norm. dF/F')

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\axon_GCaMP\pooled_ordered_heatmap_RO_aligned{}'.format(ext),
                dpi=300)