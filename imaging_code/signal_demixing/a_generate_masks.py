# -*- coding: utf-8 -*-
"""
Created originally by Yingxue Wang
Modified on 2 Sept 2025

using linear regression to demix motion artefacts from imaging recordings 
    (dual colour)
modified from Yingxue's original pipeline

@author: Dinghao Luo
"""


#%% imports 
import sys 
import gc

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import Motion_Per_ROI
import Generate_masks

p_data = r'Z:\Dinghao\2p_recording\A126i\A126i-20250616\A126i-20250616-01\suite2p\plane0'
file_name = r'\data.bin'
file_name2 = r'\data_chan2.bin'

path1 = Path(p_data, file_name)
path2 = Path(p_data, file_name2)

path_result = Path(p_data) / 'result' / 'masks'
path_result.mkdir(parents=True, exist_ok=True)

num_frames = 10000
height = 512
width = 512

# we load using memmap to reduce RAM usage; originally FilterImage was used 
mov = np.memmap(path1, dtype='float32', mode='r',
                shape=(num_frames, height, width))
movr = np.memmap(path2, dtype='float32', mode='r',
                 shape=(num_frames, height, width))


#%% raw movie traces 
# trim movies of borders 
edge_pixel_num = 15
shape = mov.shape
xlim = shape[1]
ylim = shape[2]
mov_tmp_r = movr[:, edge_pixel_num:ylim-edge_pixel_num, :]
mov_tmp_r = mov_tmp_r[:, :, edge_pixel_num:xlim-edge_pixel_num]
mov_tmp_g = mov[:, edge_pixel_num:ylim-edge_pixel_num, :]
mov_tmp_g = mov_tmp_g[:, :, edge_pixel_num:xlim-edge_pixel_num]

# raw traces whole-field
mov_mean_trace = mov_tmp_r.mean(axis=(1,2))
mov_mean_trace_g = mov_tmp_g.mean(axis=(1,2))


#%% mean images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle('Mean image Ch1 vs Ch2', fontsize=16)

im1 = ax1.imshow(np.mean(mov_tmp_g, axis=0), cmap='gray')
ax1.set_title('Mean_Ch1')
fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04, label='F')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

im2 = ax2.imshow(np.mean(mov_tmp_r, axis=0), cmap='gray')
ax2.set_title('Mean_Ch2')
fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04, label='F')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

fig.savefig(Path(path_result, 'mean_images.png'),
            dpi=300,
            bbox_inches='tight')


#%% masks
# create global axon mask
mean_img_red = movr.mean(axis=0)
global_axon_mask = Generate_masks.generate_and_save_axon_mask(
    mean_img=mean_img_red, 
    output_dir=path_result,
    output_filename_base='global_axon_mask',
    tophat_radius=5, 
    intensity_quantile=0.97, 
    min_size=10
)

# create global dLight mask
mean_img_green = mov.mean(axis=0)
global_dlight_mask = Generate_masks.generate_and_save_dlight_mask(
    mean_img=mean_img_green,
    output_dir=path_result,
    output_filename_base='global_dlight_mask',
    gaussian_sigma=1.5,
    peak_min_distance=5,
    adaptive_block_size=5
)

# create global axon and dLight mask
global_axon_dlight_mask = Generate_masks.AND_mask(
    mask1=global_axon_mask,
    mask2=global_dlight_mask,
    mask1_name='axon',
    mask2_name='dlight' ,
    output_dir=path_result,
    output_filename_base='global_axon_and_dlight_mask'
)

# create global axon or dLight mask
global_axon_or_dlight_mask = Generate_masks.OR_mask(
    mask1=global_axon_mask,
    mask2=global_dlight_mask,
    mask1_name='axon',
    mask2_name='dlight' ,
    output_dir=path_result,
    output_filename_base='global_axon_or_dlight_mask'
)

# create dilated global axon and dLight mask
global_axon_mask_dilated = Generate_masks.dilate_mask(global_axon_mask, iterations=4)
global_axon_or_dlight_mask = Generate_masks.AND_mask(
    mask1=global_axon_mask_dilated,
    mask2=global_dlight_mask,
    mask1_name='dilated_axon',
    mask2_name='dlight' ,
    output_dir=path_result,
    output_filename_base='dilated_global_axon_and_dlight_mask'
)

# create dilated global axon or dLight mask
global_axon_or_dlight_mask = Generate_masks.OR_mask(
    mask1=global_axon_mask_dilated,
    mask2=global_dlight_mask,
    mask1_name='dilated_axon',
    mask2_name='dlight' ,
    output_dir=path_result,
    output_filename_base='dilated_global_axon_or_dlight_mask'
)


grid_size = 16


#%% pixel count heatmap 
pixel_counts = Motion_Per_ROI.plot_pixel_count_heatmap(
        global_axon_mask=global_axon_mask,
        grid_size=grid_size,
        mean_img=global_axon_mask)


#%% dump
del mov
del movr

gc.collect()