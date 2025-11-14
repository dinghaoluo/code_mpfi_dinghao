# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:33:23 2025

pixel-wise stim.-response map 

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 
import tifffile
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm 
from scipy.stats import ttest_rel

import rec_list
paths = rec_list.pathdLightLCOpto + rec_list.pathdLightLCOptoCtrl + rec_list.pathdLightLCOptoInh


#%% parameters 
SAMP_FREQ = 30
BEF = 2 
AFT = 10  # in seconds 

TAXIS = np.arange(-BEF*SAMP_FREQ, AFT*SAMP_FREQ) / SAMP_FREQ

BASELINE_IDX = (TAXIS >= -1.0) & (TAXIS <= -.15)

STIM_IDX = (TAXIS >= 1.15) & (TAXIS < 2.0)


#%% path stems
all_sess_stem = Path('Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto/all_sessions')
tmap_stem = Path('Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto/t_maps')


#%% main 
for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    whether_ctrl = '_ctrl' if path in rec_list.pathdLightLCOptoCtrl else ''
    savepath = all_sess_stem / f'{recname}{whether_ctrl}'
        
    # if (savepath / 'processed_data' / f'{recname}_tmap.npy').exists():
    #     print(f'{recname} has been processed... skipped')
    #     continue
    
    aligned_F = np.load(
        savepath / 'processed_data' / f'{recname}_pixel_F_aligned.npy'
        )
    ref = np.load(
        savepath / 'processed_data' / f'{recname}_ref_mat_ch1.npy'
        )
    ref2 = np.load(
        savepath / 'processed_data' / f'{recname}_ref_mat_ch2.npy'
        )
    
    # extract activity from defined periods
    baseline = aligned_F[:, BASELINE_IDX, :, :]  # shape: (pulses, time, y, x)
    stim = aligned_F[:, STIM_IDX, :, :]
    
    # average within each time window
    baseline_mean = np.nanmean(baseline, axis=1)  # shape: (pulses, y, x)
    stim_mean = np.nanmean(stim, axis=1)
    
    # perform paired t-test per pixel
    tmap = np.zeros_like(stim_mean[0])  # shape: (y, x)
    
    for i in range(tmap.shape[0]):
        for j in range(tmap.shape[1]):
            stim_vals = stim_mean[:, i, j]
            base_vals = baseline_mean[:, i, j]
    
            # only do t-test if not all nan
            if np.isfinite(stim_vals).all() and np.isfinite(base_vals).all():
                tmap[i, j], _ = ttest_rel(stim_vals, base_vals)
            else:
                tmap[i, j] = np.nan
                
    # tmap plotting 
    vmin = min(np.nanmin(tmap), -.001)
    vmax = max(np.nanmax(tmap), .001)
    
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    im = ax.imshow(tmap, cmap='RdBu_r', norm=norm, interpolation='none')
    cbar = plt.colorbar(im, shrink=.5, label='t statistic')
    cbar.set_ticks([vmin, 0, vmax])
    
    plt.axis('off')
    plt.tight_layout()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            tmap_stem / f'{recname}_tmap{ext}',
            dpi=300,
            bbox_inches='tight'
        )
        
    cmap = cm.get_cmap('RdBu_r')

    rgba_tmap = cmap(norm(tmap))  # shape (y,x,4), floats in 0–1
    
    # convert to 8-bit RGB for TIFF
    rgb_tmap = (rgba_tmap[..., :3] * 255).astype(np.uint8)
            
    tifffile.imwrite(tmap_stem / f'{recname}_tmap.tiff', rgb_tmap)
    
    # save the tmap 
    np.save(savepath / 'processed_data' / f'{recname}_tmap.npy', tmap)