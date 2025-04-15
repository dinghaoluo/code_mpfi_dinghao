# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:33:23 2025

pixel-wise stim.-response map 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_rel, t
import sys 
import os 

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathdLightLCOpto + rec_list.pathdLightLCOptoCtrl


#%% parameters 
SAMP_FREQ = 30
BEF = 2 
AFT = 10  # in seconds 
TAXIS = np.arange(-BEF*SAMP_FREQ, AFT*SAMP_FREQ) / SAMP_FREQ

BASELINE_IDX = (TAXIS >= -1.0) & (TAXIS <= -.15)
STIM_IDX = (TAXIS >= 1.15) & (TAXIS < 2.0)


#%% main 
for path in paths:
    recname = path.split('\\')[-1]
    print(f'\n{recname}')
    
    whether_ctrl = '_ctrl' if path in rec_list.pathdLightLCOptoCtrl else ''
    savepath = os.path.join(
        r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions',
        f'{recname}{whether_ctrl}'
        )
    
    tmappath = r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\t_maps'
    
    if os.path.exists(rf'{tmappath}\{recname}_tmap.png'):
        print(f'{recname} has been processed... skipped')
        continue
    
    aligned_dFF_pix = np.load(
        rf'{savepath}\processed_data\{recname}_pixelwise_dFF.npy'
        )
    ref = np.load(
        rf'{savepath}\processed_data\{recname}_ref_mat_ch1.npy'
        )
    ref2 = np.load(
        rf'{savepath}\processed_data\{recname}_ref_mat_ch2.npy'
        )
    
    # extract activity from defined periods
    baseline = aligned_dFF_pix[:, BASELINE_IDX, :, :]  # shape: (pulses, time, y, x)
    stim = aligned_dFF_pix[:, STIM_IDX, :, :]
    
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
    
    n_trials = stim_mean.shape[0]
    p_thresh = 0.001
    t_thresh = t.ppf(1 - p_thresh, df=n_trials - 1)  # one-tailed
    
    sig_mask = tmap > t_thresh  # only positive, one-sided            
    
    # plotting 
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    ax.imshow(ref, cmap='gray', interpolation='none')
    
    overlay = np.zeros((*sig_mask.shape, 4))  # RGBA
    overlay[..., 0] = 1.0  # red
    overlay[..., 3] = sig_mask * 0.2  # alpha transparency where sig
    
    ax.imshow(overlay, interpolation='none')
    ax.set(xlim=(0, 512), ylim=(0, 512))
    plt.axis('off')
    plt.tight_layout()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'{tmappath}\masked\{recname}_tmap_ch1{ext}',
            dpi=300,
            bbox_inches='tight'
            )
    
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    ax.imshow(ref2, cmap='gray', interpolation='none')
    
    overlay = np.zeros((*sig_mask.shape, 4))  # RGBA
    overlay[..., 0] = 1.0  # red
    overlay[..., 3] = sig_mask * 0.2  # alpha transparency where sig
    
    ax.imshow(overlay, interpolation='none')
    ax.set(xlim=(0, 512), ylim=(0, 512))
    plt.axis('off')
    plt.tight_layout()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'{tmappath}\masked\{recname}_tmap_ch2{ext}',
            dpi=300,
            bbox_inches='tight'
            )
    
    # tmap plotting 
    vmin = np.nanmin(tmap)
    vmax = np.nanmax(tmap)
    
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    
    fig, ax = plt.subplots(figsize=(4, 3.5))
    
    im = ax.imshow(tmap, cmap='RdBu_r', norm=norm, interpolation='none')
    cbar = plt.colorbar(im, shrink=.5, label='t value')
    cbar.set_ticks([vmin, 0, vmax])
    
    ax.set(xlim=(0, 512), ylim=(0, 512))
    plt.axis('off')
    plt.tight_layout()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'{tmappath}\{recname}_tmap{ext}',
            dpi=300,
            bbox_inches='tight'
        )
        
    plt.close('all')