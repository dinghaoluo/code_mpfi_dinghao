# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 18:07:16 2025

analyse the dispersion of dLight signal after stim.
    dependent on extraction with HPC_dLight_LC_opto_extract.py

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathdLightLCOpto


#%% parameters 
EDGE = 6  # pixels (to remove)
ALPHA = 0.05


#%% helper 
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C


#%% path stems 
all_sess_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions')
save_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\dispersion_analysis')


#%% main
for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    pixel_RI_bins_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_RI_bins.npy'

    if not pixel_RI_bins_path.exists():
        print('No pixel_RI_bins; skipped')
        continue

    # load data
    print('Loading data...')
    pixel_RI_bins = np.load(pixel_RI_bins_path, allow_pickle=True)  # (512,512,40)    
    
    # smooth data 
    pixel_RI_bins = gaussian_filter(pixel_RI_bins, sigma=(0, 1, 1))
    
    # binning
    bins = np.arange(pixel_RI_bins.shape[-1])
    
    vmin = np.nanpercentile(np.nanmean(pixel_RI_bins[:, :, :10], axis=2), 1)
    vmax = np.nanpercentile(np.nanmean(pixel_RI_bins[:, :, :10], axis=2), 99)
    
    # edge case check
    if vmin >= 0:
        vmin = -.001
    if vmax <= 0:
        vmax = .001

    # plot binned pixel RI maps
    print('Plotting whole-field dispersion plots...')
    fig, axes = plt.subplots(5, 8 , figsize=(14, 10))
    axes = axes.ravel()
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    
    for ax, idx in zip(axes, bins):
        im = ax.imshow(pixel_RI_bins[:, :, idx],
                       norm=norm, cmap='RdBu_r')
        ax.set_title(f'Stim. + {round(idx*.1, 1)} s')
        ax.axis('off')
    
    fig.suptitle(recname)
    
    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.25,
        ticks=[vmin, 0, vmax]
    )
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            save_stem / 'all_sessions' / f'{recname}_pixel_RI_bins{ext}',
            dpi=300,
            bbox_inches='tight'
        )
    plt.close(fig)