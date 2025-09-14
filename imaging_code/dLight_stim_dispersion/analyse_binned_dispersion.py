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

import rec_list
paths = rec_list.pathdLightLCOpto


#%% parameters 
EDGE = 6  # pixels (to remove)


#%% path stems 
all_sess_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions')


#%% main
recname = 'A146i-20250910-01'
pixel_dFF_bins = np.load(
    all_sess_stem / recname / f'processed_data/{recname}_pixel_dFF_bins.npy', 
    allow_pickle=True
    )
pixel_dFF_bins = pixel_dFF_bins[EDGE:-EDGE, EDGE:-EDGE, :]

slices = np.arange(20)
global_min = pixel_dFF_bins[:, :, slices].min()
global_max = pixel_dFF_bins[:, :, slices].max()

fig, axes = plt.subplots(4, 5 , figsize=(10, 8))
axes = axes.ravel()
norm = TwoSlopeNorm(vcenter=0, vmin=global_min, vmax=global_max)

for ax, idx in zip(axes, slices):
    im = ax.imshow(pixel_dFF_bins[:, :, idx],
                   norm=norm, cmap='RdBu_r')
    ax.set_title(f'stim. + {round(idx*.2, 1)} s')
    ax.axis('off')
    
fig.suptitle(recname)

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.25)

for ext in ['.png', '.pdf']:
    fig.savefig(
        all_sess_stem / recname / f'{recname}_pixel_dFF_bins{ext}',
        dpi=300,
        bbox_inches='tight'
        )