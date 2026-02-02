# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:05:43 2026

Compare the response indices of ROIs against those of the neuropil

@author: Dinghao Luo
"""


#%% imports
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon 
from scipy.ndimage import binary_dilation

from plotting_functions import plot_violin_with_scatter
from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathdLightLCOpto


#%% helper
def _build_roi_mask(roi_dict):
    """build combined mask from all ROIs"""
    mask = np.zeros((512, 512), dtype=bool)
    for roi_id, roi in roi_dict.items():
        mask[roi['ypix'], roi['xpix']] = True
    return mask


#%% path stems
all_sess_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions')
save_stem     = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\ROI_vs_neuropil')


#%% parameters
# how far away from ROI to count as neuropil 
DISTANCE_FROM_ROI = 9  # 9 pixels ~ 5 um

ALPHA  = 0.05
MIN_RI = 0.1


#%% main
# initialise containers
all_ROI_RIs      = []
all_neuropil_RIs = []


# loop
for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    pixel_RI_bin_path  = all_sess_stem / recname / f'processed_data/{recname}_pixel_RI_bins.npy'
    pixel_RI_stim_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_RI_stim.npy'
    roi_dict_path      = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'

    if not pixel_RI_bin_path.exists():
        print('No pixel_RI_bin; skipped')
        continue
    if not pixel_RI_stim_path.exists():
        print('No pixel_RI_stim; skipped')
        continue
    if not roi_dict_path.exists():
        print('No roi_dict; skipped')
        continue

    # load data
    print('Loading data...')
    pixel_RI_bin  = np.load(pixel_RI_bin_path, allow_pickle=True)
    pixel_RI_stim = np.load(pixel_RI_stim_path, allow_pickle=True)
    roi_dict      = np.load(roi_dict_path, allow_pickle=True).item()

    # ---- identify releasing ROIs ----
    releasing = {}
    
    for rid, roi in roi_dict.items():
        vals  = pixel_RI_stim[roi['ypix'], roi['xpix'], :]
        means = np.nanmean(vals, axis=0)  # mean over pixels 
        means = [mean for mean in means if np.isfinite(mean)]  # filtering first 
        if len(means) > 2:
            _, p = wilcoxon(means, alternative='greater')
            if p < ALPHA and np.mean(means) > MIN_RI:
                releasing[rid] = roi
    
    if len(releasing) == 0:
        print('No releasing ROI; skipped')
        continue 
    # ---- identification ends ----

    # build mask of all ROIs
    releasing_mask = _build_roi_mask(releasing)
    
    # build dilated mask and then the anti-mask 
    ROI_mask = _build_roi_mask(roi_dict)
    ROI_dilated = binary_dilation(ROI_mask, iterations=DISTANCE_FROM_ROI)
    anti_ROI_mask = ~ROI_dilated 
    
    # get med of pixel_RI_stim 
    pixel_RI_stim_med = np.nanmedian(pixel_RI_stim, axis=2)
    
    # get ROI and neuropil RI medians
    ROI_RI_med      = np.nanmedian(pixel_RI_stim_med[releasing_mask])
    neuropil_RI_med = np.nanmedian(pixel_RI_stim_med[anti_ROI_mask])
    
    # append 
    all_ROI_RIs.append(ROI_RI_med)
    all_neuropil_RIs.append(neuropil_RI_med)
    
    # sanity check
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    im0 = ax[0].imshow(pixel_RI_stim_med, cmap='viridis')
    ax[0].set_title(f'{recname}\npixel RI')
    ax[0].axis('off')
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    
    ax[1].imshow(ROI_mask, cmap='Reds')
    ax[1].set_title('ROI mask')
    ax[1].axis('off')
    
    ax[2].imshow(anti_ROI_mask, cmap='Blues')
    ax[2].set_title(f'neuropil (>{DISTANCE_FROM_ROI}px from ROI)')
    ax[2].axis('off')
    
    plt.tight_layout()
    
    fig.savefig(save_stem / f'{recname}.png',
                dpi=300,
                bbox_inches='tight')
    
    plt.close(fig)
    

#%% statistics 
plot_violin_with_scatter(all_neuropil_RIs, all_ROI_RIs, 
                         'grey', 'darkgreen',
                         xticklabels=['Neuropil', 'ROI'],
                         ylabel='RI',
                         save=True,
                         print_statistics=True,
                         savepath=save_stem / 'neuropil_vs_ROI_violin')