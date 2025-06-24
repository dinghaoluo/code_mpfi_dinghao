# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:36:30 2025

Analyse t-map signal inside vs. outside manually curated fibre ROIs.

@author: Dinghao Luo
"""

#%% imports 
import numpy as np
import os
import sys 
import matplotlib.pyplot as plt 
from scipy.stats import ttest_1samp
from matplotlib.colors import TwoSlopeNorm

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
from plotting_functions import plot_violin_with_scatter
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list

paths = rec_list.pathdLightLCOpto


#%% main 
roi_means = []
bg_means = []

for path in paths:
    recname = os.path.basename(path)
    print(f'analysing {recname}...')

    proc_dir = os.path.join(
        r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions',
        recname,
        'processed_data'
    )
    tmap_path = os.path.join(proc_dir, f'{recname}_tmap.npy')
    roi_dict_path = os.path.join(proc_dir, f'{recname}_ROI_dict.npy')

    if not (os.path.exists(tmap_path) and os.path.exists(roi_dict_path)):
        print(f'missing data for {recname} — skipped')
        continue

    tmap = np.load(tmap_path)
    roi_dict = np.load(roi_dict_path, allow_pickle=True).item()

    releasing_rois = {}

    # build ROI mask and collect releasing ROIs
    roi_mask = np.zeros_like(tmap, dtype=bool)
    releasing_mask = np.zeros_like(tmap, dtype=bool)

    for roi_id, roi in roi_dict.items():
        roi_mask[roi['ypix'], roi['xpix']] = True

        roi_vals = tmap[roi['ypix'], roi['xpix']]
        if np.all(np.isfinite(roi_vals)) and len(roi_vals) > 2:
            t_stat, p_val = ttest_1samp(roi_vals, popmean=0, alternative='greater')
            if p_val < 0.05:
                releasing_rois[roi_id] = roi
                releasing_mask[roi['ypix'], roi['xpix']] = True

    bg_mask = ~roi_mask & np.isfinite(tmap)

    roi_vals = tmap[roi_mask]
    bg_vals = tmap[bg_mask]

    roi_mean = np.nanmean(roi_vals)
    bg_mean = np.nanmean(bg_vals)

    roi_means.append(roi_mean)
    bg_means.append(bg_mean)

    # plot 2×2
    fig, axs = plt.subplots(2, 2, figsize=(6, 5))

    # top-left: all ROIs
    roi_overlay = np.zeros((*roi_mask.shape, 4))
    roi_overlay[..., :3] = 1.0  # white
    roi_overlay[..., 3] = roi_mask * 1.0
    axs[0, 0].imshow(np.zeros_like(roi_mask), cmap='gray')
    axs[0, 0].imshow(roi_overlay, interpolation='none')
    axs[0, 0].set_title('All ROIs')
    axs[0, 0].axis('off')

    # bottom-left: releasing ROIs
    releasing_overlay = np.zeros((*roi_mask.shape, 4))
    releasing_overlay[..., :3] = 1.0  # white
    releasing_overlay[..., 3] = releasing_mask * 1.0
    axs[1, 0].imshow(np.zeros_like(releasing_mask), cmap='gray')
    axs[1, 0].imshow(releasing_overlay, interpolation='none')
    axs[1, 0].set_title('Releasing ROIs')
    axs[1, 0].axis('off')

    # top-right: full t-map
    vmin, vmax = np.nanmin(tmap), np.nanmax(tmap)
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    im1 = axs[0, 1].imshow(tmap, cmap='RdBu_r', norm=norm, interpolation='none')
    axs[0, 1].set_title('t-map')
    axs[0, 1].axis('off')
    fig.colorbar(im1, ax=axs[0, 1], shrink=0.6, label='t-value')

    # bottom-right: t > 0 only
    tmap_thresh = np.where(tmap > 0, tmap, np.nan)
    im2 = axs[1, 1].imshow(tmap_thresh, cmap='Reds', interpolation='none')
    axs[1, 1].set_title('t > 0 only')
    axs[1, 1].axis('off')
    fig.colorbar(im2, ax=axs[1, 1], shrink=0.6, label='t-value')

    fig.suptitle(recname)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\correlation_analysis\tmap_x_release\{recname}_maps{ext}',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()
    
    releasing_roi_dict_path = os.path.join(proc_dir, f'{recname}_releasing_ROI_dict.npy')
    np.save(releasing_roi_dict_path, releasing_rois)
        
plot_violin_with_scatter(bg_means, roi_means, 
                         'grey', 'darkgreen',
                         xticklabels=['background', 'ROI'],
                         ylabel='stim. v baseline DA',
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\correlation_analysis\tmap_x_release\summary_violinplot')