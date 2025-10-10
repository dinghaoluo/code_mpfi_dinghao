# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 2025

systematic ROI dilation analysis for dLight LC opto
    builds ROI mask directly from ROI_dict,
    dilates it (1–5 px), and
    compares dLight signal distributions with boxplots
    across time bins (0–4 s)

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import ttest_1samp, friedmanchisquare, wilcoxon 
from scipy.ndimage import binary_dilation

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathdLightLCOpto


#%% helper
def build_roi_mask(roi_dict):
    """build combined mask from all ROIs"""
    mask = np.zeros((512, 512), dtype=bool)
    for roi_id, roi in roi_dict.items():
        mask[roi['ypix'], roi['xpix']] = True
    return mask


#%% path stems
all_sess_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions')
save_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\dispersion_analysis')


#%% parameters
STRUCT = np.ones((3, 3), dtype=bool)
DILATE_STEP = 2
MAX_DILATE = 10
EDGE = 6

WINDOW_BINS = {
    '0-1s': range(0, 10),
    '1-2s': range(10, 20),
    '2-3s': range(20, 30),
    '3-4s': range(30, 40),
}


#%% main
# initialise containers
dilation_results_all = {w: {} for w in WINDOW_BINS.keys()}

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    pixel_dFF_bins_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_dFF_bins.npy'
    roi_dict_path = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'

    if not pixel_dFF_bins_path.exists() or not roi_dict_path.exists():
        print('missing arrays, skipped')
        continue

    pixel_dFF_bins = np.load(pixel_dFF_bins_path, allow_pickle=True)  # (512,512,40)
    roi_dict = np.load(roi_dict_path, allow_pickle=True).item()

    # identify releasing ROIs (using whole 0–4 s median just as before)
    pixel_dFF_med = np.median(pixel_dFF_bins, axis=2)
    releasing_rois = {}
    for roi_id, roi in roi_dict.items():
        roi_vals = pixel_dFF_med[roi['ypix'], roi['xpix']]
        if np.all(np.isfinite(roi_vals)) and len(roi_vals) > 2:
            _, p_val = ttest_1samp(roi_vals, popmean=0, alternative='greater')
            if p_val < 0.05:
                releasing_rois[roi_id] = roi

    # per ROI, per window
    for roi_id, roi in releasing_rois.items():
        base_mask = np.zeros((512, 512), dtype=bool)
        base_mask[roi['ypix'], roi['xpix']] = True

        dilated_masks = {}
        ring_masks = {}
        for dilation in range(0, MAX_DILATE+1, DILATE_STEP):
            if dilation == 0:
                dilated_masks[0] = base_mask
                ring_masks[0] = base_mask
            else:
                dilated_masks[dilation] = binary_dilation(base_mask, structure=STRUCT, iterations=dilation)
                ring_masks[dilation] = dilated_masks[dilation] & ~dilated_masks[dilation-DILATE_STEP]

            # now extract per-window medians
            for wname, bins in WINDOW_BINS.items():
                vals = pixel_dFF_bins[:, :, bins][ring_masks[dilation]]  # shape: (#pixels × #bins)
                val_median = np.median(vals)  # collapse across pixels & time bins

                if roi_id not in dilation_results_all[wname]:
                    dilation_results_all[wname][roi_id] = {}
                if dilation not in dilation_results_all[wname][roi_id]:
                    dilation_results_all[wname][roi_id][dilation] = []

                dilation_results_all[wname][roi_id][dilation].append(val_median)
                
        # plot ring masks 
        fig, axes = plt.subplots(1, len(ring_masks), figsize=(12, 2))
        if len(ring_masks) == 1:
            axes = [axes]
        for j, (d, mask) in enumerate(sorted(ring_masks.items())):
            axes[j].imshow(mask.astype(int), cmap='gray')
            axes[j].set_title(f'{d}px', fontsize=8)
            axes[j].axis('off')
        fig.suptitle(f'{recname} – ROI {roi_id}', fontsize=10)
        plt.tight_layout()

        savepath = save_stem / 'all_ring_masks' / f'{recname}_ROI{roi_id}_rings'
        for ext in ['.png', '.pdf']:
            fig.savefig(f'{savepath}{ext}', dpi=200)
        plt.close(fig)


#%% statistics + plotting
for wname, results in dilation_results_all.items():
    print(f'\nCurrent window: {wname}')

    # collapse to medians per ROI × dilation
    roi_meds = {}
    for roi_id, ring_dLight in results.items():
        dvals = {d: np.median(vals) for d, vals in ring_dLight.items()}
        if all(np.isfinite(val) for val in dvals.values()):
            roi_meds[roi_id] = dvals

    # collect per-dilation values
    per_dilation = {d: [] for d in sorted(next(iter(roi_meds.values())).keys())}
    for roi_id, dvals in roi_meds.items():
        for d, val in dvals.items():
            per_dilation[d].append(val)

    dilations = sorted(per_dilation.keys())
    data = [np.array(per_dilation[d]).ravel() for d in dilations]

    # Friedman test across dilations
    vals_matrix = np.array([[dvals[d] for d in dilations] for dvals in roi_meds.values()])
    friedman_stat, friedman_p = friedmanchisquare(
        *[vals_matrix[:, j] for j in range(vals_matrix.shape[1])]
    )
    print(f'Friedman test across dilations ({wname}): χ²={friedman_stat:.2f}, p={friedman_p:.3e}')

    # pairwise Wilcoxon tests vs base (0 px)
    pvals = {}
    for j, d in enumerate(dilations[1:], start=1):
        stat, p = wilcoxon(vals_matrix[:,0], vals_matrix[:,j], alternative='greater')
        pvals[d] = p
        print(f'd=0 vs d={d} ({wname}): Wilcoxon W={stat}, p={p:.3e}')

    # plotting 
    fig, ax = plt.subplots(figsize=(4,3))
    parts = ax.violinplot(data, positions=dilations, widths=1.2,
                          showmeans=False, showextrema=False)

    # style violins
    for pc in parts['bodies']:
        pc.set_facecolor('darkgreen')
        pc.set_edgecolor('none')
        pc.set_alpha(0.6)

    # thin ROI trajectories
    for roi_id, dvals in roi_meds.items():
        xs = dilations
        ys = [dvals[d] for d in xs]
        ax.plot(xs, ys, color='gray', alpha=0.08, linewidth=0.8)
        ax.scatter(xs, ys, color='black', s=1, alpha=0.07)
        
    # cosmetics
    ax.set_xticks(dilations)
    ax.set_xlabel('dilation (px)')
    ax.set_ylabel('median dLight (a.u.)')
    ax.set_ylim((-.05, .8))
    ax.set_title(f'ROI-median dLight per dilation ring ({wname})\n'
                 f'Friedman χ²={friedman_stat:.2f}, p={friedman_p:.3e}')
    
    for s in ['top', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
    
    # annotate
    y_max = max([max(vals) for vals in data if len(vals) > 0])
    y_step = 0.02
    for i, d in enumerate(dilations[1:], start=1):
        p = pvals[d]
        y = y_max + i*y_step
        if p < 0.0001:
            star = '****'
        elif p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'
        else:
            star = 'n.s.'
        ax.text((dilations[0]+d)/2, y + 0.01*y_max, star,
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(save_stem / f'dilation_analysis_{wname}{ext}',
                    dpi=300,
                    bbox_inches='tight')

    
#%% inspect example ring masks 
example_ids = list(releasing_rois.keys())[-10:]

fig, axes = plt.subplots(len(example_ids), MAX_DILATE+1, figsize=(3*(MAX_DILATE+1), 3*len(example_ids)))

if len(example_ids) == 1:
    axes = [axes]

for r, roi_id in enumerate(example_ids):
    for d, dilation in enumerate(range(0, MAX_DILATE+1, DILATE_STEP)):
        ax = axes[r][d] if len(example_ids) > 1 else axes[d]
        
        #---change this line to change the mask type
        mask_img = ring_masks[dilation].astype(int)
        
        ax.imshow(mask_img, cmap='gray')
        ax.set_title(f'ROI {roi_id}, d={dilation}')
        ax.axis('off')

plt.tight_layout()
plt.show()