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
from tqdm import tqdm  
from scipy.stats import ttest_1samp, wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.ndimage import binary_dilation

from common_functions import mpl_formatting
mpl_formatting()

from common.mask.utils_mask import generate_adaptive_membrane_mask

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
save_stem     = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\dispersion_analysis')


#%% parameters
STRUCT = np.ones((3, 3), dtype=bool)
DILATE_STEP = 1
MAX_DILATE = 20
EDGE = 6

ALPHA  = 0.05
MIN_RI = 0.1

WINDOW_BINS = {
    '0-1 s':   range(0, 10),
    '1-2 s':   range(10, 20),
    '2-3 s':   range(20, 30),
    '3-4 s':   range(30, 40),
}

# for visualisation 
CROP_PAD = MAX_DILATE + 2


#%% main
# initialise containers
dilation_results_all = {w: {} for w in WINDOW_BINS.keys()}

for path in paths:
    recname = Path(path).name
    
    if 'A114' in recname or 'A116' in recname:
        continue 
    
    print(f'\n{recname}')

    ref_ch1_path       = all_sess_stem / recname / f'processed_data/{recname}_ref_mat_ch1.npy'
    ref_ch2_path       = all_sess_stem / recname / f'processed_data/{recname}_ref_mat_ch2.npy'
    pixel_RI_path      = all_sess_stem / recname / f'processed_data/{recname}_pixel_RI_bins.npy'
    pixel_RI_stim_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_RI_stim.npy'
    roi_dict_path      = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'

    if not ref_ch1_path.exists():
        print('No mean image for ch1; skipped')
        continue
    if not ref_ch2_path.exists():
        print('No mean image for ch2; skipped')
        continue
    if not pixel_RI_path.exists():
        print('No pixel_RI_bins; skipped')
        continue
    if not pixel_RI_stim_path.exists():
        print('No pixel_dFF_stim; skipped')
        continue
    if not roi_dict_path.exists():
        print('No roi_dict; skipped')
        continue

    # load data
    print('Loading data...')
    ref_ch1       = np.load(ref_ch1_path, allow_pickle=True)
    ref_ch2       = np.load(ref_ch2_path, allow_pickle=True)
    pixel_RI_bins = np.load(pixel_RI_path, allow_pickle=True)  # (512,512,40)
    pixel_RI_stim = np.load(pixel_RI_stim_path, allow_pickle=True)
    roi_dict      = np.load(roi_dict_path, allow_pickle=True).item()
    
    # ---- identify releasing ROIs ----
    releasing_rois = {}

    for rid, roi in roi_dict.items():
        vals  = pixel_RI_stim[roi['ypix'], roi['xpix'], :]
        means = np.nanmean(vals, axis=0)  # mean over pixels 
        means = [mean for mean in means if np.isfinite(mean)]  # filtering first 
        if len(means) > 2:
            _, p = ttest_1samp(means, 0, alternative='greater')
            if p < ALPHA and np.mean(means) > MIN_RI:
                releasing_rois[rid] = roi

    if len(releasing_rois) == 0:
        print('No releasing ROI; skipped')
        continue 
    # ---- identification ends ----
    
    # threshold based on dLight expression, 26 Mar 2026
    thres_mask = generate_adaptive_membrane_mask(ref_ch1, visualize=0)

    # per ROI, per window
    # first collect all pixel idx of all releasing ROIs (for collision exclusion)
    all_roi_y = [y for roi in releasing_rois.values() for y in roi['ypix']]
    all_roi_x = [x for roi in releasing_rois.values() for x in roi['xpix']]
    all_roi_mask = np.zeros((512, 512), dtype=bool)
    all_roi_mask[all_roi_y, all_roi_x] = True

    CROP_PAD = MAX_DILATE + 8

    # now loop over all the rois with collision exclusion
    for roi_id, roi in tqdm(releasing_rois.items(),
                            total=len(releasing_rois),
                            desc='Looping over ROIs'):
        base_mask = np.zeros((512, 512), dtype=bool)
        base_mask[roi['ypix'], roi['xpix']] = True

        dilated_masks = {}
        ring_masks = {}

        for dilation in range(0, MAX_DILATE + 1, DILATE_STEP):
            if dilation == 0:
                dilated_masks[0] = base_mask
                ring_masks[0] = base_mask & thres_mask
            else:
                dm = binary_dilation(base_mask, structure=STRUCT, iterations=dilation)

                # collision exclusion
                dm = dm & ~all_roi_mask

                dilated_masks[dilation] = dm
                ring_masks[dilation] = dm & ~dilated_masks[dilation - DILATE_STEP]
                ring_masks[dilation] = ring_masks[dilation] & thres_mask

            # now extract per-window means from thresholded ring only
            for wname, bins in WINDOW_BINS.items():
                vals = pixel_RI_bins[:, :, bins][ring_masks[dilation]]
                val_mean = np.nanmean(vals)

                if roi_id not in dilation_results_all[wname]:
                    dilation_results_all[wname][roi_id] = {}
                if dilation not in dilation_results_all[wname][roi_id]:
                    dilation_results_all[wname][roi_id][dilation] = []

                dilation_results_all[wname][roi_id][dilation] = val_mean

        # plot ring masks on cropped ch2 background
        sorted_items = sorted(ring_masks.items())

        row1 = sorted_items[:10]
        row2 = sorted_items[10:]
        ncols = max(len(row1), len(row2))

        fig, axes = plt.subplots(2, ncols, figsize=(1.6 * ncols, 3.4))

        if ncols == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        # fixed crop box per ROI, based on original ROI mask
        ypix = np.array(roi['ypix'])
        xpix = np.array(roi['xpix'])

        y0 = max(0, np.min(ypix) - CROP_PAD)
        y1 = min(ref_ch2.shape[0], np.max(ypix) + CROP_PAD + 1)
        x0 = max(0, np.min(xpix) - CROP_PAD)
        x1 = min(ref_ch2.shape[1], np.max(xpix) + CROP_PAD + 1)

        ref_ch2_crop = ref_ch2[y0:y1, x0:x1]

        # first row
        for j, (d, mask) in enumerate(row1):
            ax = axes[0, j]
            mask_crop = mask[y0:y1, x0:x1]

            ax.imshow(ref_ch2_crop, cmap='gray', interpolation='none')

            overlay = np.zeros((*mask_crop.shape, 4), dtype=float)
            overlay[mask_crop, 0] = 0.0
            overlay[mask_crop, 1] = 0.39215686
            overlay[mask_crop, 2] = 0.0
            overlay[mask_crop, 3] = 1.0

            ax.imshow(overlay, interpolation='none')
            ax.set_title(f'{d}px', fontsize=8)
            ax.axis('off')

        # second row
        for j, (d, mask) in enumerate(row2):
            ax = axes[1, j]
            mask_crop = mask[y0:y1, x0:x1]

            ax.imshow(ref_ch2_crop, cmap='gray', interpolation='none')

            overlay = np.zeros((*mask_crop.shape, 4), dtype=float)
            overlay[mask_crop, 0] = 0.0
            overlay[mask_crop, 1] = 0.39215686
            overlay[mask_crop, 2] = 0.0
            overlay[mask_crop, 3] = 1.0

            ax.imshow(overlay, interpolation='none')
            ax.set_title(f'{d}px', fontsize=8)
            ax.axis('off')

        # hide unused axes
        for j in range(len(row1), ncols):
            axes[0, j].axis('off')
        for j in range(len(row2), ncols):
            axes[1, j].axis('off')

        fig.suptitle(f'{recname} – ROI {roi_id}', fontsize=10)
        plt.tight_layout()

        savepath = save_stem / 'all_ring_masks' / f'{recname}_ROI{roi_id}_rings'
        savepath.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['.png', '.pdf']:
            fig.savefig(f'{savepath}{ext}', dpi=550)
        plt.close(fig)


#%% plotting with Jingyu's median plot
# global y range for plotting 
global_ymin, global_ymax = -0.05, 0.5

for wname, results in dilation_results_all.items():
    print(f'\nCurrent window: {wname}')

    # build ROI × dilation matrix
    roi_meds = {}
    for roi_id, ring_dLight in results.items():
        if all(np.isfinite(val) for val in ring_dLight.values()):
            roi_meds[roi_id] = ring_dLight

    if not roi_meds:
        print(f'No valid ROIs for {wname}, skipped.')
        continue

    dilations = sorted(next(iter(roi_meds.values())).keys())
    roi_ids = list(roi_meds.keys())
    roi_map = np.array([[roi_meds[roi_id][d] for d in dilations] for roi_id in roi_ids])
    groups = [roi_map[:, i][~np.isnan(roi_map[:, i])] for i in range(len(dilations))]

    # all pairwise comparisons
    pvals = {}  # key: (d1, d2)
    
    for i, d1 in enumerate(dilations):
        for j, d2 in enumerate(dilations):
            if j <= i:
                continue  # avoid duplicates & self-comparisons
    
            vals1 = roi_map[:, i]
            vals2 = roi_map[:, j]
    
            common_mask = np.isfinite(vals1) & np.isfinite(vals2)
            common_n = np.sum(common_mask)
    
            if common_n > 5:
                stat, p = wilcoxon(vals1[common_mask], vals2[common_mask])
                test_used = 'Wilcoxon'
            else:
                stat, p = mannwhitneyu(vals1, vals2)
                test_used = 'Mann–Whitney'
    
            pvals[(d1, d2)] = (p, test_used, common_n)

    # FDR correction
    if pvals:
        p_raw = np.array([v[0] for v in pvals.values()])
        reject, p_corr, _, _ = multipletests(p_raw, method='fdr_bh')
    else:
        reject, p_corr = np.array([]), np.array([])

    # store paired stats 
    stats_out = {}
    for ((d1, d2), (p, test_used, n)), pc, rej in zip(pvals.items(), p_corr, reject):
        line = f'{d1}px vs {d2}px: p={p:.2e}, p_corr={pc:.2e}'
        print(line)
        stats_out[(d1, d2)] = {
            'text': line,
            'rej': bool(rej)
        }

    # plotting
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=300)
    bp = ax.boxplot(groups, labels=dilations, patch_artist=True, showfliers=False)

    # Jingyu style
    for box in bp['boxes']:
        box.set(facecolor='lightblue', edgecolor='none', alpha=0.7)
    for median in bp['medians']:
        median.set(color='teal', linewidth=1.5)
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.5)

    # jittered points overlay
    rng = np.random.default_rng(0)
    for i, vals in enumerate(groups, start=1):
        if len(vals) == 0:
            continue
        x = rng.normal(i, 0.02, size=len(vals))
        ax.plot(x, vals, '.', alpha=0.2, markersize=2, color='slategrey')

    # cosmetics
    ax.spines[['top', 'right']].set_visible(False)
    ax.set(xlabel='Dilation (px)', ylabel='Mean dLight (a.u.)')
    ax.set_ylim(global_ymin, global_ymax)

    # bracket annotation: all significant pairs, stacked by span
    yrange = global_ymax - global_ymin
    tick_h = 0.02 * yrange
    level_gap = 0.05 * yrange
    
    xpos = {lab: i for i, lab in enumerate(dilations, start=1)}
    ymax_per_group = [np.nanmax(v) if len(v) else np.nan for v in groups]
    
    # collect all pairs 
    all_pairs = [
        (d1, d2, stats)
        for (d1, d2), stats in stats_out.items()
    ]
    
    # sort by span (short → long)
    all_pairs.sort(key=lambda x: xpos[x[1]] - xpos[x[0]])
    
    current_level = 0
    for d1, d2, stats in all_pairs:
        x1, x2 = xpos[d1], xpos[d2]
    
        involved_max = np.nanmax([
            ymax_per_group[x1 - 1],
            ymax_per_group[x2 - 1]
        ])
    
        y = involved_max + (current_level + 1) * level_gap
        
        # style switch
        is_sig = stats['rej']
        color = 'black' if is_sig else '0.6'
        lw = 1.0 if is_sig else 0.6
        alpha = 1.0 if is_sig else 0.6
        
        # bracket
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + tick_h, y + tick_h, y],
            color=color,
            lw=lw,
            alpha=alpha,
            clip_on=False
        )
        
        ax.text(
            (x1 + x2) / 2,
            y + tick_h * 1.3,
            stats['text'],
            ha='center',
            va='bottom',
            fontsize=6,
            color=color,
            alpha=alpha
        )
        current_level += 1
    
    # expand ylim to fit all brackets
    ax.set_ylim(
        global_ymin,
        global_ymax + (current_level + 2) * level_gap
    )