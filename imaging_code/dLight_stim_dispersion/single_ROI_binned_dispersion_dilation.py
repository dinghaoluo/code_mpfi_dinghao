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
from scipy.stats import ttest_1samp, friedmanchisquare, wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
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

ALPHA = 0.05

WINDOW_BINS = {
    '0-0.5 s': range(0, 5),
    '0.5-1 s': range(5, 10),
    '1-1.5 s': range(10, 15),
    '1.5-2 s': range(15, 20),
    '2-2.5 s': range(20, 25),
    '2.5-3 s': range(25, 30),
    '3-3.5 s': range(30, 35),
    '3.5-4 s': range(35, 40)
}


#%% main
# initialise containers
dilation_results_all = {w: {} for w in WINDOW_BINS.keys()}

for path in paths[:20]:
    recname = Path(path).name
    print(f'\n{recname}')

    pixel_dFF_bins_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_dFF_bins.npy'
    roi_dict_path = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'

    if not pixel_dFF_bins_path.exists():
        print('No pixel_dFF_bins; skipped')
        continue
    if not roi_dict_path.exists():
        print('No roi_dict; skipped')
        continue

    # load data
    print('loading data...')
    pixel_dFF_bins = np.load(pixel_dFF_bins_path, allow_pickle=True)  # (512,512,40)
    roi_dict = np.load(roi_dict_path, allow_pickle=True).item()
    
    # identify releasing ROIs (using whole 0–4 s median)
    print(f'identifying releasing ROIs with alpha={ALPHA}...')
    pixel_dFF_med = np.median(pixel_dFF_bins, axis=2)
    releasing_rois = {}
    for roi_id, roi in roi_dict.items():
        roi_vals = pixel_dFF_med[roi['ypix'], roi['xpix']]
        if np.all(np.isfinite(roi_vals)) and len(roi_vals) > 2:
            _, p_val = ttest_1samp(roi_vals, popmean=0, alternative='greater')
            if p_val < ALPHA:
                releasing_rois[roi_id] = roi

    # per ROI, per window
    # first we collect all pixel idx of all ROIs (for collision exclusion)
    all_roi_y = [y for roi in releasing_rois.values() for y in roi['ypix']]
    all_roi_x = [x for roi in releasing_rois.values() for x in roi['xpix']]
    all_roi_mask = np.zeros((512, 512), dtype=bool)
    all_roi_mask[all_roi_y, all_roi_x] = True
    
    # now we loop over all the rois with collision exclusion
    print(f'looping over {len(releasing_rois)} ROIs...')
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
                dilated_masks[dilation] = dilated_masks[dilation] & ~all_roi_mask  # collision exclusion
                
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
# for wname, results in dilation_results_all.items():
#     print(f'\nCurrent window: {wname}')

#     # collapse to medians per ROI × dilation
#     roi_meds = {}
#     for roi_id, ring_dLight in results.items():
#         dvals = {d: np.median(vals) for d, vals in ring_dLight.items()}
#         if all(np.isfinite(val) for val in dvals.values()):
#             roi_meds[roi_id] = dvals

#     # collect per-dilation values
#     per_dilation = {d: [] for d in sorted(next(iter(roi_meds.values())).keys())}
#     for roi_id, dvals in roi_meds.items():
#         for d, val in dvals.items():
#             per_dilation[d].append(val)

#     dilations = sorted(per_dilation.keys())
#     data = [np.array(per_dilation[d]).ravel() for d in dilations]

#     # Friedman test across dilations
#     vals_matrix = np.array([[dvals[d] for d in dilations] for dvals in roi_meds.values()])
#     friedman_stat, friedman_p = friedmanchisquare(
#         *[vals_matrix[:, j] for j in range(vals_matrix.shape[1])]
#     )
#     print(f'Friedman test across dilations ({wname}): χ²={friedman_stat:.2f}, p={friedman_p:.3e}')

#     # pairwise Wilcoxon tests vs base (0 px)
#     pvals = {}
#     for j, d in enumerate(dilations[1:], start=1):
#         stat, p = wilcoxon(vals_matrix[:,0], vals_matrix[:,j], alternative='greater')
#         pvals[d] = p
#         print(f'd=0 vs d={d} ({wname}): Wilcoxon W={stat}, p={p:.3e}')

#     # plotting 
#     fig, ax = plt.subplots(figsize=(4,3))
#     parts = ax.violinplot(data, positions=dilations, widths=1.2,
#                           showmeans=False, showextrema=False)

#     # style violins
#     for pc in parts['bodies']:
#         pc.set_facecolor('darkgreen')
#         pc.set_edgecolor('none')
#         pc.set_alpha(0.6)

#     # thin ROI trajectories
#     for roi_id, dvals in roi_meds.items():
#         xs = dilations
#         ys = [dvals[d] for d in xs]
#         ax.plot(xs, ys, color='gray', alpha=0.08, linewidth=0.8)
#         ax.scatter(xs, ys, color='black', s=1, alpha=0.07)
        
#     # cosmetics
#     ax.set_xticks(dilations)
#     ax.set_xlabel('dilation (px)')
#     ax.set_ylabel('median dLight (a.u.)')
#     ax.set_ylim((-.05, .8))
#     ax.set_title(f'ROI-median dLight per dilation ring ({wname})\n'
#                  f'Friedman χ²={friedman_stat:.2f}, p={friedman_p:.3e}')
    
#     for s in ['top', 'right', 'bottom']:
#         ax.spines[s].set_visible(False)
    
#     # annotate
#     y_max = max([max(vals) for vals in data if len(vals) > 0])
#     y_step = 0.02
#     for i, d in enumerate(dilations[1:], start=1):
#         p = pvals[d]
#         y = y_max + i*y_step
#         if p < 0.0001:
#             star = '****'
#         elif p < 0.001:
#             star = '***'
#         elif p < 0.01:
#             star = '**'
#         elif p < 0.05:
#             star = '*'
#         else:
#             star = 'n.s.'
#         ax.text((dilations[0]+d)/2, y + 0.01*y_max, star,
#                 ha='center', va='bottom', fontsize=10)

#     plt.tight_layout()
#     plt.show()
    
#     for ext in ['.png', '.pdf']:
#         fig.savefig(save_stem / f'dilation_analysis_{wname}{ext}',
#                     dpi=300,
#                     bbox_inches='tight')

    
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


#%% plotting with Jingyu's median plot
# global y range for plotting 
global_ymin, global_ymax = -0.05, 0.5

for wname, results in dilation_results_all.items():
    print(f'\nCurrent window: {wname}')

    # build ROI × dilation matrix
    roi_meds = {}
    for roi_id, ring_dLight in results.items():
        dvals = {d: np.median(vals) for d, vals in ring_dLight.items()}
        if all(np.isfinite(val) for val in dvals.values()):
            roi_meds[roi_id] = dvals

    if not roi_meds:
        print(f'No valid ROIs for {wname}, skipped.')
        continue

    dilations = sorted(next(iter(roi_meds.values())).keys())
    roi_ids = list(roi_meds.keys())
    roi_map = np.array([[roi_meds[roi_id][d] for d in dilations] for roi_id in roi_ids])
    groups = [roi_map[:, i][~np.isnan(roi_map[:, i])] for i in range(len(dilations))]

    # pairwise tests vs base
    pvals = {}
    for i, d in enumerate(dilations):
        if d == 0:
            continue
        vals0 = roi_map[:, 0]
        valsk = roi_map[:, i]
        common_mask = np.isfinite(vals0) & np.isfinite(valsk)
        common_n = np.sum(common_mask)
        if common_n > 5:
            stat, p = wilcoxon(vals0[common_mask], valsk[common_mask], alternative='greater')
            test_used = 'Wilcoxon'
        else:
            stat, p = mannwhitneyu(vals0, valsk, alternative='greater')
            test_used = 'Mann–Whitney'
        pvals[d] = (p, test_used, common_n)

    # FDR correction
    if len(pvals) > 0:
        p_raw = np.array([p for p, _, _ in pvals.values()])
        reject, p_corr, _, _ = multipletests(p_raw, method='fdr_bh')
    else:
        reject, p_corr = np.array([]), np.array([])

    stats_out = {}
    for (d, (p, test_used, n)), pc, rej in zip(pvals.items(), p_corr, reject):
        line = f'dilation {d} vs 0: {test_used} p={p:.3e}, p_corr={pc:.3e}, n={n}'
        print(line)
        stats_out[d] = {'text': line, 'rej': bool(rej)}


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
    ax.set(xlabel='dilation (px)', ylabel='median dLight (a.u.)')
    ax.set_ylim(global_ymin, global_ymax)

    # bracket annotation inline
    yrange = global_ymax - global_ymin
    base_gap = 0.1 * yrange
    tick_h = 0.03 * yrange
    level_gap = 0.08 * yrange
    ymax_per_group = [np.nanmax(v) if len(v) else np.nan for v in groups]
    xpos = {lab: i for i, lab in enumerate(dilations, start=1)}

    current_level = 0
    for d in dilations:
        if d == 0 or d not in stats_out:
            continue
        x1, x2 = xpos[0], xpos[d]
        involved_max = np.nanmax([ymax_per_group[x1-1], ymax_per_group[x2-1]])
        y = global_ymax + current_level * level_gap

        # draw bracket
        ax.plot([x1, x1, x2, x2], [y, y+tick_h, y+tick_h, y], color='black', linewidth=1.0, clip_on=False)
        ax.text((x1+x2)/2, y + tick_h * 3, stats_out[d]['text'],
                ha='center', va='top', fontsize=6.5, color='black')

        current_level += 1

    ax.set_ylim(global_ymin, global_ymax + base_gap + (current_level+1)*level_gap + tick_h)
    plt.tight_layout()

    for ext in ['.png', '.pdf']:
        fig.savefig(save_stem / f'dilation_boxplot_FDRbracket_{wname}{ext}',
                    dpi=300, bbox_inches='tight')
    plt.show()