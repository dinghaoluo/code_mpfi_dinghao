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
from scipy.ndimage import binary_dilation

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
save_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\dilation_analysis')


#%% main
dilation_results = {d: [] for d in range(6)}  # store per-session traces
EDGE = 6

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    pixel_dFF_bins_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_dFF_bins.npy'
    roi_dict_path = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'

    if not pixel_dFF_bins_path.exists() or not roi_dict_path.exists():
        print('missing arrays, skipped')
        continue

    pixel_dFF_bins = np.load(pixel_dFF_bins_path, allow_pickle=True)
    roi_dict = np.load(roi_dict_path, allow_pickle=True).item()

    # build ROI mask (all ROIs)
    roi_mask = build_roi_mask(roi_dict)

    # dilations 0–5
    for dilation in range(6):
        if dilation > 0:
            struct = np.ones((3, 3), dtype=bool)
            dilated_mask = binary_dilation(roi_mask, structure=struct, iterations=dilation)
        else:
            dilated_mask = roi_mask

        roi_vals_binned = []
        for tbin in range(20):
            roi_vals_binned.append(np.nanmean(pixel_dFF_bins[:, :, tbin][dilated_mask]))

        dilation_results[dilation].append(roi_vals_binned)


#%% reorganise per-time-bin values
# structure: per_timebin[tbin][dilation] = values (sessions, filtered >= 0)
per_timebin = {tbin: {} for tbin in range(20)}
for dilation, traces in dilation_results.items():
    traces = np.array(traces)  # shape: (n_sessions, n_timebins)
    for tbin in range(traces.shape[1]):
        vals = traces[:, tbin]
        vals = vals[vals >= 0]  # remove all points < 0
        per_timebin[tbin][dilation] = vals


#%% plot boxplots for all 20 time bins
xaxis = np.arange(20) / 5  # 0.2 s bins

fig, axes = plt.subplots(4, 5, figsize=(12, 8), sharey=True)
axes = axes.ravel()

for tbin, ax in enumerate(axes):
    # gather 6 dilation levels for this time bin
    data_for_box = [per_timebin[tbin][d] for d in range(6)]
    bp = ax.boxplot(data_for_box,
                    positions=np.arange(6),
                    widths=0.6,
                    patch_artist=True)

    # color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title(f'{xaxis[tbin]:.1f} s', fontsize=8)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels([f'{d}' for d in range(6)], fontsize=6)
    if tbin % 5 == 0:
        ax.set_ylabel('dLight dF/F')

fig.suptitle('Effect of ROI dilation on dLight signal (all time bins)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
# fig.savefig(save_stem / 'dilation_boxplots.png', dpi=300)
# fig.savefig(save_stem / 'dilation_boxplots.pdf', dpi=300)


#%% save results
np.save(save_stem / 'dilation_results.npy', dilation_results, allow_pickle=True)
print(f'\nSaved results + boxplots to {save_stem}')
