# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:56:31 2025

Quantify the release probability of each axonal ROI

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path 

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ttest_1samp
from skimage.measure import find_contours 
from tqdm import tqdm 

from common import mpl_formatting
mpl_formatting()

import rec_list

paths = rec_list.pathdLightLCOpto + \
        rec_list.pathdLightLCOptoDbhBlock


#%% parameters 
ALPHA = 0.05
EDGE  = 6  # pixels 

N_SHUF = 100

SAMP_FREQ = 30  # Hz 
BEF       = 2  # s 
AFT       = 10
XAXIS     = np.arange((BEF + AFT) * SAMP_FREQ) / SAMP_FREQ - BEF  # for plotting 

baseline_idx = np.where((XAXIS >= -1.0)  & (XAXIS <= -.15))[0]


#%% path stems 
all_sess_stem = Path('Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto/all_sessions')
mice_exp_stem = Path('Z:/Dinghao/MiceExp')

        
#%% main 
all_release_probabilities = []

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    pixel_RI_path  = all_sess_stem / recname / 'processed_data' / f'{recname}_pixel_RI_stim.npy'
    roi_dict_path  = all_sess_stem / recname / 'processed_data' / f'{recname}_ROI_dict.npy'
    ref2_path      = all_sess_stem / recname / 'processed_data' / f'{recname}_ref_mat_ch2.npy'
    ref1100_path   = all_sess_stem / recname / 'processed_data' / f'{recname}_ref_mat_1100nm.npy'

    if not pixel_RI_path.exists():
        print('No pixel_dFF_bins; skipped')
        continue
    if not roi_dict_path.exists():
        print('No roi_dict; skipped')
        continue
    if not ref2_path.exists():
        print('No ref2; skipped')
        continue

    # load data
    print('Loading data...')
    pixel_RI = np.load(pixel_RI_path, allow_pickle=True)  # (512,512,40)
    roi_dict  = np.load(roi_dict_path, allow_pickle=True).item()
    
    # load 1100-nm ref. or ref2 (we used 1100-nm references for very early recordings)
    if ref1100_path.exists():
        print('Using 1100-nm ref.')
        ref2 = np.load(ref1100_path, allow_pickle=True)
    else:
        print('Using channel 2 ref.')
        ref2 = np.load(ref2_path, allow_pickle=True)
    
    # identify releasing ROIs
    print(f'Identifying releasing ROIs with alpha={ALPHA}...')
    pixel_RI_med = np.nanmedian(pixel_RI, axis=2)  # get median values of pixel dFF
    releasing_rois = {}
    for roi_id, roi in roi_dict.items():
        roi_vals = pixel_RI_med[roi['ypix'], roi['xpix']]
        if np.all(np.isfinite(roi_vals)) and len(roi_vals) > 2:
            _, p_val = ttest_1samp(roi_vals, popmean=0, alternative='greater')
            if p_val < ALPHA:
                releasing_rois[roi_id] = roi
    
    # get min and max 
    global_min = np.nanmin(pixel_RI_med[EDGE:-EDGE, EDGE:-EDGE])
    global_max = np.nanmax(pixel_RI_med[EDGE:-EDGE, EDGE:-EDGE])
    
    # plot pixel dFF map
    fig, ax = plt.subplots(figsize=(3,3))
    norm = TwoSlopeNorm(vcenter=0, vmin=global_min, vmax=global_max)
    
    im = ax.imshow(pixel_RI_med, norm=norm, cmap='RdBu_r')
    
    for roi_id, roi in roi_dict.items():
        if roi_id in releasing_rois:
            ax.scatter(roi['ypix'], roi['xpix'], s=.4, color='darkred', edgecolor='none', alpha=.1)
        else:
            ax.scatter(roi['ypix'], roi['xpix'], s=.4, color='grey', edgecolor='none', alpha=.1)
    
    handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='darkred',
                   markersize=4, label='Releasing'),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='grey',
                   markersize=4, label='Non-releasing'),
    ]
    ax.legend(handles=handles,
              loc='center left',
              bbox_to_anchor=(1.0, 0.9),
              frameon=False,
              fontsize=6,
              handletextpad=0.3)
    
    ax.set(title=recname)
    ax.axis('off')
    
    cbar = fig.colorbar(im, shrink=0.25, ticks=[global_min, 0, global_max])
    
    fig.tight_layout()
    
    out_path = all_sess_stem / recname / f'{recname}_releasing_ROI_map'
    for ext in ['.png', '.pdf']:
        fig.savefig(
            f'{out_path}{ext}',
            dpi=300,
            bbox_inches='tight'
        )
    
    
    # ----------------------------
    # single-ROI release analysis
    # ----------------------------
    print('Loading F_aligned array...')
    F_aligned_path  = all_sess_stem / recname / f'processed_data/{recname}_pixel_F_aligned.npy'    
    if not F_aligned_path.exists():
        print('Missing aligned array; skipped')
        continue
    F_aligned  = np.load(F_aligned_path, allow_pickle=True)
    
    # session containers
    sess_release_probabilities = []
    
    # iterate 
    save_stem = all_sess_stem / recname / 'single_ROI_stim_aligned'
    save_stem.mkdir(exist_ok=True)
    for roi_id, roi in tqdm(roi_dict.items(),
                            desc='Plotting single axons',
                            total=len(roi_dict)):
        # get traces 
        curr_F_pixel_traces = F_aligned[:, :, roi['ypix'], roi['xpix']]
        curr_F_traces = np.nanmean(curr_F_pixel_traces, axis=2)
        
        # determine response index 
        mean_trace = np.nanmean(curr_F_traces, axis=0)
        nan_mask = np.isnan(mean_trace)
        nan_indices = np.where(nan_mask)[0]
        last_nan = nan_indices[-1]
        response_start = (last_nan + 1 - BEF * SAMP_FREQ) / SAMP_FREQ
        response_idx = np.where((XAXIS >= response_start) & (XAXIS <= response_start + .85))[0]
 
        n_trials = curr_F_traces.shape[0]
 
        RIs = np.zeros(n_trials) 
        for tr in range(n_trials):
            trace = curr_F_traces[tr, :]
 
            base_val = np.nanmean(trace[baseline_idx])
            resp_val = np.nanmean(trace[response_idx])
 
            RIs[tr] = (resp_val - base_val) / np.abs(base_val)
            
        # release probability 
        release_trials = np.sum(RIs > 0.01)  # 1% dFF threshold
        release_prob = release_trials / n_trials
        
        if not np.isnan(release_prob):
            sess_release_probabilities.append(release_prob)
        
        
        # --------------------
        # single-ROI plotting
        # --------------------
        curr_F_mean = np.nanmean(curr_F_traces, axis=0)
        
        fig = plt.figure(figsize=(7.5, 2.2))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2], wspace=0.45)
        axs = [
            fig.add_subplot(gs[0]),
            fig.add_subplot(gs[1]),
            fig.add_subplot(gs[2]),
        ]
        
        # Determine ROI-centred crop size (auto-scaling)
        cy = int(np.mean(roi['ypix']))
        cx = int(np.mean(roi['xpix']))
        
        xmin, xmax = np.min(roi['xpix']), np.max(roi['xpix'])
        ymin, ymax = np.min(roi['ypix']), np.max(roi['ypix'])
        
        roi_extent = max(xmax - xmin, ymax - ymin)
        win = int((roi_extent / 2) + roi_extent * 0.5)  # 50% padding
        win = np.clip(win, 30, 200)
        
        # crop bounds
        y0, y1 = max(0, cy - win), min(pixel_RI_med.shape[0], cy + win)
        x0, x1 = max(0, cx - win), min(pixel_RI_med.shape[1], cx + win)
        
        # Extract local field-of-view (channel 1 dFF + channel 2 ref)
        local_img_dFF = pixel_RI_med[y0:y1, x0:x1]
        local_img_ref2 = ref2[y0:y1, x0:x1]
        
        # build ROI mask from cropped coords
        mask = np.zeros_like(local_img_dFF, dtype=np.uint8)
        local_x = roi['xpix'] - x0
        local_y = roi['ypix'] - y0
        valid = (
            (local_x >= 0) & (local_x < mask.shape[1]) &
            (local_y >= 0) & (local_y < mask.shape[0])
        )
        mask[local_y[valid], local_x[valid]] = 1
        
        # extract outline once
        contours = find_contours(mask, 0.5)
        
        axs[0].imshow(local_img_dFF, cmap='RdBu_r', norm=norm, interpolation='none')
        for c in contours:
            axs[0].plot(c[:, 1], c[:, 0],
                        color='darkgreen',
                        linestyle='--',
                        linewidth=0.8)
        axs[0].axis('off')
        
        axs[1].imshow(local_img_ref2, cmap='gray', interpolation='none')
        for c in contours:
            axs[1].plot(c[:, 1], c[:, 0],
                        color='red',
                        linestyle='--',
                        linewidth=0.7)
        axs[1].axis('off')
        
        axs[2].plot(XAXIS, curr_F_traces.T,
                    color='darkgreen', alpha=0.02)
        axs[2].plot(XAXIS, curr_F_mean,
                    color='darkgreen')
        
        axs[2].text(
            0.02, 0.98,
            f'Release prob = {release_prob:.2f}',
            transform=axs[2].transAxes,
            ha='left', va='top',
            fontsize=7,
            color='firebrick'
        )
        
        axs[2].set(
            xlabel='Time from stim. onset (s)',
            xticks=[0, 5, 10],
            ylabel='F'
        )
        for s in ['top', 'right']:
            axs[2].spines[s].set_visible(False)
        
        fig.suptitle(f'{recname} ROI {roi_id}')
        
        for ext in ['.pdf', '.png']:
            fig.savefig(
                save_stem / f'ROI_{roi_id}{ext}', 
                dpi=300, bbox_inches='tight'
            )
        
        plt.close(fig)
        # -------------------------
        # single-ROI plotting ends
        # -------------------------
    # ---------------------------------
    # single-ROI release analysis ends
    # ---------------------------------
    
    
    ## ---- session summary 
    if not sess_release_probabilities:
        print('No valid ROI for session.')
        continue
    
    all_release_probabilities.extend(sess_release_probabilities)
    
    fig, ax = plt.subplots(figsize=(2.4, 2.1))
    ax.hist(sess_release_probabilities,
            bins=np.linspace(0, 1, 11),
            color='darkgreen',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5)

    # median line
    med = np.median(sess_release_probabilities)
    ax.axvline(med, color='firebrick', linestyle='--', linewidth=1)
    ax.text(
        0.98, 0.98,
        f'median = {med:.2f}',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=7,
        color='firebrick'
    )

    ax.set(
        xlabel='Release probability',
        ylabel='Number of ROIs',
        title=f'{recname}'
    )
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    fig.tight_layout()

    # save
    summary_path = all_sess_stem / recname / f'{recname}_release_prob_hist'
    for ext in ['.png', '.pdf']:
        fig.savefig(
            f'{summary_path}{ext}',
            dpi=300,
            bbox_inches='tight'
        )
    plt.close(fig)
    
    
#%% summary
fig, ax = plt.subplots(figsize=(2.4, 2.1))
ax.hist(all_release_probabilities,
        bins=np.linspace(0,1,21),
        color='darkgreen', edgecolor='black', linewidth=0.4)

med = np.median(all_release_probabilities)
ax.axvline(med, color='firebrick', linestyle='--')
ax.text(0.98, 0.98, f'median = {med:.2f}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=7, color='firebrick')

for ext in ['.pdf', '.png']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\dLight_LC_stim_release_probabilities{ext}',
        dpi=300, bbox_inches='tight')