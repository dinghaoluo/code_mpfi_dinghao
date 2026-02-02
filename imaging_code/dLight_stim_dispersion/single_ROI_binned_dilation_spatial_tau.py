# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 2025
Modified on Wed 28 Jan 2026

systematic ROI dilation analysis for dLight LC opto
    builds ROI mask directly from ROI_dict,
    dilates it (1–5 px), and
    compares dLight signal distributions with boxplots
    across time bins (0–4 s)
Modified to calculate spatial tau based on tighter spatial binning 

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_1samp
from scipy.ndimage import binary_dilation
from scipy.optimize import curve_fit

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
save_stem     = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\dispersion_analysis\dilation_1px_spatial_tau')


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


#%% main
# initialise containers
dilation_results_all = {w: {} for w in WINDOW_BINS.keys()}
tau_results_all      = {w: {} for w in WINDOW_BINS.keys()}

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    pixel_RI_path      = all_sess_stem / recname / f'processed_data/{recname}_pixel_RI_bins.npy'
    pixel_RI_stim_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_RI_stim.npy'
    roi_dict_path      = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'

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
    
    # all ROIs
    all_ROI_mask = _build_roi_mask(roi_dict)
    
    # grand releasing-ROIs
    releasing_mask = _build_roi_mask(releasing_rois)
    
    # actual dilation 
    dilated_masks = {}
    ring_masks = {}
    
    for dilation in range(0, MAX_DILATE + 1, DILATE_STEP):
        if dilation == 0:
            dilated_masks[0] = releasing_mask
            ring_masks[0] = releasing_mask
        else:
            dm = binary_dilation(releasing_mask, structure=STRUCT, iterations=dilation)
    
            # collision exclusion against *all* rois (including releasing ones)
            dm = dm & ~all_ROI_mask
    
            dilated_masks[dilation] = dm
            ring_masks[dilation] = dm & ~dilated_masks[dilation - DILATE_STEP]
    
        # per-window mean
        for wname, bins in WINDOW_BINS.items():
            vals = pixel_RI_bins[:, :, bins][ring_masks[dilation]]
            val_mean = np.nanmean(vals)
    
            if recname not in dilation_results_all[wname]:
                dilation_results_all[wname][recname] = {}
    
            dilation_results_all[wname][recname][dilation] = val_mean
    
    # plot a subset of masks for view
    plot_dilations = list(range(0, MAX_DILATE + 1, 2))  # or 5

    fig, axes = plt.subplots(1, len(plot_dilations), figsize=(12, 2))
    for j, d in enumerate(plot_dilations):
        axes[j].imshow(ring_masks[d].astype(int), cmap='gray')
        axes[j].set_title(f'{d}px', fontsize=8)
        axes[j].axis('off')
    
    fig.suptitle(f'{recname} – grand releasing rings', fontsize=10)
    plt.tight_layout()
    
    savepath = save_stem / 'all_ring_masks' / f'{recname}_grand_releasing_rings'
    savepath.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['.png', '.pdf']:
        fig.savefig(f'{savepath}{ext}', dpi=200)
    plt.close(fig)


#%% tau estimation pass 
def _exp_decay(d, A, tau):
    return A * np.exp(-d / tau)

for wname in WINDOW_BINS.keys():
    for recname, curve in dilation_results_all[wname].items():
        dilations = np.array(sorted(curve.keys()))
        R = np.array([curve[d] for d in dilations])

        # define C as min over 0..MAX_DILATE (your rule)
        C = np.nanmin(R)
        R0 = R - C

        # validity
        valid = np.isfinite(R0) & (R0 >= 0)
        d_fit = dilations[valid]
        y_fit = R0[valid]

        if len(y_fit) >= 6 and np.nanmax(y_fit) > 0:
            try:
                popt, _ = curve_fit(
                    _exp_decay,
                    d_fit,
                    y_fit,
                    bounds=([0, 0.5], [np.inf, 50])
                )
                tau_hat = float(popt[1])
            except Exception:
                tau_hat = np.nan
        else:
            tau_hat = np.nan

        tau_results_all[wname][recname] = tau_hat

    
#%% histogram of spatial tau per time window
hist_stem = save_stem / 'tau_histograms'
hist_stem.mkdir(parents=True, exist_ok=True)

TAU_MAX = 50
BIN_WIDTH = 0.5
bins = np.arange(0, TAU_MAX + BIN_WIDTH, BIN_WIDTH)

for wname in WINDOW_BINS.keys():
    taus = tau_results_all[wname]
    vals = np.array([v for v in taus.values() if np.isfinite(v)])

    if len(vals) == 0:
        print(f'{wname}: no valid taus, skipped')
        continue

    fig, ax = plt.subplots(figsize=(2.4, 2.1))

    ax.hist(
        vals,
        bins=bins,
        color='lightblue',
        edgecolor='none',
        linewidth=0.4
    )

    med = np.median(vals)
    ax.axvline(med, color='teal', linestyle='--')
    ax.text(
        0.9, 0.98,
        f'Median = {med:.2f}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=7,
        color='teal'
    )

    ax.spines[['top', 'right']].set_visible(False)
    ax.set(
        xlabel=r'Spatial $\tau$ (px)',
        xlim=(-1,21),
        ylabel='Session count',
        title=wname
    )

    for ext in ['.pdf', '.png']:
        fig.savefig(
            hist_stem / f'tau_hist_{wname.replace(" ", "").replace("-", "")}{ext}',
            dpi=300,
            bbox_inches='tight'
        )


#%% median radial decay curves with IQR
lineplot_stem = save_stem / 'mean_radial_curves'
lineplot_stem.mkdir(parents=True, exist_ok=True)

for wname, results in dilation_results_all.items():
    print(f'\nCurrent window: {wname}')

    if not results:
        print(f'No data for {wname}, skipped.')
        continue

    recnames = list(results.keys())
    dilations = sorted(next(iter(results.values())).keys())

    mat = np.full((len(recnames), len(dilations)), np.nan)
    for i, rec in enumerate(recnames):
        for j, d in enumerate(dilations):
            mat[i, j] = results[rec].get(d, np.nan)

    # mean and SEM across sessions
    mean_curve = np.nanmean(mat, axis=0)
    sem_curve  = sem(mat, axis=0, nan_policy='omit')

    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=300)

    ax.errorbar(
        dilations,
        mean_curve,
        yerr=sem_curve,
        fmt='_',              # short horizontal line for the mean
        color='teal',
        ecolor='lightblue',
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        markersize=6
    )

    tau_vals = np.array([
        v for v in tau_results_all[wname].values()
        if np.isfinite(v)
    ])

    # label taus
    tau_median = np.median(tau_vals)

    ax.axvline(
        tau_median,
        color='lightblue',
        linestyle='--',
        linewidth=1.2,
        alpha=0.9
    )

    ax.text(
        tau_median,
        max(mean_curve) * 1.1,
        rf'$\tau$={round(tau_median, 2)}',
        ha='center',
        va='top',
        fontsize=8,
        color='lightblue'
    )
    
    ax.spines[['top', 'right']].set_visible(False)
    ax.set(
        xlabel='Dilation (px)',
        xlim=(-1,21),
        ylabel='Mean dLight RI',
        title=wname
    )

    # annotate n (sessions contributing at any dilation)
    n_sess = np.sum(np.any(np.isfinite(mat), axis=1))
    ax.text(
        0.98, 0.98,
        f'n = {n_sess}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=7
    )

    for ext in ['.pdf', '.png']:
        fig.savefig(
            lineplot_stem / f'mean_radial_curve_{wname.replace(" ", "").replace("-", "")}{ext}',
            dpi=300,
            bbox_inches='tight'
        )