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
from scipy.stats import ttest_1samp, sem
from scipy.optimize import curve_fit

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathdLightLCOpto


#%% parameters 
ALPHA    = 0.05
R2_THRES = 0.7
N_BINS   = 40
XAXIS    = np.arange(N_BINS) / 10

NUM_ATTEMPTS = 100


#%% helper 
def exp_decay_fixed(t, A, tau, t0, B):
    return A * np.exp(-(t - t0) / tau) + B


#%% path stems 
all_sess_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions')
save_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\dispersion_analysis')


#%% main
all_roi_vals_binned = []
all_shuff_vals_binned = []
taus_real = []          # per-ROI tau values per session
taus_real_all = []      # tau values across all sessions

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    pixel_RI_path      = all_sess_stem / recname / f'processed_data/{recname}_pixel_RI_bins.npy'
    pixel_RI_stim_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_RI_stim.npy'
    roi_path           = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'
    
    if not pixel_RI_path.exists():
        print('No pixel_RI array; skipped')
        continue 
    if not pixel_RI_stim_path.exists():
        print('No pixel_RI_stim array; skipped')
        continue 
    if not roi_path.exists():
        print('No roi_dict; skipped')
        continue

    pixel_RI_bins = np.load(pixel_RI_path, allow_pickle=True)
    pixel_RI_stim = np.load(pixel_RI_stim_path, allow_pickle=True)
    roi_dict      = np.load(roi_path, allow_pickle=True).item()

    H, W, _ = pixel_RI_bins.shape

    # ---- identify releasing ROIs ----
    pixel_med = np.nanmedian(pixel_RI_stim, axis=2)
    releasing = {}

    for rid, roi in roi_dict.items():
        vals = pixel_med[roi['ypix'], roi['xpix']]
        if np.all(np.isfinite(vals)) and len(vals) > 2:
            _, p = ttest_1samp(vals, 0, alternative='greater')
            if p < ALPHA:
                releasing[rid] = roi

    if len(releasing) == 0:
        print('No releasing ROI; skipped')
        continue 

    # original mask to avoid colllision
    orig_mask = np.zeros((H, W), bool)
    for r in releasing.values():
        orig_mask[r['ypix'], r['xpix']] = True

    # main loop
    for rid, roi in releasing.items():
        ypix = roi['ypix']
        xpix = roi['xpix']

        # real trace
        real_trace = np.array([
            np.nanmean(pixel_RI_bins[:, :, tb][ypix, xpix])
            for tb in range(N_BINS)
            ])
        if not np.all(np.isfinite(real_trace)):
            continue
        
        # shuffled trace 
        roi_h = ypix.max() - ypix.min() + 1
        roi_w = xpix.max() - xpix.min() + 1

        norm_y = ypix - ypix.min()
        norm_x = xpix - xpix.min()

        max_y0 = H - roi_h
        max_x0 = W - roi_w
        
        best_overlap = np.inf
        best_coords = None

        for _ in range(NUM_ATTEMPTS):
            y0 = np.random.randint(0, max_y0 + 1)
            x0 = np.random.randint(0, max_x0 + 1)

            new_y = norm_y + y0
            new_x = norm_x + x0

            overlap = np.sum(orig_mask[new_y, new_x])

            if overlap == 0:
                best_coords = (new_y, new_x)
                break

            if overlap < best_overlap:
                best_overlap = overlap
                best_coords = (new_y, new_x)

        new_y, new_x = best_coords

        shuffled_trace = np.array([
            np.nanmean(pixel_RI_bins[:, :, tb][new_y, new_x])
            for tb in range(N_BINS)
            ])
        
        # fit tau 
        first_valid = np.where(np.isfinite(real_trace))[0][0]
        t0 = XAXIS[first_valid]      # no need for peak index if unnecessary
        B  = real_trace.min()
        
        t_fit = XAXIS[first_valid:]
        y_fit = real_trace[first_valid:]
        
        # nested function for curve_fit — only A and tau are free
        def fitfun(t, A, tau):
            return exp_decay_fixed(t, A, tau, t0, B)
        
        try:
            popt, _ = curve_fit(
                fitfun,
                t_fit,
                y_fit,
                p0=[real_trace[first_valid] - B, 1.0],
                bounds=([0, 0], [np.inf, np.inf])
            )
            A_fit, tau_fit = popt
            
            # predict fitted decay
            y_pred = fitfun(t_fit, A_fit, tau_fit)
            
            # compute R2
            ss_res = np.sum((y_fit - y_pred)**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # require R2 > thres
            if R2 < R2_THRES:
                continue
            
            taus_real_all.append(tau_fit)
        
        except RuntimeError:
            continue
        
        all_roi_vals_binned.append(real_trace)
        all_shuff_vals_binned.append(shuffled_trace)


#%% tau plot
real_mat = np.vstack(all_roi_vals_binned)      # shape: nROI × nTime
shuf_mat = np.vstack(all_shuff_vals_binned)

real_mean = np.nanmean(real_mat, axis=0)
real_sem  = sem(real_mat, axis=0, nan_policy='omit')

shuf_mean = np.nanmean(shuf_mat, axis=0)
shuf_sem  = sem(shuf_mat, axis=0, nan_policy='omit')

fig, ax = plt.subplots(figsize=(3, 2.4))

ax.plot(XAXIS, real_mean, color='darkgreen', label='LC axons')
ax.fill_between(
    XAXIS,
    real_mean - real_sem,
    real_mean + real_sem,
    color='darkgreen', alpha=.3, linewidth=0, edgecolor='none'
)

ax.plot(XAXIS, shuf_mean, color='grey', label='shuffled')
ax.fill_between(
    XAXIS,
    shuf_mean - shuf_sem,
    shuf_mean + shuf_sem,
    color='grey', alpha=.3, linewidth=0, edgecolor='none'
)

ax.set(
    xlabel='Time from stim.-offset (s)',
    ylabel='RI',
    title='LC axon dLight (per-ROI avg)'
)
ax.legend(frameon=False, fontsize=7)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        save_stem / f'RI_decay_real_shuff{ext}',
        dpi=300,
        bbox_inches='tight'
    )


#%% tau histogram
bins = np.linspace(0, 4, 21)

# use the correct list of taus
if len(taus_real_all) == 0:
    print('Warning: no tau values extracted — histogram empty!')
    taus_to_plot = []
else:
    taus_to_plot = taus_real_all

med_tau = np.median(taus_to_plot) if len(taus_to_plot) > 0 else np.nan

fig, ax = plt.subplots(figsize=(3, 2.4))

ax.hist(taus_to_plot, bins=bins,
        color='darkgreen', edgecolor='black')

ax.set(xlabel='Tau (s)',
       ylabel='ROI count',
       title='Distribution of decay constants')

if not np.isnan(med_tau):
    ax.axvline(med_tau, color='red', linestyle='--',
               label=f'Median = {med_tau:.2f}s')
    ax.legend(frameon=False, fontsize=7)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(save_stem / f'RI_decay_tau_hist{ext}',
                dpi=300,
                bbox_inches='tight')