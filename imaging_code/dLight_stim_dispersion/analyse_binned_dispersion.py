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
from scipy.stats import ttest_1samp, sem
from scipy.optimize import curve_fit

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathdLightLCOpto


#%% parameters 
EDGE = 6  # pixels (to remove)


#%% helper 
def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C


#%% path stems 
all_sess_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions')
save_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\dispersion_analysis')


#%% main
all_roi_vals_binned = []
all_shuff_roi_vals_binned = []

for path in paths:
    recname = Path(path).name

    pixel_dFF_bins_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_dFF_bins.npy'
    roi_dict_path = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'

    if not pixel_dFF_bins_path.exists() or not roi_dict_path.exists():
        print(f'\n{recname} does not have the required data arrays; skipped')
        continue
    else:
        print(f'\n{recname}')

    pixel_dFF_bins = np.load(pixel_dFF_bins_path, allow_pickle=True)
    pixel_dFF_bins_edge_removed = pixel_dFF_bins[EDGE:-EDGE, EDGE:-EDGE, :]
    
    slices = np.arange(20)
    
    global_min = pixel_dFF_bins_edge_removed[:, :, slices].min()
    global_max = pixel_dFF_bins_edge_removed[:, :, slices].max()
    
    fig, axes = plt.subplots(4, 5 , figsize=(10, 8))
    axes = axes.ravel()
    norm = TwoSlopeNorm(vcenter=0, vmin=global_min, vmax=global_max)
    
    for ax, idx in zip(axes, slices):
        im = ax.imshow(pixel_dFF_bins_edge_removed[:, :, idx],
                       norm=norm, cmap='RdBu_r')
        ax.set_title(f'stim. + {round(idx*.2, 1)} s')
        ax.axis('off')
    
    fig.suptitle(recname)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.25)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            save_stem / 'all_sessions' / f'{recname}_pixel_dFF_bins{ext}',
            dpi=300,
            bbox_inches='tight'
            )
    plt.close(fig)
        
    roi_dict = np.load(roi_dict_path, allow_pickle=True).item()
    
    # take the first bin as the peak release map to detect releasing ROIs
    releasing_rois = {}
    
    release_map_peak = pixel_dFF_bins[:, :, 0]
    roi_mask = np.zeros_like(release_map_peak, dtype=bool)
    releasing_mask = np.zeros_like(release_map_peak, dtype=bool)

    for roi_id, roi in roi_dict.items():
        roi_mask[roi['ypix'], roi['xpix']] = True

        roi_vals = releasing_mask[roi['ypix'], roi['xpix']]
        if np.all(np.isfinite(roi_vals)) and len(roi_vals) > 2:
            t_stat, p_val = ttest_1samp(roi_vals, popmean=0, alternative='greater')
            if p_val < 0.05:
                releasing_rois[roi_id] = roi
                releasing_mask[roi['ypix'], roi['xpix']] = True
    
    roi_vals_binned = []
    shuff_roi_vals_binned = []
    
    # compute true ROI traces
    for tbin in range(20):
        roi_vals_binned.append(np.nanmean(pixel_dFF_bins[:, :, tbin][roi_mask]))
    
    # compute shuffled ROI traces
    shuffled_mask = np.zeros_like(roi_mask, dtype=bool)
    for roi_id, roi in roi_dict.items():
        # get ROI pixel coordinates
        ypix = roi['ypix']
        xpix = roi['xpix']
        
        # pick a random shift within bounds (512 × 512, minus ROI size)
        max_y_shift = 512 - (ypix.max() - ypix.min()) - 1
        max_x_shift = 512 - (xpix.max() - xpix.min()) - 1
        
        y_shift = np.random.randint(0, max_y_shift)
        x_shift = np.random.randint(0, max_x_shift)
        
        new_y = ypix - ypix.min() + y_shift
        new_x = xpix - xpix.min() + x_shift
        
        shuffled_mask[new_y, new_x] = True
    
    for tbin in range(20):
        shuff_roi_vals_binned.append(
            np.nanmean(pixel_dFF_bins[:, :, tbin][shuffled_mask])
        )
    
    all_roi_vals_binned.append(roi_vals_binned)
    all_shuff_roi_vals_binned.append(shuff_roi_vals_binned)
    

#%% statistics 
roi_vals_mean_trace = np.mean(all_roi_vals_binned, axis=0)
shuff_roi_vals_mean_trace = np.mean(all_shuff_roi_vals_binned, axis=0)

roi_vals_sem_trace = sem(all_roi_vals_binned, axis=0)
shuff_roi_vals_sem_trace = sem(all_shuff_roi_vals_binned, axis=0)

xaxis = np.arange(20) / 5

fig, ax = plt.subplots(figsize=(3.1,2.4))

ax.plot(xaxis, roi_vals_mean_trace, c='darkgreen', label='LC axons')
ax.fill_between(xaxis, roi_vals_mean_trace+roi_vals_sem_trace,
                       roi_vals_mean_trace-roi_vals_sem_trace,
                color='darkgreen', edgecolor='none', alpha=.3)

ax.plot(xaxis, shuff_roi_vals_mean_trace, color='grey', label='shuf. LC axons')
ax.fill_between(xaxis, shuff_roi_vals_mean_trace+shuff_roi_vals_sem_trace,
                       shuff_roi_vals_mean_trace-shuff_roi_vals_sem_trace,
                color='grey', edgecolor='none', alpha=.3)

# fit exponetial for tau calculation 
fit_mask = xaxis >= xaxis[np.argmax(roi_vals_mean_trace)]  # only after peak
t_fit = xaxis[fit_mask]
y_fit = roi_vals_mean_trace[fit_mask]

# initial guesses: amplitude=max-min, tau=1s, offset=last point
p0 = [y_fit.max() - y_fit.min(), 1.0, y_fit.min()]

# fit
popt, pcov = curve_fit(exp_decay, t_fit, y_fit, p0=p0)
A_fit, tau_fit, C_fit = popt
tau_seconds = tau_fit  # this is your decay constant in seconds

print(f'Estimated tau (decay constant): {tau_seconds:.3f} s')

# plot fitted curve on top
t_fine = np.linspace(xaxis.min(), xaxis.max(), 200)
fit_curve = exp_decay(t_fine, *popt)
ax.plot(t_fine, fit_curve, '--', c='black', lw=1, alpha=.5, label=f'Fit τ={tau_seconds:.2f}s')

ax.legend(frameon=False, fontsize=6)

ax.set(xlabel='Time from stim.-onset (s)',
       ylabel='dLight dF/F',
       title='dLight dispersion rel. stim.')

for ext in ['.png', '.pdf']:
    fig.savefig(
        save_stem / f'dispersion_axon_shuf{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    
    
#%% tau histogram 
taus_real = []

for roi_trace in all_roi_vals_binned:
    y = np.array(roi_trace)
    t_fit = xaxis[xaxis >= xaxis[np.argmax(y)]]
    y_fit = y[xaxis >= xaxis[np.argmax(y)]]
    try:
        popt, _ = curve_fit(exp_decay, t_fit, y_fit,
                            p0=[y_fit.max()-y_fit.min(), 1.0, y_fit.min()],
                            bounds=([0,0,-np.inf],[np.inf,np.inf,np.inf]))
        taus_real.append(popt[1])
    except RuntimeError:
        continue

plt.figure(figsize=(3,2.4))
bins = np.linspace(.8, 4.5, 20)  # adjust if needed
plt.hist(taus_real, bins=bins, color='darkgreen', alpha=0.7, edgecolor='black')

plt.xlabel('Tau (s)')
plt.ylabel('Session count')
plt.title('Distribution of decay constants')
plt.axvline(np.median(taus_real), color='red', linestyle='--', label=f'Median = {np.median(taus_real):.2f}s')
plt.legend(frameon=False, fontsize=6)

plt.tight_layout()
plt.savefig(save_stem / 'tau_hist.png', dpi=300)
plt.savefig(save_stem / 'tau_hist.pdf', dpi=300)