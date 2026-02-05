# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 18:07:16 2025
Modified on 22 Jan 2026

analyse the dispersion of dLight signal after stim.
    dependent on extraction with HPC_dLight_LC_opto_extract.py
Modified to work on ROI vs neuropil over bins 

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import wilcoxon, sem
from scipy.optimize import curve_fit
from scipy.ndimage import binary_dilation

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

def _exp_decay_fixed(t, A, tau, t0, B):
    return A * np.exp(-(t - t0) / tau) + B


#%% paths and parameters 
dLight_stem   = Path('Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto')
all_sess_stem = dLight_stem / 'all_sessions'
save_stem     = dLight_stem / 'ROI_vs_neuropil'

# how far away from ROI to count as neuropil 
DISTANCE_FROM_ROI = 9  # 9 pixels ~ 5 um

ALPHA    = 0.05
MIN_RI   = 0.1
R2_THRES = 0.7
N_BINS   = 40
XAXIS    = np.arange(N_BINS) / 10


#%% main
all_ROI_RI_bins      = []
all_neuropil_RI_bins = []

all_ROI_RI_taus      = []
all_neuropil_RI_taus = [] 


# main loop 
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
    
    # build mask of releasing ROIs
    releasing_mask = _build_roi_mask(releasing)
    
    # build dilated mask and then the anti-mask 
    ROI_mask      = _build_roi_mask(roi_dict)
    ROI_dilated   = binary_dilation(ROI_mask, iterations=DISTANCE_FROM_ROI)
    anti_ROI_mask = ~ROI_dilated 
    
    # ------------------
    # EXTRACT RI TRACES
    # ------------------
    # ... from ROI_mask 
    ROI_RI_bins      = np.nanmean(pixel_RI_bins[releasing_mask, :], axis=0)
    
    # ... and from anti_ROI_mask 
    neuropil_RI_bins = np.nanmean(pixel_RI_bins[anti_ROI_mask, :], axis=0)
    
    # append first 
    all_ROI_RI_bins.append(ROI_RI_bins)
    all_neuropil_RI_bins.append(neuropil_RI_bins)
    
    # now we fit tau 
    first_valid = np.where(np.isfinite(ROI_RI_bins))[0][0]
    
    t0   = XAXIS[first_valid]
    B    = ROI_RI_bins[first_valid:].min()
    Bneu = neuropil_RI_bins[first_valid:].min() 
    
    t_fit    = XAXIS[first_valid:]
    y_fit    = ROI_RI_bins[first_valid:]
    yneu_fit = neuropil_RI_bins[first_valid:]
    
    # nested functions for curve_fit — only A and tau are free
    def _fitfun(t, A, tau):
        return _exp_decay_fixed(t, A, tau, t0, B)
    def _fitfun_neu(t, A, tau):
        return _exp_decay_fixed(t, A, tau, t0, Bneu)
    
    
    # fit ROIs first 
    try:
        popt, _ = curve_fit(
            _fitfun,
            t_fit,
            y_fit,
            p0=[ROI_RI_bins[first_valid] - B, 1.0],
            bounds=([0, 0], [np.inf, np.inf])
        )
        A_fit, tau_fit = popt
        
        # predict fitted decay
        y_pred = _fitfun(t_fit, A_fit, tau_fit)
        
        # compute R2
        ss_res = np.sum((y_fit - y_pred)**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # require R2 > thres
        if R2 < R2_THRES:
            continue
        
        all_ROI_RI_taus.append(tau_fit)
    
    except RuntimeError:
        continue
    
    
    # then do neuropil 
    try:
        popt, _ = curve_fit(
            _fitfun_neu,
            t_fit,
            yneu_fit,
            p0=[neuropil_RI_bins[first_valid] - Bneu, 1.0],
            bounds=([0, 0], [np.inf, np.inf])
        )
        A_fit, tau_fit = popt
        
        # predict fitted decay
        y_pred = _fitfun_neu(t_fit, A_fit, tau_fit)
        
        # compute R2
        ss_res = np.sum((yneu_fit - y_pred)**2)
        ss_tot = np.sum((yneu_fit - np.mean(yneu_fit))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # require R2 > thres
        if R2 < R2_THRES:
            continue
        
        all_neuropil_RI_taus.append(tau_fit)
    
    except RuntimeError:
        continue
    # -----------------------
    # EXTRACT RI TRACES ENDS
    # -----------------------


#%% tau plot
ROI_mat      = np.vstack(all_ROI_RI_bins)      # shape: nROI × nTime
neuropil_mat = np.vstack(all_neuropil_RI_bins)

ROI_mean = np.nanmean(ROI_mat, axis=0)
ROI_sem  = sem(ROI_mat, axis=0, nan_policy='omit')

neuropil_mean = np.nanmean(neuropil_mat, axis=0)
neuropil_sem  = sem(neuropil_mat, axis=0, nan_policy='omit')


# plotting 
fig, ax = plt.subplots(figsize=(3, 2.4))

ax.plot(XAXIS, ROI_mean, color='darkgreen', label='LC axons')
ax.fill_between(
    XAXIS,
    ROI_mean - ROI_sem,
    ROI_mean + ROI_sem,
    color='darkgreen', alpha=.3, linewidth=0, edgecolor='none'
)

ax.plot(XAXIS, neuropil_mean, color='grey', label='Neuropil')
ax.fill_between(
    XAXIS,
    neuropil_mean - neuropil_sem,
    neuropil_mean + neuropil_sem,
    color='grey', alpha=.3, linewidth=0, edgecolor='none'
)

ax.set(
    xlabel='Time from stim.-offset (s)',
    ylabel='RI',
    title='LC axon dLight'
)
ax.legend(frameon=False, fontsize=7)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        save_stem / f'RI_ROI_vs_neuropil{ext}',
        dpi=300,
        bbox_inches='tight'
    )


#%% tau histogram
bins = np.linspace(0, 4, 21)

ROI_taus = np.array(all_ROI_RI_taus)
NEU_taus = np.array(all_neuropil_RI_taus)

fig, ax = plt.subplots(figsize=(3, 2.4))

ax.hist(
    ROI_taus,
    bins=bins,
    color='darkgreen',
    alpha=0.6,
    edgecolor='none',
    label='ROI'
    )

ax.hist(
    NEU_taus,
    bins=bins,
    color='grey',
    alpha=0.6,
    edgecolor='none',
    label='Neuropil'
    )

# medians + IQRs
if ROI_taus.size > 0:
    q1_roi, med_roi, q3_roi = np.percentile(ROI_taus, [25, 50, 75])
    ax.axvline(med_roi, color='darkgreen', linestyle='--', lw=1)
    ax.text(
        0.02, 0.95,
        f'ROI median = {med_roi:.2f}s\nIQR = [{q1_roi:.2f}, {q3_roi:.2f}]',
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=7, color='darkgreen'
    )

if NEU_taus.size > 0:
    q1_neu, med_neu, q3_neu = np.percentile(NEU_taus, [25, 50, 75])
    ax.axvline(med_neu, color='grey', linestyle='--', lw=1)
    ax.text(
        0.02, 0.85,
        f'Neuropil median = {med_neu:.2f}s\nIQR = [{q1_neu:.2f}, {q3_neu:.2f}]',
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=7, color='grey'
    )

ax.set(
    xlabel='Tau (s)',
    ylabel='ROI count',
    title='Decay constant distribution'
    )

ax.legend(frameon=False, fontsize=7)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        save_stem / f'ROI_vs_neuropil_decay_tau_hist{ext}',
        dpi=300,
        bbox_inches='tight'
    )
