# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:48:04 2025

control:
    dLight expression vs response ratio to stimulation
    includes ROI-specific r_axon correlations
    permutation test: null distribution of R² medians across sessions

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathdLightLCOpto


#%% path stems
all_sess_stem = Path(r'Z:\\Dinghao\\code_dinghao\\HPC_dLight_LC_opto\\all_sessions')
save_stem = Path(r'Z:\\Dinghao\\code_dinghao\\HPC_dLight_LC_opto\\controls')


#%% parameters 
N_PERM = 1000  # permutations per session


#%% main
all_r = []          # full-field correlations
all_r_axon = []     # LC-axon restricted correlations
null_r2_medians = []       # shuffle medians for r
null_r2_axon_medians = []  # shuffle medians for r_axon

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    ref_ch1_path = all_sess_stem / recname / f'processed_data/{recname}_ref_mat_ch1.npy'
    roi_dict_path = all_sess_stem / recname / f'processed_data/{recname}_ROI_dict.npy'
    pixel_dFF_stim_path = all_sess_stem / recname / f'processed_data/{recname}_pixel_dFF_stim.npy'

    if not (ref_ch1_path.exists() and pixel_dFF_stim_path.exists() and roi_dict_path.exists()):
        print('missing arrays, skipped')
        continue

    ref1 = np.load(ref_ch1_path, allow_pickle=True)
    roi_dict = np.load(roi_dict_path, allow_pickle=True).item()
    pixel_dFF = np.load(pixel_dFF_stim_path, allow_pickle=True)
    pixel_dFF_med = np.nanmedian(pixel_dFF, axis=2)

    # ROI mask
    all_roi_y = [y for roi in roi_dict.values() for y in roi['ypix']]
    all_roi_x = [x for roi in roi_dict.values() for x in roi['xpix']]
    all_roi_mask = np.zeros((512, 512), dtype=bool)
    all_roi_mask[all_roi_y, all_roi_x] = True

    # full-field correlation
    r = np.corrcoef(ref1.ravel(), pixel_dFF_med.ravel())[0, 1]
    all_r.append(r)

    # ROI-specific correlation
    r_axon = np.corrcoef(ref1[all_roi_mask], pixel_dFF_med[all_roi_mask])[0, 1]
    all_r_axon.append(r_axon)

    # regression (for plotting)
    m, b, *_ = linregress(ref1.ravel(), pixel_dFF_med.ravel())

    # permutation tests
    x_all = ref1.ravel()
    y_all = pixel_dFF_med.ravel()
    y_ax = pixel_dFF_med[all_roi_mask]

    null_r2 = np.empty(N_PERM)
    null_r2_axon = np.empty(N_PERM)
    for i in range(N_PERM):
        y_shuf = np.random.permutation(y_all)
        null_r2[i] = np.corrcoef(x_all, y_shuf)[0, 1] ** 2

        y_ax_shuf = np.random.permutation(y_ax)
        null_r2_axon[i] = np.corrcoef(ref1[all_roi_mask], y_ax_shuf)[0, 1] ** 2

    null_r2_medians.append(np.nanmedian(null_r2))
    null_r2_axon_medians.append(np.nanmedian(null_r2_axon))

    # single-session scatter plot: all pixels
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    ax.scatter(ref1, pixel_dFF_med,
               color='darkgreen', edgecolor='none', s=5, alpha=0.6, rasterized=True)

    xfit = np.linspace(np.nanmin(ref1), np.nanmax(ref1), 200)
    yfit = b + m * xfit
    ax.plot(xfit, yfit, color='k', lw=1)

    ax.text(0.05, 0.95,
            f'$r = {r:.2f}$\n$r_{{axon}} = {r_axon:.2f}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)

    ax.set_xlabel('dLight intensity')
    ax.set_ylabel('Median stim. RI')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9, length=2, width=0.6)
    fig.tight_layout()
    plt.show()

    for ext in ['.png', '.pdf']:
        fig.savefig(
            save_stem / f'single_session_dLight_expression_corr/{recname}_scatter{ext}',
            dpi=300,
            bbox_inches='tight'
        )

    # single-session scatter plot: axon ROI pixels only
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    ax.scatter(ref1[all_roi_mask], pixel_dFF_med[all_roi_mask],
               color='mediumseagreen', edgecolor='none', s=8, alpha=0.7, rasterized=True)

    # regression line for axon pixels
    m_ax, b_ax, *_ = linregress(ref1[all_roi_mask], pixel_dFF_med[all_roi_mask])
    xfit_ax = np.linspace(np.nanmin(ref1[all_roi_mask]), np.nanmax(ref1[all_roi_mask]), 200)
    yfit_ax = b_ax + m_ax * xfit_ax
    ax.plot(xfit_ax, yfit_ax, color='k', lw=1)

    ax.text(0.05, 0.95, f'$r_{{axon}} = {r_axon:.2f}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)

    ax.set_xlabel('dLight intensity (axon ROIs)')
    ax.set_ylabel('Median stim. RI (axon ROIs)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9, length=2, width=0.6)
    fig.tight_layout()
    plt.show()

    for ext in ['.png', '.pdf']:
        fig.savefig(
            save_stem / f'single_session_dLight_expression_corr/{recname}_axon_scatter{ext}',
            dpi=300,
            bbox_inches='tight'
        )


#%% summary histograms
r2 = np.square(all_r)
r2_axon = np.square(all_r_axon)

med_r2 = np.median(r2)
iqr_r2 = np.percentile(r2, 75) - np.percentile(r2, 25)
med_r2_axon = np.median(r2_axon)
iqr_r2_axon = np.percentile(r2_axon, 75) - np.percentile(r2_axon, 25)

print(f'All pixels: median R² = {med_r2:.4f} ± {iqr_r2/2:.4f}')
print(f'Axon ROIs: median R² = {med_r2_axon:.4f} ± {iqr_r2_axon/2:.4f}')

r2_max = max(r2.max(), r2_axon.max())
bins = np.linspace(0, r2_max, 100)

fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.2))
axs = axs.flatten()

# all pixels
axs[0].hist(r2, bins=bins, color='darkgreen', alpha=0.7,
            weights=np.ones_like(r2) / len(r2))
axs[0].set_title('All pixels', fontsize=9)
axs[0].text(0.95, 0.93, f'median = {med_r2:.4f}',
            transform=axs[0].transAxes, ha='right', va='top', fontsize=8)

# axon ROIs
axs[1].hist(r2_axon, bins=bins, color='mediumseagreen', alpha=0.7,
            weights=np.ones_like(r2_axon) / len(r2_axon))
axs[1].set_title('Axon ROIs', fontsize=9)
axs[1].text(0.95, 0.93, f'median = {med_r2_axon:.4f}',
            transform=axs[1].transAxes, ha='right', va='top', fontsize=8)

for ax in axs:
    ax.set_xlim([0, np.percentile(np.concatenate([r2, r2_axon]), 99)])
    ax.set_xlabel('$R^2$', fontsize=9)
    ax.set_ylabel('Proportion of sessions', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8, length=2, width=0.6)

fig.tight_layout(w_pad=1.0)

for ext in ['.png', '.pdf']:
    fig.savefig(save_stem / f'all_sessions_dLight_expression_R2_summary{ext}',
                dpi=300, bbox_inches='tight')

plt.show()