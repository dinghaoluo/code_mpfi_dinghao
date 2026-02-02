# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:16:12 2026

Quantify the proportions of ROIs with significant release in each session

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from common import mpl_formatting
mpl_formatting()

import rec_list

paths = rec_list.pathdLightLCOpto + \
        rec_list.pathdLightLCOptoDbhBlock


#%% parameters 
ALPHA  = 0.05
MIN_RI = 0.1


#%% path stems 
dLight_stem   = Path('Z:/Dinghao/code_dinghao/HPC_dlight_LC_opto')
all_sess_stem = dLight_stem / 'all_sessions'
mice_exp_stem = Path('Z:/Dinghao/MiceExp')

        
#%% main 
all_release_proportions   = []  # proportions of axon ROIs that release in each session
all_release_probabilities = []  # release probabilities of all axon ROIs

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    pixel_RI_stim_path  = all_sess_stem / recname / 'processed_data' / f'{recname}_pixel_RI_stim.npy'
    roi_dict_path       = all_sess_stem / recname / 'processed_data' / f'{recname}_ROI_dict.npy'

    if not pixel_RI_stim_path.exists():
        print('No pixel_dFF_bins; skipped')
        continue
    if not roi_dict_path.exists():
        print('No roi_dict; skipped')
        continue

    # load data
    print('Loading data...')
    pixel_RI_stim = np.load(pixel_RI_stim_path, allow_pickle=True)  # (512,512,40)
    roi_dict      = np.load(roi_dict_path, allow_pickle=True).item()
    
    # ---- identify releasing ROIs ----
    releasing_rois = {}
    
    for rid, roi in roi_dict.items():
        vals  = pixel_RI_stim[roi['ypix'], roi['xpix'], :]
        means = np.nanmean(vals, axis=0)  # mean over pixels 
        means = [mean for mean in means if np.isfinite(mean)]  # filtering first 
        if len(means) > 2:
            _, p = wilcoxon(means, alternative='greater')
            if p < ALPHA and np.mean(means) > MIN_RI:
                releasing_rois[rid] = roi
    
    if len(releasing_rois) == 0:
        print('No releasing ROI; skipped')
        continue 
    # ---- identification ends ----
            
    # get proportion of releasing ROIs
    curr_prop = len(releasing_rois) / len(roi_dict)
    all_release_proportions.append(curr_prop)
    
    
#%% plotting 
props = np.array(all_release_proportions)

# bins in proportion space [0, 1]
bins = np.linspace(0, 1, 21)

fig, ax = plt.subplots(figsize=(3, 2.4))

ax.hist(
    props,
    bins=bins,
    color='darkgreen',
    alpha=0.6,
    edgecolor='none'
)

# median + IQR
if props.size > 0:
    q1, med, q3 = np.percentile(props, [25, 50, 75])

    ax.axvline(med, color='darkgreen', linestyle='--', lw=1)

    ax.text(
        0.02, 0.95,
        f'median = {med:.4f}\nIQR = [{q1:.4f}, {q3:.4f}]',
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=7, color='darkgreen'
    )

ax.set(
    xlabel='proportion of releasing ROIs per session',
    ylabel='session count',
    title='session-wise release proportion'
)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        dLight_stem / f'session_release_proportion_hist{ext}',
        dpi=300,
        bbox_inches='tight'
    )