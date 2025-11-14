# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:37:15 2025

summarise dLight + LC activation data 

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
from scipy.stats import sem
import matplotlib.pyplot as plt 

from plotting_functions import plot_violin_with_scatter
from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathdLightLCOpto


#%% path stems
dLight_stim_stem = Path('Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto')
all_sess_stem = dLight_stim_stem / 'all_sessions'


#%% parameters 
SAMP_FREQ = 30  # Hz
BEF = 2
AFT = 10

XAXIS = np.arange((BEF+AFT) * SAMP_FREQ) / SAMP_FREQ - BEF

BASELINE_WIN = [SAMP_FREQ * 1, int(SAMP_FREQ * 1.85)]
STIM_WIN     = [int(SAMP_FREQ * (2+1.15)), int(SAMP_FREQ * (2+2))]


#%% main 
all_dFF  = []
all_dFF2 = []

for path in paths:
    recname = Path(path).name
    print(recname)
    
    dFF_path  = all_sess_stem / recname / 'processed_data' / f'{recname}_wholefield_dFF_stim.npy'
    dFF2_path = all_sess_stem / recname / 'processed_data' / f'{recname}_wholefield_dFF2_stim.npy'
    
    dFF  = np.load(dFF_path, allow_pickle=True)
    dFF2 = np.load(dFF2_path, allow_pickle=True)
    
    # centring and filtering 
    dFF  = dFF  - np.nanmean(dFF[:30])
    dFF2 = dFF2 - np.nanmean(dFF2[:30])
    dFF[60:110]  = np.nan
    dFF2[60:110] = np.nan
    
    all_dFF.append(dFF)
    all_dFF2.append(dFF2)
    
all_dFF  = np.array(all_dFF)
all_dFF2 = np.array(all_dFF2)
    
    
#%% data wrangling
dFF_mean  = np.nanmean(all_dFF, axis=0)
dFF_sem   = sem(all_dFF, axis=0)
dFF2_mean = np.nanmean(all_dFF2, axis=0)
dFF2_sem  = sem(all_dFF2, axis=0)

dFF_baselines  = np.nanmean(all_dFF[:, BASELINE_WIN[0] : BASELINE_WIN[1]], axis=1)
dFF_stims      = np.nanmean(all_dFF[:, STIM_WIN[0] : STIM_WIN[1]], axis=1)
dFF2_baselines = np.nanmean(all_dFF2[:, BASELINE_WIN[0] : BASELINE_WIN[1]], axis=1)
dFF2_stims     = np.nanmean(all_dFF2[:, STIM_WIN[0] : STIM_WIN[1]], axis=1)

# nan filtering 
nan_mask = ~np.isnan(dFF_baselines) & ~np.isnan(dFF_stims)
dFF_baselines = dFF_baselines[nan_mask]
dFF_stims     = dFF_stims[nan_mask]

nan_mask2 = ~np.isnan(dFF2_baselines) & ~np.isnan(dFF2_stims)
dFF2_baselines = dFF2_baselines[nan_mask2]
dFF2_stims     = dFF2_stims[nan_mask2]


#%% plotting 
fig, ax = plt.subplots(figsize=(3,2.5))

ax.plot(XAXIS, dFF_mean, color='darkgreen', label='dLight')
ax.plot(XAXIS, all_dFF.T, color='darkgreen', alpha=.03, linewidth=0.5)
# ax.fill_between(XAXIS, dFF_mean+dFF_sem,
#                        dFF_mean-dFF_sem,
#                        alpha=.3, color='darkgreen', edgecolor='none')
ax.plot(XAXIS, dFF2_mean, color='darkred', label='tdTomato')
ax.plot(XAXIS, all_dFF2.T, color='darkred', alpha=.03, linewidth=0.5)
# ax.fill_between(XAXIS, dFF2_mean+dFF2_sem,
#                        dFF2_mean-dFF2_sem,
#                        alpha=.3, color='darkred', edgecolor='none')

ax.set(xlabel='Time from stim. (s)',
       ylabel='dF/F',
       xticks=(0, 5, 10))
ax.legend(frameon=False)

for ext in ['.png', '.pdf']:
    fig.savefig(dLight_stim_stem / f'dLight_LC_stim_summary{ext}',
                dpi=300, bbox_inches='tight')
    

#%% plotting violinplots 
plot_violin_with_scatter(dFF_baselines, dFF_stims, 
                         'darkgreen', 'royalblue',
                         save=True,
                         savepath=dLight_stim_stem / 'dLight_LC_stim_violin')

plot_violin_with_scatter(dFF2_baselines, dFF2_stims, 
                         'darkred', 'royalblue',
                         save=True,
                         savepath=dLight_stim_stem / 'dLight_LC_stim_violin_ch2')


#%% Dbh inhibitor 
paths = rec_list.pathdLightLCOptoDbhBlock


#%% main 
all_dFF  = []
all_dFF2 = []

for path in paths:
    recname = Path(path).name
    print(recname)
    
    dFF_path  = all_sess_stem / recname / 'processed_data' / f'{recname}_wholefield_dFF_stim.npy'
    dFF2_path = all_sess_stem / recname / 'processed_data' / f'{recname}_wholefield_dFF2_stim.npy'
    
    dFF  = np.load(dFF_path, allow_pickle=True)
    dFF2 = np.load(dFF2_path, allow_pickle=True)
    
    # centring and filtering 
    dFF  = dFF  - np.nanmean(dFF[:30])
    dFF2 = dFF2 - np.nanmean(dFF2[:30])
    dFF[60:110]  = np.nan
    dFF2[60:110] = np.nan
        
    # append 
    all_dFF.append(dFF)
    all_dFF2.append(dFF2)
    
all_dFF  = np.array(all_dFF)
all_dFF2 = np.array(all_dFF2)
    
    
#%% data wrangling
dFF_mean  = np.nanmean(all_dFF, axis=0)
dFF_sem   = sem(all_dFF, axis=0)
dFF2_mean = np.nanmean(all_dFF2, axis=0)
dFF2_sem  = sem(all_dFF2, axis=0)

dFF_baselines  = np.nanmean(all_dFF[:, BASELINE_WIN[0] : BASELINE_WIN[1]], axis=1)
dFF_stims      = np.nanmean(all_dFF[:, STIM_WIN[0] : STIM_WIN[1]], axis=1)
dFF2_baselines = np.nanmean(all_dFF2[:, BASELINE_WIN[0] : BASELINE_WIN[1]], axis=1)
dFF2_stims     = np.nanmean(all_dFF2[:, STIM_WIN[0] : STIM_WIN[1]], axis=1)

# nan filtering 
nan_mask = ~np.isnan(dFF_baselines) & ~np.isnan(dFF_stims)
dFF_baselines = dFF_baselines[nan_mask]
dFF_stims     = dFF_stims[nan_mask]

nan_mask2 = ~np.isnan(dFF2_baselines) & ~np.isnan(dFF2_stims)
dFF2_baselines = dFF2_baselines[nan_mask2]
dFF2_stims     = dFF2_stims[nan_mask2]


#%% plotting 
fig, ax = plt.subplots(figsize=(3,2.5))

ax.plot(XAXIS, dFF_mean, color='darkgreen', label='dLight')
ax.plot(XAXIS, all_dFF.T, color='darkgreen', alpha=.03)
ax.fill_between(XAXIS, dFF_mean+dFF_sem,
                       dFF_mean-dFF_sem,
                       alpha=.3, color='darkgreen', edgecolor='none')
ax.plot(XAXIS, dFF2_mean, color='darkred', label='tdTomato')
ax.plot(XAXIS, all_dFF2.T, color='darkred', alpha=.03)
ax.fill_between(XAXIS, dFF2_mean+dFF2_sem,
                       dFF2_mean-dFF2_sem,
                       alpha=.3, color='darkred', edgecolor='none')

ax.set(xticks=(0, 5, 10))
ax.legend(frameon=False)

for ext in ['.png', '.pdf']:
    fig.savefig(dLight_stim_stem / f'dLight_LC_stim_Dbh_inh_summary{ext}',
                dpi=300, bbox_inches='tight')
    

#%% plotting violinplots 
plot_violin_with_scatter(dFF_baselines, dFF_stims, 
                         'darkgreen', 'royalblue',
                         save=True,
                         savepath=dLight_stim_stem / 'dLight_LC_stim_Dbh_inh_violin')

plot_violin_with_scatter(dFF2_baselines, dFF2_stims, 
                         'darkred', 'royalblue',
                         save=True,
                         savepath=dLight_stim_stem / 'dLight_LC_stim_Dbh_inh_violin_ch2')