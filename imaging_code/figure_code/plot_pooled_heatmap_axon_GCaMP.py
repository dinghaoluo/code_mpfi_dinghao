# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:52:50 2024
Modified on Thur 24 Apr 2025 to add a size filter 

plot the heatmap of pooled ROI activity of axon-GCaMP animals 

@author: Dinghao Luo
"""


#%% imports 
from pathlib import Path

import numpy as np 
from scipy.stats import sem
import matplotlib.pyplot as plt
import pandas as pd 

from common import mpl_formatting, smooth_convolve, normalise
mpl_formatting()

import rec_list
paths = rec_list.pathLCHPCGCaMP


#%% paths and parameters
axon_GCaMP_stem = Path('Z:/Dinghao/code_dinghao/LCHPC_axon_GCaMP')

SAMP_FREQ = 30  # Hz
BEF = 1
AFT = 4  # s 

RUN_ONSET_BIN = 90  # sample 

XAXIS = np.arange(-BEF * SAMP_FREQ, AFT * SAMP_FREQ) / SAMP_FREQ

ROI_size_threshold = 500  # pixel count 


#%% load data 
df = pd.read_pickle(axon_GCaMP_stem / 'LCHPC_axon_GCaMP_all_profiles.pkl')


#%% main 
proc_path = axon_GCaMP_stem / 'all_sessions'

# initialise with the first session
recname = Path(paths[0]).name 
print(recname)

run_aligned_path     = proc_path / recname / f'{recname}_all_run_mean.npy'
run_aligned_ch2_path = proc_path / recname / f'{recname}_all_run_ch2_mean.npy'
coord_path           = proc_path / recname / 'processed_data' / 'valid_ROIs_coord_dict.npy'

temp_dict       = np.load(run_aligned_path, allow_pickle=True).item()
temp_dict_ch2   = np.load(run_aligned_ch2_path, allow_pickle=True).item()
temp_coord_dict = np.load(coord_path, allow_pickle=True).item()

pooled_ROIs = np.row_stack(
    [temp_dict[key][RUN_ONSET_BIN - SAMP_FREQ * BEF : RUN_ONSET_BIN + SAMP_FREQ * AFT]
     for key in temp_dict
     if len(temp_coord_dict[key][0]) > ROI_size_threshold]
    )
pooled_ROIs_ch2 = np.row_stack(
    [temp_dict_ch2[key][RUN_ONSET_BIN - SAMP_FREQ * BEF : RUN_ONSET_BIN + SAMP_FREQ * AFT]
     for key in temp_dict_ch2
     if len(temp_coord_dict[key][0]) > ROI_size_threshold]
    )

all_RO_peak = np.row_stack(
    [temp_dict[key][RUN_ONSET_BIN - SAMP_FREQ * BEF : RUN_ONSET_BIN + SAMP_FREQ * AFT]
     for key in temp_dict
     if len(temp_coord_dict[key][0]) > ROI_size_threshold
     and df.loc[f'{recname} {key}']['run_onset_peak']]
    )
all_RO_peak_ch2 = np.row_stack(
    [temp_dict_ch2[key][RUN_ONSET_BIN - SAMP_FREQ * BEF : RUN_ONSET_BIN + SAMP_FREQ * AFT]
     for key in temp_dict_ch2
     if len(temp_coord_dict[key][0]) > ROI_size_threshold
     and df.loc[f'{recname} {key}']['run_onset_peak']]
    )


for path in paths[1:]:
    recname = Path(path).name
    print(recname)
    
    run_aligned_path     = proc_path / recname / f'{recname}_all_run_mean.npy'
    run_aligned_ch2_path = proc_path / recname / f'{recname}_all_run_ch2_mean.npy'
    coord_path           = proc_path / recname / 'processed_data' / 'valid_ROIs_coord_dict.npy'
    
    temp_dict       = np.load(run_aligned_path, allow_pickle=True).item()
    temp_dict_ch2   = np.load(run_aligned_ch2_path, allow_pickle=True).item()
    temp_coord_dict = np.load(coord_path, allow_pickle=True).item()
    
    temp_array = np.row_stack(
        [temp_dict[key][RUN_ONSET_BIN - SAMP_FREQ * BEF : RUN_ONSET_BIN + SAMP_FREQ * AFT]
         for key in temp_dict
         if len(temp_coord_dict[key][0]) > ROI_size_threshold]
        )
    temp_array_ch2 = np.row_stack(
        [temp_dict_ch2[key][RUN_ONSET_BIN - SAMP_FREQ * BEF : RUN_ONSET_BIN + SAMP_FREQ * AFT]
         for key in temp_dict_ch2
         if len(temp_coord_dict[key][0]) > ROI_size_threshold]
        )
    
    try:
        temp_RO_peak = np.row_stack(
            [temp_dict[key][RUN_ONSET_BIN - SAMP_FREQ * BEF : RUN_ONSET_BIN + SAMP_FREQ * AFT]
             for key in temp_dict
             if len(temp_coord_dict[key][0]) > ROI_size_threshold
             and df.loc[f'{recname} {key}']['run_onset_peak']]
            )
        temp_RO_peak_ch2 = np.row_stack(
            [temp_dict_ch2[key][RUN_ONSET_BIN - SAMP_FREQ * BEF : RUN_ONSET_BIN + SAMP_FREQ * AFT]
             for key in temp_dict_ch2
             if len(temp_coord_dict[key][0]) > ROI_size_threshold
             and df.loc[f'{recname} {key}']['run_onset_peak']]
            )
    except ValueError:
        temp_RO_peak     = []
        temp_RO_peak_ch2 = []
    
    # stack to previously saved array
    pooled_ROIs     = np.vstack((pooled_ROIs, temp_array))
    pooled_ROIs_ch2 = np.vstack((pooled_ROIs_ch2, temp_array_ch2))
    
    if len(temp_RO_peak) > 0:
        all_RO_peak     = np.vstack((all_RO_peak, temp_RO_peak))
        all_RO_peak_ch2 = np.vstack((all_RO_peak_ch2, temp_RO_peak_ch2))
    
    
#%% noramlisation 
tot_rois = pooled_ROIs.shape[0]
pooled_ROIs = normalise(smooth_convolve(pooled_ROIs))


#%% plotting 
keys = np.argsort([np.argmax(pooled_ROIs[roi, :]) for roi in range(tot_rois)])
im_matrix = pooled_ROIs[keys, :]

fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from run onset (s)',
       ylabel='ROI #')
ax.set_aspect('equal')
fig.suptitle('LC-CA1 GCaMP')

im_ordered = ax.imshow(im_matrix, 
                       cmap='viridis', aspect='auto', extent=(-1, 4, 0, tot_rois))
plt.colorbar(im_ordered, shrink=.5, ticks=[0,1], label='norm. dF/F')

for ext in ['.png', '.pdf']:
    fig.savefig(
        axon_GCaMP_stem / f'pooled_ordered_heatmap_RO_aligned{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    

#%% run onset peaks?
run_onset_peaks = df['run_onset_peak']

print(f'Percentage of run-onset-peaking axon ROIs: {sum(run_onset_peaks) / len(run_onset_peaks) * 100}%')


#%% mean trace of run-onset-peaking ROIs
mean_RO_peak     = np.mean(all_RO_peak, axis=0)
sem_RO_peak      = sem(all_RO_peak, axis=0)
mean_RO_peak_ch2 = np.mean(all_RO_peak_ch2, axis=0)
sem_RO_peak_ch2  = sem(all_RO_peak_ch2, axis=0)


# plotting 
fig, ax = plt.subplots(figsize=(2.4, 1.8))

ax.plot(XAXIS, mean_RO_peak, color='darkgreen', lw=1)
ax.fill_between(XAXIS,
                mean_RO_peak + sem_RO_peak,
                mean_RO_peak - sem_RO_peak,
                color='darkgreen', alpha=.3, edgecolor='none')
ax.plot(XAXIS, mean_RO_peak_ch2, color='red', lw=1)
ax.fill_between(XAXIS,
                mean_RO_peak_ch2 + sem_RO_peak_ch2,
                mean_RO_peak_ch2 - sem_RO_peak_ch2,
                color='firebrick', alpha=.3, edgecolor='none')

ax.set(
    xlabel='Time from run onset (s)',
    ylabel='dF/F',
    title='Run-onset-peaking axon ROIs'
)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

for ext in ['.png', '.pdf']:
    fig.savefig(
        axon_GCaMP_stem / f'pooled_mean_trace_RO_peaking{ext}',
        dpi=300,
        bbox_inches='tight'
    )