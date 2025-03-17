# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:53:25 2025

burst analysis for LC cells 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import sys 
import pandas as pd 
import matplotlib.pyplot as plt 
import os

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% load dataframe 
df = pd.read_pickle(
    r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl'
    )


#%% functions 
def detect_bursts_from_isi(ISIs, 
                           burst_thresh=None, 
                           end_thresh=None, 
                           min_spikes=3):
    """
    detects bursts using inter-spike intervals (isi) and percentile-based thresholds.

    parameters:
    - ISIs: numpy array of inter-spike intervals in seconds
    - burst_thresh: threshold for burst onset (if None, should be set externally)
    - end_thresh: threshold for burst termination (if None, should be set externally)
    - min_spikes: minimum number of spikes required to classify a burst (default: 2)

    returns:
    - bursts: list of tuples (burst_start_index, burst_end_index) where each tuple represents a detected burst
    """
    bursts = []
    burst_start = None
    spike_count = 0  

    for i, isi in enumerate(ISIs):
        if isi < burst_thresh:  
            if burst_start is None:
                burst_start = i  
                spike_count = 2  
            else:
                spike_count += 1  
        elif isi > end_thresh and burst_start is not None:  
            if spike_count >= min_spikes:
                bursts.append((burst_start, i))  
            burst_start = None  
            spike_count = 0  

    if burst_start is not None and spike_count >= min_spikes:
        bursts.append((burst_start, len(ISIs)))

    return bursts


def percentile_based_thresholds(ISIs, 
                                burst_percentile=10, 
                                end_factor=2.5,
                                SAMP_FREQ=20000):
    """
    computes burst onset and termination thresholds based on percentiles.

    parameters:
    - ISIs: numpy array of inter-spike intervals in seconds
    - burst_percentile: percentile threshold for burst onset (default: 10th percentile)
    - end_factor: multiplier for burst termination threshold (default: 2.5x burst threshold)

    returns:
    - burst_thresh: threshold for burst onset
    - end_thresh: threshold for burst termination
    """
    MAX_THRESH = 0.2 * SAMP_FREQ
    
    burst_thresh = np.percentile(ISIs, burst_percentile)
    burst_thresh = min(burst_thresh, MAX_THRESH)
    
    end_thresh = end_factor * burst_thresh
    return burst_thresh, end_thresh


def plot_bursts(spike_times, 
                bursts, 
                cluname, 
                save_dir, 
                window_size=3, 
                max_bursts_per_fig=50):
    """
    plots spikes with bursts highlighted in black and tonic spikes in grey, centering each burst on its own subplot.
    if a cell has too many bursts, splits them into multiple figures.

    parameters:
    - spike_times: list or numpy array of spike timestamps in seconds
    - bursts: list of tuples (burst_start_index, burst_end_index) from detect_bursts_from_isi
    - cluname: string, name of the cell (used for figure title)
    - save_dir: directory where figures will be saved
    - window_size: time window around each burst center (default: 3 seconds)
    - max_bursts_per_fig: maximum number of bursts per figure (default: 50)

    returns:
    - None (saves the figure)
    """
    if len(spike_times) == 0 or len(bursts) == 0:
        print(f'no spikes or bursts to plot for {cluname}.')
        return

    # ensure save directory exists
    save_dir = rf'{save_dir}'

    spike_times = np.array(spike_times)

    num_total_bursts = len(bursts)
    num_figs = int(np.ceil(num_total_bursts / max_bursts_per_fig))

    for fig_idx in range(num_figs):
        start_burst = fig_idx * max_bursts_per_fig
        end_burst = min((fig_idx + 1) * max_bursts_per_fig, num_total_bursts)
        bursts_subset = bursts[start_burst:end_burst]

        num_bursts = len(bursts_subset)
        fig, axes = plt.subplots(num_bursts, 1, figsize=(5, 0.5 * num_bursts), sharex=True)

        if num_bursts == 1:
            axes = [axes]

        for i, (start_idx, end_idx) in enumerate(bursts_subset):
            ax = axes[i]

            burst_center = spike_times[(start_idx + end_idx) // 2]
            start_time = burst_center - window_size / 2
            end_time = burst_center + window_size / 2

            spikes_in_window = spike_times[(spike_times >= start_time) & (spike_times < end_time)]
            burst_spikes = spike_times[start_idx:end_idx + 1]
            tonic_spikes = np.setdiff1d(spikes_in_window, burst_spikes)

            ax.scatter(tonic_spikes - burst_center, np.ones_like(tonic_spikes), color='grey', s=10, alpha=0.6)
            ax.scatter(burst_spikes - burst_center, np.ones_like(burst_spikes), color='black', s=10)

            ax.set_xlim(-window_size / 2, window_size / 2)
            ax.set_yticks([])

            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)

            if i == num_bursts - 1:
                ax.set_xlabel('time from burst (s)', fontsize=8)
            else:
                ax.set_xticklabels([])

            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
            ax.tick_params(axis='x', labelsize=8)

        fig.suptitle(f'{cluname} (part {fig_idx + 1})', fontsize=10)
        plt.tight_layout()

        # generate and save figure
        plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        plt.close()


#%% main 
for path in paths:
    recname = path[-17:]
    print(f'processing {recname}')
    
    sess_folder = rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}'
    burst_folder = r'Z:\Dinghao\code_dinghao\LC_ephys\single_cell_bursts'
    os.makedirs(burst_folder, exist_ok=True)

    ISI_dict = np.load(rf'{sess_folder}\{recname}_all_ISIs.npy', allow_pickle=True).item()
    spike_dict = np.load(rf'{sess_folder}\{recname}_all_spikes.npy', allow_pickle=True).item()

    burst_dict = {}

    for cluname, ISIs in ISI_dict.items():
        identity = df.loc[cluname]['identity']
        if identity != 'other':  
            burst_thresh, end_thresh = percentile_based_thresholds(ISIs)
            bursts = detect_bursts_from_isi(ISIs, burst_thresh, end_thresh)

            spike_times = [t/20000 for t in spike_dict[cluname]]
            save_path = os.path.join(burst_folder, f'{cluname}_{identity}.png')

            plot_bursts(spike_times, bursts, cluname, save_path)

            burst_dict[cluname] = bursts

    # save burst dictionary for session
    np.save(rf'{sess_folder}\{recname}_all_bursts.npy', 
            burst_dict)