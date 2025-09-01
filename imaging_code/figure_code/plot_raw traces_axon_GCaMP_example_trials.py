# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:17:07 2025

plot example trials for example ROI 

@author: Dinghao
"""

#%% imports
import os
import sys 

import numpy as np
import matplotlib.pyplot as plt
import pickle

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
from imaging_pipeline_functions import calculate_dFF

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, smooth_convolve
mpl_formatting()


#%% parameters 
path = r'Z:\Dinghao\2p_recording\A101i\A101i-20241107\A101i-20241107-03'

# see target trials etc below 

smoothing_sigma_ms = 100
speed_smooth_sigma_ms = 300

speed_samp_hz = 1000
track_len_cm = 180


#%% helper 
def trial_slice(k):
    start = run_onsets_idx[k]
    # default end at next onset if exists
    if k + 1 < run_onsets_idx.size:
        end = run_onsets_idx[k + 1]
    else:
        # find when distances goes back to 0 (after onset) because of 180 cm reset
        after = distances[start + 1:]
        zeros = np.flatnonzero(after == 0.0)
        end = (start + 1 + zeros[0]) if zeros.size > 0 else times.size
    return start, end


#%% beh
F = np.load(os.path.join(path, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')

recname = os.path.basename(path.rstrip('\\/'))
beh_pkl = rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\LCHPCGCaMP\{recname}.pkl'
with open(beh_pkl, 'rb') as f:
    beh = pickle.load(f)

frame_times_ms = np.asarray(beh['frame_times'])
first_frame_time = int(frame_times_ms[0])
last_frame_time  = int(frame_times_ms[-1])

t_ms_full = np.asarray(beh['upsampled_timestamps_ms'])
v_cms_full = np.asarray(beh['upsampled_speed_cm_s'], dtype=float)

mask = (t_ms_full > first_frame_time) & (t_ms_full < last_frame_time)
times = t_ms_full[mask]           # ms
speeds = v_cms_full[mask]         # cm/s

run_onsets_all = np.asarray(beh['run_onsets'])
run_onsets = run_onsets_all[(run_onsets_all > first_frame_time) & (run_onsets_all < last_frame_time)]

run_onsets_idx = np.searchsorted(times, run_onsets)
distances = np.zeros_like(times, dtype=float)

trial_ptr = 0
accumulating = False
dist_cm = 0.0
dt_s = 1.0 / speed_samp_hz  # if irregular sampling, compute from np.diff(times)/1000

for i in range(times.size):
    if trial_ptr < run_onsets_idx.size and i == run_onsets_idx[trial_ptr]:
        dist_cm = 0.0
        accumulating = True
        trial_ptr += 1

    if accumulating:
        dist_cm += speeds[i] * dt_s
        if dist_cm >= track_len_cm:
            dist_cm = 0.0
            accumulating = False

    distances[i] = dist_cm


#%% run 
target_roi_id = 869
example_trial = 61

F_roi = np.array(F[target_roi_id, :], dtype=float)
n_frames = min(frame_times_ms.size, F_roi.size)
F_roi = F_roi[:n_frames]
ft_ms = frame_times_ms[:n_frames].astype(float)

F_roi_interp = np.interp(times.astype(float), ft_ms, F_roi)
F_roi_interp = smooth_convolve(F_roi_interp, smoothing_sigma_ms)

speeds_smoothed = smooth_convolve(speeds, speed_smooth_sigma_ms)

# plot 
num_trials = run_onsets_idx.size
i0 = max(0, example_trial - 1)
i1 = example_trial
i2 = min(num_trials - 1, example_trial + 1)
chosen = [i0, i1, i2]
chosen = sorted(set(chosen), key=chosen.index)

# one set of axes
fig, axs = plt.subplots(3, 1, figsize=(3, 2), sharex=True)
fig.subplots_adjust(hspace=0.3)

baseline_pre_s = 3

offset = 0  # time offset in seconds so trials are concatenated
for bi, k in enumerate(chosen):
    s, e = trial_slice(k)

    # if this is the first trial, back up by baseline_pre_s
    if bi == 0:
        # number of samples to go back
        n_back = int(baseline_pre_s * speed_samp_hz)
        s = max(0, s - n_back)

    # build time axis relative to offset
    t_seg = (times[s:e] - times[s]) / 1000.0 + offset

    axs[0].plot(t_seg, distances[s:e] / 100.0, color='black', linewidth=1.0)
    axs[1].plot(t_seg, F_roi_interp[s:e], color='green', linewidth=1.0)
    axs[2].plot(t_seg, speeds_smoothed[s:e], color=(0.4, 0.5, 0.0), linewidth=1.0)

    offset = t_seg[-1]

axs[0].set_ylabel('distance (m)')
axs[0].set_title(f'roi {target_roi_id}, trials {chosen}')
axs[1].set_ylabel('dF/F')
axs[2].set_ylabel('speed (cm/s)')
axs[2].set_xlabel('time (s, concatenated trials)')

save_dir = r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\raw_trace_distance_speed_examples'
os.makedirs(save_dir, exist_ok=True)
outpath = os.path.join(save_dir, f'roi_{target_roi_id}_trials_{chosen[0]}_{chosen[-1]}')

for i in range(3):
    for s in ['top', 'right']:
        axs[i].spines[s].set_visible(False)
        
for ext in ['.png', '.pdf']:
    fig.savefig(f'{outpath}{ext}', dpi=300, bbox_inches='tight')