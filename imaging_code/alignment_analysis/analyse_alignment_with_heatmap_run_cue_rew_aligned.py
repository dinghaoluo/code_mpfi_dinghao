# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 17:53:46 2025
Modified on 5 Dec 2025 
Modified on 19 Jan 2026 

plot population heatmap, but aligned to cue and rew
modified to include statistics
modified from the original LC ephys version to work on axon-GCaMP

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import pickle 
from scipy.stats import sem 
from tqdm import tqdm
from statsmodels.stats.proportion import proportions_ztest

import rec_list
paths = rec_list.pathLCHPCGCaMP

from common import normalise, mpl_formatting
mpl_formatting()


#%% load data 
axon_GCaMP_stem = Path('Z:/Dinghao/code_dinghao/LCHPC_axon_GCaMP')
all_sess_stem   = axon_GCaMP_stem / 'all_sessions'
pop_map_stem    = axon_GCaMP_stem / 'population_maps'

beh_stem        = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LCHPCGCaMP')
 
ROI_size_threshold = 500  # pixel count 


#%% load data 
axon_profiles_all = pd.read_pickle(axon_GCaMP_stem / 'LCHPC_axon_GCaMP_all_profiles.pkl')
axon_profiles     = axon_profiles_all[
    (axon_profiles_all['roi_type'] == 'primary') &
    (axon_profiles_all['size'] >= ROI_size_threshold)]  # only primary and > threshold size

peak_list = [axon for axon in axon_profiles.index if axon_profiles['run_onset_peak'][axon]]


#%% main 
all_run = []
all_cue = []
all_rew = []

# per-session peaks (indices)
sess_peaks_run = {}
sess_peaks_cue = {}
sess_peaks_rew = {}

# peak time list 
peak_run_peak_time = []

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    sess_path = all_sess_stem / recname
    
    # load beh file 
    beh_path = beh_stem / f'{recname}.pkl'
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)
    
    # fine stim start 
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    try:
        stim_start = stim_conds.index('2')
    except ValueError:
        stim_start = len(stim_conds)
    
    
    # load cell profiles  
    try:
        trains_run = np.load(sess_path / f'{recname}_all_run.npy',
                             allow_pickle=True).item()
        trains_cue = np.load(sess_path / f'{recname}_all_cue.npy',
                             allow_pickle=True).item()
        trains_rew = np.load(sess_path / f'{recname}_all_rew.npy',
                             allow_pickle=True).item()
    except FileNotFoundError:
        print(f'{recname}: missing files, skipping')
        continue

    # get list of cells 
    axon_list = [axon for axon in list(trains_run.keys()) 
                 if f'{recname} {axon}' in list(axon_profiles.index)]  # filter through valid ROIs
    
    for axon in tqdm(axon_list, total=len(axon_list)):
        curr_trains_run = trains_run[axon][:, 90-30 : 90+30*4]
        curr_trains_cue = trains_cue[axon][:, 90-30 : 90+30*4]
        curr_trains_rew = trains_rew[axon][:, 90-30 : 90+30*4]

        mean_run = np.mean(curr_trains_run, axis=0)
        mean_cue = np.mean(curr_trains_cue, axis=0)
        mean_rew = np.mean(curr_trains_rew, axis=0)

        # per-session peak indices
        sess_peaks_run.setdefault(recname, []).append(np.argmax(mean_run))
        sess_peaks_cue.setdefault(recname, []).append(np.argmax(mean_cue))
        sess_peaks_rew.setdefault(recname, []).append(np.argmax(mean_rew))

        all_run.append(mean_run)
        all_cue.append(mean_cue)
        all_rew.append(mean_rew)
        
        if f'{recname} {axon}' in peak_list:
            peak_run_peak_time.append(np.argmax(mean_run[:60]))  # restrict to 2 seconds around run onset


# sorting
run_argmax = [np.argmax(axon) for axon in all_run]
run_sorted_idx = np.argsort(run_argmax)
all_run_sorted = np.array([normalise(all_run[axon]) for axon in run_sorted_idx])

cue_argmax = [np.argmax(axon) for axon in all_cue]
cue_sorted_idx = np.argsort(cue_argmax)
all_cue_sorted = np.array([normalise(all_cue[axon]) for axon in cue_sorted_idx])

rew_argmax = [np.argmax(axon) for axon in all_rew]
rew_sorted_idx = np.argsort(rew_argmax)
all_rew_sorted = np.array([normalise(all_rew[axon]) for axon in rew_sorted_idx])
    
    
#%% same but with viridis 
# run-aligned
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from run onset (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Dbh+ axons')

image = ax.imshow(all_run_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. dF/F')

for ext in ['.png', '.pdf']:
    fig.savefig(
        pop_map_stem / f'run_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    

# reward-aligned 
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from reward (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Dbh+ axons')

image = ax.imshow(all_rew_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. dF/F')

for ext in ['.png', '.pdf']:
    fig.savefig(
        pop_map_stem / f'rew_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    

# cue-aligned 
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from cue (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Dbh+ axons')

image = ax.imshow(all_cue_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. dF/F')

for ext in ['.png', '.pdf']:
    fig.savefig(
        pop_map_stem / f'cue_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
#%% statistics 
center = 30
window = 15  # .5-s window 

peaks_run = np.argmax(all_run_sorted, axis=1)
peaks_cue = np.argmax(all_cue_sorted, axis=1)
peaks_rew = np.argmax(all_rew_sorted, axis=1)

p_run = np.mean((peaks_run >= center - window) & (peaks_run <= center + window))
p_cue = np.mean((peaks_cue >= center - window) & (peaks_cue <= center + window))
p_rew = np.mean((peaks_rew >= center - window) & (peaks_rew <= center + window))


# organise values for plotting
proportions = [p_run, p_cue, p_rew]
labels = ['run', 'cue', 'rew']
x = np.arange(len(labels))


#%% per-session proportions
sess_p_run = []
sess_p_cue = []
sess_p_rew = []

for recname in paths:
    recname = recname[-17:]  # same as above

    peaks = np.array(sess_peaks_run.get(recname, []))
    if peaks.size > 0:
        sess_p_run.append(np.mean((peaks >= center - window) & (peaks <= center + window)))
    # skip sessions with no tagged cells
    peaks = np.array(sess_peaks_cue.get(recname, []))
    if peaks.size > 0:
        sess_p_cue.append(np.mean((peaks >= center - window) & (peaks <= center + window)))
    peaks = np.array(sess_peaks_rew.get(recname, []))
    if peaks.size > 0:
        sess_p_rew.append(np.mean((peaks >= center - window) & (peaks <= center + window)))
        

#%% plot
fig, ax = plt.subplots(figsize=(2, 3))

y = np.arange(len(labels))

# bars
ax.barh(y, proportions, height=0.35, color='darkgreen')

# y-axis labels
ax.set_yticks(y)
ax.set_yticklabels(labels)

# fix x scaling so bars are visible
ax.set_xlim(0, 1)

y_run, y_cue, y_rew = y

bump = 0.02   # how far to move zero-points off the axis


# --- run ---
if len(sess_p_run) > 0:
    vals = np.array(sess_p_run, float)
    vals[vals == 0] = bump    # inline fix
    ax.scatter(vals,
               np.full(len(vals), y_run),
               s=8, color='darkgreen', edgecolors='k')


# --- cue ---
if len(sess_p_cue) > 0:
    vals = np.array(sess_p_cue, float)
    vals[vals == 0] = bump
    ax.scatter(vals,
               np.full(len(vals), y_cue),
               s=8, color='darkgreen', edgecolors='k')


# --- rew ---
if len(sess_p_rew) > 0:
    vals = np.array(sess_p_rew, float)
    vals[vals == 0] = bump
    ax.scatter(vals,
               np.full(len(vals), y_rew),
               s=8, color='darkgreen', edgecolors='k')


ax.set_yticks(x)
ax.set_yticklabels(labels)
# ax.set_xlim(0, 1)
ax.set_xlabel('prop. with peak ±0.25 s')
ax.set_title('Peak proximity to alignment')

for spine in ['top', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        pop_map_stem / f'peak_proximity_bar{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
#%% statistics for alignment test
n = len(all_run_sorted)

# convert proportions back to counts
k_run = int(p_run * n)
k_cue = int(p_cue * n)
k_rew = int(p_rew * n)

print('\n--- counts and proportions ---')
print(f'run-aligned:    {k_run}/{n}  ({k_run/n*100:.2f}%)')
print(f'cue-aligned:    {k_cue}/{n}  ({k_cue/n*100:.2f}%)')
print(f'reward-aligned: {k_rew}/{n}  ({k_rew/n*100:.2f}%)')

# Two-proportion z-tests
# run vs cue
stat_run_cue, p_run_cue = proportions_ztest(
    [k_run, k_cue],
    [n, n]
)

# run vs reward
stat_run_rew, p_run_rew = proportions_ztest(
    [k_run, k_rew],
    [n, n]
)

# cue vs reward
stat_cue_rew, p_cue_rew = proportions_ztest(
    [k_cue, k_rew],
    [n, n]
)

print('\n--- Two-proportion z-tests ---')
print(f'run vs cue:    z = {stat_run_cue:.3f}, p = {p_run_cue:.3e}')
print(f'run vs reward: z = {stat_run_rew:.3f}, p = {p_run_rew:.3e}')
print(f'cue vs reward: z = {stat_cue_rew:.3f}, p = {p_cue_rew:.3e}')


#%% peak time 
mean_peak_run_peak_time = np.mean(peak_run_peak_time) / 30 - 1  # -1 because aligned to run with a pre of 1 s
sem_peak_run_peak_time  = sem(peak_run_peak_time) / 30

print(f'Peak time: {mean_peak_run_peak_time} ± {sem_peak_run_peak_time} s')