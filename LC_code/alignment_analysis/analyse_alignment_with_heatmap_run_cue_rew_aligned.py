# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 17:53:46 2025
Modified on 5 Dec 2025 

plot population heatmap, but aligned to cue and rew
modified to include statistics

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import pickle 
from tqdm import tqdm
from scipy.stats import wilcoxon, ttest_rel, sem

import rec_list
paths = rec_list.pathLC

from common_functions import normalise, mpl_formatting, colour_putative, colour_tagged
mpl_formatting()


#%% paths 
LC_stem       = Path('Z:/Dinghao/code_dinghao/LC_ephys')
all_sess_stem = LC_stem / 'all_sessions'

beh_stem      = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')


#%% helper
def _print_paired_stats(x, y, label_x, label_y):
    """
    print mean ± sem, median [IQR], wilcoxon and paired t-test.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # keep only paired entries
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    print(f'\n{label_x} vs {label_y}')
    print(f'n = {len(x)}')

    # mean ± sem
    print(f'{label_x}: mean ± sem = {np.mean(x):.3f} ± {sem(x):.3f}')
    print(f'{label_y}: mean ± sem = {np.mean(y):.3f} ± {sem(y):.3f}')

    # median [IQR]
    mx, q1x, q3x = np.median(x), *np.percentile(x, [25, 75])
    my, q1y, q3y = np.median(y), *np.percentile(y, [25, 75])
    print(f'{label_x}: median [IQR] = {mx:.3f} [{q1x:.3f}, {q3x:.3f}]')
    print(f'{label_y}: median [IQR] = {my:.3f} [{q1y:.3f}, {q3y:.3f}]')

    # tests
    wstat, wp = wilcoxon(x, y)
    tstat, tp = ttest_rel(x, y)

    print(f'wilcoxon: W = {wstat:.3f}, p = {wp:.3e}')
    print(f'paired t: t = {tstat:.3f}, p = {tp:.3e}')


#%% load data 
cell_profiles = pd.read_pickle(LC_stem / 'LC_all_cell_profiles.pkl')

tag_list = [clu for clu in cell_profiles.index if cell_profiles['identity'][clu]=='tagged']
put_list = [clu for clu in cell_profiles.index if cell_profiles['identity'][clu]=='putative']

peak_list = [clu for clu in cell_profiles.index if cell_profiles['run_onset_peak'][clu]]


#%% main 
all_tagged_run = []
all_tagged_cue = []
all_tagged_rew = []
all_putative_run = []
all_putative_cue = []
all_putative_rew = []
all_pooled_run = []
all_pooled_cue = []
all_pooled_rew = []

# per-session peaks (indices)
sess_tagged_peaks_run = {}
sess_tagged_peaks_cue = {}
sess_tagged_peaks_rew = {}
sess_put_peaks_run = {}
sess_put_peaks_cue = {}
sess_put_peaks_rew = {}

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
        trains_run = np.load(sess_path / f'{recname}_all_trains_run.npy',
                             allow_pickle=True).item()
        trains_cue = np.load(sess_path / f'{recname}_all_trains_cue.npy',
                             allow_pickle=True).item()
        trains_rew = np.load(sess_path / f'{recname}_all_trains_rew.npy',
                             allow_pickle=True).item()
    except FileNotFoundError:
        print(f'{recname}: missing files, skipping')
        continue

    # get list of cells 
    clu_list = list(trains_cue.keys())
    
    for clu in tqdm(clu_list, total=len(clu_list)):
        if clu in tag_list or clu in put_list:
            curr_trains_run = trains_run[clu][:stim_start, 3750-1250:3750+1250*4]
            curr_trains_cue = trains_cue[clu][:stim_start, 3750-1250:3750+1250*4]
            curr_trains_rew = trains_rew[clu][:stim_start, 3750-1250:3750+1250*4]

            mean_run = np.mean(curr_trains_run, axis=0)
            mean_cue = np.mean(curr_trains_cue, axis=0)
            mean_rew = np.mean(curr_trains_rew, axis=0)

            all_pooled_run.append(mean_run)
            all_pooled_cue.append(mean_cue)
            all_pooled_rew.append(mean_rew)

            # per-session peak indices
            if clu in tag_list:
                sess_tagged_peaks_run.setdefault(recname, []).append(np.argmax(mean_run))
                sess_tagged_peaks_cue.setdefault(recname, []).append(np.argmax(mean_cue))
                sess_tagged_peaks_rew.setdefault(recname, []).append(np.argmax(mean_rew))

                all_tagged_run.append(mean_run)
                all_tagged_cue.append(mean_cue)
                all_tagged_rew.append(mean_rew)

            if clu in put_list:
                sess_put_peaks_run.setdefault(recname, []).append(np.argmax(mean_run))
                sess_put_peaks_cue.setdefault(recname, []).append(np.argmax(mean_cue))
                sess_put_peaks_rew.setdefault(recname, []).append(np.argmax(mean_rew))

                all_putative_run.append(mean_run)
                all_putative_cue.append(mean_cue)
                all_putative_rew.append(mean_rew)
            
            if clu in peak_list:
                peak_run_peak_time.append(np.argmax(mean_run[:2500]))  # restrict to 2 seconds around run onset
            
# sorting
pooled_run_argmax = [np.argmax(clu) for clu in all_pooled_run]
pooled_run_sorted_idx = np.argsort(pooled_run_argmax)
all_pooled_run_sorted = np.array([normalise(all_pooled_run[clu]) for clu in pooled_run_sorted_idx])

tagged_run_argmax = [np.argmax(clu) for clu in all_tagged_run]
tagged_run_sorted_idx = np.argsort(tagged_run_argmax)
all_tagged_run_sorted = np.array([normalise(all_tagged_run[clu]) for clu in tagged_run_sorted_idx])

putative_run_argmax = [np.argmax(clu) for clu in all_putative_run]
putative_run_sorted_idx = np.argsort(putative_run_argmax)
all_putative_run_sorted = np.array([normalise(all_putative_run[clu]) for clu in putative_run_sorted_idx])

pooled_cue_argmax = [np.argmax(clu) for clu in all_pooled_cue]
pooled_cue_sorted_idx = np.argsort(pooled_cue_argmax)
all_pooled_cue_sorted = np.array([normalise(all_pooled_cue[clu]) for clu in pooled_cue_sorted_idx])

tagged_cue_argmax = [np.argmax(clu) for clu in all_tagged_cue]
tagged_cue_sorted_idx = np.argsort(tagged_cue_argmax)
all_tagged_cue_sorted = np.array([normalise(all_tagged_cue[clu]) for clu in tagged_cue_sorted_idx])

putative_cue_argmax = [np.argmax(clu) for clu in all_putative_cue]
putative_cue_sorted_idx = np.argsort(putative_cue_argmax)
all_putative_cue_sorted = np.array([normalise(all_putative_cue[clu]) for clu in putative_cue_sorted_idx])

pooled_rew_argmax = [np.argmax(clu) for clu in all_pooled_rew]
pooled_rew_sorted_idx = np.argsort(pooled_rew_argmax)
all_pooled_rew_sorted = np.array([normalise(all_pooled_rew[clu]) for clu in pooled_rew_sorted_idx])

tagged_rew_argmax = [np.argmax(clu) for clu in all_tagged_rew]
tagged_rew_sorted_idx = np.argsort(tagged_rew_argmax)
all_tagged_rew_sorted = np.array([normalise(all_tagged_rew[clu]) for clu in tagged_rew_sorted_idx])

putative_rew_argmax = [np.argmax(clu) for clu in all_putative_rew]
putative_rew_sorted_idx = np.argsort(putative_rew_argmax)
all_putative_rew_sorted = np.array([normalise(all_putative_rew[clu]) for clu in putative_rew_sorted_idx])


#%% plotting 
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from run onset (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Pooled Dbh+ cells')

image = ax.imshow(all_pooled_run_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_pooled_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\pooled_run_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    

fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from run onset (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Tagged Dbh+ cells')

image = ax.imshow(all_tagged_run_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_tagged_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\tagged_run_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from run onset (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Putative Dbh+ cells')

image = ax.imshow(all_putative_run_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_putative_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\putative_run_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from cue (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Tagged Dbh+ cells')

image = ax.imshow(all_tagged_cue_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_tagged_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\tagged_cue_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from cue (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Putative Dbh+ cells')

image = ax.imshow(all_putative_cue_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_putative_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\putative_cue_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from reward (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Tagged Dbh+ cells')

image = ax.imshow(all_tagged_rew_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_tagged_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\tagged_rew_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from reward (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Putative Dbh+ cells')

image = ax.imshow(all_putative_rew_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_putative_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\putative_rew_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
    
#%% same but with viridis 
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from run onset (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Pooled Dbh+ cells')

image = ax.imshow(all_pooled_run_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_pooled_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\pooled_run_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    

fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from run onset (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Tagged Dbh+ cells')

image = ax.imshow(all_tagged_run_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_tagged_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\tagged_run_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from run onset (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Putative Dbh+ cells')

image = ax.imshow(all_putative_run_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_putative_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\putative_run_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from cue (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Tagged Dbh+ cells')

image = ax.imshow(all_tagged_cue_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_tagged_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\tagged_cue_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from cue (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Putative Dbh+ cells')

image = ax.imshow(all_putative_cue_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_putative_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\putative_cue_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from reward (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Tagged Dbh+ cells')

image = ax.imshow(all_tagged_rew_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_tagged_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\tagged_rew_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='Time from reward (s)',
       ylabel='Cell #')
ax.set_aspect('equal')
fig.suptitle('Putative Dbh+ cells')

image = ax.imshow(all_putative_rew_sorted, 
                  aspect='auto', cmap='viridis', interpolation='none',
                  extent=[-1, 4, 1, len(all_putative_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='Norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\putative_rew_aligned_viridis{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
#%% statistics 
center = 1250
window = 313

# tagged
peaks_tagged_run = np.argmax(all_tagged_run_sorted, axis=1)
peaks_tagged_cue = np.argmax(all_tagged_cue_sorted, axis=1)
peaks_tagged_rew = np.argmax(all_tagged_rew_sorted, axis=1)

p_tagged_run = np.mean((peaks_tagged_run >= center - window) & (peaks_tagged_run <= center + window))
p_tagged_cue = np.mean((peaks_tagged_cue >= center - window) & (peaks_tagged_cue <= center + window))
p_tagged_rew = np.mean((peaks_tagged_rew >= center - window) & (peaks_tagged_rew <= center + window))

# putative
peaks_put_run = np.argmax(all_putative_run_sorted, axis=1)
peaks_put_cue = np.argmax(all_putative_cue_sorted, axis=1)
peaks_put_rew = np.argmax(all_putative_rew_sorted, axis=1)

p_put_run = np.mean((peaks_put_run >= center - window) & (peaks_put_run <= center + window))
p_put_cue = np.mean((peaks_put_cue >= center - window) & (peaks_put_cue <= center + window))
p_put_rew = np.mean((peaks_put_rew >= center - window) & (peaks_put_rew <= center + window))

# organise values for plotting
proportions = [
    [p_tagged_run, p_put_run],
    [p_tagged_cue, p_put_cue],
    [p_tagged_rew, p_put_rew]
]
labels = ['run', 'cue', 'rew']
x = np.arange(len(labels))


#%% per-session proportions
sess_p_tagged_run = []
sess_p_tagged_cue = []
sess_p_tagged_rew = []
sess_p_put_run = []
sess_p_put_cue = []
sess_p_put_rew = []

for recname in paths:
    recname = recname[-17:]  # same as above
    # tagged
    peaks = np.array(sess_tagged_peaks_run.get(recname, []))
    if peaks.size > 0:
        sess_p_tagged_run.append(np.mean((peaks >= center - window) & (peaks <= center + window)))
    # skip sessions with no tagged cells
    peaks = np.array(sess_tagged_peaks_cue.get(recname, []))
    if peaks.size > 0:
        sess_p_tagged_cue.append(np.mean((peaks >= center - window) & (peaks <= center + window)))
    peaks = np.array(sess_tagged_peaks_rew.get(recname, []))
    if peaks.size > 0:
        sess_p_tagged_rew.append(np.mean((peaks >= center - window) & (peaks <= center + window)))

    # putative
    peaks = np.array(sess_put_peaks_run.get(recname, []))
    if peaks.size > 0:
        sess_p_put_run.append(np.mean((peaks >= center - window) & (peaks <= center + window)))
    peaks = np.array(sess_put_peaks_cue.get(recname, []))
    if peaks.size > 0:
        sess_p_put_cue.append(np.mean((peaks >= center - window) & (peaks <= center + window)))
    peaks = np.array(sess_put_peaks_rew.get(recname, []))
    if peaks.size > 0:
        sess_p_put_rew.append(np.mean((peaks >= center - window) & (peaks <= center + window)))


#%% plot
fig, ax = plt.subplots(figsize=(2, 3))
height = 0.35

for i in range(len(labels)):
    ax.barh(y=i - height/2, width=proportions[i][0], height=height,
            label='tagged' if i == 0 else "", color=colour_tagged)
    ax.barh(y=i + height/2, width=proportions[i][1], height=height,
            label='putative' if i == 0 else "", color=colour_putative)
    
ytag_run = 0 - height/2
yput_run = 0 + height/2

ytag_cue = 1 - height/2
yput_cue = 1 + height/2

ytag_rew = 2 - height/2
yput_rew = 2 + height/2

bump = 0.02   # how far to move zero-points off the axis

# --- run ---
if len(sess_p_tagged_run) > 0:
    vals = np.array(sess_p_tagged_run, float)
    vals[vals == 0] = bump    # inline fix
    ax.scatter(vals,
               np.full(len(vals), ytag_run),
               s=8, color=colour_tagged, edgecolors='k')

if len(sess_p_put_run) > 0:
    vals = np.array(sess_p_put_run, float)
    vals[vals == 0] = bump
    ax.scatter(vals,
               np.full(len(vals), yput_run),
               s=8, color=colour_putative, edgecolors='k')

# --- cue ---
if len(sess_p_tagged_cue) > 0:
    vals = np.array(sess_p_tagged_cue, float)
    vals[vals == 0] = bump
    ax.scatter(vals,
               np.full(len(vals), ytag_cue),
               s=8, color=colour_tagged, edgecolors='k')

if len(sess_p_put_cue) > 0:
    vals = np.array(sess_p_put_cue, float)
    vals[vals == 0] = bump
    ax.scatter(vals,
               np.full(len(vals), yput_cue),
               s=8, color=colour_putative, edgecolors='k')

# --- rew ---
if len(sess_p_tagged_rew) > 0:
    vals = np.array(sess_p_tagged_rew, float)
    vals[vals == 0] = bump
    ax.scatter(vals,
               np.full(len(vals), ytag_rew),
               s=8, color=colour_tagged, edgecolors='k')

if len(sess_p_put_rew) > 0:
    vals = np.array(sess_p_put_rew, float)
    vals[vals == 0] = bump
    ax.scatter(vals,
               np.full(len(vals), yput_rew),
               s=8, color=colour_putative, edgecolors='k')

ax.set_yticks(x)
ax.set_yticklabels(labels)
# ax.set_xlim(0, 1)
ax.set_xlabel('prop. with peak ±0.25 s')
ax.set_title('Peak proximity to alignment')
ax.legend(frameon=False, loc='upper right')

for spine in ['top', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\peak_proximity_bar{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
#%% statistics for alignment test
# TAGGED Dbh+ cells (session-level)
print('\n--- TAGGED LC (session-level) ---')

_print_paired_stats(
    sess_p_tagged_run, sess_p_tagged_cue,
    'run', 'cue'
)

_print_paired_stats(
    sess_p_tagged_run, sess_p_tagged_rew,
    'run', 'reward'
)

_print_paired_stats(
    sess_p_tagged_cue, sess_p_tagged_rew,
    'cue', 'reward'
)


# PUTATIVE Dbh+ cells (session-level)
print('\n--- PUTATIVE LC (session-level) ---')

_print_paired_stats(
    sess_p_put_run, sess_p_put_cue,
    'run', 'cue'
)

_print_paired_stats(
    sess_p_put_run, sess_p_put_rew,
    'run', 'reward'
)

_print_paired_stats(
    sess_p_put_cue, sess_p_put_rew,
    'cue', 'reward'
)