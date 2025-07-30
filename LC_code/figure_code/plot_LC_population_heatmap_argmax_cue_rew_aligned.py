# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 17:53:46 2025

plot population heatmap, but aligned to cue and rew

@author: Dinghao Luo
"""

#%% imports 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import pickle 
from tqdm import tqdm

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import normalise, mpl_formatting
mpl_formatting()


#%% load data 
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )
tag_list = [clu for clu in cell_profiles.index if cell_profiles['identity'][clu]=='tagged']
put_list = [clu for clu in cell_profiles.index if cell_profiles['identity'][clu]=='putative']


#%% main 
all_tagged_run = []
all_tagged_cue = []
all_tagged_rew = []
all_putative_run = []
all_putative_cue = []
all_putative_rew = []

for path in paths:
    recname = path[-17:]
    print(recname)
    
    sess_folder = rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}'
    
    # load beh file 
    with open(rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\LC\{recname}.pkl', 'rb') as f:
        beh = pickle.load(f)
    
    # fine stim start 
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    try:
        stim_start = stim_conds.index('2')
    except ValueError:
        stim_start = len(stim_conds)
    
    # load cell profiles  
    try:
        trains_run = np.load(rf'{sess_folder}\{recname}_all_trains_run.npy',
                             allow_pickle=True).item()
        trains_cue = np.load(rf'{sess_folder}\{recname}_all_trains_cue.npy',
                             allow_pickle=True).item()
        trains_rew = np.load(rf'{sess_folder}\{recname}_all_trains_rew.npy',
                             allow_pickle=True).item()
    except FileNotFoundError:
        print(f'{recname}: missing files, skipping')
        continue

    # get list of cells 
    clu_list = list(trains_cue.keys())
    
    for clu in tqdm(clu_list, total=len(clu_list)):
        if clu in tag_list:
            curr_trains_run = trains_run[clu][:stim_start, 3750-1250:3750+1250*4]
            curr_trains_cue = trains_cue[clu][:stim_start, 3750-1250:3750+1250*4]
            curr_trains_rew = trains_rew[clu][:stim_start, 3750-1250:3750+1250*4]
            all_tagged_run.append(np.mean(curr_trains_run, axis=0))
            all_tagged_cue.append(np.mean(curr_trains_cue, axis=0))
            all_tagged_rew.append(np.mean(curr_trains_rew, axis=0))
        if clu in put_list:
            curr_trains_run = trains_run[clu][:stim_start, 3750-1250:3750+1250*4]
            curr_trains_cue = trains_cue[clu][:stim_start, 3750-1250:3750+1250*4]
            curr_trains_rew = trains_rew[clu][:stim_start, 3750-1250:3750+1250*4]
            all_putative_run.append(np.mean(curr_trains_run, axis=0))
            all_putative_cue.append(np.mean(curr_trains_cue, axis=0))
            all_putative_rew.append(np.mean(curr_trains_rew, axis=0))
            
# sorting
tagged_run_argmax = [np.argmax(clu) for clu in all_tagged_run]
tagged_run_sorted_idx = np.argsort(tagged_run_argmax)
all_tagged_run_sorted = np.array([normalise(all_tagged_run[clu]) for clu in tagged_run_sorted_idx])

putative_run_argmax = [np.argmax(clu) for clu in all_putative_run]
putative_run_sorted_idx = np.argsort(putative_run_argmax)
all_putative_run_sorted = np.array([normalise(all_putative_run[clu]) for clu in putative_run_sorted_idx])

tagged_cue_argmax = [np.argmax(clu) for clu in all_tagged_cue]
tagged_cue_sorted_idx = np.argsort(tagged_cue_argmax)
all_tagged_cue_sorted = np.array([normalise(all_tagged_cue[clu]) for clu in tagged_cue_sorted_idx])

putative_cue_argmax = [np.argmax(clu) for clu in all_putative_cue]
putative_cue_sorted_idx = np.argsort(putative_cue_argmax)
all_putative_cue_sorted = np.array([normalise(all_putative_cue[clu]) for clu in putative_cue_sorted_idx])

tagged_rew_argmax = [np.argmax(clu) for clu in all_tagged_rew]
tagged_rew_sorted_idx = np.argsort(tagged_rew_argmax)
all_tagged_rew_sorted = np.array([normalise(all_tagged_rew[clu]) for clu in tagged_rew_sorted_idx])

putative_rew_argmax = [np.argmax(clu) for clu in all_putative_rew]
putative_rew_sorted_idx = np.argsort(putative_rew_argmax)
all_putative_rew_sorted = np.array([normalise(all_putative_rew[clu]) for clu in putative_rew_sorted_idx])


#%% plotting 
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from run-onset (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('tagged Dbh+ cells')

image = ax.imshow(all_tagged_run_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_tagged_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\tagged_run_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from run-onset (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('putative Dbh+ cells')

image = ax.imshow(all_putative_run_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_putative_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\putative_run_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from cue (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('tagged Dbh+ cells')

image = ax.imshow(all_tagged_cue_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_tagged_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\tagged_cue_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from cue (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('putative Dbh+ cells')

image = ax.imshow(all_putative_cue_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_putative_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\putative_cue_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from reward (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('tagged Dbh+ cells')

image = ax.imshow(all_tagged_rew_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_tagged_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 80])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\tagged_rew_aligned{ext}',
        dpi=300,
        bbox_inches='tight')
    
    
fig, ax = plt.subplots(figsize=(2.6,2.1))
ax.set(xlabel='time from reward (s)',
       ylabel='cell #')
ax.set_aspect('equal')
fig.suptitle('putative Dbh+ cells')

image = ax.imshow(all_putative_rew_sorted, 
                  aspect='auto', cmap='Greys', interpolation='none',
                  extent=[-1, 4, 1, len(all_putative_run_sorted)])
plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 100, 200])

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\putative_rew_aligned{ext}',
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


#%% plot
fig, ax = plt.subplots(figsize=(2, 3))
height = 0.35

for i in range(len(labels)):
    ax.barh(y=i - height/2, width=proportions[i][0], height=height,
            label='tagged' if i == 0 else "", color='royalblue')
    ax.barh(y=i + height/2, width=proportions[i][1], height=height,
            label='putative' if i == 0 else "", color='darkorange')

ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.set_xlim(0, 1)
ax.set_xlabel('prop. with peak ±0.25 s')
ax.set_title('Peak proximity to alignment')
ax.legend(frameon=False, loc='upper right')

# remove top, right, bottom spines
for spine in ['top', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\population_maps\peak_proximity_bar{ext}',
        dpi=300,
        bbox_inches='tight')