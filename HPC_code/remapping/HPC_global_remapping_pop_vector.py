# -*- coding: utf-8 -*-
"""
Created on Fri May  9 19:43:16 2025

checking global remapping between baseline and ctrl/stim 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
import sys 
import pandas as pd 
import os 
from scipy.stats import pearsonr

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, normalise_to_all
from plotting_functions import plot_violin_with_scatter
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLC = rec_list.pathHPCLCopt
pathHPCLCterm = rec_list.pathHPCLCtermopt


#%% load dataframe 
print('loading dataframe...')
cell_profiles = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 

df_ON = df_pyr[df_pyr['class']=='run-onset ON']
df_OFF = df_pyr[df_pyr['class']=='run-onset OFF']


#%% main 
r_A_B = []
r_AB_ctrl = []
r_AB_stim = []
r_ctrl_stim = []

r_A_B_ON = []
r_AB_ctrl_ON = []
r_AB_stim_ON = []
r_ctrl_stim_ON = []

r_A_B_OFF = []
r_AB_ctrl_OFF = []
r_AB_stim_OFF = []
r_ctrl_stim_OFF = []

for path in pathHPCLCterm:
    recname = path[-17:]
    print(f'\n{recname}')
    
    trains = np.load(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}\{recname}_all_trains.npy',
        allow_pickle=True
        ).item()
    
    if os.path.exists(
            rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl'
            ):
        with open(
                rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl',
                'rb'
                ) as f:
            beh = pickle.load(f)
    else:
        with open(
                rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLCterm\{recname}.pkl',
                'rb'
                ) as f:
            beh = pickle.load(f)
    
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds)
                if cond!='0']
    ctrl_idx = [trial+2 for trial in stim_idx]
    baseline_idx = list(np.arange(stim_idx[0]))
    
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    curr_df_pyr_ON = curr_df_pyr[curr_df_pyr['class']=='run-onset ON']
    curr_df_pyr_OFF = curr_df_pyr[curr_df_pyr['class']=='run-onset OFF']
    pyr_list = curr_df_pyr.index.tolist()
    pyr_ON_list = curr_df_pyr_ON.index.tolist()
    pyr_OFF_list = curr_df_pyr_OFF.index.tolist()
    
    r_A_B_curr = []
    r_A_B_curr_ON = []
    r_A_B_curr_OFF = []
    for i in range(5):  # hard-coded for now; bootstrapping for baseline half-halves
        np.random.shuffle(baseline_idx)
        
        # split into two equal halves
        midpoint = len(baseline_idx) // 2
        A_idx = baseline_idx[:midpoint]
        B_idx = baseline_idx[midpoint:]
        
        pop_vector_A = []
        pop_vector_B = []
        pop_vector_A_ON = []
        pop_vector_B_ON = []
        pop_vector_A_OFF = []
        pop_vector_B_OFF = []
        for cluname in pyr_list:  # accumulate pop vectors 
            mean_A = list(np.mean(trains[cluname][A_idx, :], axis=0))
            mean_B = list(np.mean(trains[cluname][B_idx, :], axis=0))
            pop_vector_A.extend(mean_A)
            pop_vector_B.extend(mean_B)
                        
            if cluname in pyr_ON_list:
                pop_vector_A_ON.extend(mean_A)
                pop_vector_B_ON.extend(mean_B)
            if cluname in pyr_OFF_list:
                pop_vector_A_OFF.extend(mean_A)
                pop_vector_B_OFF.extend(mean_B)
        
        r, p = pearsonr(pop_vector_A, pop_vector_B)  # half-half corr
        r_A_B_curr.append(r)
        if len(pyr_ON_list)>0:
            r, p = pearsonr(pop_vector_A_ON, pop_vector_B_ON)
            r_A_B_curr_ON.append(r)
        if len(pyr_ON_list)>0:
            r, p = pearsonr(pop_vector_A_OFF, pop_vector_B_OFF)
            r_A_B_curr_OFF.append(r)
    
    r_A_B.append(np.mean(r_A_B_curr))  # mean over bootstrapped r's
    if len(pyr_ON_list)>0:
        r_A_B_ON.append(np.mean(r_A_B_curr_ON))
    if len(pyr_ON_list)>0:
        r_A_B_OFF.append(np.mean(r_A_B_curr_OFF))
    
    # next, calculate AB ctrl and AB stim
    pop_vector_AB = []
    pop_vector_ctrl = []
    pop_vector_stim = []
    pop_vector_AB_ON = []
    pop_vector_ctrl_ON = []
    pop_vector_stim_ON = []
    pop_vector_AB_OFF = []
    pop_vector_ctrl_OFF = []
    pop_vector_stim_OFF = []
    for cluname in pyr_list:
        mean_AB = list(np.mean(trains[cluname][baseline_idx, :], axis=0))
        mean_ctrl = list(np.mean(trains[cluname][ctrl_idx, :], axis=0))
        mean_stim = list(np.mean(trains[cluname][stim_idx, :], axis=0))
        pop_vector_AB.extend(mean_AB)
        pop_vector_ctrl.extend(mean_ctrl)
        pop_vector_stim.extend(mean_stim)
        
        if cluname in pyr_ON_list:
            pop_vector_AB_ON.extend(mean_AB)
            pop_vector_ctrl_ON.extend(mean_ctrl)
            pop_vector_stim_ON.extend(mean_stim)
        if cluname in pyr_OFF_list:
            pop_vector_AB_OFF.extend(mean_AB)
            pop_vector_ctrl_OFF.extend(mean_ctrl)
            pop_vector_stim_OFF.extend(mean_stim)
        
    r, p = pearsonr(pop_vector_AB, pop_vector_ctrl)
    r_AB_ctrl.append(r)
    r, p = pearsonr(pop_vector_AB, pop_vector_stim)
    r_AB_stim.append(r)
    r, p = pearsonr(pop_vector_ctrl, pop_vector_stim)
    r_ctrl_stim.append(r)
    
    if len(pyr_ON_list)>0:
        r, p = pearsonr(pop_vector_AB_ON, pop_vector_ctrl_ON)
        r_AB_ctrl_ON.append(r)
        r, p = pearsonr(pop_vector_AB_ON, pop_vector_stim_ON)
        r_AB_stim_ON.append(r)
        r, p = pearsonr(pop_vector_ctrl_ON, pop_vector_stim_ON)
        r_ctrl_stim_ON.append(r)
        
    if len(pyr_OFF_list)>0:
        r, p = pearsonr(pop_vector_AB_OFF, pop_vector_ctrl_OFF)
        r_AB_ctrl_OFF.append(r)
        r, p = pearsonr(pop_vector_AB_OFF, pop_vector_stim_OFF)
        r_AB_stim_OFF.append(r)
        r, p = pearsonr(pop_vector_ctrl_OFF, pop_vector_stim_OFF)
        r_ctrl_stim_OFF.append(r)
        

#%% statistics 
from scipy.stats import ranksums 
# create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
conditions = ['overall', 'ON', 'OFF']

# prepare data lists
data_sets = [
    [r_A_B, r_AB_ctrl, r_AB_stim, r_ctrl_stim],
    [r_A_B_ON, r_AB_ctrl_ON, r_AB_stim_ON, r_ctrl_stim_ON],
    [r_A_B_OFF, r_AB_ctrl_OFF, r_AB_stim_OFF, r_ctrl_stim_OFF]
]

group_names = ['A vs B', 'AB ctrl', 'AB stim', 'ctrl vs stim']

# plot each subplot
for ax, condition, data in zip(axes, conditions, data_sets):
    ax.boxplot(data, labels=group_names)
    ax.set_title(f'{condition} comparisons')
    ax.set_ylabel('value')
    
    # perform ranksum tests and annotate
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            stat, p = ranksums(data[i], data[j])
            y_max = max(max(data[i]), max(data[j]))
            # ax.plot(np.arange(1,5), data, ls='dashed')
            ax.text((i + j) / 2 + 1, y_max * 1.05, f'p={p:.3f}',
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()