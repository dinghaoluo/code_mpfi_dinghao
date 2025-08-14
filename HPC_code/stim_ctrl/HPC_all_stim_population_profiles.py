# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:32:36 2025

Quantify the effect of LC stim on HPC population activity after run

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import pickle
import sys 
import os 
import matplotlib.pyplot as plt

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

from plotting_functions import plot_violin_with_scatter


#%% load data 
print('loading dataframes...')

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
    )
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']


#%% parameters 
SAMP_FREQ = 1250  # Hz 
PRE = 1  # s 
POST = 4  # s 

TAXIS = np.linspace(-PRE, POST, (PRE+POST) * SAMP_FREQ)


#%% main
population_ctrl_means = []
population_stim_means = []

for path in paths:
    recname = path[-17:]
    print(f'\n\n{recname}')

    trains = np.load(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}\{recname}_all_trains.npy',
        allow_pickle=True
        ).item()

    if os.path.exists(
            rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl'):
        with open(rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl', 'rb') as f:
            beh = pickle.load(f)
    else:
        with open(rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLCterm\{recname}.pkl', 'rb') as f:
            beh = pickle.load(f)

    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds) if cond != '0']
    ctrl_idx = [trial + 2 for trial in stim_idx]

    pulse_times = beh['pulse_times'][1:]
    run_onsets = beh['run_onsets'][1:len(pulse_times)+1]
    pulse_onsets = [p[0] - r if p else [] for p, r in zip(pulse_times, run_onsets)]
    filtered_stim_idx = []
    filtered_ctrl_idx = []
    for stim_i, ctrl_i in zip(stim_idx, ctrl_idx):
        offset = pulse_onsets[stim_i]
        if isinstance(offset, (int, float)) and offset <= SAMP_FREQ:
            filtered_stim_idx.append(stim_i)
            filtered_ctrl_idx.append(ctrl_i)
    print(f'filtered out {len(stim_idx) - len(filtered_stim_idx)} bad trial pairs')

    if len(filtered_stim_idx) == 0:
        print('no trials left; abort')
        continue

    curr_df_pyr = df_pyr[df_pyr['recname'] == recname]

    for cluname, row in tqdm(curr_df_pyr.iterrows(),
                             total=len(curr_df_pyr)):
        train = trains[cluname]

        # get list of mean
        ctrl_means = []
        stim_means = []
        for idx in filtered_ctrl_idx:
            ctrl_means.append(np.mean(train[idx][3750 : 3750 + POST*SAMP_FREQ]))
        for idx in filtered_stim_idx:
            stim_means.append(np.mean(train[idx][3750 : 3750 + POST*SAMP_FREQ]))
    
    population_ctrl_means.append(np.mean(ctrl_means))
    population_stim_means.append(np.mean(stim_means))


#%%
plot_violin_with_scatter(population_ctrl_means, population_stim_means,
                         'grey', 'royalblue')