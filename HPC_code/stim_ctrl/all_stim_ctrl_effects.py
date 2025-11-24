# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:04:59 2025

Estimate the first point of stimulation effects for CA1 cells 

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import sem
import pickle 
import pandas as pd 

from behaviour_functions import detect_run_onsets_teensy
from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathHPCLCopt


#%% path stems
all_sess_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions')
all_beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments')


#%% load data
print('loading dataframes...')
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')
df_pyr = cell_profiles[cell_profiles['cell_identity'] == 'pyr']


#%% main 
for path in paths[:1]:
    recname = Path(path).name
    print(f'\n{recname}')

    train_path = all_sess_stem / recname / f'{recname}_all_trains.npy'
    trains = np.load(train_path, allow_pickle=True).item()

    if (all_beh_stem / 'HPCLC' / f'{recname}.pkl').exists():
        with open(all_beh_stem / 'HPCLC' / f'{recname}.pkl', 'rb') as f:
            beh = pickle.load(f)
    else:
        with open(all_beh_stem / 'HPCLCterm' / f'{recname}.pkl', 'rb') as f:
            beh = pickle.load(f)
            
    # get stim and ctrl idx
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds) if cond != '0']
    ctrl_idx = [trial+2 for trial in stim_idx]
    
    # get stim times and truncate pulses 
    pulse_times = np.array(beh['pulse_times'])
    diffs = np.diff(pulse_times)
    split_idx = np.where(diffs >= 1000)[0] + 1
    pulse_trains = np.split(pulse_times, split_idx)
    
    # get first pulse for each train 
    stim_times = [pulse_train[0] for pulse_train in pulse_trains]
    
    # get run onset times
    run_onsets = beh['run_onsets']
    
    curr_df_pyr = df_pyr[df_pyr['recname'] == recname]

    for idx, session in curr_df_pyr.iterrows():
        cluname = idx