# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:35:35 2024
Modified on 21 Jan 2025 Tuesday 

process and save behaviour files as dataframes
for non-imaging experiments

modified: 
    modified for use on immobile sessions

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import sys
import os
import pandas as pd
from time import time 
from datetime import timedelta

# import pre-processing functions 
sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import behaviour_functions as bf


#%% recording list
sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list

pathIm = rec_list.pathIm
prefix = 'immobile'


#%% container
fname = r'Z:\Dinghao\code_dinghao\behaviour\all_{}_sessions.pkl'.format(prefix)
if os.path.exists(fname):
    df = pd.read_pickle(fname)
    print(f'df loaded from {fname}')
    processed_sess = df.index.tolist()
else:
    processed_sess = []
    sess = {
        'speed_times': [],
        'speed_distances': [],
        'lick_times': [],
        'lick_distances': [],
        'lick_maps': [],
        'start_cue_times': [],
        'start_cue_frames': [],
        'reward_times': [],
        'reward_distances': [],
        'reward_frames': [],
        'run_onsets': [],
        'run_onset_frames': [],
        'lick_selectivities': [],
        'trial_statements': [],
        'full_stops': [],
        'bad_trials': [],
        'frame_times': []
        }
    df = pd.DataFrame(sess)


#%% main
for pathname in paths:    
    txt_path = pathname 
    behavioural_data = bf.process_behavioural_data_imaging(txt_path)
    df.loc[recname] = np.array([behavioural_data['speed_times'],
                                behavioural_data['speed_distances'],
                                behavioural_data['lick_times'],
                                behavioural_data['lick_distances'],
                                behavioural_data['lick_maps'],
                                behavioural_data['start_cue_times'],
                                behavioural_data['start_cue_frames'],
                                behavioural_data['reward_times'],
                                behavioural_data['reward_distances'],
                                behavioural_data['reward_frames'],
                                behavioural_data['run_onsets'],
                                behavioural_data['run_onset_frames'],
                                behavioural_data['lick_selectivities'],
                                behavioural_data['trial_statements'],
                                behavioural_data['full_stops'],
                                behavioural_data['bad_trials'],
                                behavioural_data['frame_times']
                                    ],
                                dtype='object')

    print('session finished ({})\n'.format(str(timedelta(seconds=int(time()-start)))))


#%% save dataframe 
df.to_pickle(r'Z:\Dinghao\code_dinghao\behaviour\all_{}_sessions.pkl'.format(prefix))