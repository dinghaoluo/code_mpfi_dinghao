# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:35:35 2024

process and save behaviour files as dataframes
for non-imaging experiments

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
if r'Z:\Dinghao\code_dinghao' not in sys.path:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list

list_to_process = input('Process which list? 1. HPCLC, 2. HPCLCterm, 3. LC\n')

if list_to_process == '1':
    paths = rec_list.pathHPCLCopt
    prefix = 'HPCLC'
elif list_to_process == '2':
    paths = rec_list.pathHPCLCtermopt
    prefix = 'HPCLCterm'
elif list_to_process == '3':
    paths = rec_list.pathLC
    prefix = 'LC'
else:
    raise Exception('not a valid input; only 1, 2, 3 are supported')


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
        'speed_distance': [],
        'lick_times': [],
        'lick_distance': [],
        'start_cue_times': [],
        'reward_times': [],
        'reward_distance': [],
        'run_onsets': [],
        'lick_selectivities': [],
        'trial_statements': [],
        'full_stops': [],
        'frame_times': []
        }
    df = pd.DataFrame(sess)


#%% main
for pathname in paths:    
    recname = pathname[-17:]
    if recname not in processed_sess:
        print(recname)
    else:
        print(f'{recname} has already been processed...\n')
        continue

    start = time()

    txt_path = r'Z:\Dinghao\MiceExp\ANMD{}\{}\{}\{}T.txt'.format(recname[1:5], recname[-17:-3], recname[-17:], recname[-17:])

    behavioural_data = bf.process_behavioural_data(txt_path)

    df.loc[recname] = np.array([behavioural_data['speed_times'],
                                behavioural_data['speed_distance'],
                                behavioural_data['lick_times'],
                                behavioural_data['lick_distance'],
                                behavioural_data['start_cue_times'],
                                behavioural_data['reward_times'],
                                behavioural_data['reward_distance'],
                                behavioural_data['run_onsets'],
                                behavioural_data['lick_selectivities'],
                                behavioural_data['trial_statements'],
                                behavioural_data['full_stops'],
                                behavioural_data['frame_times']
                                    ],
                                dtype='object')

    print('session finished ({})\n'.format(str(timedelta(seconds=int(time()-start)))))


#%% save dataframe 
df.to_pickle(r'Z:\Dinghao\code_dinghao\behaviour\all_{}_sessions.pkl'.format(prefix))