# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:35:35 2024

process and save behaviour files as dataframes

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import sys
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

list_to_process = input(
    'Process which list? 1. HPCLC, 2. HPCLCterm, 3. LC, 4. HPCGRABNE, 5. HPCLCGCaMP, 6. PROCESS ALL\n'
    )

if list_to_process == '1':
    paths = rec_list.pathHPCLCopt
    prefix = 'HPCLC'
elif list_to_process == '2':
    paths = rec_list.pathHPCLCtermopt
    prefix = 'HPCLCterm'
elif list_to_process == '3':
    paths = rec_list.pathLC
    prefix = 'LC'
elif list_to_process == '4':
    paths = rec_list.pathHPCGRABNE
    prefix = 'HPCGRABNE'
elif list_to_process == '5':
    paths = rec_list.pathLCHPCGCaMP
    prefix = 'LCHPCGCaMP'


#%% main 
def process_all(prefix, list_to_process, paths):
    fname = rf'Z:\Dinghao\code_dinghao\behaviour\all_{prefix}_sessions.pkl'
    # if os.path.exists(fname):
    if False:  # always re-process everything, 23 Apr 2025 Dinghao 
        df = pd.read_pickle(fname)
        print(f'df loaded from {fname}')
        processed_sess = df.index.tolist()
    else:
        processed_sess = []
        if list_to_process in ['1', '2', '3']:
            sess = {
                'speed_times': [],
                'speed_times_aligned': [],
                'speed_distances': [],
                'lick_times': [],
                'lick_times_aligned': [],
                'lick_distances': [],
                'lick_maps': [],
                'start_cue_times': [],
                'reward_times': [],
                'reward_distances': [],
                'run_onsets': [],
                'lick_selectivities': [],
                'trial_statements': [],
                'full_stops': [],
                'bad_trials': [],
                'frame_times': [],
                'speed_times_full': []
                }
        elif list_to_process in ['4', '5']:
            sess = {
                'speed_times': [],
                'speed_times_aligned': [],
                'speed_distances': [],
                'lick_times': [],
                'lick_times_aligned': [],
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
                'frame_times': [],
                'speed_times_full': []
                }
        df = pd.DataFrame(sess)
    
    for pathname in paths:    
        recname = pathname[-17:]
        if recname not in processed_sess:
            print(recname)
        else:
            print(f'{recname} has already been processed...\n')
            continue  # now we re-process everything
    
        start = time()
    
        if list_to_process in ['1', '2', '3']:
            txt_path = r'Z:\Dinghao\MiceExp\ANMD{}\{}\{}\{}T.txt'.format(recname[1:5], recname[-17:-3], recname[-17:], recname[-17:])
            behavioural_data = bf.process_behavioural_data(txt_path)
            df.loc[recname] = np.array([behavioural_data['speed_times'],
                                        behavioural_data['speed_times_aligned'],
                                        behavioural_data['speed_distances'],
                                        behavioural_data['lick_times'],
                                        behavioural_data['lick_times_aligned'],
                                        behavioural_data['lick_distances'],
                                        behavioural_data['lick_maps'],
                                        behavioural_data['start_cue_times'],
                                        behavioural_data['reward_times'],
                                        behavioural_data['reward_distances'],
                                        behavioural_data['run_onsets'],
                                        behavioural_data['lick_selectivities'],
                                        behavioural_data['trial_statements'],
                                        behavioural_data['full_stops'],
                                        behavioural_data['bad_trials'],
                                        behavioural_data['frame_times'],
                                        behavioural_data['speed_times_full']
                                            ],
                                        dtype='object')
        elif list_to_process in ['4', '5']:
            txt_path = r'Z:\Dinghao\MiceExp\ANMD{}\{}{}T.txt'.format(recname[1:4], recname[:4], recname[5:])
            behavioural_data = bf.process_behavioural_data_imaging(txt_path)
            df.loc[recname] = np.array([behavioural_data['speed_times'],
                                        behavioural_data['speed_times_aligned'],
                                        behavioural_data['speed_distances'],
                                        behavioural_data['lick_times'],
                                        behavioural_data['lick_times_aligned'],
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
                                        behavioural_data['frame_times'],
                                        behavioural_data['speed_times_full']
                                            ],
                                        dtype='object')
    
        print('session finished ({})\n'.format(str(timedelta(seconds=int(time()-start)))))
    
    df.to_pickle(r'Z:\Dinghao\code_dinghao\behaviour\all_{}_sessions.pkl'.format(prefix))
    
if __name__ == '__main__':
    if list_to_process in ['1', '2', '3', '4', '5']:
        process_all(prefix, list_to_process, paths)
    elif list_to_process == '6':
        lists = ['HPCLC', 'HPCLCterm', 'LC', 'HPCGRABNE', 'LCHPCGCaMP']
        paths_list = [
            rec_list.pathHPCLCopt,
            rec_list.pathHPCLCtermopt,
            rec_list.pathLC,
            rec_list.pathHPCGRABNE,
            rec_list.pathLCHPCGCaMP
            ]
        for i in range(5):
            print(f'processing {lists[i]}...\n')
            process_all(lists[i], str(i+1), paths_list[i])
    else:
        raise Exception('not a valid input; only 1, 2, 3, 4, 5 and 6 are supported')