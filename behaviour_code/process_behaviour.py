# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:35:35 2024
Modified on Wed 30 Apr 18:02:32 2025

process and save behaviour files as dataframes, session-by-session
modifications:
    - now uses pickle to save single session dictionaries, instead of 
        dataframes like before

@author: Dinghao Luo
"""


#%% imports 
import pickle 
import sys
import os
from time import time 
from datetime import timedelta

# import pre-processing functions 
sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\behaviour_code\utils')
import behaviour_functions as bf


#%% recording list
sys.path.append(r'Z:\Dinghao\code_dinghao')
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
    output_folder = os.path.join(
        r'Z:\Dinghao\code_dinghao\behaviour\all_experiments', prefix
        )
    os.makedirs(output_folder, exist_ok=True)

    for pathname in paths:    
        recname = pathname[-17:]
        print(f'\nprocessing {recname}...')

        start = time()

        # process 
        if list_to_process in ['1', '2', '3']:
            txt_path = (rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}'
                        rf'\{recname[-17:-3]}\{recname[-17:]}'
                        rf'\{recname[-17:]}T.txt')
            behavioural_data = bf.process_behavioural_data(txt_path)
        elif list_to_process in ['4', '5']:
            txt_path = (rf'Z:\Dinghao\MiceExp\ANMD{recname[1:4]}'
                        rf'\{recname[:4]}{recname[5:]}T.txt')
            behavioural_data = bf.process_behavioural_data_imaging(txt_path)
            
        print(f'session finished ({str(timedelta(seconds=int(time()-start)))})')
        start = time()

        # ... and save
        with open(os.path.join(output_folder, f'{recname}.pkl'), 'wb') as f:
            pickle.dump(behavioural_data, f)

        print(f'session saved ({str(timedelta(seconds=int(time()-start)))})')


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
