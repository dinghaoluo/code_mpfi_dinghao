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
import sys
import os
import pickle 
from time import time 
from datetime import timedelta

# import pre-processing functions 
sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\behaviour_code\utils')
import behaviour_functions as bf


#%% recording list
sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list

paths = rec_list.pathLCHPCGCaMPImmobile
prefix = 'LCHPCGCaMPImmobile'


#%% main
output_folder = os.path.join(
    r'Z:\Dinghao\code_dinghao\behaviour\all_experiments', prefix
    )
os.makedirs(output_folder, exist_ok=True)

for pathname in paths:    
    recname = pathname[-17:]
    print(f'\nprocessing {recname}...')

    start = time()
    
    txt_path = (rf'Z:\Dinghao\MiceExp\ANMD{recname[1:4]}'
                rf'\{recname[:4]}{recname[5:]}T.txt')
    behavioural_data = bf.process_behavioural_data_immobile_imaging(txt_path)
    
    print(f'session finished ({str(timedelta(seconds=int(time()-start)))})')
    start = time()

    # ... and save
    with open(os.path.join(output_folder, f'{recname}.pkl'), 'wb') as f:
        pickle.dump(behavioural_data, f)

    print(f'session saved ({str(timedelta(seconds=int(time()-start)))})')