# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 17:08:30 2025

call demixing functions on recordings 

@author: Dinghao Luo
"""

#%% imports 
import sys
from pathlib import Path

import numpy as np
import pickle 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\signal_demixing')
import demixing_support_functions as dsf

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathdLightLCOptoInh

paths = [path for path in paths if 'A140' in path]


#%% main 
for path in paths[5:]:
    recname = path.split('\\')[-1]
    print(recname)
    
    p_data = rf'Z:\Dinghao\2p_recording\{recname[:5]}\{recname[:14]}\{recname}\suite2p\plane0'
    file_name = r'data.bin'
    file_name2 = r'data_chan2.bin'
    
    path1 = Path(p_data, file_name)
    path2 = Path(p_data, file_name2)
    
    path_result = Path(p_data, 'result')
    path_result.mkdir(parents=True, exist_ok=True)
    path_masks = Path(path_result, 'masks')
    path_masks.mkdir(parents=True, exist_ok=True)
    
    height = 512
    width = 512
    
    # beh
    beh_file = rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCdLightLCOptoInh\{recname}.pkl'
    with open(beh_file, 'rb') as f:
        beh = pickle.load(f)
    stim_idx = [i for i, t in enumerate(beh['trial_statements']) if t[15]!='0']
    first_stim = stim_idx[0]
    first_stim_frame = beh['run_onset_frames'][first_stim]
    
    # we load using memmap to reduce RAM usage; originally FilterImage was used 
    mov = np.memmap(path1, dtype='int16', mode='r',
                    shape=(first_stim_frame, height, width)).astype(np.float32)
    movr = np.memmap(path2, dtype='int16', mode='r',
                     shape=(first_stim_frame, height, width)).astype(np.float32)
    
    # masks 
    print('generating masks...')
    dsf.generate_masks(mov, movr, path_masks)
    
    # beh
    print('extracting behaviour events...')
    dsf.process_behavior(beh_file, path_result)
    
    # visualisation 
    beh_npz = Path(path_result, f'{recname}_processed_behavior.npz')
    print('visualising grids...')
    dsf.run_event_locked_analysis(mov, movr, path_result, path_masks, beh_npz)
    
    # single trial 
    print('running single-trial regression (dilated)...')
    dsf.run_regression_single_trial_dilated(mov, movr, path_result, path_masks, beh_npz)