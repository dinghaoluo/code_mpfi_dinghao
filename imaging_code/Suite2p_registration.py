# -*- coding: utf-8 -*-
"""
Created on 4 June 16:30:51 2024
Modified on 8 July 2025

**NOTE: THIS SCRIPT RUNS THE CUSTOMISED SUITE2P-WANG-LAB INSTEAD OF SUITE2P**
**SUITE2P-WANG-LAB CAN BE ACCESSED HERE: 
    https://github.com/the-wang-lab/suite2p-wang-lab**
    
Modified to work on any directory list and as a general-purpose batch registra-
    tion processor 

@author: Dinghao Luo
"""


#%% imports 
from pathlib import Path

import suite2p_functions as s2f

import rec_list

# modified to automatically process all
paths = (
    rec_list.pathHPCGRABNE + 
    rec_list.pathGRABNELCOpto + 
    rec_list.pathGRABNELCOptoDbhBlock + 
    rec_list.pathGRABNETone + 
    rec_list.pathGRABNEToneDbhBlock + 
    rec_list.pathdLightLCOpto +
    rec_list.pathdLightLCOptoDbhBlock + 
    rec_list.pathdLightLCOptoCtrl + 
    rec_list.pathdLightLCOptoInh
    )
    

#%% run all sessions
for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    reg_path = Path(path) / 'suite2p'
    reg_path_alt = Path(path) / 'processed'  # in some older recordings
    if reg_path.exists() or reg_path_alt.exists():  # if registed
        print('Session already registered; skipped')
        continue
    else:
        s2f.register(path)
        
    # if 'no tiffs' is raised, most likely it is due to typos in pathnames