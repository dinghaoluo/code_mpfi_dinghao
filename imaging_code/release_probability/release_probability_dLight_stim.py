# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:56:31 2025

Quantify the release probability of each axonal ROI

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path 

import numpy as np 
import matplotlib.pyplot as plt 

import rec_list

paths = rec_list.pathdLightLCOpto + \
        rec_list.pathdLightLCOptoDbhBlock


#%% path stems 
all_sess_stem = Path('Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto/all_sessions')

        
#%% main 
for path in paths:
    recname = Path(path).name
    
    roi_path = all_sess_stem / recname / 'processed_data' / f'{recname}_ROI_dict.npy'
    
    if not roi_path.exists():
        print(f'{recname} has no ROI dict')