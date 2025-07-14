# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 20:26:59 2025

plot example sess ref's and release map 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
from PIL import Image 


#%% main 
recname = 'A126i-20250606-01'

ref1 = np.load(
    rf'Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto/all_sessions/{recname}/processed_data/{recname}_ref_mat_ch1.npy',
    allow_pickle=True)
ref2 = np.load(
    rf'Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto/all_sessions/{recname}/processed_data/{recname}_ref_mat_ch2.npy',
    allow_pickle=True)
release_map = np.load(
    rf'Z:/Dinghao/code_dinghao/HPC_dLight_LC_opto/all_sessions/{recname}/processed_data/{recname}_release_map.npy',
    allow_pickle=True
    )

ref1_i = Image.fromarray(ref1)
ref1_i.save(rf'Z:\Dinghao\paper\figures_for_yingxue\{recname}_ref1.tif')

ref2_i = Image.fromarray(ref2)
ref2_i.save(rf'Z:\Dinghao\paper\figures_for_yingxue\{recname}_ref2.tif')

release_map_i = Image.fromarray(release_map)
release_map_i.save(rf'Z:\Dinghao\paper\figures_for_yingxue\{recname}_release_map.tif')