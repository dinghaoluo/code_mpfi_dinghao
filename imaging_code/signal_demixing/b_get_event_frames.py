"""
Created originally by Yingxue Wang
Modified on 2 Sept 2025

using linear regression to demix motion artefacts from imaging recordings 
    (dual colour)
modified from Yingxue's original pipeline

@author: Dinghao Luo
"""

#%% imports 
import sys 
import importlib

import Align_Beh_Imaging 
importlib.reload(Align_Beh_Imaging)


#%% parameters 
num_used_frames = 10_000


#%% main 
beh_file = r'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCdLightLCOpto\A126i-20250616-01.pkl'
ourput_dir = r'Z:\Dinghao\2p_recording\A126i\A126i-20250616\A126i-20250616-01\suite2p\plane0\result'

Align_Beh_Imaging.process_and_save_behavioral_data(
    input_pkl_path=beh_file,
    output_dir=ourput_dir,
    num_imaging_frames=num_used_frames,
    imaging_rate=30.0,
    do_plots=True
)