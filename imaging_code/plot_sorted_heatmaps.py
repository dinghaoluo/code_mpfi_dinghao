# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:32:01 2024

plot run-onset- and reward-aligned sorted heatmaps of each session 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 


#%% load data 
behav = pd.read_pickle('Z:/Dinghao/code_dinghao/behaviour/all_GRABNE_sessions.pkl')


#%%
grid_traces = np.load('Z:/Dinghao/2p_recording/A093i/A093i-20240620/A093i-20240620-01_grid_extract/grid_traces_31.npy',
                      allow_pickle=True)

