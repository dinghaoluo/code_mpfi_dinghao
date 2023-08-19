# -*- coding: utf-8 -*-
"""
Created on Sat 5 Aug 14:23:46 2023

plot sequence given firing rate profiles and place cell classification (from MATLAB pipeline)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 


#%% read data
# We only need the firing rate profiles from my Python pipeline (HPC[LC]_all_train)
# and the classification results from the MATLAB preprocessing pipeline
info = np.load('Z:/Dinghao/code_dinghao/HPC_all/HPC_all_info.npy', 
               allow_pickle=True).item()
classification = sio.loadmat('Z:/Dinghao/MiceExp/ANMD063r/A063r-20230713/A063r-20230713-01/A063r-20230713-01_DataStructure_mazeSection1_TrialType1_FieldWidthLR_20mm_L_Run0.mat')
place_cells = classification['fieldStructSess'][0][0]['neuronFieldAll']
