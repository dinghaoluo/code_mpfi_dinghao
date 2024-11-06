# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:52:50 2024

plot the heatmap of pooled ROI activity of axon-GCaMP animals 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot 
import sys

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import normalise, mpl_formatting
mpl_formatting()

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLCGCaMP = rec_list.pathHPCLCGCaMP


#%% load data 
pooled_ROIs = np.load()