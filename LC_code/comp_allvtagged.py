# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:44:54 2023

plot all v tagged FWHM and firing rates (compare)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import sys

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% load files
wfs = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
              allow_pickle=True).item()