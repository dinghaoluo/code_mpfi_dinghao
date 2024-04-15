# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:42:06 2024

save place cell profiles for each session

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pandas as pd 
import sys
import h5py
import mat73
import os

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCtermopt + rec_list.pathHPCLCopt


#%% dict to contain profiles 
profiles = {'place cells': []}


#%% load information
for 