# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:37:09 2023

analyse and plot single-trial tonic v phasic activity
question: does tonic activity increase in trials with smaller peaks?

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 

# import cluster 2 clunames
cluster2 = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_tagged_clustered_fromall.npy',
                   allow_pickle=True).item()['cluster 2']
