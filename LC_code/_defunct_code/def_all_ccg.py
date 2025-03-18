# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:22:09 2023

process all CCG and save into .npy

@author: Dinghao Luo
"""


#%% import 
import numpy as np 
import matplotlib.pyplot as plt
import sys
import mat73

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% parameters
CCGrange_ms = 500  # millisecond
CCGrange = CCGrange_ms * 2  # 20 kHz sampling rate


#%% load
all_ccg = {}

for pathname in pathLC:
    print(pathname[-17:])
    
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    CCGname = filename + '_CCG_Run0.mat'
    
    CCGfile = mat73.loadmat(CCGname)
    
    CCGval = CCGfile['CCGSess'][0]['ccgVal']
    CCGt = CCGfile['CCGSess'][0]['ccgT']
    
    CCGmidt = (len(CCGt)-1)/2
    tot_clu = CCGval.shape[1]
    
    for cluind in range(tot_clu):
        cluname = pathname[-17:]+' clu'+str(cluind+2)
        t1 = int(CCGmidt-CCGrange/2); t2 = int(CCGmidt+CCGrange/2)
        all_ccg[cluname] = CCGval[t1:t2, cluind, cluind]
        
        
#%% save 
print('successfully processed {} cells'.format(len(all_ccg)))
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_ccg.npy', 
        all_ccg)