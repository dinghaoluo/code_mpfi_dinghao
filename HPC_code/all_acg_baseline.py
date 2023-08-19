# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:22:09 2023

process all ACG's and save into .npy

@author: Dinghao Luo
"""


#%% import 
import numpy as np 
import sys
import mat73

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% load
all_acg_baseline = {}

for pathname in pathLC:
    print(pathname[-17:])
    
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    CCGname = filename + '_CCG_Ctrl_Run0_mazeSess1.mat'
    
    CCGfile = mat73.loadmat(CCGname)
    
    CCGval = CCGfile['CCGSessCtrl']['ccgVal']
    CCGt = CCGfile['CCGSessCtrl']['ccgT']
    
    CCGmidt = (len(CCGt)-1)/2
    tot_clu = CCGval.shape[1]
    
    for cluind in range(tot_clu):
        cluname = pathname[-17:]+' clu'+str(cluind+2)
        all_acg_baseline[cluname] = CCGval[:, cluind, cluind]
        
        
#%% save 
print('successfully processed {} cells'.format(len(all_acg_baseline)))
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_acg_baseline.npy', 
        all_acg_baseline)