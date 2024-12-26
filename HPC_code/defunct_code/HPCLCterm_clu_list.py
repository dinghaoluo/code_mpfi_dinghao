# -*- coding: utf-8 -*-
"""
Created on Mon 10 July 10:02:32 2023

get clu_list and save as .npy

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import sys
import h5py
import mat73
import os

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCtermopt


#%% MAIN 
for pathname in pathHPC:
    clu_list = []
    
    recname = pathname[-17:]
    print(recname)
    
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    BehavLFP = mat73.loadmat('{}.mat'.format(pathname+pathname[-18:]+'_BehavElectrDataLFP'))
    Clu = BehavLFP['Clu']
    shank = Clu['shank']
    localClu = Clu['localClu']
        
    tot_clu = len(shank)
    
    for clu in range(tot_clu):
        clu_name = '{} clu{} {} {}'.format(pathname[-17:], clu+2, int(shank[clu]), int(localClu[clu]))
        clu_list.append(clu_name)
    
    clu_list = np.array(clu_list)
    
    outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
    if not os.path.exists(outdirroot):
        os.makedirs(outdirroot)
    np.save('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_clu_list_{}.npy'.format(recname, recname), 
            clu_list)
    print('processed and saved to Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy\n'.format(recname, recname))