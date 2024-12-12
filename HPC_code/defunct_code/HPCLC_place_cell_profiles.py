# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:42:06 2024

save place cell profiles for each session

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pandas as pd 
import scipy.io as sio
import sys

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% dict to contain profiles 
profiles = {'place cells': [],
            'total place cells': []}

df = pd.DataFrame(profiles)


#%% main
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
    place_cells = list(classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0])
    tot_pc = len(place_cells)
    
    df.loc[recname] = np.asarray([place_cells, tot_pc], dtype='object')  # 'object' data type is necessary to encode series of different-dimension lists


#%% save dataframe 
df.to_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_place_cell_profiles.csv')