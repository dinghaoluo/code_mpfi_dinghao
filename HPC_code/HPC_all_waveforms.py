# -*- coding: utf-8 -*-
"""
Created on Mon 10 July 14:57:34 2023
Modified on Mon 23 Dec 2024:
    - process all paths

summarise all cell waveforms from HPC recordings 

@author: Dinghao Luo
"""


#%% imports
import numpy as np
import sys
import os 
import mat73
import scipy.io as sio
from datetime import timedelta 
from time import time 

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLC = rec_list.pathHPCLCopt
pathHPCLCterm = rec_list.pathHPCLCtermopt
paths = pathHPCLC + pathHPCLCterm


#%% MAIN
all_cells = {}

for pathname in paths:
    recname = pathname[-17:]
    print(recname)
    
    t0 = time()
    
    # load NeuronQuality file
    nq_file = sio.loadmat(os.path.join(pathname, pathname[-17:]+'.NeuronQuality.mat'))
    
    # assume 6 shanks and load spikes into dict
    clu_count = 2
    for shank in range(1, 7):
        try:
            avspk = nq_file['nqShank{}'.format(shank)]['AvSpk'][0][0]
        except KeyError:
            print(f'this recording does not have shank {shank}')
            continue
        tot_clu = avspk.shape[0]
        
        for clu in range(tot_clu):
            cluname = f'{recname} clu{clu_count} {shank} int{clu+2}'
            clu_count+=1
            
            all_cells[cluname] = avspk[clu, :]
    
    print('done; saving...')
    sess_folder = r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{}'.format(recname)
    os.makedirs(sess_folder, exist_ok=True)
    np.save(r'{}\{}_all_waveforms.npy'.format(sess_folder, recname), 
            all_cells)
    print(f'saved to {sess_folder} ({str(timedelta(seconds=int(time() - t0)))})\n')