# -*- coding: utf-8 -*-
"""
Created on Fri 20 Dec 17:30:12 2024

plot rasters of HPC cells in ctrl and stim trials 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 
import scipy.io as sio 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLC = rec_list.pathHPCLCopt
pathHPCLCterm = rec_list.pathHPCLCtermopt
paths = pathHPCLC + pathHPCLCterm


#%% parameters 
time_bef = 1  # second 
time_aft = 4
samp_freq = 1250  # Hz 


#%% main 
for pathname in paths:
    recname = pathname[-17:]
    print(recname)
    
    # make root folders for figures
    root = r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{}\rasters_ctrl_stim'.format(recname)
    os.makedirs(f'{root}_pyr', exist_ok=True)
    os.makedirs(f'{root}_int', exist_ok=True)
    
    # load rasters for this recording 
    raster_file = np.load(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{}\{}_all_rasters.npy'
        .format(recname, recname),
        allow_pickle=True
        ).item()
        
    tot_time = 1250 + 5000  # 1 s before, 4 s after 
    
    # if each neurones is an interneurone or a pyramidal cell 
    info = sio.loadmat(
        '{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'
        .format(pathname, recname)
        )
    rec_info = info['rec'][0][0]
    intern_map = rec_info['isIntern'][0]  # 1 for int, 0 for pyr
    
    # behaviour parameters 
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat(
        '{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
        .format(pathname, recname)
        )
    stim_idx = np.where(behPar['behPar']['stimOn'][0][0][0]!=0)[0]
    ctrl_idx = stim_idx + 2
    
    for i, (key, value) in enumerate(raster_file.items()):
        cluname = key
        cell_identity = 'int' if intern_map[i] else 'pyr'
    
        ctrl_matrix = value[ctrl_idx]
        stim_matrix = value[stim_idx]
        
        # check 
        if ctrl_matrix.shape != stim_matrix.shape:
            raise ValueError('ctrl and stim matrices do not match in shape!')
        
        # plotting 
        fig, axs = plt.subplots(2, 1, figsize=(2.1,2.1))
        fig.suptitle(cluname, fontsize=10)
        
        for line in range(len(ctrl_idx)):
            axs[0].scatter(np.where(ctrl_matrix[line]==1)[0]/samp_freq-3,
                           [line+1]*int(sum(ctrl_matrix[line])),
                           c='grey', ec='none', s=2)
            axs[1].scatter(np.where(stim_matrix[line]==1)[0]/samp_freq-3,
                           [line+1]*int(sum(stim_matrix[line])),
                           c='royalblue', ec='none', s=2)
                        
        for i in range(2):
            axs[i].set(xticks=[0,2,4], xlim=(-1, 4),
                       ylabel='trial #')
            for p in ['top', 'right']:
                axs[i].spines[p].set_visible(False)
            
        # only set xlabel for ax 1
        axs[1].set(xlabel='time from run-onset (s)')

        fig.tight_layout()
        plt.show()
        
        # save figure
        for ext in ['.png', '.pdf']:
            fig.savefig(f'{root}_{cell_identity}\{cluname}{ext}',
                        dpi=300,
                        bbox_inches='tight')

        plt.close(fig)