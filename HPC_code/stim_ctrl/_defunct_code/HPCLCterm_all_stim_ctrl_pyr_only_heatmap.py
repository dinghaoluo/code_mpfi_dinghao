# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:14:01 2024

spike rate map for single place cells across trials

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import scipy.io as sio 
import sys 
import matplotlib.pyplot as plt 

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if (r'Z:\Dinghao\code_mpfi_dinghao\iutils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCtermopt


#%% main 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    # load spike train 
    trains = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
                          allow_pickle=True).item().values())
    
    # load place cells
    FieldSpCorrAligned = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
    place_cells = FieldSpCorrAligned['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]
    tot_pc = len(place_cells)
    
    # behaviour parameters 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    ctrl_trials = stim_trials+2
    
    # plot single session pc only
    if tot_pc==0:  # in case there is no place cell in this session
        continue
    
    for pc in place_cells:
        curr_train = trains[pc-2]
        ctrl_map = np.zeros((len(ctrl_trials), 5*1250))
        stim_map = np.zeros((len(stim_trials), 5*1250))
        for i, trial in enumerate(ctrl_trials):
            trial_length = len(curr_train[trial])-2500
            if trial_length>0:
                if trial_length<5*1250:
                    ctrl_map[i, :trial_length] = curr_train[trial][2500:trial_length+2500]
                else:
                    ctrl_map[i,:] = curr_train[trial][2500:2500+5*1250]
        for i, trial in enumerate(stim_trials):
            trial_length = len(curr_train[trial])-2500
            if trial_length>0:
                if trial_length<5*1250:
                    stim_map[i, :trial_length] = curr_train[trial][2500:trial_length+2500]
                else:
                    stim_map[i,:] = curr_train[trial][2500:2500+5*1250]
                
        fig, axs = plt.subplots(1,2, figsize=(4,2.4))
        axs[0].imshow(ctrl_map, aspect='auto', interpolation='none',
                      extent=[-1,4,1,len(ctrl_trials)+1])
        axs[1].imshow(stim_map, aspect='auto', interpolation='none',
                      extent=[-1,4,1,len(ctrl_trials)+1])
        axs[0].set(title='ctrl')
        axs[1].set(title='stim')
        for i in range(2):
            axs[i].set(xlabel='time (s)', ylabel='trial #',
                       xticks=[0,2,4], xticklabels=[0,2,4])
        fig.suptitle('{}\nclu{}'.format(recname, pc))
        fig.tight_layout()
        plt.show()
        
        fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\single_place_cell_heatmaps\HPCLCterm\{}_clu{}.png'.format(recname, pc),
                    dpi=300,
                    bbox_inches='tight')
        
        plt.close(fig)