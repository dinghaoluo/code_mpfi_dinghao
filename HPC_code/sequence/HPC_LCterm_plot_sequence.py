# -*- coding: utf-8 -*-
"""
Created on Sat 5 Aug 14:23:46 2023

plot sequence given firing rate profiles and place cell classification (from MATLAB pipeline)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import sys

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCtermopt


#%% main 
# We only need the firing rate profiles from my Python pipeline (HPC all train)
# and the classification results from the MATLAB preprocessing pipeline
for pathname in pathHPC[7:]:
    recname = pathname[-17:]
    print(recname)
    
    classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
    place_cells = classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]
    tot_pc = len(place_cells)
    if tot_pc == 0:
        print('session has no detected place cells under current criteria\n')
        continue
    print('session has {} detected place cells'.format(tot_pc))
    
    trains = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
                          allow_pickle=True).item().values())
    tot_trial = len(trains[0])
    
    # behaviour parameters
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2
    
    profile_cont = np.zeros((tot_pc, 5*1250))
    profile_stim = np.zeros((tot_pc, 5*1250))
    for i, cell in enumerate(place_cells): 
        # take average 
        temp_cont = np.zeros((len(cont_trials), 5*1250))
        temp_stim = np.zeros((len(stim_trials), 5*1250))
        for ind, trial in enumerate(cont_trials):
            trial_length = len(trains[cell-2][trial])-2500
            if trial_length<5*1250 and trial_length>0:
                temp_cont[ind, :trial_length] = trains[cell-2][trial][2500:2500+1250*5]
            elif trial_length>0:
                temp_cont[ind, :] = trains[cell-2][trial][2500:2500+1250*5]
        for ind, trial in enumerate(stim_trials):
            trial_length = len(trains[cell-2][trial])-2500
            if trial_length<5*1250 and trial_length>0:
                temp_stim[ind, :trial_length] = trains[cell-2][trial][2500:2500+5*1250]
            elif trial_length>0:
                temp_stim[ind, :] = trains[cell-2][trial][2500:2500+5*1250]
        
        profile_cont[i,:] = normalise(np.mean(temp_cont, axis=0))
        profile_stim[i,:] = normalise(np.mean(temp_stim, axis=0))
        
    # order stuff by argmax
    max_pt = {}  # argmax for conts for all pyrs
    for i in range(tot_pc):
        max_pt[i] = np.argmax(profile_cont[i,:])
    def helper(x):
        return max_pt[x]
    ord_ind = sorted(np.arange(tot_pc), key=helper)
    
    im_mat_cont = np.zeros((tot_pc, 5*1250))
    im_mat_stim = np.zeros((tot_pc, 5*1250))
    for i, ind in enumerate(ord_ind): 
        im_mat_cont[i,:] = profile_cont[ind,:]
        im_mat_stim[i,:] = profile_stim[ind,:]
       
    # stimcont sequence 
    fig, ax = plt.subplots(figsize=(4.5,3))
    image_cont = ax.imshow(im_mat_cont, 
                           aspect='auto', cmap='Greys', interpolation='none',
                           extent=(-1, 4, 0, tot_pc))
    plt.colorbar(image_cont, shrink=.5)
    yticks = range(1, tot_pc+1, 5)
    ax.set(yticks=yticks,
           ylabel='cell #', xlabel='time (s)',
           title='{} stimcont'.format(recname))
    
    plt.show()
    
    # save figure
    outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
    outdir = '{}\sequence_pyr_stimcont.png'.format(outdirroot)
    fig.savefig('{}'.format(outdir),
                dpi=500,
                bbox_inches='tight')
    
    plt.close(fig)
    
    # stim sequence ordered by stimcont
    fig, ax = plt.subplots(figsize=(4.5,3))
    image_stim = ax.imshow(im_mat_stim, 
                           aspect='auto', cmap='Greys', interpolation='none',
                           extent=(-1, 4, 0, tot_pc))
    plt.colorbar(image_stim, shrink=.5)
    yticks = range(1, tot_pc+1, 5)
    ax.set(yticks=yticks,
           ylabel='cell #', xlabel='time (s)',
           title='{} stim'.format(recname))
    
    plt.show()
    
    # save figure
    outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
    outdir = '{}\sequence_pyr_stim_by_stimcont.png'.format(outdirroot)
    fig.savefig('{}'.format(outdir),
                dpi=500,
                bbox_inches='tight')
    
    plt.close(fig)