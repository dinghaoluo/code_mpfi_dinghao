# -*- coding: utf-8 -*-
"""
Created on Tue 26 Sep 14:05:31 2023

pyramidal cell heatmap for each session

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import mat73
import scipy.io as sio 
import sys


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% load functions
if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% main 
for pathname in pathHPC[:14]:
    recname = pathname[-17:]
    
    # spikes by time
    file = mat73.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_convSpikesAligned_msess1_Run1.mat'.format(pathname, recname))
    spikes_time = file['filteredSpikeArrayRun_LasttoCurTr']
    
    tot_time = 5 * 1250  # 5 seconds in 1250 Hz
    tot_trial = spikes_time[0].shape[0]
    
    # if each neurones is an interneurone or a pyramidal cell 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    
    # behaviour parameters 
    beh_info = info['beh'][0][0]
    stim_ind = np.where(beh_info['pulseMethod']!=0)[1]
    stim_start = stim_ind[0]; stim_end = stim_ind[-1]
    stim_trials = np.arange(stim_start, stim_end)
    
    tot_clu = len(pyr_id)
    tot_pyr = sum(pyr_id)  # how many pyramidal cells are in this recording
    pyr_mat = np.zeros((tot_pyr, tot_time))
    pyr_mat_cont = np.zeros((tot_pyr, tot_time))
    pyr_mat_stim = np.zeros((tot_pyr, tot_time))
    
    pyr_counter = 0 
    argmax_pyr = []; argmax_pyr_cont = []; argmax_pyr_stim = []
    for i in range(tot_clu):
        if pyr_id[i]==True:
            temp = np.zeros((tot_trial, tot_time))  # temporary to contain all trials of one clu
            temp_cont = np.zeros((stim_start, tot_time))
            temp_stim = np.zeros((len(stim_trials), tot_time))
            for trial in range(tot_trial):
                temp[trial, :] = spikes_time[i][trial,2750:9000]
            for trial in range(stim_start):
                temp_cont[trial, :] = spikes_time[i][trial,2750:9000]
            for ind, trial in enumerate(stim_trials):
                temp_stim[ind, :] = spikes_time[i][trial,2750:9000]
            pyr_mat[pyr_counter, :] = normalise(np.mean(temp, axis=0))
            pyr_mat_cont[pyr_counter, :] = normalise(np.mean(temp_cont, axis=0))
            pyr_mat_stim[pyr_counter, :] = normalise(np.mean(temp_stim, axis=0))
            argmax_pyr.append(np.argmax(pyr_mat[pyr_counter, :]))
            argmax_pyr_cont.append(np.argmax(pyr_mat_cont[pyr_counter, :]))
            argmax_pyr_stim.append(np.argmax(pyr_mat_stim[pyr_counter, :]))
            
            pyr_counter+=1

    temp = list(np.arange(tot_pyr))
    peak_ordered, pyr_id_ordered = zip(*sorted(zip(argmax_pyr, temp)))
    pyr_mat_ordered = np.zeros((tot_pyr, tot_time))
    for i in range(tot_pyr):
        curr_id = pyr_id_ordered[i]
        pyr_mat_ordered[i,:] = pyr_mat[curr_id,:]
        
    peak_ordered_cont, pyr_id_ordered_cont = zip(*sorted(zip(argmax_pyr_cont, temp)))
    pyr_mat_ordered_cont = np.zeros((tot_pyr, tot_time))
    for i in range(tot_pyr):
        curr_id = pyr_id_ordered_cont[i]  # use the same order 
        pyr_mat_ordered_cont[i,:] = pyr_mat_cont[curr_id,:]
        
    peak_ordered_stim, pyr_id_ordered_stim = zip(*sorted(zip(argmax_pyr_stim, temp)))
    pyr_mat_ordered_stim = np.zeros((tot_pyr, tot_time))
    for i in range(tot_pyr):
        curr_id = pyr_id_ordered_stim[i]  # use the same order
        pyr_mat_ordered_stim[i,:] = pyr_mat_stim[curr_id,:]

    # plot heatmap 
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    image_cont = axs[0].imshow(pyr_mat_ordered_cont, aspect='auto', cmap='turbo',
                               extent=[-1, 4, 0, tot_pyr])
    plt.colorbar(image_cont)
    axs[0].set(title='control')
    
    image_stim = axs[1].imshow(pyr_mat_ordered_stim, aspect='auto', cmap='turbo',
                               extent=[-1, 4, 0, tot_pyr])
    plt.colorbar(image_stim)
    axs[1].set(title='stim')
    
    for i in range(2):
        axs[i].set(xlabel='time from run (s)', ylabel='pyr cell #')
    
    fig.suptitle(recname)
    
    plt.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_cont_stim_runaligned_heatmap\{}.png'.format(recname),
                dpi=500,
                bbox_inches='tight')
    
    plt.close(fig)