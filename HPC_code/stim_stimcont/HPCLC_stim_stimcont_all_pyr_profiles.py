# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:34:28 2024

plot profiles for all pyramidal cells in the HPC-LC stimulation sessions, 
    comparing stim-cont and stim trials 

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


#%% parameters
# method for sorting cells; 1: use argmax(), 0: use pre-post firing rate ratio
argmax = 0

# run HPC-LC or HPC-LCterm
HPC_LC = 1


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
if HPC_LC:
    pathHPC = rec_list.pathHPCLCopt
elif not HPC_LC:
    pathHPC = rec_list.pathHPCLCtermopt


#%% lists to contain profiles
stim = []
stim_cont = []


#%% switch 
if argmax:
    print('using argmax() to sort pyramidal cells...')
    
    #%% get profiles and place in lists
    for pathname in pathHPC:
        recname = pathname[-17:]
        print(recname)
        
        # # We do not care about whether a cell is a PC or not for now
        # classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
        # place_cells = classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]
        # tot_pc = len(place_cells)
        # if tot_pc == 0:
        #     print('session has no detected place cells under current criteria\n')
        #     continue
        # print('session has {} detected place cells'.format(tot_pc))
        
        trains = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
                              allow_pickle=True).item().values())
        tot_trial = len(trains[0])
        
        # determine if each cell is pyramidal or intern 
        info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
        rec_info = info['rec'][0][0]
        intern_id = rec_info['isIntern'][0]
        pyr_id = [not(clu) for clu in intern_id]
        tot_pyr = sum(pyr_id)
        
        # behaviour parameters
        info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
        beh_info = info['beh'][0][0]
        behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
        stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
        stim_trials = np.where(stimOn!=0)[0]+1
        cont_trials = stim_trials+2
    
        # put each averaged profile into the separate lists
        for i, if_pyr in enumerate(pyr_id):
            if if_pyr:
                # take average 
                temp_cont = np.zeros((len(cont_trials), 5*1250))
                temp_stim = np.zeros((len(stim_trials), 5*1250))
                for ind, trial in enumerate(cont_trials):
                    trial_length = len(trains[i][trial])-2500
                    if trial_length<5*1250 and trial_length>0:
                        temp_cont[ind, :trial_length] = trains[i][trial][2500:2500+1250*5]
                    elif trial_length>0:
                        temp_cont[ind, :] = trains[i][trial][2500:2500+1250*5]
                for ind, trial in enumerate(stim_trials):
                    trial_length = len(trains[i][trial])-2500
                    if trial_length<5*1250 and trial_length>0:
                        temp_stim[ind, :trial_length] = trains[i][trial][2500:2500+5*1250]
                    elif trial_length>0:
                        temp_stim[ind, :] = trains[i][trial][2500:2500+5*1250]
                
                stim_cont.append(normalise(np.mean(temp_cont, axis=0)))
                stim.append(normalise(np.mean(temp_stim, axis=0)))
                
    tot_clu = len(stim_cont)
    
    
    #%% convert the lists into arrays 
    stim_cont = np.asarray(stim_cont)
    stim = np.asarray(stim)
    
    
    #%% ordering based on peak (argmax)
    # order stim_cont
    stim_cont_max_pt = {}  # argmax for conts for all pyrs
    for i in range(tot_clu):
        stim_cont_max_pt[i] = np.argmax(stim_cont[i,:])
    def helper(x):
        return stim_cont_max_pt[x]
    stim_cont_ord_ind = sorted(np.arange(tot_clu), key=helper)
    
    # order stim
    stim_max_pt = {}  # argmax for conts for all pyrs
    for i in range(tot_clu):
        stim_max_pt[i] = np.argmax(stim[i,:])
    def helper(x):
        return stim_max_pt[x]
    stim_ord_ind = sorted(np.arange(tot_clu), key=helper)
    
    im_mat_stim_cont = np.zeros((tot_clu, 5*1250))
    im_mat_stim = np.zeros((tot_clu, 5*1250))
    for i, ind in enumerate(stim_cont_ord_ind): 
        im_mat_stim_cont[i,:] = stim_cont[ind,:]
    for i, ind in enumerate(stim_ord_ind): 
        im_mat_stim[i,:] = stim[ind,:]
    
    
    #%% plotting (argmax)
    fig, ax = plt.subplots(figsize=(5,4))
    fig.suptitle('stim_cont_all_pyr')
    ax.imshow(im_mat_stim_cont, interpolation='none', cmap='Greys', aspect='auto', 
              extent=[-1, 4, 1, tot_clu])
    
    fig, ax = plt.subplots(figsize=(5,4))
    fig.suptitle('stim_all_pyr')
    ax.imshow(im_mat_stim, interpolation='none', cmap='Greys', aspect='auto', 
              extent=[-1, 4, 1, tot_clu])


#%% switch 2
else:
    print('using pre-post spike rate ratios to sort pyramidal cells...')
    
    #%% get profiles and place in lists
    for pathname in pathHPC:
        recname = pathname[-17:]
        print(recname)
        
        # # We do not care about whether a cell is a PC or not for now
        # classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
        # place_cells = classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]
        # tot_pc = len(place_cells)
        # if tot_pc == 0:
        #     print('session has no detected place cells under current criteria\n')
        #     continue
        # print('session has {} detected place cells'.format(tot_pc))
        
        trains = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
                              allow_pickle=True).item().values())
        tot_trial = len(trains[0])
        
        # determine if each cell is pyramidal or intern 
        info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
        rec_info = info['rec'][0][0]
        intern_id = rec_info['isIntern'][0]
        pyr_id = [not(clu) for clu in intern_id]
        tot_pyr = sum(pyr_id)
        
        # behaviour parameters
        info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
        beh_info = info['beh'][0][0]
        behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
        stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
        stim_trials = np.where(stimOn!=0)[0]+1
        cont_trials = stim_trials+2
    
        # put each averaged profile into the separate lists
        for i, if_pyr in enumerate(pyr_id):
            if if_pyr:
                # take average 
                temp_cont = np.zeros((len(cont_trials), 6*1250))
                temp_stim = np.zeros((len(stim_trials), 6*1250))
                for ind, trial in enumerate(cont_trials):
                    trial_length = len(trains[i][trial])-1250
                    if trial_length<6*1250 and trial_length>0:
                        temp_cont[ind, :trial_length] = trains[i][trial][1250:1250+6*1250]  # use [-2, 4] to include -1.5 to -.5 seconds 
                    elif trial_length>0:
                        temp_cont[ind, :] = trains[i][trial][1250:1250+6*1250]
                for ind, trial in enumerate(stim_trials):
                    trial_length = len(trains[i][trial])-1250
                    if trial_length<6*1250 and trial_length>0:
                        temp_stim[ind, :trial_length] = trains[i][trial][1250:1250+6*1250]
                    elif trial_length>0:
                        temp_stim[ind, :] = trains[i][trial][1250:1250+6*1250]
                
                stim_cont.append(np.nanmean(temp_cont, axis=0))  # we don't have to normalise here 
                stim.append(np.nanmean(temp_stim, axis=0))
                
    tot_clu = len(stim_cont)
    
    
    #%% convert the lists into arrays 
    stim_cont = np.asarray(stim_cont)
    stim = np.asarray(stim)
    
    
    #%% ordering based on peak (argmax)
    buffer_count_stim_cont = 0
    buffer_count_stim = 0
    
    # order stim_cont
    stim_cont_pp_ratio = {}  # argmax for conts for all pyrs
    stim_cont_rise_count = 0
    stim_cont_down_count = 0
    for i in range(tot_clu):
        curr_ratio = np.nanmean(stim_cont[i,625:1875])/np.nanmean(stim_cont[i,3125:4375])  # .5~1.5s
        stim_cont_pp_ratio[i] = curr_ratio
        if curr_ratio < .9:  # rise-cells 
            stim_cont_rise_count+=1
        elif curr_ratio > 1.1:  # down-cells 
            stim_cont_down_count+=1
        if np.isnan(stim_cont_pp_ratio[i]) or max(stim_cont[i,:])==0:  # if everything is 0
            stim_cont_pp_ratio[i] = 100  # if divided by 0
            buffer_count_stim_cont+=1
    def helper(x):
        return stim_cont_pp_ratio[x]
    stim_cont_pp_ord_ind = sorted(np.arange(tot_clu), key=helper)
    
    # order stim
    stim_pp_ratio = {}  # argmax for conts for all pyrs
    stim_rise_count = 0
    stim_down_count = 0
    for i in range(tot_clu):
        curr_ratio = np.nanmean(stim[i,625:1875])/np.nanmean(stim[i,3125:4375])
        stim_pp_ratio[i] = curr_ratio
        if curr_ratio < .9:  # rise-cells 
            stim_rise_count+=1
        elif curr_ratio > 1.1:  # down-cells 
            stim_down_count+=1
        if np.isnan(stim_pp_ratio[i]):
            stim_pp_ratio[i] = 100
            buffer_count_stim+=1
    def helper(x):
        return stim_pp_ratio[x]
    stim_pp_ord_ind = sorted(np.arange(tot_clu), key=helper)
    
    line_count_stim_cont = 0
    line_count_stim = 0
    im_mat_stim_cont_pp = np.zeros((tot_clu-buffer_count_stim_cont, 6*1250))
    im_mat_stim_pp = np.zeros((tot_clu-buffer_count_stim, 6*1250))
    for i, ind in enumerate(stim_cont_pp_ord_ind): 
        if stim_cont_pp_ratio[i] != 100:
            im_mat_stim_cont_pp[line_count_stim_cont,:] = normalise(stim_cont[ind,:])
            line_count_stim_cont+=1
    for i, ind in enumerate(stim_pp_ord_ind): 
        if stim_pp_ratio[i] != 100:
            im_mat_stim_pp[line_count_stim,:] = normalise(stim[ind,:])
            line_count_stim+=1
    
    
    #%% plotting (argmax)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set(title='stim_cont_all_pyr_pre_post_ratio', xlabel='time (s)', ylabel='pyr #')
    image_stim_cont = ax.imshow(im_mat_stim_cont_pp, interpolation='none', cmap='Greys', aspect='auto', 
                                extent=[-2, 4, 1, tot_clu-buffer_count_stim_cont])
    plt.colorbar(image_stim_cont, shrink=.5)
    if HPC_LC:
        fig.suptitle('HPC_LC')
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_all_pyr_pre_post_ratio.png',
                    dpi=500, bbox_inches='tight')
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_all_pyr_pre_post_ratio.pdf',
                    bbox_inches='tight')
    elif not HPC_LC:
        fig.suptitle('HPC_LCterm')
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_all_pyr_pre_post_ratio.png',
                    dpi=500, bbox_inches='tight')
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_all_pyr_pre_post_ratio.pdf',
                    bbox_inches='tight')
        
    
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set(title='stim_all_pyr_pre_post_ratio', xlabel='time (s)', ylabel='pyr #')
    image_stim = ax.imshow(im_mat_stim_pp, interpolation='none', cmap='Greys', aspect='auto', 
                           extent=[-2, 4, 1, tot_clu-buffer_count_stim])
    plt.colorbar(image_stim, shrink=.5)
    if HPC_LC:
        fig.suptitle('HPC_LC')
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_cont_all_pyr_pre_post_ratio.png',
                    dpi=500, bbox_inches='tight')
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_cont_all_pyr_pre_post_ratio.pdf',
                    bbox_inches='tight')
    elif not HPC_LC:
        fig.suptitle('HPC_LCterm')
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_cont_all_pyr_pre_post_ratio.png',
                    dpi=500, bbox_inches='tight')
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_cont_all_pyr_pre_post_ratio.pdf',
                    bbox_inches='tight')