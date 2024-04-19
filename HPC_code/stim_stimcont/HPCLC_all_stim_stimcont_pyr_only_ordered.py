# -*- coding: utf-8 -*-
"""
Created on Thu 30 Nov 18:10:37 2023

plot heatmaps showing all the pyramidal cells ordered by peak FR

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 
import scipy.io as sio 

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
pathHPC = rec_list.pathHPCLCopt


#%% main 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    # load trains for this recording 
    all_info = np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname),
                       allow_pickle=True).item()
    
    tot_time = 8 * 1250  # 8 seconds in 1250 Hz
    
    trains = list(all_info.values())
    clu_list = list(all_info.keys())
    tot_trial = len(trains[0])
    
    # if each neurones is an interneurone or a pyramidal cell 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    
    # depth 
    depth = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Depth.mat'.format(pathname, recname))['depthNeu'][0]
    rel_depth = depth['relDepthNeu'][0][0]
    
    # behaviour parameters 
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2
    
    tot_clu = len(pyr_id)
    tot_pyr = sum(pyr_id)  # how many pyramidal cells are in this recording
    pyr_cont = np.zeros((tot_pyr, tot_time))
    pyr_stim = np.zeros((tot_pyr, tot_time))
    
    pyr_count = 0
    for i in range(tot_clu):
        if pyr_id[i]==True:
            temp_cont = np.zeros((len(cont_trials), tot_time))
            temp_stim = np.zeros((len(stim_trials), tot_time))
            for ind, trial in enumerate(cont_trials):
                trial_length = len(trains[i][trial])
                if trial_length<tot_time and trial_length>0:
                    temp_cont[ind, :trial_length] = trains[i][trial][:tot_time]
                elif trial_length>0:
                    temp_cont[ind, :] = trains[i][trial][:tot_time]
            for ind, trial in enumerate(stim_trials):
                trial_length = len(trains[i][trial])
                if trial_length<tot_time and trial_length>0:
                    temp_stim[ind, :trial_length] = trains[i][trial][:tot_time]
                elif trial_length>0:
                    temp_stim[ind, :] = trains[i][trial][:tot_time]

            # mean profile 
            mean_prof_cont = np.mean(temp_cont, axis=0)*1250
            mean_prof_stim = np.mean(temp_stim, axis=0)*1250
        
            # put into arrays
            pyr_cont[pyr_count,:] = normalise(mean_prof_cont)
            pyr_stim[pyr_count,:] = normalise(mean_prof_stim)
            
            pyr_count-=-1
    
    # make folders if not exist
    outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
    if not os.path.exists(outdirroot):
        os.makedirs(outdirroot)
    
    
    # order stuff by argmax
    max_pt_cont = {}  # argmax for conts for all pyrs
    max_pt_stim = {}
    for i in range(tot_pyr):
        max_pt_cont[i] = np.argmax(pyr_cont[i])
        max_pt_stim[i] = np.argmax(pyr_stim[i])

    def helper_cont(x):
        return max_pt_cont[x]
    def helper_stim(x):
        return max_pt_stim[x]
    ord_ind_cont = sorted(np.arange(tot_pyr), key=helper_cont)
    ord_ind_stim = sorted(np.arange(tot_pyr), key=helper_stim)
    
    
    # control image
    im_mat_cont = np.zeros((tot_pyr, tot_time))
    for i, ordind in enumerate(ord_ind_cont):
        im_mat_cont[i, :] = pyr_cont[ordind]
    
    fig, ax = plt.subplots(figsize=(4,4))
    fig.suptitle(recname)
    ax.set(title='all pyr, cont')
    
    image_cont = ax.imshow(im_mat_cont, aspect='auto', interpolation='none',
                           extent=[-3,5,1,tot_pyr])
    
    fig.tight_layout()
    plt.show()
    outdir = '{}\ordered_stimcont_pyr_only.png'.format(outdirroot)
    fig.savefig('{}'.format(outdir),
                dpi=500,
                bbox_inches='tight')
    plt.close(fig)
    
    
    # stim image
    im_mat_stim = np.zeros((tot_pyr, tot_time))
    for i, ordind in enumerate(ord_ind_stim):
        im_mat_stim[i, :] = pyr_stim[ordind]
    
    fig, ax = plt.subplots(figsize=(4,4))
    fig.suptitle(recname)
    ax.set(title='all pyr, stim')
    
    image_stim = ax.imshow(im_mat_stim, aspect='auto', interpolation='none',
                           extent=[-3,5,1,tot_pyr])
    
    fig.tight_layout()
    plt.show()
    outdir = '{}\ordered_stim_pyr_only.png'.format(outdirroot)
    fig.savefig('{}'.format(outdir),
                dpi=500,
                bbox_inches='tight')
    plt.close(fig)
    
    
    # stim image ordered by cont
    im_mat_stim_cont = np.zeros((tot_pyr, tot_time))
    for i, ordind in enumerate(ord_ind_cont):
        im_mat_stim_cont[i, :] = pyr_stim[ordind]
    
    fig, ax = plt.subplots(figsize=(4,4))
    fig.suptitle(recname)
    ax.set(title='all pyr, stim ord. by cont')
    
    image_stim_cont = ax.imshow(im_mat_stim_cont, aspect='auto', interpolation='none',
                                extent=[-3,5,1,tot_pyr])
    
    fig.tight_layout()
    plt.show()
    outdir = '{}\ordered_stim_ord_stimcont_pyr_only.png'.format(outdirroot)
    fig.savefig('{}'.format(outdir),
                dpi=500,
                bbox_inches='tight')
    plt.close(fig)