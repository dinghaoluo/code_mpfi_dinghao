# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:44:27 2023

compare all HPC cell's spiking profile between cont and stim 

@author: Dinghao
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 
import scipy.io as sio 

import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


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
    raster_file = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(recname),
                       allow_pickle=True).item()
    
    rasters = list(raster_file.values())
    clu_list = list(raster_file.keys())
    tot_time = 1250 + 5000  # 1 s before, 4 s after 
    
    # if each neurones is an interneurone or a pyramidal cell 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    
    # # depth 
    # depth = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Depth.mat'.format(pathname, recname))['depthNeu'][0]
    # rel_depth = depth['relDepthNeu'][0][0]
    
    # behaviour parameters 
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2
    
    tot_clu = len(pyr_id)
    tot_pyr = sum(pyr_id)  # how many pyramidal cells are in this recording
    pyr_cont = {}
    pyr_stim = {}
    
    for i in range(tot_clu):
        if pyr_id[i]==True:
            cluname = clu_list[i]
            # d = rel_depth[i]
            temp_cont = np.zeros((len(cont_trials), tot_time))
            temp_stim = np.zeros((len(stim_trials), tot_time))
            for ind, trial in enumerate(cont_trials):
                trial_length = len(rasters[i][trial])-2500
                if trial_length<tot_time and trial_length>0:
                    temp_cont[ind, :trial_length] = rasters[i][trial][2500:8750]
                elif trial_length>0:
                    temp_cont[ind, :] = rasters[i][trial][2500:8750]
            for ind, trial in enumerate(stim_trials):
                trial_length = len(rasters[i][trial])-2500
                if trial_length<tot_time and trial_length>0:
                    temp_stim[ind, :trial_length] = rasters[i][trial][2500:8750]
                elif trial_length>0:
                    temp_stim[ind, :] = rasters[i][trial][2500:8750]
            
            # plotting 
            fig, axs = plt.subplots(2, 1, figsize=(3.1,2.8))
            axs[0].set(title='ctrl')
            axs[1].set(title='stim')
            fig.suptitle(cluname, fontsize=10)
            
            for i in range(len(cont_trials)):
                axs[0].scatter(np.where(temp_cont[i]==1)[0]/1250-1, 
                               [i+1]*int(sum(temp_cont[i])),
                               c='grey', ec='none', s=2)

            for i in range(len(stim_trials)):
                axs[1].scatter(np.where(temp_stim[i]==1)[0]/1250-1, 
                               [i+1]*int(sum(temp_stim[i])),
                               c='grey', ec='none', s=2)
                            
            for i in range(2):
                for p in ['top', 'right']:
                    axs[i].spines[p].set_visible(False)
                axs[i].set(xlabel='time (s)', ylabel='trial #',
                           xticks=[0,2,4], xlim=(-1, 4),
                           yticks=[1,20])

            fig.tight_layout()
            plt.show()
            
            # make folders if not exist
            outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
            if not os.path.exists(outdirroot):
                os.makedirs(outdirroot)
            outdir = '{}\stim_stimcont_pyr_only_rasters'.format(outdirroot)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            fig.savefig('{}\{}.png'.format(outdir, cluname),
                        dpi=500,
                        bbox_inches='tight')
            fig.savefig('{}\{}.pdf'.format(outdir, cluname),
                        bbox_inches='tight')
            
            plt.close(fig)