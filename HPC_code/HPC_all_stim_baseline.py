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
from scipy.stats import sem


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% load functions
# if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
#     sys.path.append('Z:\Dinghao\code_dinghao\common')
# from common import normalise


#%% main 
for pathname in pathHPC:
    recname = pathname[17:]
    
    # load trains for this recording 
    all_info = np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname),
                       allow_pickle=True).item()
    
    tot_time = 5 * 1250  # 5 seconds in 1250 Hz
    
    trains = list(all_info.values())
    clu_list = list(all_info.keys())
    tot_trial = len(trains[0])
    
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
    pyr = {}
    pyr_cont = {}
    pyr_stim = {}
    
    pyr_counter = 0 
    for i in range(tot_clu):
        if pyr_id[i]==True:
            cluname = clu_list[i]
            temp = np.zeros((tot_trial, tot_time))  # temporary to contain all trials of one clu
            temp_cont = np.zeros((stim_start, tot_time))
            temp_stim = np.zeros((len(stim_trials), tot_time))
            for trial in range(tot_trial):
                trial_length = len(trains[i][trial])-2500
                if trial_length<tot_time:
                    temp[trial, :trial_length] = trains[i][trial][2500:8750]
                else:
                    temp[trial, :] = trains[i][trial][2500:8750]
            for trial in range(stim_start):
                trial_length = len(trains[i][trial])-2500
                if trial_length<tot_time:
                    temp_cont[trial, :trial_length] = trains[i][trial][2500:8750]
                else:
                    temp_cont[trial, :] = trains[i][trial][2500:8750]
            for ind, trial in enumerate(stim_trials):
                trial_length = len(trains[i][trial])-2500
                if trial_length<tot_time:
                    temp_stim[ind, :trial_length] = trains[i][trial][2500:8750]
                else:
                    temp_stim[ind, :] = trains[i][trial][2500:8750]
            pyr[cluname] = np.mean(temp, axis=0)*1250
            pyr_cont[cluname] = np.mean(temp_cont, axis=0)*1250
            pyr_stim[cluname] = np.mean(temp_stim, axis=0)*1250
            
            pyr_counter+=1
            
            # plotting 
            fig, ax = plt.subplots(figsize=(4,3))
            xaxis = np.arange(-1250, 5000)/1250
            
            mean_prof_cont = np.mean(temp_cont, axis=0)*1250
            std_prof_cont = sem(temp_cont, axis=0)*1250
            mean_prof_stim = np.mean(temp_stim, axis=0)*1250
            std_prof_stim = sem(temp_stim, axis=0)*1250
            contln, = ax.plot(xaxis, mean_prof_cont, color='grey')
            ax.fill_between(xaxis, mean_prof_cont+std_prof_cont,
                                   mean_prof_cont-std_prof_cont,
                                   alpha=.25, color='grey')
            stimln, = ax.plot(xaxis, mean_prof_stim, color='royalblue')
            ax.fill_between(xaxis, mean_prof_stim+std_prof_stim,
                                   mean_prof_stim-std_prof_stim,
                                   alpha=.25)
            
            ax.legend([contln, stimln], ['control', 'stim'], 
                      frameon=False, fontsize=10)
            for p in ['top', 'right']:
                ax.spines[p].set_visible(False)
            
            fig.suptitle(cluname)
            
            fig.tight_layout()
            plt.show()
            plt.close(fig)
            
            # make folders if not exist
            outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
            if not os.path.exists(outdirroot):
                os.makedirs(outdirroot)
            outdir = '{}\cont_v_stim_pyr'.format(outdirroot)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            fig.savefig('{}\{}'.format(outdir, cluname),
                        dpi=500,
                        bbox_inches='tight')