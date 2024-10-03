# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:44:27 2023

compare all HPC cell's spiking profile between cont and stim 

@author: Dinghao
"""


#%% imports 
import numpy as np 
import pandas as pd 
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


#%% load 
df = pd.read_pickle('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_diff_profiles_pyr_only.pkl') 
MI = np.array(list(df['MI']))
shuf_MI = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_shuf_MI_pyr_only.npy', allow_pickle=True)


#%% MI percentile 
MI_perc = np.percentile(shuf_MI, [5, 95])


#%% main 
tot_time = 1250 + 5000  # 1 s before, 4 s after 
recname = df['recname'][0]
pathname = r'Z:\Dinghao\MiceExp\ANMD{}r\{}\{}'.format(recname[1:4], recname[:-3], recname)
raster_file = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(recname),
                      allow_pickle=True).item()
info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
beh_info = info['beh'][0][0]
behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
stim_trials = np.where(stimOn!=0)[0]+1
cont_trials = stim_trials+2

for cluname, row in df.iterrows():
    if row['recname'] != recname:  # reload rec info 
        recname = row['recname']
        pathname = r'Z:\Dinghao\MiceExp\ANMD{}r\{}\{}'.format(recname[1:4], recname[:-3], recname)
        raster_file = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(recname),
                              allow_pickle=True).item()
        info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
        beh_info = info['beh'][0][0]
        behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
        stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
        stim_trials = np.where(stimOn!=0)[0]+1
        cont_trials = stim_trials+2
    
    if row['MI']>MI_perc[1] or row['MI']<MI_perc[0]:
        temp_cont = np.zeros((len(cont_trials), tot_time))
        temp_stim = np.zeros((len(stim_trials), tot_time))
        for ind, trial in enumerate(cont_trials):
            trial_length = len(raster_file[cluname][trial])-2500
            if trial_length<tot_time and trial_length>0:
                temp_cont[ind, :trial_length] = raster_file[cluname][trial][2500:8750]
            elif trial_length>0:
                temp_cont[ind, :] = raster_file[cluname][trial][2500:8750]
        for ind, trial in enumerate(stim_trials):
            trial_length = len(raster_file[cluname][trial])-2500
            if trial_length<tot_time and trial_length>0:
                temp_stim[ind, :trial_length] = raster_file[cluname][trial][2500:8750]
            elif trial_length>0:
                temp_stim[ind, :] = raster_file[cluname][trial][2500:8750]
            
        # plotting 
        fig, axs = plt.subplots(2, 1, figsize=(3.1,2.8))
        axs[0].set(title='ctrl.')
        axs[1].set(title='stim.')
        if row['MI']>MI_perc[1]:
            fig.suptitle(cluname+' act.', fontsize=10)
        elif row['MI']<MI_perc[0]:
            fig.suptitle(cluname+' inh.', fontsize=10)
        
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
        outdir = r'Z:\Dinghao\code_dinghao\HPC_all\single_cell_stim_ctrl_raster_signif_only\HPCLCterm\{}.png'.format(cluname)

        fig.savefig(outdir,
                    dpi=300,
                    bbox_inches='tight')
        
        plt.close(fig)