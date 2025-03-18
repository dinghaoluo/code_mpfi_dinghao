# -*- coding: utf-8 -*-
"""
Created on Sat 5 Aug 15:00:46 2023

pyr stim analysis

stim_trial includes all stim trials with 020
base_trial includes all stim trials + 2

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import scipy.io as sio 
import matplotlib.pyplot as plt 
import sys


#%% read and run 
# We need the Info.mat to tell us whether a neurone is GABAergic
# and we need the firing rate profiles

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt

all_info = np.load('Z:/Dinghao/code_dinghao/HPC_all/HPC_all_info.npy',
                   allow_pickle=True).item()


#%% main
mean_stims_rate = []
mean_bases_rate = []

for pathname in pathHPC:
    animal = pathname[19:27]
    date = pathname[28:42]
    exp = pathname[43:]
    struct = pathname[43:]+'_DataStructure_mazeSection1_TrialType1_Info.mat'
    
    # load pyr cell indices
    info = sio.loadmat('Z:/Dinghao/MiceExp/{}/{}/{}/{}'.format(animal, date, exp, struct))
    intern = info['rec'][0]['isIntern'][0][0]
    pyr = np.where(intern==0)[0]
    pyr = ['{} clu{}'.format(exp, clu) for clu in pyr]
    
    # stim parameters
    pulseMethod = info['beh']['pulseMethod'][0][0][0][1:]
    stim_trial = np.squeeze(np.where(pulseMethod!=0))
    base_trial = stim_trial+2

    # other parameters 
    tot_trial = len(pulseMethod)
    
    # lists
    mean_stims = []
    mean_bases = []
    
    for clu in pyr:
        trunc_trials = []
        
        for trial in range(tot_trial):
            trunc_trials.append(all_info[clu][trial][2500:7500])  # -1~4s
        
        mean_stim = []
        mean_base = []
        
        for trial in range(stim_trial[0], stim_trial[-1]+3):
            if trial in stim_trial:
                mean_stim.append(trunc_trials[trial])
            elif trial in base_trial:
                mean_base.append(trunc_trials[trial])
                
        std_stim = np.nanstd(mean_stim, axis=0)
        mean_stim = np.nanmean(mean_stim, axis=0)
        std_base = np.nanstd(mean_base, axis=0)
        mean_base = np.nanmean(mean_base, axis=0)
        
        # plot each pyramidal stim vs base
        fig, [ax1, ax2] = plt.subplots(1,2, figsize=(5,3))
        fig.suptitle(clu)
        xaxis = np.arange(-1250, 3750)/1250
        ax1.set(title='stim avg. prof',
                xlabel='time (sample)', ylabel='spike rate (Hz)')
        ax1.plot(xaxis, mean_stim)
        ax1.fill_between(xaxis, mean_stim+std_stim,
                                mean_stim-std_stim,
                         alpha=.25)
        ax2.set(title='base avg. prof',
                xlabel='time (sample)', ylabel='spike rate (Hz)')
        ax2.plot(xaxis, mean_base)
        ax2.fill_between(xaxis, mean_base+std_base,
                                mean_base-std_base,
                         alpha=.25)
        
        mean_stims.append(mean_stim)
        mean_bases.append(mean_base)
        
    mean_stims_rate.append(np.mean([np.mean(clu) for clu in mean_stims]) * 1250)
    mean_bases_rate.append(np.mean([np.mean(clu) for clu in mean_bases]) * 1250)