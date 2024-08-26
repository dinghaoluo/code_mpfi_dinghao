# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:22:43 2024

population deviation poisson 

@author: Dinghao Luo
"""


#%% imports 
import sys
import scipy.io as sio 
import numpy as np 
import matplotlib.pyplot as plt 
from math import log 
from scipy.stats import poisson, zscore


#%% run HPC-LC or HPC-LCterm
HPC_LC = 1

# load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
if HPC_LC:
    pathHPC = rec_list.pathHPCLCopt
elif not HPC_LC:
    pathHPC = rec_list.pathHPCLCtermopt


#%% main (single-trial act./inh. proportions)
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    rasters = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(recname), 
                           allow_pickle=True).item().values())
    
    # determine if each cell is pyramidal or intern 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    spike_rate = rec_info['firingRate'][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    tot_pyr = sum(pyr_id)
    
    # behaviour parameters
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    ctrl_trials = stim_trials+2
    tot_trial = len(stimOn)

    # for each trial we quantify deviation from Poisson
    fig, ax = plt.subplots(figsize=(3,2.5)); xaxis = np.arange(-3,5)
    pop_deviation_ctrl = []; pop_deviation_stim = []
    for trial in ctrl_trials:
        single_deviation = np.zeros((tot_pyr, 8))
        cell_counter = 0
        for i, if_pyr in enumerate(pyr_id):
            if if_pyr:
                curr_raster = rasters[i][trial]
                for t in range(8):  # 3 seconds before, 5 seconds after 
                    curr_bin = sum(curr_raster[t*1250:(t+1)*1250])
                    single_deviation[cell_counter, t] = -log(poisson.pmf(curr_bin, spike_rate[i]))
                cell_counter+=1
        pop_deviation_ctrl.append(np.sum(single_deviation, axis=0))
        ax.plot(xaxis, np.sum(single_deviation, axis=0), lw=1, alpha=.1, c='grey')
    for trial in stim_trials:
        single_deviation = np.zeros((tot_pyr, 8))
        cell_counter = 0
        for i, if_pyr in enumerate(pyr_id):
            if if_pyr:
                curr_raster = rasters[i][trial]
                for t in range(8):  # 3 seconds before, 5 seconds after 
                    curr_bin = sum(curr_raster[t*1250:(t+1)*1250])
                    single_deviation[cell_counter, t] = -log(poisson.pmf(curr_bin, spike_rate[i]))
                cell_counter+=1
        pop_deviation_stim.append(np.sum(single_deviation, axis=0))
        ax.plot(xaxis, np.sum(single_deviation, axis=0), lw=1, alpha=.1, c='steelblue')
    pop_deviation_ctrl = np.asarray(pop_deviation_ctrl)
    pop_deviation_stim = np.asarray(pop_deviation_stim)
    ax.plot(xaxis, np.mean(pop_deviation_ctrl, axis=0), lw=2, c='k')
    ax.plot(xaxis, np.mean(pop_deviation_stim, axis=0), lw=2, c='royalblue')
    ax.set(title=recname,
           xlabel='time (s)',
           ylabel='pop. deviation')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.tight_layout()