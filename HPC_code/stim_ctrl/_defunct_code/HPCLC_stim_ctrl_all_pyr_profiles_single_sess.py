# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:02:12 2024

plot profiles for all pyramidal cells in the HPC-LC stimulation sessions, 
    comparing stim-cont and stim trials
    ** single sessions 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import sys 
from scipy.stats import wilcoxon, ttest_rel
# import scipy.stats as st

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if (r'Z:\Dinghao\code_mpfi_dinghao\iutils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf

# if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
#     sys.path.append('Z:\Dinghao\code_dinghao\common')
# from common import normalise


#%% parameters
# run HPC-LC or HPC-LCterm
HPC_LC = 0


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

stim_cont_rise_count = []
stim_cont_down_count = []
stim_rise_count = []
stim_down_count = []


#%% get profiles and place in lists
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
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

    # rise and down counts for the current session
    curr_stim_cont_rise_count = 0 
    curr_stim_cont_down_count = 0 
    curr_stim_rise_count = 0 
    curr_stim_down_count = 0   

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
            
            temp_cont_mean = np.nanmean(temp_cont, axis=0)
            temp_stim_mean = np.nanmean(temp_stim, axis=0)
            
            stim_cont.append(temp_cont_mean)  # we don't have to normalise here 
            stim.append(temp_stim_mean)
            
            curr_stim_cont_ratio = np.nanmean(temp_cont_mean[625:1875])/np.nanmean(temp_cont_mean[3125:4375])
            curr_stim_ratio = np.nanmean(temp_stim_mean[625:1875])/np.nanmean(temp_stim_mean[3125:4375])
            if curr_stim_cont_ratio < .8:
                curr_stim_cont_rise_count+=1
            elif curr_stim_cont_ratio > 1.25:
                curr_stim_cont_down_count+=1
            if curr_stim_ratio < .8:
                curr_stim_rise_count+=1
            elif curr_stim_ratio > 1.25:
                curr_stim_down_count+=1
            
    stim_cont_rise_count.append(curr_stim_cont_rise_count/tot_pyr)
    stim_cont_down_count.append(curr_stim_cont_down_count/tot_pyr)
    stim_rise_count.append(curr_stim_rise_count/tot_pyr)
    stim_down_count.append(curr_stim_down_count/tot_pyr)
    
tot_clu = len(stim_cont)


#%% plotting 
if HPC_LC:
    pf.plot_violin_with_scatter(stim_cont_rise_count, stim_rise_count, 
                                'grey', 'royalblue',
                                xticklabels=['ctrl.', 'stim.'], 
                                ylabel='% run-onset pyr.',
                                save=True,
                                savepath=r'Z:\Dinghao\code_dinghao\HPC_all\HPCLC_stim_ctrl_start_cells',
                                dpi=300)
    pf.plot_violin_with_scatter(stim_cont_down_count, stim_down_count, 
                                'grey', 'royalblue',
                                xticklabels=['ctrl.', 'stim.'], 
                                ylabel='% run-onset inh. pyr.',
                                save=True,
                                savepath=r'Z:\Dinghao\code_dinghao\HPC_all\HPCLC_stim_ctrl_start_inh_cells',
                                dpi=300)
elif not HPC_LC:
    pf.plot_violin_with_scatter(stim_cont_rise_count, stim_rise_count, 
                                'grey', 'royalblue',
                                xticklabels=['ctrl.', 'stim.'], 
                                ylabel='% run-onset pyr.',
                                save=True,
                                savepath=r'Z:\Dinghao\code_dinghao\HPC_all\HPCLCterm_stim_ctrl_start_cells',
                                dpi=300)
    pf.plot_violin_with_scatter(stim_cont_down_count, stim_down_count, 
                                'grey', 'royalblue',
                                xticklabels=['ctrl.', 'stim.'], 
                                ylabel='% run-onset inh. pyr.',
                                save=True,
                                savepath=r'Z:\Dinghao\code_dinghao\HPC_all\HPCLCterm_stim_ctrl_start_inh_cells',
                                dpi=300)