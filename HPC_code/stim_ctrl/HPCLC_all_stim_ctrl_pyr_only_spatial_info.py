# -*- coding: utf-8 -*-
"""
Created on Wed 14 Aug 17:11:14 2024

compare spatial information (adapt) between stim and ctrl 

@author: Dinghao
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 
import scipy.io as sio 
from scipy.stats import sem, ttest_rel, wilcoxon

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if (r'Z:\Dinghao\code_mpfi_dinghao\iutils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% containers 
all_sp_info_stim_pc_only = []; all_sp_info_ctrl_pc_only = []
all_sp_info_stim_all_pyr = []; all_sp_info_ctrl_all_pyr = []


#%% main 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    # load place cells
    FieldSpCorrAligned = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
    place_cell = FieldSpCorrAligned['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]
    tot_pc = len(place_cell)
    
    # load single-cell spatial information 
    SpInfoAligned = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_SpInfoAligned_msess1_Run0.mat'.format(pathname, recname))
    sp_info_stim = SpInfoAligned['spatialInfoSessStim'][0][0][0]['adaptSpatialInfo'][0][0]
    sp_info_ctrl = SpInfoAligned['spatialInfoSessStimCtrl'][0][0][0]['adaptSpatialInfo'][0][0]
    sp_info_stim_pc = sp_info_stim[place_cell-1]
    sp_info_ctrl_pc = sp_info_ctrl[place_cell-1]
    
    all_sp_info_stim_pc_only.extend(sp_info_stim_pc)
    all_sp_info_ctrl_pc_only.extend(sp_info_ctrl_pc)
    all_sp_info_stim_all_pyr.extend(sp_info_stim)
    all_sp_info_ctrl_all_pyr.extend(sp_info_ctrl)
    
    # plot single session pc only
    if len(sp_info_ctrl_pc)!=0:  # in case there is no place cell in this session
        savepath_pc_only = 'Z:\Dinghao\code_dinghao\HPC_all\spatial_info\{}_spatial_info_stim_ctrl_pc_only'.format(recname)
        pf.plot_violin_with_scatter(sp_info_ctrl_pc, sp_info_stim_pc, 'grey', 'royalblue', 
                                    xticklabels=['ctrl', 'stim'], ylabel='spatial info. (bits/s)',
                                    save=True, savepath=savepath_pc_only)
        
    # plot single session all
    savepath_all_pyr = 'Z:\Dinghao\code_dinghao\HPC_all\spatial_info\{}_spatial_info_stim_ctrl_all_pyr'.format(recname)
    pf.plot_violin_with_scatter(sp_info_ctrl, sp_info_stim, 'grey', 'royalblue', 
                                xticklabels=['ctrl', 'stim'], ylabel='spatial info. (bits/s)',
                                save=True, savepath=savepath_all_pyr)

#%% pooled 
all_save_path_pc_only = r'Z:\Dinghao\code_dinghao\HPC_all\spatial_info\all_spatial_info_stim_ctrl_pc_only'
pf.plot_violin_with_scatter(all_sp_info_ctrl_pc_only, all_sp_info_stim_pc_only, 'grey', 'royalblue', 
                            xticklabels=['ctrl', 'stim'], ylabel='spatial info. (bits/s)',
                            save=True, savepath=all_save_path_pc_only, dpi=300)