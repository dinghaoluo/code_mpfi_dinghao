# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:40:52 2025

plot spatial information/temporal information (ctrl. v stim.) for 
    CA1 recordings with LC/LCterm stimulation

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import pandas as pd 


#%% load dataframe 
df = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')


#%% functions
sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from plotting_functions import plot_violin_with_scatter
from common import mpl_formatting 
mpl_formatting()


#%% filter dataframes 
df_HPCLC = df[(df['rectype']=='HPCLC') & (df['cell_identity']=='pyr')]
df_HPCLCterm = df[(df['rectype']=='HPCLCterm') & (df['cell_identity']=='pyr')]


#%% extract info 
SI_ctrl = [group.tolist() for name, group in df_HPCLC.groupby('recname')['SI']]
SI_stim = [group.tolist() for name, group in df_HPCLC.groupby('recname')['SI_stim']]

SI_ctrl_mean_single_cell = []
SI_stim_mean_single_cell = []

for ctrl, stim in zip(SI_ctrl, SI_stim):
    tot_clu = len(ctrl)
    for clu in range(tot_clu):
        SI_ctrl_mean_single_cell.append(np.mean(ctrl[clu]))
        SI_stim_mean_single_cell.append(np.mean(stim[clu]))
        

#%% plotting 
plot_violin_with_scatter(SI_ctrl_mean_single_cell, SI_stim_mean_single_cell, 
                         'grey', 'royalblue')