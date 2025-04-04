# -*- coding: utf-8 -*- 
"""
Created on Thu Apr  3 14:27:47 2025

look at the spiking profile of persistent run-onset ON and newly recruited 
    run-onset ON cells

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys 
from scipy.stats import sem
import os 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt


#%% load data 
print('loading dataframes...')

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
    )
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 


#%% main 
persistent_ON_ctrl = []
persistent_ON_stim = []
new_ON_ctrl = []
new_ON_stim = []

for index, session in df_pyr.iterrows():
    print(index)
    
    if session['class_ctrl']=='run-onset ON' and session['class_stim']=='run-onset ON':
        if 0.15<session['spike_rate']<7:
            persistent_ON_ctrl.append(session['prof_ctrl_mean'])
            persistent_ON_stim.append(session['prof_stim_mean'])
    
    if session['class_ctrl']!='run-onset ON' and session['class_stim']=='run-onset ON':
        if 0.15<session['spike_rate']<7:
            new_ON_ctrl.append(session['prof_ctrl_mean'])
            new_ON_stim.append(session['prof_stim_mean'])
        
        
persistent_ON_ctrl_mean = np.mean(persistent_ON_ctrl, axis=0)
persistent_ON_stim_mean = np.mean(persistent_ON_stim, axis=0)
new_ON_ctrl_mean = np.mean(new_ON_ctrl, axis=0)
new_ON_stim_mean = np.mean(new_ON_stim, axis=0)

persistent_ON_ctrl_sem = sem(persistent_ON_ctrl, axis=0)
persistent_ON_stim_sem = sem(persistent_ON_stim, axis=0)
new_ON_ctrl_sem = sem(new_ON_ctrl, axis=0)
new_ON_stim_sem = sem(new_ON_stim, axis=0)


#%% plotting 
TAXIS = np.arange(-1250, -1250+6*1250)/1250

fig, ax = plt.subplots(figsize=(3,2.5))

# ax.plot(TAXIS,
#         persistent_ON_ctrl_mean[2500:2500+1250*5],
#         color='grey')
ax.plot(TAXIS,
        persistent_ON_stim_mean[2500:2500+1250*6],
        color='royalblue')
# ax.fill_between(TAXIS,
#                 persistent_ON_ctrl_mean[2500:2500+1250*5] + persistent_ON_ctrl_sem[2500:2500+1250*5],
#                 persistent_ON_ctrl_mean[2500:2500+1250*5] - persistent_ON_ctrl_sem[2500:2500+1250*5],
#                 color='grey', edgecolor='none', alpha=.25)
ax.fill_between(TAXIS,
                persistent_ON_stim_mean[2500:2500+1250*6] + persistent_ON_stim_sem[2500:2500+1250*6],
                persistent_ON_stim_mean[2500:2500+1250*6] - persistent_ON_stim_sem[2500:2500+1250*6],
                color='royalblue', edgecolor='none', alpha=.25)

# ax.plot(TAXIS,
#         new_ON_ctrl_mean[2500:2500+1250*5],
#         color='limegreen')
ax.plot(TAXIS,
        new_ON_stim_mean[2500:2500+1250*6],
        color='green')
# ax.fill_between(TAXIS,
#                 new_ON_ctrl_mean[2500:2500+1250*5] + new_ON_ctrl_sem[2500:2500+1250*5],
#                 new_ON_ctrl_mean[2500:2500+1250*5] - new_ON_ctrl_sem[2500:2500+1250*5],
#                 color='limegreen', edgecolor='none', alpha=.25)
ax.fill_between(TAXIS,
                new_ON_stim_mean[2500:2500+1250*6] + new_ON_stim_sem[2500:2500+1250*6],
                new_ON_stim_mean[2500:2500+1250*6] - new_ON_stim_sem[2500:2500+1250*6],
                color='green', edgecolor='none', alpha=.25)