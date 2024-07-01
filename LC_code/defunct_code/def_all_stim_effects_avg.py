# -*- coding: utf-8 -*-
"""
Created on Mon 20 Nov 14:26:42 2023

stimulation effects with firing rate profile 

@author: Dinghao Luo 
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats import sem
import pandas as pd
plt.rcParams['font.family'] = 'Arial' 


#%% load data 
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
clu_list = list(cell_prop.index)

tag_list = []; put_list = []
tag_rop_list = []; put_rop_list = []
for clu in cell_prop.index:
    tg = cell_prop['tagged'][clu]
    pt = cell_prop['putative'][clu]
    rop = cell_prop['peakness'][clu]
    
    if tg:
        tag_list.append(clu)
        if rop:
            tag_rop_list.append(clu)
    if pt:
        put_list.append(clu)
        if rop:
            put_rop_list.append(clu)
            
            
#%% main
for cluname in tag_list[50:]:
    print(cluname)
    train = all_train[cluname]
    
    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    behInfo = sio.loadmat(filename)['beh']
    stim_trial = np.squeeze(np.where(behInfo['pulseMethod'][0][0][0]!=0))-1
    stim_cont = stim_trial+2
    tot_trial = len(behInfo['pulseMethod'][0][0][0])

    # plotting
    fig, ax = plt.subplots(figsize=(4,3))
    ax.set(xlabel='time (s)', ylabel='trial # by first licks',
           xlim=(-1, 3))
    for p in ['top', 'right']:
        ax.spines[p].set_visible(False)

    curr_stims = [t[2500:7500]*1250 for i, t in enumerate(train) if i in stim_trial]
    curr_conts = [t[2500:7500]*1250 for i, t in enumerate(train) if i in stim_cont]
    
    mean_stims = np.mean(curr_stims, axis=0)
    mean_conts = np.mean(curr_conts, axis=0)
    sem_stims = sem(curr_stims, axis=0)
    sem_conts = sem(curr_conts, axis=0)
    
    xaxis = np.arange(1250*4)/1250-1 
    sl, = ax.plot(xaxis, mean_stims, c='royalblue')
    cl, = ax.plot(xaxis, mean_conts, c='grey')
    ax.fill_between(xaxis, mean_stims+sem_stims,
                           mean_stims-sem_stims,
                           alpha=.3, color='royalblue')
    ax.fill_between(xaxis, mean_conts+sem_conts,
                           mean_conts-sem_conts,
                           alpha=.3, color='grey')
    
    # axs['A'].legend(handles=[fl], frameon=False, fontsize=10)
    # axs['A'].set(yticks=[1, 50, 100])