# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:11:43 2023

time-warped run-onset-first-lick profile of all neurones

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
import scipy.io as sio
import pandas as pd
from scipy.signal import resample 

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
warped_dict = {}

for cluname in clu_list:
    print(cluname)
    train = all_train[cluname]
    
    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    alignRun = sio.loadmat(filename)
    
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]
    
    behParf = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    behPar = sio.loadmat(behParf)
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stimOn_ind = np.where(stimOn!=0)[0]
    bad_beh_ind = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] if l-starts[trial] > 1250]  # only if the animal does not lick in the first second (carry-over licks)
        if len(lk)==0:
            first_licks.append(10000+3750)
        else:
            first_licks.extend(lk[0]-starts[trial]+3750)
            
    # default to noStim so that no contamination happens
    noStim = True
    
    # filter out stim and bad trials starting from 0
    trial_list = np.arange(tot_trial)
    trial_list = [t for t in trial_list if t not in stimOn_ind and t not in bad_beh_ind]
    tot_trial = len(trial_list)  # reset tot_trial 
    
    # downsample each trial between RO and 1st lick to 1 sec (1250 samples)
    warped_mat = np.zeros((len(trial_list), 2500))
    for i, trial in enumerate(trial_list):
        curr_1stlick = first_licks[trial]
        curr_train = train[trial]
        curr_bef_RO = curr_train[3125:3750]
        curr_mid = resample(curr_train[3750:curr_1stlick], 1250)  # actual time-warping 
        curr_aft_1stlick = curr_train[curr_1stlick:curr_1stlick+625]
        
        curr_warped = np.concatenate((curr_bef_RO, curr_mid, curr_aft_1stlick))
        length = len(curr_warped)
        
        warped_mat[i,:length] = curr_warped

    mean_warped = np.reshape(np.mean(warped_mat, axis=0), (1, 2500))
    warped_dict[cluname] = mean_warped 

    # plotting
    fig, axs = plt.subplot_mosaic('A;A;A;B',figsize=(3,3))
    
    axs['A'].imshow(warped_mat, aspect='auto', extent=[-0.5, 1.5, 1, tot_trial+1], cmap='Greys')
    axs['A'].set(xticks=[0, 1])
    
    axs['B'].imshow(mean_warped, aspect='auto', extent=[-0.5, 1.5, 0, 1], cmap='Greys')
    axs['B'].set(yticks=[], xticks=[0, 1])
    
    fig.suptitle(cluname)
    fig.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_warped_RO_first_licks\{}.png'.format(cluname),
                bbox_inches='tight',
                dpi=500)

    plt.close(fig)
    

#%% save 
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_warped.npy', 
        warped_dict)