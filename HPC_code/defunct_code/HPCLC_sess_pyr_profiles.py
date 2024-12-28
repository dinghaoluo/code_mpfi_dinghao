# -*- coding: utf-8 -*-
"""
Created on Tue Aug  27 10:39:28 2024

plot profiles for all pyramidal cells in the HPC-LC stimulation in each session

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import sys 

sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise, mpl_formatting
mpl_formatting()


#%% parameters
# run HPC-LC or HPC-LCterm
HPC_LC = 1


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
if HPC_LC:
    pathHPC = rec_list.pathHPCLCopt
elif not HPC_LC:
    pathHPC = rec_list.pathHPCLCtermopt


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

    # put each averaged profile into the separate matrices 
    temp_avg = np.zeros((tot_pyr, 8*1250))
    temp_avg_baseline = np.zeros((tot_pyr, 8*1250))
    cell_counter = 0
    for i, if_pyr in enumerate(pyr_id):
        if if_pyr:
            # take average
            temp = np.zeros((tot_trial, 8*1250))
            temp_baseline = np.zeros((stim_trials[0], 8*1250))
            for trial in range(tot_trial):
                trial_length = len(trains[i][trial])
                if trial_length<8*1250 and trial_length>0:
                    temp[trial, :trial_length] = trains[i][trial][:8*1250]  # use [-3, 5]
                    if trial<stim_trials[0]:
                        temp_baseline[trial, :trial_length] = trains[i][trial][:8*1250]
                elif trial_length>0:
                    temp[trial, :] = trains[i][trial][:8*1250]
                    if trial<stim_trials[0]:
                        temp_baseline[trial, :trial_length] = trains[i][trial][:8*1250]
            temp_avg[cell_counter,:] = np.mean(temp, axis=0)
            temp_avg_baseline[cell_counter,:] = np.mean(temp_baseline, axis=0)
            
            cell_counter+=1
            
    # order stim_cont
    pp_ratio = {}  # argmax for conts for all pyrs
    pp_ratio_baseline = {}
    rise_count = 0; down_count = 0
    for i in range(tot_pyr):
        curr_ratio = np.nanmean(temp_avg[i,1875:3125])/np.nanmean(temp_avg[i,4375:5625])  # .5~1.5s
        curr_ratio_baseline = np.nanmean(temp_avg_baseline[i,1875:3125])/np.nanmean(temp_avg_baseline[i,4375:5625])
        pp_ratio[i] = curr_ratio
        pp_ratio_baseline[i] = curr_ratio_baseline
        if curr_ratio < .8:  # rise-cells 
            rise_count+=1
        elif curr_ratio > 1.25:  # down-cells 
            down_count+=1
        if np.isnan(curr_ratio) or max(temp_avg[i,:])==0:  # if everything is 0
            pp_ratio[i] = 100  # if divided by 0
        if np.isnan(curr_ratio_baseline) or max(temp_avg_baseline[i,:])==0:
            pp_ratio_baseline[i] = 100
    def helper(x):
        return pp_ratio[x]
    def helper_baseline(x):
        return pp_ratio_baseline[x]
    pp_ord_ind = sorted(np.arange(tot_pyr), key=helper)
    pp_ord_ind_baseline = sorted(np.arange(tot_pyr), key=helper_baseline)
    
    im_ord = np.zeros((tot_pyr, 8*1250))
    im_ord_baseline = np.zeros((tot_pyr, 8*1250))
    for i, ind in enumerate(pp_ord_ind): 
        if pp_ratio[i] != 100:
            im_ord[i,:] = normalise(temp_avg[ind,:])
    for i, ind in enumerate(pp_ord_ind_baseline): 
        if pp_ratio_baseline[i] != 100:
            im_ord_baseline[i,:] = normalise(temp_avg_baseline[ind,:])
    
    
    # plotting (pre-post)
    fig, ax = plt.subplots(figsize=(2.5,2.2))
    ax.set(title='all pyr pre-post R', xlabel='time (s)', ylabel='pyr #',
           xticks=[-2,0,2,4], xlim=(-3,5))
    image_all = ax.imshow(im_ord, interpolation='none', cmap='Greys', aspect='auto', 
                          extent=[-3, 5, 1, tot_pyr])
    plt.colorbar(image_all, shrink=.5)
    # fig.suptitle('HPC_LC')
    fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_profiles\RO_act_inh\{}_all.png'.format(recname),
                dpi=300, bbox_inches='tight')
        
    
    fig, ax = plt.subplots(figsize=(2.5,2.2))
    ax.set(title='baseline pyr pre-post R', xlabel='time (s)', ylabel='pyr #',
           xticks=[-2,0,2,4], xlim=(-3,5))
    image_baseline = ax.imshow(im_ord_baseline, interpolation='none', cmap='Greys', aspect='auto', 
                          extent=[-3, 5, 1, tot_pyr])
    plt.colorbar(image_baseline, shrink=.5)
    # fig.suptitle('HPC_LC')
    fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_profiles\RO_act_inh\{}_baseline.png'.format(recname),
                dpi=300, bbox_inches='tight')