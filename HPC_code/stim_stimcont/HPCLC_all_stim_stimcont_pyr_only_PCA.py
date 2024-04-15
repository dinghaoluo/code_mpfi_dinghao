# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:44:27 2023

single session PCA, compare all HPC cell's spiking profile between cont and stim 

@author: Dinghao
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 
import scipy.io as sio 
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# define default plotly renderer
pio.renderers.default = "browser"


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% basic parameters 
tot_time = 5 * 1250  # 5 seconds in 1250 Hz
comp_window = [4375, 5625]  # 0.5-1.5 seconds 
# cont_window = [, 6250]  # 3-4 seconds


#%% containers 
all_shuf_dist = []
all_dist = []


#%% main 
for pathname in pathHPC:
    recname = pathname[-17:]
    
    print(recname)
    
    # load trains for this recording 
    all_info = np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname),
                       allow_pickle=True).item()
    
    trains = list(all_info.values())
    clu_list = list(all_info.keys())
    tot_trial = len(trains[0])
    
    # if each neurones is an interneurone or a pyramidal cell 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    
    # behaviour parameters 
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2
    
    tot_clu = len(pyr_id)
    tot_pyr = sum(pyr_id)  # how many pyramidal cells are in this recording
    pyr_cont = {}
    pyr_cont_cont = {}
    pyr_stim = {}
    pyr_stim_cont = {}
    
    X = np.zeros((tot_clu, len(stim_trials)*2))
    for i in range(tot_clu):
        if pyr_id[i]==True:
            cluname = clu_list[i]
            temp_all = np.zeros(len(stim_trials)*2)

            for ind, trial in enumerate(cont_trials):
                temp_all[ind] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
            for ind, trial in enumerate(stim_trials):
                temp_all[len(stim_trials)+ind] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
            
            X[i, :] = temp_all
            
    X = np.transpose(X)
    X[np.isnan(X)] = 0
    y = np.arange(len(stim_trials)*2)
    
    labels = ['grey']*len(stim_trials)+['royalblue']*len(stim_trials)
            
    # PCA stuff
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # split the dataset into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, random_state=42, test_size=0.25, shuffle=True)

    # Run PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_norm)

    # plot first and second principal component axes
    fig = plt.figure()
    
    ax = plt.axes(projection ='3d')
    ax.scatter(xs=X_pca[:, 0], ys=X_pca[:, 1], zs=X_pca[:, 2], c=labels)
    
    # mean 
    ax.scatter(xs=np.mean(X_pca[:len(stim_trials), 0]),
               ys=np.mean(X_pca[:len(stim_trials), 0]),
               zs=np.mean(X_pca[:len(stim_trials), 0]),
               color='black')
    ax.scatter(xs=np.mean(X_pca[len(stim_trials):, 0]),
               ys=np.mean(X_pca[len(stim_trials):, 0]),
               zs=np.mean(X_pca[len(stim_trials):, 0]),
               color='darkblue')
    
    # legend
    st = ax.scatter([],[], c='royalblue')
    ct = ax.scatter([],[], c='grey')
    ax.legend([st, ct], ['stim', 'baseline'], frameon=False)
    
    ax.set(title='PCA, stim v stim-cont, {}'.format(recname))
    
    fig.tight_layout()
            
    # make folders if not exist
    outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
    outdir = '{}\stim_stim_cont_pyr_only_PCA'.format(outdirroot)
            
    fig.savefig(outdir,
                dpi=500,
                bbox_inches='tight')
    
    stim_cent = np.array([np.mean(X_pca[:len(stim_trials), 0]),
                          np.mean(X_pca[:len(stim_trials), 0]),
                          np.mean(X_pca[:len(stim_trials), 0])])
    cont_cent = np.array([np.mean(X_pca[len(stim_trials):, 0]),
                          np.mean(X_pca[len(stim_trials):, 0]),
                          np.mean(X_pca[len(stim_trials):, 0])])
    
    squared_dist = np.sum((cont_cent-stim_cent)**2, axis=0)
    dist = np.sqrt(squared_dist)
    
    
    # shuffled
    shuf_number = 500
    
    dist_shuf = np.zeros(shuf_number)
    for shuf in range(shuf_number):
        shuf_1 = np.random.randint(low=0, high=tot_trial, size=len(stim_trials))
        shuf_2 = np.random.randint(low=0, high=tot_trial, size=len(stim_trials))
        
        X_shuf = np.zeros((tot_clu, len(stim_trials)*2))
        for i in range(tot_clu):
            if pyr_id[i]==True:
                cluname = clu_list[i]
                temp_all = np.zeros(len(stim_trials)*2)

                for ind, trial in enumerate(shuf_1):
                    temp_all[ind] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
                for ind, trial in enumerate(shuf_2):
                    temp_all[len(stim_trials)+ind] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
                
                X_shuf[i, :] = temp_all
            
        X_shuf = np.transpose(X_shuf)
        X_shuf[np.isnan(X_shuf)] = 0
        y = np.arange(len(stim_trials)*2)
                
        # PCA stuff
        X_norm_shuf = scaler.fit_transform(X_shuf)
    
        # split the dataset into train and test datasets
        X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = train_test_split(X_norm_shuf, y, random_state=42, test_size=0.25, shuffle=True)
    
        # Run PCA
        pca = PCA(n_components=3)
        X_shuf_pca = pca.fit_transform(X_norm_shuf)
        
        shuf_1_cent = np.array([np.mean(X_shuf_pca[:len(stim_trials), 0]),
                                np.mean(X_shuf_pca[:len(stim_trials), 0]),
                                np.mean(X_shuf_pca[:len(stim_trials), 0])])
        shuf_2_cent = np.array([np.mean(X_shuf_pca[len(stim_trials):, 0]),
                                np.mean(X_shuf_pca[len(stim_trials):, 0]),
                                np.mean(X_shuf_pca[len(stim_trials):, 0])])
        
        squared_dist_shuf = np.sum((shuf_2_cent-shuf_1_cent)**2, axis=0)
        dist_shuf[shuf] = np.sqrt(squared_dist_shuf)
        
    # get 99.9 percentile of shuffle
    shuf_98 = np.percentile(dist_shuf, 98)  # alpha=.5, two-tailed
    all_shuf_dist.append(shuf_98)
    all_dist.append(dist)
    
    # histogram for distance comparison
    fig, ax = plt.subplots(figsize=(4,3))
    
    ax.hist(dist_shuf, bins=25, ec='dimgrey', color='grey', alpha=.85, density=True)
    ax.vlines(dist, 0, 1.5, color='royalblue', linewidth=3)
    ax.vlines(shuf_98, 0, 1.5, color='dimgrey', linewidth=3)
    
    ax.set(yticks=[], yticklabels=[],
           xlabel='distance btw. embedded means',
           ylim=(0,1.1))
    
    fig.suptitle('stim v stim-cont, {}'.format(recname))
    
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)

    fig.tight_layout() 
    
    fig.savefig('{}_hist_dist'.format(outdir),
                dpi=500,
                bbox_inches='tight')
    

#%% plotting 
fig, ax = plt.subplots(figsize=(3,4))

ax.bar([1,2],[np.mean(all_shuf_dist), np.mean(all_dist)])