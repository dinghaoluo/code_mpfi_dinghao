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


#%% load functions
# if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
#     sys.path.append('Z:\Dinghao\code_dinghao\common')
# from common import normalise


#%% basic parameters 
tot_time = 5 * 1250  # 5 seconds in 1250 Hz
comp_window = [1875, 3750]  # 0.5-2 seconds 
cont_window = [5000, 6250]  # 4-5 seconds


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
    stim_start = stim_trials[0]; stim_end = stim_trials[-1]
    
    tot_clu = len(pyr_id)
    tot_pyr = sum(pyr_id)  # how many pyramidal cells are in this recording
    pyr_cont = {}
    pyr_cont_cont = {}
    pyr_stim = {}
    pyr_stim_cont = {}
    
    X = np.zeros((tot_clu, stim_start+len(stim_trials)))
    for i in range(tot_clu):
        if pyr_id[i]==True:
            cluname = clu_list[i]
            temp_all = np.zeros(stim_start+len(stim_trials))
            temp_cont = np.zeros(stim_start)
            temp_stim = np.zeros(len(stim_trials))

            for trial in range(stim_start):
                temp_cont[trial] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
                temp_all[trial] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
            for ind, trial in enumerate(stim_trials):
                temp_stim[ind] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
                temp_all[stim_start+ind] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
            pyr_cont[cluname] = temp_cont
            pyr_stim[cluname] = temp_stim
            
            X[i, :] = temp_all
            
    X = np.transpose(X)
    X[np.isnan(X)] = 0
    y = np.arange(stim_start+len(stim_trials))
    
    labels = ['grey']*stim_start+['royalblue']*len(stim_trials)
            
    # PCA stuff
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # split the dataset into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, random_state=13, test_size=0.25, shuffle=True)

    # Run PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_norm)

    # plot first and second principal component axes
    fig = plt.figure()
    
    ax = plt.axes(projection ='3d')
    ax.scatter(xs=X_pca[:, 0], ys=X_pca[:, 1], zs=X_pca[:, 2], c=labels)
    
    # legend
    st = ax.scatter([],[], c='royalblue')
    ct = ax.scatter([],[], c='grey')
    ax.legend([st, ct], ['stim', 'baseline'], frameon=False)
    
    ax.set(title='PCA, stim v baseline, {}'.format(recname))
    
    fig.tight_layout()
            
    # make folders if not exist
    outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
    if not os.path.exists(outdirroot):
        os.makedirs(outdirroot)
    outdir = '{}\stim_baseline_pyr_only_PCA'.format(outdirroot)
            
    fig.savefig(outdir,
                dpi=500,
                bbox_inches='tight')