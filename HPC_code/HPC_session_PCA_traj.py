# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:26:17 2024

perform PCA on a trial-by-trial basis, after fitting principal components to 
    averaged baseline activity

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 
import scipy.io as sio 
from scipy.stats import sem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# import pre-processing functions 
if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise

#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% parameters 
tot_samp = 1250*4


#%% containers 
dist_cs_all = []


#%% distance function 
def dist(p1, p2):
    # takes in 2 points and calculate Euc. dist.
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


#%% loop 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    # output routes
    outdirroot = r'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
    if not os.path.exists(outdirroot):
        os.makedirs(outdirroot)
    
    # load data
    trains = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
                          allow_pickle=True).item().values())
        
    # if each neurones is an interneurone or a pyramidal cell 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    tot_pyr = sum(pyr_id)
    
    # behaviour parameters 
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2
    
    # select trials with trial lengths between 3.5 and 4.5 seconds 
    use_trials = []
    for t in range(len(trains[0])):
        trial_length = len(trains[0][t][3750:])
        if trial_length<1250*5 and trial_length>1250*3:
            use_trials.append(t)
    tot_trial = len(use_trials)
    
    stim_trials = [t for t in stim_trials if t in use_trials]
    cont_trials = [t for t in cont_trials if t in use_trials]
    
    if len(stim_trials)==0 or len(cont_trials)==0:
        continue
    
    X_stim = np.zeros((tot_pyr, tot_samp*len(stim_trials)))
    X_cont = np.zeros((tot_pyr, tot_samp*len(cont_trials)))
    X_all = np.zeros((tot_pyr, tot_samp*tot_trial))
    
    pyr_counter = 0
    for ind, pyr in enumerate(pyr_id):
        if pyr:  # if pyramidal cell
            for i, trial in enumerate(use_trials):
                curr_train_pad = np.zeros(1250*4)
                curr_train = trains[ind][trial][3750:]  # from 0 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*4:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*4]
                # put this into X
                X_all[pyr_counter, i*tot_samp:(i+1)*tot_samp] = curr_train_pad
                            
            for i, trial in enumerate(stim_trials):
                curr_train_pad = np.zeros(1250*4)
                curr_train = trains[ind][trial][3750:]  # from 0 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*4:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*4]
                # put this into X
                X_stim[pyr_counter, i*tot_samp:(i+1)*tot_samp] = curr_train_pad
            
            for i, trial in enumerate(cont_trials):
                curr_train_pad = np.zeros(1250*4)
                curr_train = trains[ind][trial][3750:]  # from 0 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*4:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*4]
                # put this into X
                X_cont[pyr_counter, i*tot_samp:(i+1)*tot_samp] = curr_train_pad
                
            pyr_counter+=1
            
    
    # PCA stuff 
    # Run PCA
    pca = PCA(n_components=3)
    pca.fit(np.transpose(X_all))
    
    
    # fitting every single trial to these axes
    fig = plt.figure(figsize=(5,7))
    ax = plt.axes(projection ='3d')
    
    X_cont_avg = np.zeros((tot_pyr, tot_samp))
    X_stim_avg = np.zeros((tot_pyr, tot_samp))
    X_all_avg = np.zeros((tot_pyr, tot_samp))
    for t in range(len(cont_trials)):
        X_cont_avg+=X_cont[:, t*tot_samp:(t+1)*tot_samp]
    for t in range(len(stim_trials)):
        X_stim_avg+=X_stim[:, t*tot_samp:(t+1)*tot_samp]
    for t in range(tot_trial):
        X_all_avg+=X_all[:, t*tot_samp:(t+1)*tot_samp]
    X_cont_avg/=len(cont_trials)
    X_stim_avg/=len(stim_trials)
    X_all_avg/=tot_trial
    X_cont_avg = np.transpose(X_cont_avg)
    X_stim_avg = np.transpose(X_stim_avg)
    X_all_avg = np.transpose(X_all_avg)
    X_cont_avg_pca = pca.transform(X_cont_avg)
    X_stim_avg_pca = pca.transform(X_stim_avg)
    X_all_avg_pca = pca.transform(X_all_avg)
    
    show = 1250*3
    t = np.arange(show)
    sc = ax.scatter(xs=X_cont_avg_pca[:show, 0], ys=X_cont_avg_pca[:show, 1], zs=X_cont_avg_pca[:show, 2], s=3, c=t, cmap='summer')
    ss = ax.scatter(xs=X_stim_avg_pca[:show, 0], ys=X_stim_avg_pca[:show, 1], zs=X_stim_avg_pca[:show, 2], s=3, c=t, cmap='autumn')
    sa = ax.scatter(xs=X_all_avg_pca[:show, 0], ys=X_all_avg_pca[:show, 1], zs=X_all_avg_pca[:show, 2], s=3, c=t, cmap='gist_gray')
    
    plt.colorbar(sc, shrink=.2, ticks=[0, 3000], pad=0)
    plt.colorbar(ss, shrink=.2, ticks=[], pad=0)
    plt.colorbar(sa, shrink=.2, ticks=[], pad=.2)
    
    fig.tight_layout()
    ax.set(title=recname)
    fig.savefig(r'{}\stim_stimcont_pca.png'.format(outdirroot),
                dpi=500,
                bbox_inches='tight')
    
    # distance profiles 
    dist_cs = np.zeros(1250*4)
    for i in range(1250*4):
        dist_cs[i] = dist(X_stim_avg_pca[i,:], X_cont_avg_pca[i,:])
    
    dist_cs_all.append(normalise(dist_cs))
    
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(dist_cs)
    fig.savefig(r'{}\stim_stimcont_pca_dist.png'.format(outdirroot),
                dpi=500,
                bbox_inches='tight')
    

#%% close everything 
plt.close(fig)


#%% average distance 
avg_dist_cs = np.mean(dist_cs_all, axis=0)
sem_dist_cs = sem(dist_cs_all, axis=0)
xaxis = np.arange(5000)/1250

fig, ax = plt.subplots(figsize=(3,3))

ax.plot(xaxis, avg_dist_cs)
ax.fill_between(xaxis, avg_dist_cs+sem_dist_cs,
                       avg_dist_cs-sem_dist_cs,
                       alpha=.3)