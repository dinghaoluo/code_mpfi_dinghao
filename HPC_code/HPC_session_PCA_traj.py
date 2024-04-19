# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:26:17 2024

perform PCA on averaged spike trains and calculate distances between points 
    on the different trajectories

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
from timeit import default_timer

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
xaxis = np.arange(5000)/1250
n_shuf = 100


#%% distance function 
def dist(p1, p2):
    # takes in 2 points and calculate Euc. dist.
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def shuffle_mean(train, n_shuf=100):
    """
    Parameters
    ----------
    train : 1d-numpy array
        downsampled spike rate vector.
    n_shuf : int
        how many times to shuffle.

    Returns
    -------
    shuf_train : 1d-numpy array
        meaned shuffled train.
    """    
    # Setting up multiprocessing
    # num_workers = int(0.8 * os.cpu_count())
    # chunksize = max(1, n_shuf / num_workers)
    # train = [list(train) for s in range(n_shuf)]
    
    # print('starting parallel processing')
    # with Pool(num_workers) as p:
    #     shuf_train = p.map(lambda x : np.roll(x, -np.random.randint(1, 25)), train, chunksize)

    shift = np.random.randint(1, 1250*4, n_shuf)
    shuf_array = np.zeros((n_shuf, 1250*4))
 
    for i in range(n_shuf):
        shuf_array[i,:] = np.roll(train, -shift[i])
        
    shuf_train = np.mean(shuf_array, axis=0)
    
    return shuf_train


#%% containers 
dist_cs_all = []  # ctrl.-stim. distances 
dist_ca_all = []  # ctrl.-all distances
dist_sa_all = []  # stim.-all distances 
dist_shuf_all = []  # shuf.-all distances 


#%% loop 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    # time 
    start = default_timer()
    
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
    X_all_shuf = np.zeros((tot_pyr, tot_samp*tot_trial))
    
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
                # shuffle
                X_all_shuf[pyr_counter, i*tot_samp:(i+1)*tot_samp] = shuffle_mean(curr_train_pad, n_shuf=n_shuf)
                            
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
            
    # Run PCA
    pca = PCA(n_components=3)
    pca.fit(np.transpose(X_all))
    
    # fitting every single trial to these axes
    fig = plt.figure(figsize=(5,7))
    ax = plt.axes(projection ='3d')
    
    X_cont_avg = np.zeros((tot_pyr, tot_samp))
    X_stim_avg = np.zeros((tot_pyr, tot_samp))
    X_all_avg = np.zeros((tot_pyr, tot_samp))
    X_all_shuf_avg = np.zeros((tot_pyr, tot_samp))
    for t in range(len(cont_trials)):
        X_cont_avg+=X_cont[:, t*tot_samp:(t+1)*tot_samp]
    for t in range(len(stim_trials)):
        X_stim_avg+=X_stim[:, t*tot_samp:(t+1)*tot_samp]
    for t in range(tot_trial):
        X_all_avg+=X_all[:, t*tot_samp:(t+1)*tot_samp]
        X_all_shuf_avg+=X_all_shuf[:, t*tot_samp:(t+1)*tot_samp]
    X_cont_avg/=len(cont_trials)
    X_stim_avg/=len(stim_trials)
    X_all_avg/=tot_trial
    X_all_shuf_avg/=tot_trial
    X_cont_avg = np.transpose(X_cont_avg)
    X_stim_avg = np.transpose(X_stim_avg)
    X_all_avg = np.transpose(X_all_avg)
    X_all_shuf_avg = np.transpose(X_all_shuf_avg)
    X_cont_avg_pca = pca.transform(X_cont_avg)
    X_stim_avg_pca = pca.transform(X_stim_avg)
    X_all_avg_pca = pca.transform(X_all_avg)
    X_all_shuf_avg_pca = pca.transform(X_all_shuf_avg)
    
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
    dist_ca = np.zeros(1250*4)
    dist_sa = np.zeros(1250*4)
    dist_shuf = np.zeros(1250*4)
    for i in range(1250*4):
        dist_cs[i] = dist(X_stim_avg_pca[i,:], X_cont_avg_pca[i,:])
        dist_sa[i] = dist(X_stim_avg_pca[i,:], X_all_avg_pca[i,:])
        dist_ca[i] = dist(X_cont_avg_pca[i,:], X_all_avg_pca[i,:])
        dist_shuf[i] = dist(X_all_shuf_avg_pca[i,:], X_all_avg_pca[i,:])
    
    dist_cs_all.append(normalise(dist_cs))
    dist_sa_all.append(normalise(dist_sa))
    dist_ca_all.append(normalise(dist_ca))
    dist_shuf_all.append(normalise(dist_shuf))
    
    fig, axs = plt.subplots(1,3, figsize=(10,3))
    
    axs[0].plot(xaxis, dist_cs, c='grey')
    axs[1].plot(xaxis, dist_ca, c='orange')
    axs[2].plot(xaxis, dist_sa, c='green')
    
    for p in range(3):
        axs[p].set(xlabel='time (s)', xticks=[0,2,4])
        axs[p].plot(xaxis, dist_shuf, c='red')
        for s in ['top', 'right']:
            axs[p].spines[s].set_visible(False)
    axs[0].set(ylabel='dist. stim.-ctrl.', title='stim.-ctrl.')
    axs[1].set(ylabel='dist. ctrl.-all', title='ctrl.-all')
    axs[2].set(ylabel='dist. stim.-all', title='stim.-all')
    
    fig.tight_layout()
    
    fig.savefig(r'{}\stim_stimcont_pca_dist.png'.format(outdirroot),
                dpi=500,
                bbox_inches='tight')
    
    print('Complete. Single-session time: {} s.\nStarting next session.'.format(default_timer()-start))
    

#%% close everything 
plt.close(fig)


#%% average distance 
avg_dist_cs = np.mean(dist_cs_all, axis=0)
sem_dist_cs = sem(dist_cs_all, axis=0)

fig, ax = plt.subplots(figsize=(3.5,3))

ax.plot(xaxis, avg_dist_cs, c='darkgreen')
ax.fill_between(xaxis, avg_dist_cs+sem_dist_cs,
                       avg_dist_cs-sem_dist_cs,
                       alpha=.3, color='darkgreen', edgecolor='none')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlabel='time (s)', xticks=[0,2,4],
       ylabel='norm. dist. stim.-ctrl.', yticks=[.3,.5,.7],
       title='PCA norm. dist. stim.-ctrl.')

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\dimensionality_reduction\pca_dist_stim_stimcont.png',
            dpi=500,
            bbox_inches='tight')