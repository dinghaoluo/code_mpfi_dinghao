# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:23:34 2024
Modified on Mon 3 Sept 14:19 2024

Naive Bayesian decoding of hippocampus population

***note on feature scaling***
    I do not want to make any assumption in terms of how much each neurone 
    contribute to the population representation, and therefore do not want to 
    scale the features when training the GNB decoder, since doing so would 
    imply an assumption of uniform contributions.
    
this code has been modified to focus on the pyramidal cells that get inhibited 
    after run-onset (pyr-inh population)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 
import sys
import os
import scipy.io as sio
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from timeit import default_timer
from scipy.stats import ttest_rel, sem, wilcoxon
# from multiprocessing import Pool

# plot output parameters
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% parameters 
n_shuf = 100  # how many times to bootstrap activity
tot_bin = 25


#%% pre-post ratio dataframe 
df = pd.read_pickle('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.pkl') 


#%% containers
all_error_all_pyr = []
all_error_pyrinh = []
all_error_shuf = []

all_error_all_pyr_ab_mean = []
all_error_pyrinh_ab_mean = []
all_error_shuf_ab_mean = []
all_error_early_all_pyr_mean = []
all_error_early_pyrinh_mean = []
all_error_early_shuf_mean = []
all_error_late_all_pyr_mean = []
all_error_late_pyrinh_mean = []
all_error_late_shuf_mean = []


#%% function 
def skipping_average(v, skip_step=250):
    """
    v : a numerical vector
    skip_step : length of intervals; default to 200 ms (250 samples, 1250 Hz)
    """
    n_int = np.floor_divide(len(v), skip_step)
    v_ret = np.zeros(n_int)
    for i in range(n_int):
        v_ret[i] = np.nanmean(v[i*skip_step:(i+1)*skip_step])
    
    return v_ret

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

    shift = np.random.randint(1, 25, n_shuf)
    shuf_array = np.zeros((n_shuf, 25))
 
    for i in range(n_shuf):
        shuf_array[i,:] = np.roll(train, -shift[i])
        
    shuf_train = np.mean(shuf_array, axis=0)
    
    return shuf_train


#%% loop 
for pathname in pathHPC[:-1]:
    recname = pathname[-17:]
    print(recname)
    
    # time 
    start = default_timer()

    # load data
    trains = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
                          allow_pickle=True).item().values())
    tot_trial = len(trains[0])
    
    # if each neurones is an interneurone or a pyramidal cell 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    tot_pyr = sum(pyr_id)
    
    # classify act./inh. neurones 
    pyract = []; pyrinh = []
    for cluname, row in df.iterrows():
        if cluname.split(' ')[0]==recname:
            clu_ID = int(cluname.split(' ')[1][3:])
            if row['ctrl_pre_post']>=1.25:
                pyrinh.append(clu_ID-2)
            if row['ctrl_pre_post']<=.8:
                pyract.append(clu_ID-2)  # HPCLC_all_train.py adds 2 to the ID, so we subtracts 2 here 
    tot_pyrinh = len(pyrinh)
    if tot_pyrinh<1:
        continue
    
    # behaviour parameters 
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2
    
    # here tot_test_trial is the number of trials in either ctrl or stim
    tot_test_trial = len(stim_trials)
    
    # preprocess data into matrices for training and testing 
    """
    What we are doing here: 
    For every single trial we want to have 25 time bins, 200 ms each, i.e. 250
    samples each. Therefore we first initialise a container matrix with the 
    dimensions tot_pyr x n time bins. The number of time bins is calculated based
    on the number of trials in either ctrl or stim, times 25 (per trial). After 
    that, y is initialised as basically 200 ms, 400 ms, ... 5000 ms for every 
    single trial, repeated tot_test_trial times, forming a 25xtot_test_trial-long 
    vector.
    """
    y = list(range(tot_bin))*tot_trial
    
    X_all_pyr = np.zeros((tot_pyr, tot_bin*tot_trial))
    X_pyrinh = np.zeros((tot_pyrinh, tot_bin*tot_trial))
    X_all_shuf = np.zeros((tot_pyr, tot_bin*tot_trial))
    
    # get stim and cont trial population vectors
    pyr_counter = 0
    pyrinh_counter = 0
    for clu in range(tot_pyr):
        if pyr_id[clu]==True:  # if pyramidal cell
            for trial in range(tot_trial):
                curr_train_pad = np.zeros(1250*5)
                curr_train = trains[clu][trial][3750:]  # from 0 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*5:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*5]
                curr_train_down = skipping_average(curr_train_pad)  # downsample curr_train to get our end result
                X_all_pyr[pyr_counter, trial*tot_bin:(trial+1)*tot_bin] = curr_train_down  # put this into X
                X_all_shuf[pyr_counter, trial*tot_bin:(trial+1)*tot_bin] = shuffle_mean(curr_train_down, n_shuf=n_shuf)  # shuffling, new, Dinghao, 17 Apr 24
            pyr_counter+=1
        if clu in pyrinh:
            for trial in range(tot_trial):
                curr_train_pad = np.zeros(1250*5)
                curr_train = trains[clu][trial][3750:]
                trial_length = len(curr_train)
                if trial_length<1250*5:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*5]
                curr_train_down = skipping_average(curr_train_pad)
                X_pyrinh[pyrinh_counter, trial*tot_bin:(trial+1)*tot_bin] = curr_train_down
            pyrinh_counter+=1
            
    # splitting training and testing datasets 
    X_all_pyr_train, X_all_pyr_test, y_all_pyr_train, y_all_pyr_test = train_test_split(np.transpose(X_all_pyr), y, test_size=.1)
    X_pyrinh_train, X_pyrinh_test, y_pyrinh_train, y_pyrinh_test = train_test_split(np.transpose(X_pyrinh), y, test_size=.1)
    
    # training 
    nb_model_all_pyr = GaussianNB()
    nb_model_all_pyr.fit(X_all_pyr_train, y_all_pyr_train)
    nb_model_pyrinh = GaussianNB()
    nb_model_pyrinh.fit(X_pyrinh_train, y_pyrinh_train)
    
    # evaluation (labelled)
    y_all_pyr_pred = nb_model_all_pyr.predict(X_all_pyr_test)
    error_all_pyr = np.zeros(tot_bin)  # error
    error_all_pyr_abs = np.zeros(tot_bin)  # absoluate error
    all_pyr_counter = np.zeros(tot_bin)
    for (t, p) in zip(y, y_all_pyr_pred):
        error_all_pyr[t]+=(t-p)/5  # divide by 10 to get seconds
        all_pyr_counter[t]+=1
    for i in range(tot_bin):
        error_all_pyr[i] = (error_all_pyr[i]/all_pyr_counter[i])
    
    y_pyrinh_pred = nb_model_pyrinh.predict(X_pyrinh_test)
    error_pyrinh = np.zeros(tot_bin)
    error_pyrinh_abs = np.zeros(tot_bin)
    pyrinh_counter = np.zeros(tot_bin)
    for (t, p) in zip(y, y_pyrinh_pred):
        error_pyrinh[t]+=(t-p)/5
        pyrinh_counter[t]+=1
    for i in range(tot_bin):
        error_pyrinh[i] = (error_pyrinh[i]/pyrinh_counter[i])
        
    y_shuf_pred = nb_model_all_pyr.predict(np.transpose(X_all_shuf))
    error_shuf = np.zeros(tot_bin)
    error_shuf_ab = np.zeros(tot_bin)
    shuf_counter = np.zeros(tot_bin)
    for (t, p) in zip(y, y_shuf_pred):
        error_shuf[t]+=(t-p)/5
        shuf_counter[t]+=1
    for i in range(tot_bin):
        error_shuf[i] = (error_shuf[i]/shuf_counter[i])
    
    all_error_all_pyr.append(error_all_pyr)
    all_error_pyrinh.append(error_pyrinh)
    all_error_shuf.append(error_shuf)
    
    # plotting (error, single sessions)
    fig, ax = plt.subplots(figsize=(4,4))
    
    xaxis = np.linspace(0.4, 5, 23)
    cl, = ax.plot(xaxis, error_all_pyr[:-2], c='grey')
    sl, = ax.plot(xaxis, error_pyrinh[:-2], c='royalblue')
    shl, = ax.plot(xaxis, error_shuf[:-2], c='red')
    
    ax.legend([cl, sl, shl], ['all pyr.', 'pyrinh', 'shuf.'], frameon=False)
    ax.set(xlabel='true time (s)', ylabel='error (s)')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\bayesian_decoding\all_pyr_v_inh\{}.png'.format(recname),
                dpi=200,
                bbox_inches='tight')
    
    # evaluation (proba)
    y_all_pyr_pred_proba = nb_model_all_pyr.predict_proba(np.transpose(X_all_pyr))
    y_all_pyr_proba_trial = y_all_pyr_pred_proba[:tot_bin, :]  # first trial
    y_all_pyr_proba_trial_means = np.zeros((tot_bin, tot_bin))
    for i in range(1, tot_test_trial):
        for j in range(tot_bin):
            y_all_pyr_proba_trial[j, :]+=y_all_pyr_pred_proba[tot_bin*i+j, :]
    for i in range(tot_bin):
        y_all_pyr_proba_trial_means[i,:] = y_all_pyr_proba_trial[i,:]/tot_bin
    
    y_pyrinh_pred_proba = nb_model_pyrinh.predict_proba(np.transpose(X_pyrinh))
    y_pyrinh_proba_trial = y_pyrinh_pred_proba[:tot_bin, :]  # first trial
    y_pyrinh_proba_trial_means = np.zeros((tot_bin, tot_bin))
    for i in range(1, tot_test_trial):
        for j in range(tot_bin):
            y_pyrinh_proba_trial[j, :]+=y_pyrinh_pred_proba[tot_bin*i+j, :]
    for i in range(tot_bin):
        y_pyrinh_proba_trial_means[i,:] = y_pyrinh_proba_trial[i,:]/tot_bin
        
    y_shuf_pred_proba = nb_model_all_pyr.predict_proba(np.transpose(X_all_shuf))
    y_shuf_proba_trial = y_shuf_pred_proba[:tot_bin, :]  # first trial
    y_shuf_proba_trial_means = np.zeros((tot_bin, tot_bin))
    for i in range(1, tot_test_trial):
        for j in range(tot_bin):
            y_shuf_proba_trial[j, :]+=y_shuf_pred_proba[tot_bin*i+j, :]
    for i in range(tot_bin):
        y_shuf_proba_trial_means[i,:] = y_shuf_proba_trial[i,:]/tot_bin
    
    # plotting 
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(9,3))
    
    pim1 = ax1.imshow(y_all_pyr_proba_trial_means[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    pim2 = ax2.imshow(y_pyrinh_proba_trial_means[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    pim3 = ax3.imshow(y_shuf_proba_trial_means[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    
    max1 = np.round(np.max(y_all_pyr_proba_trial_means[:-1, :-1]), 2)
    max2 = np.round(np.max(y_pyrinh_proba_trial_means[:-1, :-1]), 2)
    max3 = np.round(np.max(y_shuf_proba_trial_means[:-1, :-1]), 2)
    cb1 = plt.colorbar(pim1, shrink=.5, ticks=[0,max1])
    cb2 = plt.colorbar(pim2, shrink=.5, ticks=[0,max2])
    cb3 = plt.colorbar(pim3, shrink=.5, ticks=[0,max3])
    
    lab1 = np.argmax(y_all_pyr_proba_trial_means[:-1, :-1], axis=0)/5
    lab2 = np.argmax(y_pyrinh_proba_trial_means[:-1, :-1], axis=0)/5
    lab3 = np.argmax(y_shuf_proba_trial_means[:-1, :-1], axis=0)/5
    ax1.plot(np.linspace(0, 4.8, 24), lab1, c='white', lw=1)
    ax2.plot(np.linspace(0, 4.8, 24), lab2, c='white', lw=1)
    ax3.plot(np.linspace(0, 4.8, 24), lab3, c='white', lw=1)
    
    ax1.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    ax2.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    ax3.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    
    ax1.set(title='all pyr.', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    ax2.set(title='pyrinh', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    ax3.set(title='shuf.', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    
    fig.suptitle('post. prob. {}'.format(recname))
    
    fig.tight_layout()
    
    fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\bayesian_decoding\all_pyr_v_pyrinh\{}.png'.format(recname),
                dpi=200,
                bbox_inches='tight')
    
    print('Complete. Single-session time: {} s.\nStarting next session.'.format(default_timer()-start))

    
#%% close everything 
plt.close(fig)


#%% quantification
all_error_all_pyr_mean = np.mean(all_error_all_pyr, axis=0)
all_error_pyrinh_mean = np.mean(all_error_pyrinh, axis=0)
all_error_shuf_mean = np.mean(all_error_shuf, axis=0)
all_error_all_pyr_sem = sem(all_error_all_pyr, axis=0)
all_error_pyrinh_sem = sem(all_error_pyrinh, axis=0)
all_error_shuf_sem = sem(all_error_shuf, axis=0)


#%% plot errors 
fig, ax = plt.subplots(figsize=(4,3))

xaxis = np.linspace(0.4, 5, 23)

cl, = ax.plot(xaxis, all_error_all_pyr_mean[2:], c='grey')
sl, = ax.plot(xaxis, all_error_pyrinh_mean[2:], c='royalblue')

ax.fill_between(xaxis, all_error_all_pyr_mean[2:]+all_error_all_pyr_sem[2:],
                       all_error_all_pyr_mean[2:]-all_error_all_pyr_sem[2:],
                       color='grey', edgecolor='none', alpha=.2)
ax.fill_between(xaxis, all_error_pyrinh_mean[2:]+all_error_pyrinh_sem[2:],
                       all_error_pyrinh_mean[2:]-all_error_pyrinh_sem[2:],
                       color='royalblue', edgecolor='none', alpha=.2)

ax.legend([cl, sl], ['ctrl.', 'stim.'], frameon=False)

ax.set(xlabel='true time (s)', ylabel='error (s)')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\bayesian_decoding\all_pyr_v_pyrinh\all_pyr_pyrinh_decoding_mean_error.png',
            dpi=300,
            bbox_inches='tight')


#%% statistics
all_error_all_pyr_ab = np.abs(all_error_all_pyr)
all_error_pyrinh_ab = np.abs(all_error_pyrinh)
all_error_shuf_ab = np.abs(all_error_shuf)
all_error_all_pyr_sqz = np.mean(all_error_all_pyr_ab, axis=1)  # 1 dp per session
all_error_pyrinh_sqz = np.mean(all_error_pyrinh_ab, axis=1)
all_error_shuf_sqz = np.mean(all_error_shuf_ab, axis=1)
all_error_all_pyr_sqz_early = np.mean(all_error_all_pyr_ab[:, :13], axis=1)
all_error_pyrinh_sqz_early = np.mean(all_error_pyrinh_ab[:, :13], axis=1)
all_error_shuf_sqz_early = np.mean(all_error_shuf_ab[:, :13], axis=1)
all_error_all_pyr_sqz_late = np.mean(all_error_all_pyr_ab[:, 13:], axis=1)
all_error_pyrinh_sqz_late = np.mean(all_error_pyrinh_ab[:, 13:], axis=1)
all_error_shuf_sqz_late = np.mean(all_error_shuf_ab[:, 13:], axis=1)


#%% plotting 
fig, ax = plt.subplots(figsize=(3.5,2.4))

vp = ax.violinplot([all_error_all_pyr_sqz, all_error_pyrinh_sqz],
                   positions=[1, 2],
                   showmeans=True, showextrema=False)
ax.scatter([1.1]*len(all_error_all_pyr_sqz),
           all_error_all_pyr_sqz, s=10, c='grey', ec='none', lw=.5, alpha=.05)
ax.scatter([1.9]*len(all_error_pyrinh_sqz),
           all_error_pyrinh_sqz, s=10, c='royalblue', ec='none', lw=.5, alpha=.05)
ax.scatter(1.1,
           np.mean(all_error_all_pyr_sqz), s=30, c='grey', ec='none', lw=.5, zorder=2)
ax.scatter(1.9,
           np.mean(all_error_pyrinh_sqz), s=30, c='royalblue', ec='none', lw=.5, zorder=2)
ax.plot([1.1, 1.9], 
        [all_error_all_pyr_sqz, all_error_pyrinh_sqz], 
        color='grey', alpha=.05, linewidth=1, zorder=1)
ax.plot([1.1, 1.9], [np.mean(all_error_all_pyr_sqz), np.mean(all_error_pyrinh_sqz)],
        color='k', linewidth=2, zorder=1)
vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('royalblue')
vp['cmeans'].set_color('k')
vp['cmeans'].set_linewidth(2)
for i in [0, 1]:
    vp['bodies'][i].set_edgecolor('none')
    vp['bodies'][i].set_alpha(.75)
    b = vp['bodies'][i]
    # get the centre 
    m = np.mean(b.get_paths()[0].vertices[:,0])
    # make paths not go further right/left than the centre 
    if i==0:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
    if i==1:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)
y_range = [max(max(all_error_all_pyr_sqz), max(all_error_pyrinh_sqz)), min(min(all_error_all_pyr_sqz), min(all_error_pyrinh_sqz))]
y_range_tot = y_range[0]-y_range[1]
wilc_stat, wilc_p = wilcoxon(all_error_all_pyr_sqz, all_error_pyrinh_sqz)
ttest_stat, ttest_p = ttest_rel(all_error_all_pyr_sqz, all_error_pyrinh_sqz)
ax.plot([1, 2], [y_range[0]+y_range_tot*.15, y_range[0]+y_range_tot*.15], c='k', lw=.5)
ax.text(1.5, y_range[0]+y_range_tot*.15, 'wilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)), 
        ha='center', va='bottom', color='k', fontsize=8)


# early plot 
vpe = ax.violinplot([all_error_all_pyr_sqz_early, all_error_pyrinh_sqz_early],
                    positions=[3, 4],
                    showmeans=True, showextrema=False)
ax.scatter([3.1]*len(all_error_all_pyr_sqz_early),
           all_error_all_pyr_sqz_early, s=10, c='grey', ec='none', lw=.5, alpha=.05)
ax.scatter([3.9]*len(all_error_pyrinh_sqz_early),
           all_error_pyrinh_sqz_early, s=10, c='royalblue', ec='none', lw=.5, alpha=.05)
ax.scatter(3.1,
           np.mean(all_error_all_pyr_sqz_early), s=30, c='grey', ec='none', lw=.5, zorder=2)
ax.scatter(3.9,
           np.mean(all_error_pyrinh_sqz_early), s=30, c='royalblue', ec='none', lw=.5, zorder=2)
ax.plot([3.1, 3.9], 
        [all_error_all_pyr_sqz_early, all_error_pyrinh_sqz_early], 
        color='grey', alpha=.05, linewidth=1, zorder=1)
ax.plot([3.1, 3.9], [np.mean(all_error_all_pyr_sqz_early), np.mean(all_error_pyrinh_sqz_early)],
        color='k', linewidth=2, zorder=1)
vpe['bodies'][0].set_color('grey')
vpe['bodies'][1].set_color('royalblue')
vpe['cmeans'].set_color('k')
vpe['cmeans'].set_linewidth(2)
for i in [0, 1]:
    vpe['bodies'][i].set_edgecolor('none')
    vpe['bodies'][i].set_alpha(.35)
    b = vpe['bodies'][i]
    # get the centre 
    m = np.mean(b.get_paths()[0].vertices[:,0])
    # make paths not go further right/left than the centre 
    if i==0:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
    if i==1:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)
y_range = [max(max(all_error_all_pyr_sqz_early), max(all_error_pyrinh_sqz_early)), min(min(all_error_all_pyr_sqz_early), min(all_error_pyrinh_sqz_early))]
y_range_tot = y_range[0]-y_range[1]
wilc_stat, wilc_p = wilcoxon(all_error_all_pyr_sqz_early, all_error_pyrinh_sqz_early)
ttest_stat, ttest_p = ttest_rel(all_error_all_pyr_sqz_early, all_error_pyrinh_sqz_early)
ax.plot([3, 4], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
ax.text(3.5, y_range[0]+y_range_tot*.05, 'wilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)), 
        ha='center', va='bottom', color='k', fontsize=8)

# late plot 
vpl = ax.violinplot([all_error_all_pyr_sqz_late, all_error_pyrinh_sqz_late],
                    positions=[5,6],
                    showmeans=True, showextrema=False)
ax.scatter([5.1]*len(all_error_all_pyr_sqz_late),
           all_error_all_pyr_sqz_late, s=10, c='grey', ec='none', lw=.5, alpha=.05)
ax.scatter([5.9]*len(all_error_pyrinh_sqz_late),
           all_error_pyrinh_sqz_late, s=10, c='royalblue', ec='none', lw=.5, alpha=.05)
ax.scatter(5.1,
           np.mean(all_error_all_pyr_sqz_late), s=30, c='grey', ec='none', lw=.5, zorder=2)
ax.scatter(5.9,
           np.mean(all_error_pyrinh_sqz_late), s=30, c='royalblue', ec='none', lw=.5, zorder=2)
ax.plot([5.1, 5.9], 
        [all_error_all_pyr_sqz_late, all_error_pyrinh_sqz_late], 
        color='grey', alpha=.05, linewidth=1, zorder=1)
ax.plot([5.1, 5.9], [np.mean(all_error_all_pyr_sqz_late), np.mean(all_error_pyrinh_sqz_late)],
        color='k', linewidth=2, zorder=1)
vpl['bodies'][0].set_color('grey')
vpl['bodies'][1].set_color('royalblue')
vpl['cmeans'].set_color('k')
vpl['cmeans'].set_linewidth(2)
for i in [0, 1]:
    vpl['bodies'][i].set_edgecolor('none')
    vpl['bodies'][i].set_alpha(.5)
    b = vpl['bodies'][i]
    # get the centre 
    m = np.mean(b.get_paths()[0].vertices[:,0])
    # make paths not go further right/left than the centre 
    if i==0:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
    if i==1:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)
y_range = [max(max(all_error_all_pyr_sqz_late), max(all_error_pyrinh_sqz_late)), min(min(all_error_all_pyr_sqz_late), min(all_error_pyrinh_sqz_late))]
y_range_tot = y_range[0]-y_range[1]
wilc_stat, wilc_p = wilcoxon(all_error_all_pyr_sqz_late, all_error_pyrinh_sqz_late)
ttest_stat, ttest_p = ttest_rel(all_error_all_pyr_sqz_late, all_error_pyrinh_sqz_late)
ax.plot([5, 6], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
ax.text(5.5, y_range[0]+y_range_tot*.05, 'wilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)), 
        ha='center', va='bottom', color='k', fontsize=8)

ax.set(xticks=[1,2,3,4,5,6], 
       xticklabels=['all pyr.', 'pyrinh', 'early\nall pyr.', 'early\npyrinh', 'late\nall pyr.', 'late\npyrinh'],
       yticks=[1,2],
       ylabel='decoding error (s)',
       xlim=(.5, 6.5))
for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

# ax.set(title='all pyr. v pyrinh')

fig.tight_layout()

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\bayesian_decoding\all_pyr_v_pyrinh\all_pyr_pyrinh_decoding_mean_error_vp.png',
            dpi=300,
            bbox_inches='tight')