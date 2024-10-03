# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:23:34 2024
Modified on Mon 2 Sept 16:54 2024

Naive Bayesian decoding of hippocampus population

***note on feature scaling***
    I do not want to make any assumption in terms of how much each neurone 
    contribute to the population representation, and therefore do not want to 
    scale the features when training the GNB decoder, since doing so would 
    imply an assumption of uniform contributions.
    
this code has been modified to focus on the pyramidal cells that get activated 
    after run-onset (pyr-act population)

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
# containers
all_error_all_pyr = []
all_error_pyract = []
all_error_other = []
all_error_shuf = []

all_error_all_pyr_ab_mean = []
all_error_pyract_ab_mean = []
all_error_other_ab_mean = []
all_error_shuf_ab_mean = []
all_error_early_all_pyr_mean = []
all_error_early_pyract_mean = []
all_error_early_other_mean = []
all_error_early_shuf_mean = []
all_error_late_all_pyr_mean = []
all_error_late_pyract_mean = []
all_error_late_other_mean = []
all_error_late_shuf_mean = []

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
    tot_pyract = len(pyract)
    tot_other = tot_pyr-len(pyract)
    
    
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
    X_pyract = np.zeros((tot_pyract, tot_bin*tot_trial))
    X_other = np.zeros((tot_other, tot_bin*tot_trial))
    X_all_shuf = np.zeros((tot_pyr, tot_bin*tot_trial))
    
    # get stim and cont trial population vectors
    pyr_counter = 0
    pyract_counter = 0
    other_counter = 0
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
            if clu in pyract:
                for trial in range(tot_trial):
                    curr_train_pad = np.zeros(1250*5)
                    curr_train = trains[clu][trial][3750:]
                    trial_length = len(curr_train)
                    if trial_length<1250*5:
                        curr_train_pad[:trial_length] = curr_train
                    else:
                        curr_train_pad = curr_train[:1250*5]
                    curr_train_down = skipping_average(curr_train_pad)
                    X_pyract[pyract_counter, trial*tot_bin:(trial+1)*tot_bin] = curr_train_down
                pyract_counter+=1
            else:  # all other pyrs
                for trial in range(tot_trial):
                    curr_train_pad = np.zeros(1250*5)
                    curr_train = trains[clu][trial][3750:]
                    trial_length = len(curr_train)
                    if trial_length<1250*5:
                        curr_train_pad[:trial_length] = curr_train
                    else:
                        curr_train_pad = curr_train[:1250*5]
                    curr_train_down = skipping_average(curr_train_pad)
                    X_other[other_counter, trial*tot_bin:(trial+1)*tot_bin] = curr_train_down
                other_counter+=1
            
    # splitting training and testing datasets 
    X_all_pyr_train, X_all_pyr_test, y_all_pyr_train, y_all_pyr_test = train_test_split(np.transpose(X_all_pyr), y, test_size=.2)
    X_pyract_train, X_pyract_test, y_pyract_train, y_pyract_test = train_test_split(np.transpose(X_pyract), y, test_size=.2)
    X_other_train, X_other_test, y_other_train, y_other_test = train_test_split(np.transpose(X_other), y, test_size=.2)
    
    # training 
    nb_model_all_pyr = GaussianNB()
    nb_model_all_pyr.fit(X_all_pyr_train, y_all_pyr_train)
    nb_model_pyract = GaussianNB()
    nb_model_pyract.fit(X_pyract_train, y_pyract_train)
    nb_model_other = GaussianNB()
    nb_model_other.fit(X_other_train, y_other_train)
    
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
    
    y_pyract_pred = nb_model_pyract.predict(X_pyract_test)
    error_pyract = np.zeros(tot_bin)
    error_pyract_abs = np.zeros(tot_bin)
    pyract_counter = np.zeros(tot_bin)
    for (t, p) in zip(y, y_pyract_pred):
        error_pyract[t]+=(t-p)/5
        pyract_counter[t]+=1
    for i in range(tot_bin):
        error_pyract[i] = (error_pyract[i]/pyract_counter[i])
        
    y_other_pred = nb_model_other.predict(X_other_test)
    error_other = np.zeros(tot_bin)
    error_other_abs = np.zeros(tot_bin)
    other_counter = np.zeros(tot_bin)
    for (t, p) in zip(y, y_other_pred):
        error_other[t]+=(t-p)/5
        other_counter[t]+=1
    for i in range(tot_bin):
        error_other[i] = (error_other[i]/other_counter[i])
        
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
    all_error_pyract.append(error_pyract)
    all_error_other.append(error_other)
    all_error_shuf.append(error_shuf)
    
    # plotting (error, single sessions)
    fig, ax = plt.subplots(figsize=(4,4))
    
    xaxis = np.linspace(0.4, 5, 23)
    al, = ax.plot(xaxis, error_all_pyr[:-2], c='darkgreen')
    actl, = ax.plot(xaxis, error_pyract[:-2], c='darkorange')
    othl, = ax.plot(xaxis, error_other[:-2], c='grey')
    shl, = ax.plot(xaxis, error_shuf[:-2], c='red')
    
    ax.legend([al, actl, othl, shl], ['all pyr.', 'pyract', 'other', 'shuf.'], frameon=False)
    ax.set(xlabel='true time (s)', ylabel='error (s)')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\bayesian_decoding\all_pyr_v_pyract\{}.png'.format(recname),
                dpi=200,
                bbox_inches='tight')
    
    # evaluation (proba)
    y_all_pyr_pred_proba = nb_model_all_pyr.predict_proba(X_all_pyr_test)
    y_all_pyr_pred_sum = np.zeros((tot_bin, tot_bin))
    for i, tbin in enumerate(y_all_pyr_test):
        y_all_pyr_pred_sum[tbin,:]+=y_all_pyr_pred_proba[i,:]
    for tbin in range(tot_bin):
        y_all_pyr_pred_sum[tbin,:]/=y_all_pyr_test.count(tbin)
    
    y_pyract_pred_proba = nb_model_pyract.predict_proba(X_pyract_test)
    y_pyract_pred_sum = np.zeros((tot_bin, tot_bin))
    for i, tbin in enumerate(y_pyract_test):
        y_pyract_pred_sum[tbin,:]+=y_pyract_pred_proba[i,:]
    for i in range(tot_bin):
        y_pyract_pred_sum[i]/=y_pyract_test.count(i)
        
    y_other_pred_proba = nb_model_other.predict_proba(X_other_test)
    y_other_pred_sum = np.zeros((tot_bin, tot_bin))
    for i, tbin in enumerate(y_other_test):
        y_other_pred_sum[tbin,:]+=y_other_pred_proba[i,:]
    for i in range(tot_bin):
        y_other_pred_sum[i]/=y_other_test.count(i)
        
    y_shuf_pred_proba = nb_model_all_pyr.predict_proba(np.transpose(X_all_shuf))
    y_shuf_pred_sum = np.zeros((tot_bin, tot_bin))
    for i, tbin in enumerate(y_all_pyr_test):
        y_shuf_pred_sum[tbin,:]+=y_shuf_pred_proba[i,:]
    for tbin in range(tot_bin):
        y_shuf_pred_sum[tbin,:]/=y_all_pyr_test.count(tbin)
    
    # plotting 
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=(12,3))
    
    pim1 = ax1.imshow(y_all_pyr_pred_sum[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    pim2 = ax2.imshow(y_pyract_pred_sum[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    pim3 = ax3.imshow(y_other_pred_sum[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    pim4 = ax4.imshow(y_shuf_pred_sum[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    
    max1 = np.round(np.max(y_all_pyr_pred_sum[:-1, :-1]), 2)
    max2 = np.round(np.max(y_pyract_pred_sum[:-1, :-1]), 2)
    max3 = np.round(np.max(y_other_pred_sum[:-1, :-1]), 2)
    max4 = np.round(np.max(y_shuf_pred_sum[:-1, :-1]), 2)
    cb1 = plt.colorbar(pim1, shrink=.5)
    cb2 = plt.colorbar(pim2, shrink=.5)
    cb3 = plt.colorbar(pim3, shrink=.5)
    cb4 = plt.colorbar(pim4, shrink=.5)
    
    lab1 = np.argmax(y_all_pyr_pred_sum[:-1, :-1], axis=0)/5
    lab2 = np.argmax(y_pyract_pred_sum[:-1, :-1], axis=0)/5
    lab3 = np.argmax(y_other_pred_sum[:-1, :-1], axis=0)/5
    lab4 = np.argmax(y_shuf_pred_sum[:-1, :-1], axis=0)/5
    ax1.plot(np.linspace(0, 4.8, 24), lab1, c='white', lw=1)
    ax2.plot(np.linspace(0, 4.8, 24), lab2, c='white', lw=1)
    ax3.plot(np.linspace(0, 4.8, 24), lab3, c='white', lw=1)
    ax4.plot(np.linspace(0, 4.8, 24), lab4, c='white', lw=1)
    
    ax1.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    ax2.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    ax3.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    ax4.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    
    ax1.set(title='all pyr.', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    ax2.set(title='pyract', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    ax3.set(title='other', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    ax4.set(title='shuf.', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    
    fig.suptitle('post. prob. {}'.format(recname))
    
    fig.tight_layout()
    
    fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\bayesian_decoding\all_pyr_v_pyract\{}_proab.png'.format(recname),
                dpi=200,
                bbox_inches='tight')
    
    print('Complete. Single-session time: {} s.\nStarting next session.'.format(default_timer()-start))

    
#%% close everything 
plt.close(fig)


#%% quantification
all_error_all_pyr_mean = np.mean(all_error_all_pyr, axis=0)
all_error_pyract_mean = np.mean(all_error_pyract, axis=0)
all_error_other_mean = np.mean(all_error_other, axis=0)
all_error_shuf_mean = np.mean(all_error_shuf, axis=0)
all_error_all_pyr_sem = sem(all_error_all_pyr, axis=0)
all_error_pyract_sem = sem(all_error_pyract, axis=0)
all_error_other_sem = sem(all_error_other, axis=0)
all_error_shuf_sem = sem(all_error_shuf, axis=0)


#%% plot errors 
fig, ax = plt.subplots(figsize=(4,3))

xaxis = np.linspace(0.4, 5, 23)

al, = ax.plot(xaxis, all_error_all_pyr_mean[2:], c='darkgreen')
actl, = ax.plot(xaxis, all_error_pyract_mean[2:], c='darkorange')
othl, = ax.plot(xaxis, all_error_other_mean[2:], c='grey')

ax.fill_between(xaxis, all_error_all_pyr_mean[2:]+all_error_all_pyr_sem[2:],
                       all_error_all_pyr_mean[2:]-all_error_all_pyr_sem[2:],
                       color='darkgreen', edgecolor='none', alpha=.2)
ax.fill_between(xaxis, all_error_pyract_mean[2:]+all_error_pyract_sem[2:],
                       all_error_pyract_mean[2:]-all_error_pyract_sem[2:],
                       color='darkorange', edgecolor='none', alpha=.2)
ax.fill_between(xaxis, all_error_other_mean[2:]+all_error_other_sem[2:],
                       all_error_other_mean[2:]-all_error_other_sem[2:],
                       color='grey', edgecolor='none', alpha=.2)

ax.legend([al, actl, othl], ['all pyr.', 'pyract', 'other'], frameon=False)

ax.set(xlabel='true time (s)', ylabel='error (s)')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\bayesian_decoding\all_pyr_v_pyract\all_pyr_pyract_decoding_mean_error.png',
            dpi=300,
            bbox_inches='tight')


#%% statistics
all_error_all_pyr_ab = np.abs(all_error_all_pyr)
all_error_pyract_ab = np.abs(all_error_pyract)
all_error_other_ab = np.abs(all_error_other)
all_error_shuf_ab = np.abs(all_error_shuf)
all_error_all_pyr_sqz = np.mean(all_error_all_pyr_ab, axis=1)  # 1 dp per session
all_error_pyract_sqz = np.mean(all_error_pyract_ab, axis=1)
all_error_other_sqz = np.mean(all_error_other_ab, axis=1)
all_error_shuf_sqz = np.mean(all_error_shuf_ab, axis=1)
all_error_all_pyr_sqz_early = np.mean(all_error_all_pyr_ab[:, :13], axis=1)
all_error_pyract_sqz_early = np.mean(all_error_pyract_ab[:, :13], axis=1)
all_error_other_sqz_early = np.mean(all_error_other_ab[:, :13], axis=1)
all_error_shuf_sqz_early = np.mean(all_error_shuf_ab[:, :13], axis=1)
all_error_all_pyr_sqz_late = np.mean(all_error_all_pyr_ab[:, 13:], axis=1)
all_error_pyract_sqz_late = np.mean(all_error_pyract_ab[:, 13:], axis=1)
all_error_other_sqz_late = np.mean(all_error_other_ab[:, 13:], axis=1)
all_error_shuf_sqz_late = np.mean(all_error_shuf_ab[:, 13:], axis=1)


#%% plotting 
fig, ax = plt.subplots(figsize=(4,2.4))

vp = ax.violinplot([all_error_all_pyr_sqz, all_error_pyract_sqz],
                   positions=[1, 2],
                   showmeans=True, showextrema=False)
ax.scatter([1.1]*len(all_error_all_pyr_sqz),
           all_error_all_pyr_sqz, s=10, c='darkgreen', ec='none', lw=.5, alpha=.05)
ax.scatter([1.9]*len(all_error_pyract_sqz),
           all_error_pyract_sqz, s=10, c='darkorange', ec='none', lw=.5, alpha=.05)
ax.scatter(1.1,
           np.mean(all_error_all_pyr_sqz), s=30, c='darkgreen', ec='none', lw=.5, zorder=2)
ax.scatter(1.9,
           np.mean(all_error_pyract_sqz), s=30, c='darkorange', ec='none', lw=.5, zorder=2)
ax.plot([1.1, 1.9], 
        [all_error_all_pyr_sqz, all_error_pyract_sqz], 
        color='darkgreen', alpha=.05, linewidth=1, zorder=1)
ax.plot([1.1, 1.9], [np.mean(all_error_all_pyr_sqz), np.mean(all_error_pyract_sqz)],
        color='k', linewidth=2, zorder=1)
vp['bodies'][0].set_color('darkgreen')
vp['bodies'][1].set_color('darkorange')
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
y_range = [max(max(all_error_all_pyr_sqz), max(all_error_pyract_sqz)), min(min(all_error_all_pyr_sqz), min(all_error_pyract_sqz))]
y_range_tot = y_range[0]-y_range[1]
wilc_stat, wilc_p = wilcoxon(all_error_all_pyr_sqz, all_error_pyract_sqz)
ttest_stat, ttest_p = ttest_rel(all_error_all_pyr_sqz, all_error_pyract_sqz)
ax.plot([1, 2], [y_range[0]+y_range_tot*.15, y_range[0]+y_range_tot*.15], c='k', lw=.5)
ax.text(1.5, y_range[0]+y_range_tot*.15, 'wilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)), 
        ha='center', va='bottom', color='k', fontsize=8)


# shuf
compvp = ax.violinplot([all_error_shuf_sqz],
                       positions=[3],
                       showmeans=True, showextrema=False)
ax.scatter([3]*len(all_error_shuf_sqz),
           all_error_shuf_sqz, s=10, c='grey', ec='none', lw=.5, alpha=.05)
ax.scatter(3, np.mean(all_error_shuf_sqz), s=30, c='grey', ec='none', lw=.5, zorder=2)
compvp['bodies'][0].set_color('grey')
compvp['bodies'][0].set_edgecolor('none')
compvp['bodies'][0].set_alpha(.75)
compvp['cmeans'].set_color('k')
compvp['cmeans'].set_linewidth(2)


# early plot 
vpe = ax.violinplot([all_error_all_pyr_sqz_early, all_error_pyract_sqz_early],
                    positions=[4, 5],
                    showmeans=True, showextrema=False)
ax.scatter([4.1]*len(all_error_all_pyr_sqz_early),
           all_error_all_pyr_sqz_early, s=10, c='darkgreen', ec='none', lw=.5, alpha=.05)
ax.scatter([4.9]*len(all_error_pyract_sqz_early),
           all_error_pyract_sqz_early, s=10, c='darkorange', ec='none', lw=.5, alpha=.05)
ax.scatter(4.1,
           np.mean(all_error_all_pyr_sqz_early), s=30, c='darkgreen', ec='none', lw=.5, zorder=2)
ax.scatter(4.9,
           np.mean(all_error_pyract_sqz_early), s=30, c='darkorange', ec='none', lw=.5, zorder=2)
ax.plot([4.1, 4.9], 
        [all_error_all_pyr_sqz_early, all_error_pyract_sqz_early], 
        color='darkgreen', alpha=.05, linewidth=1, zorder=1)
ax.plot([4.1, 4.9], [np.mean(all_error_all_pyr_sqz_early), np.mean(all_error_pyract_sqz_early)],
        color='k', linewidth=2, zorder=1)
vpe['bodies'][0].set_color('darkgreen')
vpe['bodies'][1].set_color('darkorange')
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
y_range = [max(max(all_error_all_pyr_sqz_early), max(all_error_pyract_sqz_early)), min(min(all_error_all_pyr_sqz_early), min(all_error_pyract_sqz_early))]
y_range_tot = y_range[0]-y_range[1]
wilc_stat, wilc_p = wilcoxon(all_error_all_pyr_sqz_early, all_error_pyract_sqz_early)
ttest_stat, ttest_p = ttest_rel(all_error_all_pyr_sqz_early, all_error_pyract_sqz_early)
ax.plot([4, 5], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
ax.text(4.5, y_range[0]+y_range_tot*.05, 'wilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)), 
        ha='center', va='bottom', color='k', fontsize=8)

# late plot 
vpl = ax.violinplot([all_error_all_pyr_sqz_late, all_error_pyract_sqz_late],
                    positions=[6,7],
                    showmeans=True, showextrema=False)
ax.scatter([6.1]*len(all_error_all_pyr_sqz_late),
           all_error_all_pyr_sqz_late, s=10, c='darkgreen', ec='none', lw=.5, alpha=.05)
ax.scatter([6.9]*len(all_error_pyract_sqz_late),
           all_error_pyract_sqz_late, s=10, c='darkorange', ec='none', lw=.5, alpha=.05)
ax.scatter(6.1,
           np.mean(all_error_all_pyr_sqz_late), s=30, c='darkgreen', ec='none', lw=.5, zorder=2)
ax.scatter(6.9,
           np.mean(all_error_pyract_sqz_late), s=30, c='darkorange', ec='none', lw=.5, zorder=2)
ax.plot([6.1, 6.9], 
        [all_error_all_pyr_sqz_late, all_error_pyract_sqz_late], 
        color='darkgreen', alpha=.05, linewidth=1, zorder=1)
ax.plot([6.1, 6.9], [np.mean(all_error_all_pyr_sqz_late), np.mean(all_error_pyract_sqz_late)],
        color='k', linewidth=2, zorder=1)
vpl['bodies'][0].set_color('darkgreen')
vpl['bodies'][1].set_color('darkorange')
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
y_range = [max(max(all_error_all_pyr_sqz_late), max(all_error_pyract_sqz_late)), min(min(all_error_all_pyr_sqz_late), min(all_error_pyract_sqz_late))]
y_range_tot = y_range[0]-y_range[1]
wilc_stat, wilc_p = wilcoxon(all_error_all_pyr_sqz_late, all_error_pyract_sqz_late)
ttest_stat, ttest_p = ttest_rel(all_error_all_pyr_sqz_late, all_error_pyract_sqz_late)
ax.plot([6, 6], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
ax.text(6.5, y_range[0]+y_range_tot*.05, 'wilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)), 
        ha='center', va='bottom', color='k', fontsize=8)

ax.set(xticks=[1,2,3,4,5,6,7], 
       xticklabels=['all pyr.', 'pyract', 'shuf.', 'early\nall pyr.', 'early\npyract', 'late\nall pyr.', 'late\npyract'],
       yticks=[1,2],
       ylabel='decoding error (s)',
       xlim=(.5, 7.5))
for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

# ax.set(title='all pyr. v pyract')

fig.tight_layout()

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\bayesian_decoding\all_pyr_v_pyract\all_pyr_pyract_decoding_mean_error_vp.png',
            dpi=300,
            bbox_inches='tight')