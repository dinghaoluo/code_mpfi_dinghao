# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:23:34 2024

Naive Bayesian decoding of hippocampus population

***note on feature scaling***
    I do not want to make any assumption in terms of how much each neurone 
    contribute to the population representation, and therefore do not want to 
    scale the features when training the GNB decoder, since doing so would 
    imply an assumption of uniform contributions.

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os
import scipy.io as sio
from sklearn.naive_bayes import GaussianNB
from timeit import default_timer
from scipy.stats import ttest_rel, sem
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


#%% containers
all_error_stim = []
all_error_cont = []
all_error_shuf = []

all_error_stim_ab_mean = []
all_error_cont_ab_mean = []
all_error_shuf_ab_mean = []
all_error_early_stim_mean = []
all_error_early_cont_mean = []
all_error_early_shuf_mean = []
all_error_late_stim_mean = []
all_error_late_cont_mean = []
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
    
    # output routes
    outdirroot = r'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
    if not os.path.exists(outdirroot):
        os.makedirs(outdirroot)
    outdir = r'{}\bayesian_decoding'.format(outdirroot)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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
    y = list(range(tot_bin))*tot_test_trial
    y_all = list(range(tot_bin))*tot_trial
    
    X_stim = np.zeros((tot_pyr, tot_bin*tot_test_trial))
    X_cont = np.zeros((tot_pyr, tot_bin*tot_test_trial))
    X_all = np.zeros((tot_pyr, tot_bin*tot_trial))
    X_all_shuf = np.zeros((tot_pyr, tot_bin*tot_trial))
    
    # get stim and cont trial population vectors
    pyr_counter = 0
    for ind, pyr in enumerate(pyr_id):
        if pyr:  # if pyramidal cell
            for trial in range(tot_trial):
                curr_train_pad = np.zeros(1250*5)
                curr_train = trains[ind][trial][3750:]  # from 0 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*5:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*5]
                # downsample curr_train to get our end result
                curr_train_down = skipping_average(curr_train_pad)
                # put this into X
                X_all[pyr_counter, trial*tot_bin:(trial+1)*tot_bin] = curr_train_down
                
                # shuffling, new, Dinghao, 17 Apr 24
                X_all_shuf[pyr_counter, trial*tot_bin:(trial+1)*tot_bin] = shuffle_mean(curr_train_down, n_shuf=n_shuf)
                
            for i, trial in enumerate(stim_trials):
                curr_train_pad = np.zeros(1250*5)
                curr_train = trains[ind][trial][3750:]  # from 0 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*5:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*5]
                # downsample curr_train to get our end result
                curr_train_down = skipping_average(curr_train_pad)
                # put this into X
                X_stim[pyr_counter, i*tot_bin:(i+1)*tot_bin] = curr_train_down
            
            for i, trial in enumerate(cont_trials):
                curr_train_pad = np.zeros(1250*5)
                curr_train = trains[ind][trial][3750:]  # from 0 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*5:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*5]
                # downsample curr_train to get our end result
                curr_train_down = skipping_average(curr_train_pad)
                # put this into X
                X_cont[pyr_counter, i*tot_bin:(i+1)*tot_bin] = curr_train_down
                
            pyr_counter+=1
            
    # splitting training and testing datasets 
    # X_cont_train, X_cont_test, y_cont_train, y_cont_test = train_test_split(np.transpose(X_cont), y, test_size=.3)
    # X_stim_train, X_stim_test, y_stim_train, y_stim_test = train_test_split(np.transpose(X_stim), y, test_size=.3)
    
    
    # Bayesian decoder training 
    # nb_model_cont = GaussianNB()
    # nb_model_cont.fit(X_cont_train, y_cont_train)
    
    # nb_model_stim = GaussianNB()
    # nb_model_stim.fit(X_stim_train, y_stim_train)
    
    nb_model_all = GaussianNB()
    nb_model_all.fit(np.transpose(X_all), y_all)
    
    # evaluation (labelled)
    y_cont_pred = nb_model_all.predict(np.transpose(X_cont))
    error_cont = np.zeros(tot_bin)  # error
    error_cont_ab = np.zeros(tot_bin)  # absoluate error
    cont_counter = np.zeros(tot_bin)
    for (t, p) in zip(y, y_cont_pred):
        error_cont[t]+=(t-p)/5  # divide by 10 to get seconds
        cont_counter[t]+=1
    for i in range(tot_bin):
        error_cont[i] = (error_cont[i]/cont_counter[i])
    
    y_stim_pred = nb_model_all.predict(np.transpose(X_stim))
    error_stim = np.zeros(tot_bin)
    error_stim_ab = np.zeros(tot_bin)
    stim_counter = np.zeros(tot_bin)
    for (t, p) in zip(y, y_stim_pred):
        error_stim[t]+=(t-p)/5
        stim_counter[t]+=1
    for i in range(tot_bin):
        error_stim[i] = (error_stim[i]/stim_counter[i])
        
    y_shuf_pred = nb_model_all.predict(np.transpose(X_all_shuf))
    error_shuf = np.zeros(tot_bin)
    error_shuf_ab = np.zeros(tot_bin)
    shuf_counter = np.zeros(tot_bin)
    for (t, p) in zip(y_all, y_shuf_pred):
        error_shuf[t]+=(t-p)/5
        shuf_counter[t]+=1
    for i in range(tot_bin):
        error_shuf[i] = (error_shuf[i]/shuf_counter[i])
    
    all_error_cont.append(error_cont)
    all_error_stim.append(error_stim)
    all_error_shuf.append(error_shuf)
    
    # plotting (error, single sessions)
    fig, ax = plt.subplots(figsize=(4,4))
    
    xaxis = np.linspace(0.4, 5, 23)
    cl, = ax.plot(xaxis, error_cont[:-2], c='grey')
    sl, = ax.plot(xaxis, error_stim[:-2], c='royalblue')
    shl, = ax.plot(xaxis, error_shuf[:-2], c='red')
    
    ax.legend([cl, sl, shl], ['ctrl.', 'stim.', 'shuf.'], frameon=False)
    ax.set(xlabel='true time (s)', ylabel='error (s)')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.savefig('{}\{}'.format(outdir, 'stim_stimcont_decoding_error_temporal.png'),
                dpi=500,
                bbox_inches='tight')
    
    # evaluation (proba)
    y_cont_pred_proba = nb_model_all.predict_proba(np.transpose(X_cont))
    y_cont_proba_trial = y_cont_pred_proba[:tot_bin, :]  # first trial
    y_cont_proba_trial_means = np.zeros((tot_bin, tot_bin))
    for i in range(1, tot_test_trial):
        for j in range(tot_bin):
            y_cont_proba_trial[j, :]+=y_cont_pred_proba[tot_bin*i+j, :]
    for i in range(tot_bin):
        y_cont_proba_trial_means[i,:] = y_cont_proba_trial[i,:]/tot_bin
    
    y_stim_pred_proba = nb_model_all.predict_proba(np.transpose(X_stim))
    y_stim_proba_trial = y_stim_pred_proba[:tot_bin, :]  # first trial
    y_stim_proba_trial_means = np.zeros((tot_bin, tot_bin))
    for i in range(1, tot_test_trial):
        for j in range(tot_bin):
            y_stim_proba_trial[j, :]+=y_stim_pred_proba[tot_bin*i+j, :]
    for i in range(tot_bin):
        y_stim_proba_trial_means[i,:] = y_stim_proba_trial[i,:]/tot_bin
        
    y_shuf_pred_proba = nb_model_all.predict_proba(np.transpose(X_all_shuf))
    y_shuf_proba_trial = y_shuf_pred_proba[:tot_bin, :]  # first trial
    y_shuf_proba_trial_means = np.zeros((tot_bin, tot_bin))
    for i in range(1, tot_test_trial):
        for j in range(tot_bin):
            y_shuf_proba_trial[j, :]+=y_shuf_pred_proba[tot_bin*i+j, :]
    for i in range(tot_bin):
        y_shuf_proba_trial_means[i,:] = y_shuf_proba_trial[i,:]/tot_bin
    
    # plotting 
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(9,3))
    
    pim1 = ax1.imshow(y_cont_proba_trial_means[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    pim2 = ax2.imshow(y_stim_proba_trial_means[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    pim3 = ax3.imshow(y_shuf_proba_trial_means[:-1, :-1], aspect='auto', extent=[0, 4.8, 4.8, 0])
    
    max1 = np.round(np.max(y_cont_proba_trial_means[:-1, :-1]), 2)
    max2 = np.round(np.max(y_stim_proba_trial_means[:-1, :-1]), 2)
    max3 = np.round(np.max(y_shuf_proba_trial_means[:-1, :-1]), 2)
    cb1 = plt.colorbar(pim1, shrink=.5, ticks=[0,max1])
    cb2 = plt.colorbar(pim2, shrink=.5, ticks=[0,max2])
    cb3 = plt.colorbar(pim3, shrink=.5, ticks=[0,max3])
    
    lab1 = np.argmax(y_cont_proba_trial_means[:-1, :-1], axis=0)/5
    lab2 = np.argmax(y_stim_proba_trial_means[:-1, :-1], axis=0)/5
    lab3 = np.argmax(y_shuf_proba_trial_means[:-1, :-1], axis=0)/5
    ax1.plot(np.linspace(0, 4.8, 24), lab1, c='white', lw=1)
    ax2.plot(np.linspace(0, 4.8, 24), lab2, c='white', lw=1)
    ax3.plot(np.linspace(0, 4.8, 24), lab3, c='white', lw=1)
    
    ax1.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    ax2.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    ax3.plot(np.linspace(0, 4.8, 24), np.linspace(0, 4.8, 24), c='k', ls='dashed', lw=1, alpha=.5)
    
    ax1.set(title='ctrl.', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    ax2.set(title='stim.', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    ax3.set(title='shuf.', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    
    fig.suptitle('post. prob. {}'.format(recname))
    
    fig.tight_layout()
    
    fig.savefig('{}\{}'.format(outdir, 'stim_stimcont_decoding_post_proba.png'),
                dpi=500,
                bbox_inches='tight')
    
    print('Complete. Single-session time: {} s.\nStarting next session.'.format(default_timer()-start))

    
#%% close everything 
plt.close(fig)


#%% quantification
all_error_cont_mean = np.mean(all_error_cont, axis=0)
all_error_stim_mean = np.mean(all_error_stim, axis=0)
all_error_shuf_mean = np.mean(all_error_shuf, axis=0)
all_error_cont_sem = sem(all_error_cont, axis=0)
all_error_stim_sem = sem(all_error_stim, axis=0)
all_error_shuf_sem = sem(all_error_shuf, axis=0)


#%% plot errors 
fig, ax = plt.subplots(figsize=(4,4))

xaxis = np.linspace(0.4, 5, 23)

cl, = ax.plot(xaxis, all_error_cont_mean[2:], c='grey')
sl, = ax.plot(xaxis, all_error_stim_mean[2:], c='royalblue')
shl, = ax.plot(xaxis, all_error_shuf_mean[2:], c='red')

ax.fill_between(xaxis, all_error_cont_mean[2:]+all_error_cont_sem[2:],
                       all_error_cont_mean[2:]-all_error_cont_sem[2:],
                       color='grey', edgecolor='none', alpha=.2)
ax.fill_between(xaxis, all_error_stim_mean[2:]+all_error_stim_sem[2:],
                       all_error_stim_mean[2:]-all_error_stim_sem[2:],
                       color='royalblue', edgecolor='none', alpha=.2)
ax.fill_between(xaxis, all_error_shuf_mean[2:]+all_error_shuf_sem[2:],
                       all_error_shuf_mean[2:]-all_error_shuf_sem[2:],
                       color='red', edgecolor='none', alpha=.2)

ax.legend([cl, sl, shl], ['ctrl.', 'stim.', 'shuf.'], frameon=False)

ax.set(xlabel='true time (s)', ylabel='error (s)')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.savefig('Z:/Dinghao/code_dinghao/HPC_all/bayesian_decoding/stim_stimcont_decoding_mean_error',
            dpi=500,
            bbox_inches='tight')


#%% statistics
all_error_cont_ab = np.abs(all_error_cont)
all_error_stim_ab = np.abs(all_error_stim)
all_error_shuf_ab = np.abs(all_error_shuf)
all_error_cont_sqz = np.mean(all_error_cont_ab, axis=1)  # 1 dp per session
all_error_stim_sqz = np.mean(all_error_stim_ab, axis=1)
all_error_shuf_sqz = np.mean(all_error_shuf_ab, axis=1)
all_error_cont_sqz_early = np.mean(all_error_cont_ab[:, :13], axis=1)
all_error_stim_sqz_early = np.mean(all_error_stim_ab[:, :13], axis=1)
all_error_shuf_sqz_early = np.mean(all_error_shuf_ab[:, :13], axis=1)
all_error_cont_sqz_late = np.mean(all_error_cont_ab[:, 13:], axis=1)
all_error_stim_sqz_late = np.mean(all_error_stim_ab[:, 13:], axis=1)
all_error_shuf_sqz_late = np.mean(all_error_shuf_ab[:, 13:], axis=1)

results = ttest_rel(all_error_cont_sqz, all_error_stim_sqz)
results_shuf = ttest_rel((all_error_cont_sqz+all_error_stim_sqz)/2, all_error_shuf_sqz)
results_early = ttest_rel(all_error_cont_sqz_early, all_error_stim_sqz_early)
results_late = ttest_rel(all_error_cont_sqz_late, all_error_stim_sqz_late)


#%% plotting 
fig, ax = plt.subplots(figsize=(5,3))

vp = ax.violinplot([all_error_cont_sqz, all_error_stim_sqz, all_error_shuf_sqz],
                   positions=[1, 2, 3],
                   showmedians=True, showextrema=False,
                   quantiles=[[0.25, 0.75], [0.25, 0.75], [0.25, 0.75]])
jit1 = np.random.uniform(-.04, .04, len(all_error_cont_sqz))
jit2 = np.random.uniform(-.04, .04, len(all_error_stim_sqz))
jit3 = np.random.uniform(-.04, .04, len(all_error_shuf_sqz))
ax.scatter([1]*len(all_error_cont_sqz)+jit1,
           all_error_cont_sqz, s=3, c='grey', ec='none', alpha=.5)
ax.scatter([2]*len(all_error_stim_sqz)+jit2,
           all_error_stim_sqz, s=3, c='royalblue', ec='none', alpha=.5)
ax.scatter([3]*len(all_error_shuf_sqz)+jit3,
           all_error_shuf_sqz, s=3, c='red', ec='none', alpha=.5)
vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('royalblue')
vp['bodies'][2].set_color('red')
for i in [0, 1, 2]:
    vp['bodies'][i].set_edgecolor('none')
    vp['bodies'][i].set_alpha(.2)
vp['cmedians'].set(color='darkred')
vp['cquantiles'].set(color='red', alpha=.25)

# early plot 
vpe = ax.violinplot([all_error_cont_sqz_early, all_error_stim_sqz_early],
                    positions=[4, 5],
                    showmedians=True, showextrema=False,
                    quantiles=[[0.25, 0.75], [0.25, 0.75]])
jit_e1 = np.random.uniform(-.04, .04, len(all_error_cont_sqz_early))
jit_e2 = np.random.uniform(-.04, .04, len(all_error_stim_sqz_early))
ax.scatter([4]*len(all_error_cont_sqz)+jit_e1,
           all_error_cont_sqz, s=3, c='grey', ec='none', alpha=.5)
ax.scatter([5]*len(all_error_stim_sqz)+jit_e2,
           all_error_stim_sqz, s=3, c='royalblue', ec='none', alpha=.5)
vpe['bodies'][0].set_color('grey')
vpe['bodies'][1].set_color('royalblue')
for i in [0, 1]:
    vpe['bodies'][i].set_edgecolor('none')
    vpe['bodies'][i].set_alpha(.2)
vpe['cmedians'].set(color='darkred')
vpe['cquantiles'].set(color='red', alpha=.25)

# late plot 
vpl = ax.violinplot([all_error_cont_sqz_late, all_error_stim_sqz_late],
                    positions=[6,7],
                    showmedians=True, showextrema=False,
                    quantiles=[[0.25, 0.75], [0.25, 0.75]])
jit_l1 = np.random.uniform(-.04, .04, len(all_error_cont_sqz_late))
jit_l2 = np.random.uniform(-.04, .04, len(all_error_stim_sqz_late))
ax.scatter([6]*len(all_error_cont_sqz_late)+jit_l1,
           all_error_cont_sqz_late, s=3, c='grey', ec='none', alpha=.5)
ax.scatter([7]*len(all_error_stim_sqz_late)+jit_l2,
           all_error_stim_sqz_late, s=3, c='royalblue', ec='none', alpha=.5)
vpl['bodies'][0].set_color('grey')
vpl['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vpl['bodies'][i].set_edgecolor('none')
    vpl['bodies'][i].set_alpha(.4)
vpl['cmedians'].set(color='darkred')
vpl['cquantiles'].set(color='red', alpha=.25)

ax.set(xticks=[1,2,3,4,5,6,7], 
       xticklabels=['ctrl.', 'stim.', 'shuf.', 'early\nctrl.', 'early\nstim.', 'late\nctrl.', 'late\nstim.'],
       yticks=[0,1,2],
       ylabel='error (s)',
       xlim=(.5, 7.5))
for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(title='early_t p={}\nlate_t p={}\ne_l_t p={}\nshuf p={}'.format(
       results_early[1], results_late[1], results[1], results_shuf[1]))

fig.tight_layout()

fig.savefig('Z:/Dinghao/code_dinghao/HPC_all/bayesian_decoding/{}'.format(
            'stim_stimcont_decoding_mean_error_vp'),
            dpi=500,
            bbox_inches='tight')