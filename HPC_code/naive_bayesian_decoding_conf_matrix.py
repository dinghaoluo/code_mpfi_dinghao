# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:23:34 2024

Naive Bayesian decoding of hippocampus population

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os
from scipy.stats import wilcoxon, ttest_rel, sem
import scipy.io as sio
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

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

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% function 
def skipping_average(v, skip_step=125):
    """
    v : a numerical vector
    skip_step : length of intervals; default to 100 ms (125 samples, 1250 Hz)
    """
    
    n_int = np.floor_divide(len(v), skip_step)
    v_ret = np.zeros(n_int)
    for i in range(n_int):
        v_ret[i] = np.nanmean(v[i*skip_step:(i+1)*skip_step])
    
    return v_ret


#%% containers
all_error_stim = []
all_error_cont = []

all_error_stim_mean = []
all_error_cont_mean = []
all_error_early_stim_mean = []
all_error_early_cont_mean = []
all_error_late_stim_mean = []
all_error_late_cont_mean = []


#%% loop 
for pathname in pathHPC[:1]:
    recname = pathname[-17:]
    print(recname)
    
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
    For every single trial we want to have 50 time bins, 100 ms each, i.e. 125
    samples each. Therefore we first initialise a container matrix with the 
    dimensions tot_pyr x n time bins. The number of time bins is calculated based
    on the number of trials in either ctrl or stim, times 50 (per trial). After 
    that, y is initialised as basically 100 ms, 200 ms, ... 5000 ms for every 
    single trial, repeated tot_test_trial times, forming a 50xtot_test_trial-long 
    vector.
    """
    y = list(range(-9,51))*tot_test_trial
    y_all = list(range(-9,51))*tot_trial
    
    X_stim = np.zeros((tot_pyr, 60*tot_test_trial))
    X_cont = np.zeros((tot_pyr, 60*tot_test_trial))
    X_all = np.zeros((tot_pyr, 60*tot_trial))
    
    # get stim and cont trial population vectors
    pyr_counter = 0
    for ind, pyr in enumerate(pyr_id):
        if pyr:  # if pyramidal cell
            for trial in range(tot_trial):
                curr_train_pad = np.zeros(1250*6)
                curr_train = trains[ind][trial][2500:]  # from -1 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*6:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*6]
                # downsample curr_train to get our end result
                curr_train_down = skipping_average(curr_train_pad)
                # put this into X
                X_all[pyr_counter, trial*60:(trial+1)*60] = curr_train_down
                
            for i, trial in enumerate(stim_trials):
                curr_train_pad = np.zeros(1250*6)
                curr_train = trains[ind][trial][2500:]  # from 0 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*6:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*6]
                # downsample curr_train to get our end result
                curr_train_down = skipping_average(curr_train_pad)
                # put this into X
                X_stim[pyr_counter, i*60:(i+1)*60] = curr_train_down
            
            for i, trial in enumerate(cont_trials):
                curr_train_pad = np.zeros(1250*6)
                curr_train = trains[ind][trial][2500:]  # from 0 second onwards
                trial_length = len(curr_train)
                if trial_length<1250*6:
                    curr_train_pad[:trial_length] = curr_train
                else:
                    curr_train_pad = curr_train[:1250*6]
                # downsample curr_train to get our end result
                curr_train_down = skipping_average(curr_train_pad)
                # put this into X
                X_cont[pyr_counter, i*60:(i+1)*60] = curr_train_down
                
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
    
    
    # evaluation 
    # y_cont_pred = nb_model_cont.predict(X_cont_test)
    # conf_cont = confusion_matrix(y_cont_test, y_cont_pred)
    # error_cont = np.zeros(50)  # 50 total labels
    # cont_counter = np.zeros(50)
    # for (t, p) in zip(y_cont_test, y_cont_pred):
    #     error_cont[t-1]+=(t-p)
    #     cont_counter[t-1]+=1
    # for i in range(50):
    #     error_cont[i] = error_cont[i]/cont_counter[i]
    
    # y_stim_pred = nb_model_stim.predict(X_stim_test)
    # conf_stim = confusion_matrix(y_stim_test, y_stim_pred)
    # error_stim = np.zeros(50)  # 50 total labels
    # stim_counter = np.zeros(50)
    # for (t, p) in zip(y_stim_test, y_stim_pred):
    #     error_stim[t-1]+=(t-p)
    #     stim_counter[t-1]+=1
    # for i in range(50):
    #     error_stim[i] = error_stim[i]/stim_counter[i]
        
    y_cont_pred = nb_model_all.predict(np.transpose(X_cont))
    conf_cont = confusion_matrix(y, y_cont_pred)
    error_cont = np.zeros(30)  # 30 total labels
    cont_counter = np.zeros(30)
    for (t, p) in zip(y, y_cont_pred):
        if t>5 and t<36:  # 0~4 seconds
            error_cont[t-6]+=(t-p)/10  # divide by 10 to get seconds
            cont_counter[t-6]+=1
    for i in range(30):
        error_cont[i] = (error_cont[i]/cont_counter[i])**2
    
    y_stim_pred = nb_model_all.predict(np.transpose(X_stim))
    conf_stim = confusion_matrix(y, y_stim_pred)
    error_stim = np.zeros(30)
    stim_counter = np.zeros(30)
    for (t, p) in zip(y, y_stim_pred):
        if t>5 and t<36:
            error_stim[t-6]+=(t-p)/10
            stim_counter[t-6]+=1
    for i in range(30):
        error_stim[i] = (error_stim[i]/stim_counter[i])**2
    
    all_error_cont.append(error_cont)
    all_error_stim.append(error_stim)
    
    all_error_stim_mean.append(np.mean(error_stim))
    all_error_cont_mean.append(np.mean(error_cont))
    all_error_early_stim_mean.append(np.mean(error_stim[:15]))
    all_error_early_cont_mean.append(np.mean(error_cont[:15]))
    all_error_late_stim_mean.append(np.mean(error_stim[15:]))
    all_error_late_cont_mean.append(np.mean(error_cont[15:]))
    
    
    # plotting confusion matrices 
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,3))
    
    cf1 = ax1.imshow(conf_cont, aspect='auto', extent=[-1, 5, 5, -1], cmap='Greys')
    cf2 = ax2.imshow(conf_stim, aspect='auto', extent=[-1, 5, 5, -1], cmap='Greys')
    
    plt.colorbar(cf1, shrink=.5)
    plt.colorbar(cf2, shrink=.5)
    
    ax1.set(title='ctrl', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    ax2.set(title='stim', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    
    fig.suptitle('conf. mat.')
    
    fig.savefig('{}\{}'.format(outdir, 'stim_stimcont_decoding_confusion_matrix'),
                dpi=500,
                bbox_inches='tight')
    
    
    # normalising confusion matrices by class 
    class_sum_cont = np.sum(conf_cont, axis=0)
    norm_conf_cont = np.zeros((60, 60))
    for i in range(60):
        for j in range(60):
            norm_conf_cont[j, i] = conf_cont[j, i]/class_sum_cont[i]
        norm_conf_cont[:, i] = normalise(norm_conf_cont[:, i])
            
    class_sum_stim = np.sum(conf_stim, axis=0)
    norm_conf_stim = np.zeros((60, 60))
    for i in range(60):
        for j in range(60):
            norm_conf_stim[j, i] = conf_stim[j, i]/class_sum_stim[i]
        norm_conf_stim[:, i] = normalise(norm_conf_stim[:, i])
            
    
    # plotting confusion matrices (class-normalised)
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,3))
    
    cf1n = ax1.imshow(norm_conf_cont, aspect='auto', extent=[-1, 5, 5, -1])
    cf2n = ax2.imshow(norm_conf_stim, aspect='auto', extent=[-1, 5, 5, -1])
    
    plt.colorbar(cf1n, shrink=.5, ticks=[0,1])
    plt.colorbar(cf2n, shrink=.5, ticks=[0,1])
    
    ax1.set(title='ctrl', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    ax2.set(title='stim', xlabel='true time (s)', ylabel='decoded time (s)',
            xticks=[0,2,4], yticks=[0,2,4])
    
    fig.suptitle('conf. mat. class-norm.')
    
    fig.savefig('{}\{}'.format(outdir, 'stim_stimcont_decoding_confusion_matrix_class_normalised'),
                dpi=500,
                bbox_inches='tight')
    
    
    # plotting error
    fig, ax = plt.subplots(figsize=(3,3))
    
    ctl, = ax.plot(np.linspace(.5, 3.5, 30), error_cont, 'grey')
    stl, = ax.plot(np.linspace(.5, 3.5, 30), error_stim, 'royalblue')
    
    ax.legend([ctl, stl], ['ctrl', 'stim'], frameon=False)
    
    ax.set(xlabel='time (s)', ylabel='sq. error')
    
    fig.savefig('{}\{}'.format(outdir, 'stim_stimcont_decoding_error_temporal'),
                dpi=500,
                bbox_inches='tight')
    
    
    # plotting boxplot and stats 
    fig, ax = plt.subplots(figsize=(2,3))
    
    bp = ax.boxplot([error_cont, error_stim],
                    positions=[.5, 2],
                    patch_artist=True,
                    notch='True')
    
    ax.scatter([.8]*len(error_cont), 
               error_cont, 
               s=10, c='grey', ec='none', lw=.5)
    
    ax.scatter([1.7]*len(error_stim), 
               error_stim, 
               s=10, c='royalblue', ec='none', lw=.5)
    
    ax.set(xticklabels=['ctrl', 'stim'],
           title='wilcoxon p={}\nttest_rel p={}'.format(
               np.round(wilcoxon(error_cont, error_stim)[1],6),
               np.round(ttest_rel(error_cont, error_stim)[1],6)))
    
    colors = ['grey', 'royalblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    bp['fliers'][0].set(marker ='o',
                    color ='#e7298a',
                    markersize=2,
                    alpha=0.5)
    bp['fliers'][1].set(marker ='o',
                    color ='#e7298a',
                    markersize=2,
                    alpha=0.5)
    
    for median in bp['medians']:
        median.set(color='darkred',
                    linewidth=1)
        
    fig.savefig('{}\{}'.format(outdir, 'stim_stimcont_decoding_error_stats'),
                dpi=500,
                bbox_inches='tight')
    

#%% close everything 
plt.close(fig)

    
#%% mean profiles 
fig, ax = plt.subplots(figsize=(3,4))

mean_prof_cont = np.mean(all_error_cont, axis=0)
sem_prof_cont = sem(all_error_cont, axis=0)
mean_prof_stim = np.mean(all_error_stim, axis=0)
sem_prof_stim = sem(all_error_stim, axis=0)
xaxis = np.linspace(.5, 3.5, 30)

acl, = ax.plot(xaxis, mean_prof_cont, 'grey')
asl, = ax.plot(xaxis, mean_prof_stim, 'royalblue')
ax.fill_between(xaxis, mean_prof_cont+sem_prof_cont,
                       mean_prof_cont-sem_prof_cont,
                color='grey', ec='none', alpha=.2)
ax.fill_between(xaxis, mean_prof_stim+sem_prof_stim,
                       mean_prof_stim-sem_prof_stim,
                color='royalblue', ec='none', alpha=.2)
ax.set(xlabel='time (s)', ylabel='sq. error',
       title='mean dec. err. stim-stimcont')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    

ax.legend([acl, asl], ['mean ctrl', 'mean stim'], frameon=False, loc='upper left')

fig.savefig('Z:/Dinghao/code_dinghao/HPC_all/bayesian_decoding/stim_stimcont_decoding_mean_error',
            dpi=500,
            bbox_inches='tight')


#%% violin plot
fig, ax = plt.subplots(figsize=(3,3))

# error plot 
vpe = ax.violinplot([all_error_early_cont_mean, all_error_early_stim_mean],
                    positions=[1,2],
                    showmedians=True, showextrema=False,
                    quantiles=[[0.25, 0.75], [0.25, 0.75]])
ax.scatter([1]*len(all_error_late_cont_mean),
           all_error_late_cont_mean, s=3, c='grey', ec='none', alpha=.5)
ax.scatter([2]*len(all_error_late_stim_mean),
           all_error_late_stim_mean, s=3, c='royalblue', ec='none', alpha=.5)
vpe['bodies'][0].set_color('grey')
vpe['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vpe['bodies'][i].set_edgecolor('none')
    vpe['bodies'][i].set_alpha(.2)
vpe['cmedians'].set(color='darkred')
vpe['cquantiles'].set(color='red', alpha=.25)
ax.set(title='early_wilc p={}\nlate_wilc p={}\ne_l_wilc p={}'.format(
    np.round(wilcoxon(all_error_early_cont_mean, all_error_early_stim_mean)[1], 5),
    np.round(wilcoxon(all_error_late_cont_mean, all_error_late_stim_mean)[1], 5),
    np.round(wilcoxon(all_error_early_cont_mean+all_error_early_stim_mean, 
                      all_error_late_cont_mean+all_error_late_stim_mean)[1],5)))

# late plot 
vpl = ax.violinplot([all_error_late_cont_mean, all_error_late_stim_mean],
                    positions=[3,4],
                    showmedians=True, showextrema=False,
                    quantiles=[[0.25, 0.75], [0.25, 0.75]])
ax.scatter([3]*len(all_error_late_cont_mean),
               all_error_late_cont_mean, s=3, c='grey', ec='none')
ax.scatter([4]*len(all_error_late_stim_mean),
               all_error_late_stim_mean, s=3, c='royalblue', ec='none')
vpl['bodies'][0].set_color('grey')
vpl['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vpl['bodies'][i].set_edgecolor('none')
    vpl['bodies'][i].set_alpha(.5)
vpl['cmedians'].set(color='darkred')
vpl['cquantiles'].set(color='red', alpha=.25)

ax.set(xticks=[1,2,3,4], xticklabels=['early\nctrl', 'early\nstim', 'late\ncont', 'late\nstim'],
       yticks=[0,1,2,3],
       ylabel='sq. error',
       xlim=(.5, 4.5))
for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

fig.tight_layout()

fig.savefig('Z:/Dinghao/code_dinghao/HPC_all/bayesian_decoding/{}'.format(
            'stim_stimcont_decoding_mean_error_vp'),
            dpi=500,
            bbox_inches='tight')