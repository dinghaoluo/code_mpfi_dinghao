# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:22:43 2024

correlates run-onset population drifts (single-trial) with behavioural 
    parameters (using probability in a Poisson distribution)

@author: Dinghao Luo
"""


#%% imports 
import sys
import scipy.io as sio 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from math import log 
from scipy.stats import linregress, poisson, zscore


#%% run HPC-LC or HPC-LCterm
HPC_LC = 1

# load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
if HPC_LC:
    pathHPC = rec_list.pathHPCLCopt
elif not HPC_LC:
    pathHPC = rec_list.pathHPCLCtermopt


#%% pre-post ratio dataframe 
df = pd.read_pickle('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.pkl') 


#%% main (single-trial act./inh. proportions)
all_pop_dev_z = []; all_first_lick = []
all_pop_dev_baseline_z = []; all_first_lick_baseline = []
for pathname in pathHPC:
    recname = pathname[-17:]
    if recname=='A063r-20230708-02' or recname=='A063r-20230708-01':  # lick detection problems 
        continue
    print(recname)
    
    rasters = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(recname), 
                           allow_pickle=True).item().values())
    
    # determine if each cell is pyramidal or intern 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    spike_rate = rec_info['firingRate'][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    tot_pyr = sum(pyr_id)
    
    # classify act./inh. neurones 
    pyr_act = []
    for cluname, row in df.iterrows():
        if cluname.split(' ')[0]==recname:
            clu_ID = int(cluname.split(' ')[1][3:])
            # if pp>=1.25:
            #     trial_inh+=1
            if row['ctrl_pre_post']<=.8:
                pyr_act.append(clu_ID-2)  # HPCLC_all_train.py adds 2 to the ID, so we subtracts 2 here 
    
    # behaviour parameters
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2
    tot_trial = len(stimOn)
    tot_baseline = stim_trials[0]

    # for each trial we quantify deviation from Poisson
    fig, axs = plt.subplots(1,2, figsize=(5.5,2.5)); xaxis = np.arange(-2,5)
    pop_deviation = []  # tot_trial-long
    for trial in range(tot_trial):
        single_deviation = np.zeros((len(pyr_act), 7))
        cell_counter = 0
        for pyr in pyr_act:
            curr_raster = rasters[pyr][trial]
            for t in range(7):  # 2.5 seconds before, 4.5 seconds after 
                curr_bin = sum(curr_raster[625+t*1250:625+(t+1)*1250])
                single_deviation[cell_counter, t] = -log(poisson.pmf(curr_bin, spike_rate[pyr]))
            cell_counter+=1
        pop_deviation.append(np.sum(single_deviation, axis=0))
        
        axs[0].plot(xaxis, np.sum(single_deviation, axis=0), lw=1, alpha=.1)
        axs[1].plot(xaxis, np.sum(single_deviation, axis=0), lw=1, alpha=.1)
    
    pop_deviation = np.asarray(pop_deviation)
    axs[0].plot(xaxis, np.mean(pop_deviation, axis=0), lw=2, c='k')
    axs[0].set(title='{}_all'.format(recname),
               xlabel='time (s)',
               ylabel='pop. deviation')
    for s in ['top','right']: axs[0].spines[s].set_visible(False)
    
    pop_deviation_baseline = pop_deviation[:stim_trials[0]]
    axs[1].plot(xaxis, np.mean(pop_deviation_baseline, axis=0), lw=2, c='k')
    axs[1].set(title='{}_baseline'.format(recname),
               xlabel='time (s)',
               ylabel='pop. deviation')
    for s in ['top','right']: axs[1].spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig('Z:\Dinghao\code_dinghao\HPC_all\population_deviation\{}_pyract_only.png'.format(recname),
                dpi=120)
    plt.close(fig)
    
    # behaviour
    lickfilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(recname[1:5], recname[:14], recname[:17], recname[:17])
    alignRun = sio.loadmat(lickfilename)
    
    # import bad beh trial indices
    behPar = sio.loadmat(pathname+pathname[-18:]+
                         '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
    # -1 to account for MATLAB Python difference
    delete = list(np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1)  # bad trials 
    
    # ignore all 1st trials since it is before counting starts and is an empty cell
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
        if len(lk)!=0:  # append only if there is licks in this trial
            first_licks.append((lk[0]-starts[trial])/1250)
        else:
            first_licks.append(0)
    first_licks_baseline = first_licks[:stim_trials[0]]
    
    # filtering 
    for trial in range(tot_trial):
        if first_licks[trial]>8 or first_licks[trial]==0:
            delete.append(trial)
    first_licks = [v for i, v in enumerate(first_licks) if i not in delete]
    first_licks_baseline = [v for i, v in enumerate(first_licks_baseline) if i not in delete]
    pop_deviation = [v for i, v in enumerate(pop_deviation) if i not in delete]
    pop_deviation_z = zscore(pop_deviation, axis=1)
    pop_deviation_baseline = [v for i, v in enumerate(pop_deviation_baseline) if i not in delete]
    pop_deviation_baseline_z = zscore(pop_deviation_baseline, axis=1)
    all_pop_dev_z.extend(pop_deviation_z)
    all_pop_dev_baseline_z.extend(pop_deviation_baseline_z)
    all_first_lick.extend(first_licks)
    all_first_lick_baseline.extend(first_licks_baseline)
    
    # lin-regress
    results = linregress(np.sum(pop_deviation_z[:,2:4], axis=1), first_licks)
    pval = results[3]
    slope = results[0]; intercept = results[1]; rvalue = [2]
    xmin = min(pop_deviation_z[:,2]); xmax = max(pop_deviation_z[:,2])  # for plotting 
    ymin = min(first_licks); ymax = max(first_licks)  # for plotting 
    
    fig, ax = plt.subplots(figsize=(2.2,2.4))
    ax.scatter(np.sum(pop_deviation_z[:,2:4], axis=1), first_licks, c='grey', ec='none', s=5)
    ax.plot([xmin-.1, xmax+.1], [intercept+(xmin-.1)*slope, intercept+(xmax+.1)*slope], 
            color='k', lw=1)
    ax.set(title='{}\npval={}'.format(recname, round(pval, 5)),
           xlabel='pop. deviation (std.)',
           ylabel='t. to 1st-lick (s)')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig('Z:\Dinghao\code_dinghao\HPC_all\population_deviation\{}_v_1st_licks_pyract_only.png'.format(recname),
                dpi=120)
    plt.close()


    results = linregress(np.sum(pop_deviation_baseline_z[:,2:4], axis=1), first_licks_baseline)
    pval = results[3]
    slope = results[0]; intercept = results[1]; rvalue = [2]
    xmin = min(pop_deviation_baseline_z[:,2]); xmax = max(pop_deviation_baseline_z[:,2])  # for plotting 
    ymin = min(first_licks_baseline); ymax = max(first_licks_baseline)  # for plotting 
    
    fig, ax = plt.subplots(figsize=(2.2,2.4))
    ax.scatter(np.sum(pop_deviation_baseline_z[:,2:4], axis=1), first_licks_baseline, c='grey', ec='none', s=5)
    ax.plot([xmin-.1, xmax+.1], [intercept+(xmin-.1)*slope, intercept+(xmax+.1)*slope], 
            color='k', lw=1)
    ax.set(title='{}\npval={}'.format(recname, round(pval, 5)),
           xlabel='pop. deviation (std.)',
           ylabel='t. to 1st-lick (s)')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig('Z:\Dinghao\code_dinghao\HPC_all\population_deviation\{}_v_1st_licks_baseline_pyract_only.png'.format(recname),
                dpi=120)
    plt.close()
    
    
#%% zscore
all_pop_dev_z = zscore(all_pop_dev)
all_first_lick_z = zscore(all_first_lick)

all_pop_dev_baseline_z = zscore(all_pop_dev_baseline)
all_first_lick_baseline_z = zscore(all_first_lick_baseline)


#%% plotting 

