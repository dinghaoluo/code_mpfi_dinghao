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
from scipy.stats import linregress, poisson, zscore, sem

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if (r'Z:\Dinghao\code_mpfi_dinghao\iutils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf


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


#%% parameter for binning 
tot_bins = 16


#%% pre-post ratio dataframe 
df = pd.read_pickle('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.pkl') 


#%% main (single-trial act./inh. proportions)
all_pop_dev_z = {}; all_first_lick_z = {}
all_pop_dev_baseline_z = {}; all_first_lick_baseline_z = {}

all_pyract_pop_dev_z = {}
all_pyract_pop_dev_baseline_z = {}
all_pyract_pop_dev_profile = {}
all_pyract_pop_dev_baseline_profile = {}

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
    tot_clu = len(pyr_id)
    tot_pyr = sum(pyr_id)
    
    # classify act./inh. neurones 
    pyr_act = []; pyr_inh = []
    for cluname, row in df.iterrows():
        if cluname.split(' ')[0]==recname:
            clu_ID = int(cluname.split(' ')[1][3:])
            if row['ctrl_pre_post']>=1.25:
                pyr_inh.append(clu_ID-2)
            if row['ctrl_pre_post']<=.8:
                pyr_act.append(clu_ID-2)  # HPCLC_all_train.py adds 2 to the ID, so we subtracts 2 here 
    
    # behaviour parameters
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    tot_trial = len(stimOn)
    tot_baseline = stim_trials[0]

    # for each trial we quantify dev from Poisson
    fig, axs = plt.subplots(1,2, figsize=(5.5,2.5)); xaxis = np.linspace(-2.5,5.5,tot_bins+1)[:-1]
    pop_dev = []  # tot_trial-long
    pol_pop_dev = []
    for trial in range(tot_trial):
        single_dev = np.zeros((tot_pyr, tot_bins))
        pol_single_dev = np.zeros((tot_pyr, tot_bins))
        cell_counter = 0
        for clu in range(tot_clu):
            if pyr_id[clu]==True:
                curr_raster = rasters[clu][trial]
                for tbin, t in enumerate(np.linspace(0,8,tot_bins+1)[:-1]):  # 3 seconds before, 5 seconds after 
                    curr_bin = sum(curr_raster[int(t*1250):int((t+1)*1250)])
                    single_dev[cell_counter, tbin] = -log(poisson.pmf(curr_bin, spike_rate[clu]))
                    coeff=1
                    if curr_bin<spike_rate[clu]: coeff=-1  # this is adding in polarity 
                    pol_single_dev[cell_counter, tbin] = -log(poisson.pmf(curr_bin, spike_rate[clu]))*coeff
                cell_counter+=1
        pop_dev.append(np.sum(single_dev, axis=0))
        
        pol_pop_dev.append(np.mean(pol_single_dev, axis=0))
        axs[0].plot(xaxis, np.mean(pol_single_dev, axis=0), lw=1, alpha=.1)
        if trial<stim_trials[0]:
            axs[1].plot(xaxis, np.mean(pol_single_dev, axis=0), lw=1, alpha=.1)
    axs[0].set(title='{}_all'.format(recname)); axs[1].set(title='{}_baseline'.format(recname))
    for i in [0,1]:
        axs[i].set(xlabel='time (s)', xlim=(-1,4),
                   ylabel='pop. dev.')
        for s in ['top','right']: axs[i].spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\all\{}_single_trial.png'.format(recname),
                dpi=300)
    plt.close(fig)
    
    pop_dev = np.asarray(pop_dev)
    mean_pop_dev = np.mean(pop_dev, axis=0)
    sem_pop_dev = sem(pop_dev, axis=0)
    
    fig, axs = plt.subplots(1,2, figsize=(5.5,2.5))
    pol_pop_dev = np.asarray(pol_pop_dev)
    mean_pol_pop_dev = np.mean(pol_pop_dev, axis=0)
    sem_pol_pop_dev = sem(pol_pop_dev, axis=0)
    scale_min, scale_max = pf.scale_min_max(mean_pol_pop_dev[3:14], sem_pol_pop_dev[3:14])  # find min and max for plotting purposes
    axs[0].plot(xaxis, mean_pol_pop_dev, lw=2, c='k')
    axs[0].fill_between(xaxis, mean_pol_pop_dev+sem_pol_pop_dev,
                               mean_pol_pop_dev-sem_pol_pop_dev,
                        color='k', edgecolor='none', alpha=.1)
    axs[0].set(title='{}_all'.format(recname),
               xlabel='time (s)', xlim=(-1,4),
               ylabel='pop. dev', ylim=(scale_min, scale_max))
    for s in ['top','right']: axs[0].spines[s].set_visible(False)
    
    pop_dev_baseline = pop_dev[:stim_trials[0],:]
    mean_pop_dev_baseline = np.mean(pop_dev_baseline, axis=0)
    sem_pop_dev_baseline = sem(pop_dev_baseline, axis=0)
    
    pol_pop_dev_baseline = pol_pop_dev[:stim_trials[0],:]
    mean_pol_pop_dev_baseline = np.mean(pol_pop_dev_baseline, axis=0)
    sem_pol_pop_dev_baseline = sem(pol_pop_dev_baseline, axis=0)
    scale_min, scale_max = pf.scale_min_max(mean_pol_pop_dev_baseline[3:14], sem_pop_dev_baseline[3:14])
    axs[1].plot(xaxis, mean_pol_pop_dev_baseline, lw=2, c='k')
    axs[1].fill_between(xaxis, mean_pol_pop_dev_baseline+sem_pop_dev_baseline,
                               mean_pol_pop_dev_baseline-sem_pop_dev_baseline,
                        color='k', edgecolor='none', alpha=.1)
    axs[1].set(title='{}_baseline'.format(recname),
               xlabel='time (s)', xlim=(-1,4),
               ylabel='pop. dev', ylim=(scale_min, scale_max))
    for s in ['top','right']: axs[1].spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\all\{}.png'.format(recname),
                dpi=300)
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
        if first_licks[trial]>10 or first_licks[trial]<=1.5:
            delete.append(trial)
    first_licks = [v for i, v in enumerate(first_licks) if i not in delete]
    first_licks_baseline = [v for i, v in enumerate(first_licks_baseline) if i not in delete]
    pop_dev = [v for i, v in enumerate(pop_dev) if i not in delete]
    pop_dev_z = zscore(pop_dev)
    pop_dev_baseline = [v for i, v in enumerate(pop_dev_baseline) if i not in delete]
    pop_dev_baseline_z = zscore(pop_dev_baseline)
    all_pop_dev_z[recname] = np.sum(pop_dev_z[:, 6:10],axis=1)
    all_pop_dev_baseline_z[recname] = np.sum(pop_dev_baseline_z[:, 6:10],axis=1)
    all_first_lick_z[recname] = zscore(first_licks)
    all_first_lick_baseline_z[recname] = zscore(first_licks_baseline)
    
    # lin-regress
    results = linregress(np.sum(pop_dev_z[:, 6:10],axis=1), first_licks)
    pval = results[3]
    slope = results[0]; intercept = results[1]; rvalue = [2]
    xmin = min(np.sum(pop_dev_z[:, 6:10],axis=1)); xmax = max(np.sum(pop_dev_z[:, 6:10],axis=1))  # for plotting 
    ymin = min(first_licks); ymax = max(first_licks)  # for plotting 
    
    fig, ax = plt.subplots(figsize=(2.2,2.4))
    ax.scatter(np.sum(pop_dev_z[:, 6:10],axis=1), first_licks, c='grey', ec='none', s=5)
    ax.plot([xmin-.1, xmax+.1], [intercept+(xmin-.1)*slope, intercept+(xmax+.1)*slope], 
            color='k', lw=1)
    ax.set(title='{}\npval={}'.format(recname, round(pval, 5)),
           xlabel='pop. dev (std.)',
           ylabel='t. to 1st-lick (s)')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\all\{}_v_1st_licks.png'.format(recname),
                dpi=300)
    plt.close()

    results = linregress(np.sum(pop_dev_baseline_z[:, 6:10],axis=1), first_licks_baseline)
    pval = results[3]
    slope = results[0]; intercept = results[1]; rvalue = [2]
    xmin = min(np.sum(pop_dev_baseline_z[:, 6:10],axis=1)); xmax = max(np.sum(pop_dev_baseline_z[:, 6:10],axis=1))  # for plotting 
    ymin = min(first_licks_baseline); ymax = max(first_licks_baseline)  # for plotting 
    
    fig, ax = plt.subplots(figsize=(2.2,2.4))
    ax.scatter(np.sum(pop_dev_baseline_z[:, 6:10],axis=1), first_licks_baseline, c='grey', ec='none', s=5)
    ax.plot([xmin-.1, xmax+.1], [intercept+(xmin-.1)*slope, intercept+(xmax+.1)*slope], 
            color='k', lw=1)
    ax.set(title='{}\npval={}'.format(recname, round(pval, 5)),
           xlabel='pop. dev (std.)',
           ylabel='t. to 1st-lick (s)')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\all\{}_v_1st_licks_baseline.png'.format(recname),
                dpi=300)
    plt.close()
    
    
    # pyract_only
    fig, axs = plt.subplots(1,2, figsize=(5.5,2.5))
    pyract_pop_dev = []  # tot_trial-long
    pol_pyract_pop_dev = []
    for trial in range(tot_trial):
        single_dev = np.zeros((len(pyr_act), tot_bins))
        pol_single_dev = np.zeros((len(pyr_act), tot_bins))
        cell_counter = 0
        for pyr in pyr_act:
            curr_raster = rasters[pyr][trial]
            for tbin, t in enumerate(np.linspace(0,8,tot_bins+1)[:-1]):  # 3 seconds before, 5 seconds after 
                curr_bin = sum(curr_raster[int(t*1250):int((t+1)*1250)])
                single_dev[cell_counter, tbin] = -log(poisson.pmf(curr_bin, spike_rate[pyr]))
                coeff=1
                if curr_bin<spike_rate[pyr]: coeff=-1
                pol_single_dev[cell_counter, tbin] = -log(poisson.pmf(curr_bin, spike_rate[pyr]))*coeff
            cell_counter+=1
        pyract_pop_dev.append(np.sum(single_dev, axis=0))
        
        pol_pyract_pop_dev.append(np.mean(pol_single_dev, axis=0))
        axs[0].plot(xaxis, np.mean(pol_single_dev, axis=0), lw=1, alpha=.1)
        if trial<stim_trials[0]:
            axs[1].plot(xaxis, np.mean(pol_single_dev, axis=0), lw=1, alpha=.1)
    axs[0].set(title='{}_all'.format(recname)); axs[1].set(title='{}_baseline'.format(recname))
    for i in [0,1]:
        axs[i].set(xlabel='time (s)', xlim=(-1,4),
                   ylabel='pop. dev.')
        for s in ['top','right']: axs[i].spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\{}_single_trial.png'.format(recname),
                dpi=120)
    plt.close(fig)
    
    pyract_pop_dev = np.asarray(pyract_pop_dev)
    mean_pyract_pop_dev = np.mean(pyract_pop_dev, axis=0)
    sem_pyract_pop_dev = sem(pyract_pop_dev, axis=0)
    
    fig, axs = plt.subplots(1,2, figsize=(5.5,2.5))
    pol_pyract_pop_dev = np.asarray(pol_pyract_pop_dev)
    mean_pol_pyract_pop_dev = np.mean(pol_pyract_pop_dev, axis=0)
    sem_pol_pyract_pop_dev = sem(pol_pyract_pop_dev, axis=0)
    scale_min, scale_max = pf.scale_min_max(mean_pol_pyract_pop_dev[3:14], sem_pol_pyract_pop_dev[3:14])
    axs[0].plot(xaxis, mean_pol_pyract_pop_dev, lw=2, c='k')
    axs[0].fill_between(xaxis, mean_pol_pyract_pop_dev+sem_pol_pyract_pop_dev,
                               mean_pol_pyract_pop_dev-sem_pol_pyract_pop_dev,
                        color='k', edgecolor='none', alpha=.1)
    axs[0].set(title='{}_pyract'.format(recname),
               xlabel='time (s)', xlim=(-1,4),
               ylabel='pop. dev', ylim=(scale_min, scale_max))
    for s in ['top','right']: axs[0].spines[s].set_visible(False)
    
    pyract_pop_dev_baseline = pyract_pop_dev[:stim_trials[0],:]
    mean_pyract_pop_dev_baseline = np.mean(pyract_pop_dev_baseline, axis=0)
    sem_pyract_pop_dev_baseline = sem(pyract_pop_dev_baseline, axis=0)
    
    pol_pyract_pop_dev_baseline = pol_pyract_pop_dev[:stim_trials[0],:]
    mean_pol_pyract_pop_dev_baseline = np.mean(pol_pyract_pop_dev_baseline, axis=0)
    sem_pol_pyract_pop_dev_baseline = sem(pol_pyract_pop_dev_baseline, axis=0)
    scale_min, scale_max = pf.scale_min_max(mean_pol_pyract_pop_dev_baseline[3:14], sem_pol_pyract_pop_dev_baseline[3:14])
    axs[1].plot(xaxis, mean_pol_pyract_pop_dev_baseline, lw=2, c='k')
    axs[1].fill_between(xaxis, mean_pol_pyract_pop_dev_baseline+sem_pol_pyract_pop_dev_baseline,
                               mean_pol_pyract_pop_dev_baseline-sem_pol_pyract_pop_dev_baseline,
                        color='k', edgecolor='none', alpha=.1)
    axs[1].set(title='{}_baseline_pyract'.format(recname),
               xlabel='time (s)', xlim=(-1,4),
               ylabel='pop. dev', ylim=(scale_min, scale_max))
    for s in ['top','right']: axs[1].spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\{}.png'.format(recname),
                dpi=300)
    plt.close(fig)
    
    # filtering 
    pyract_pop_dev = [v for i, v in enumerate(pyract_pop_dev) if i not in delete]
    pyract_pop_dev_z = zscore(pyract_pop_dev)
    pyract_pop_dev_baseline = [v for i, v in enumerate(pyract_pop_dev_baseline) if i not in delete]
    pyract_pop_dev_baseline_z = zscore(pyract_pop_dev_baseline)
    all_pyract_pop_dev_z[recname] = np.sum(pyract_pop_dev_z[:, 6:10],axis=1)
    all_pyract_pop_dev_baseline_z[recname] = np.sum(pyract_pop_dev_baseline_z[:, 6:10],axis=1)
    all_pyract_pop_dev_profile[recname] = pyract_pop_dev_z
    all_pyract_pop_dev_baseline_profile[recname] = pyract_pop_dev_baseline_z
    
    # lin-regress
    results = linregress(np.sum(pyract_pop_dev_z[:, 6:10],axis=1), first_licks)
    pval = results[3]
    slope = results[0]; intercept = results[1]; rvalue = [2]
    xmin = min(np.sum(pyract_pop_dev_z[:, 6:10],axis=1)); xmax = max(np.sum(pyract_pop_dev_z[:, 6:10],axis=1))  # for plotting 
    ymin = min(first_licks); ymax = max(first_licks)  # for plotting 
    
    fig, ax = plt.subplots(figsize=(2.2,2.4))
    ax.scatter(np.sum(pyract_pop_dev_z[:, 6:10],axis=1), first_licks, c='grey', ec='none', s=5)
    ax.plot([xmin-.1, xmax+.1], [intercept+(xmin-.1)*slope, intercept+(xmax+.1)*slope], 
            color='k', lw=1)
    
    ax.set(title='{}\npval={}'.format(recname, round(pval, 5)),
           xlabel='pop. dev (std.)',
           ylabel='t. to 1st-lick (s)')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\{}_v_1st_licks.png'.format(recname),
                dpi=300)
    plt.close()


    results = linregress(np.sum(pyract_pop_dev_baseline_z[:, 6:10],axis=1), first_licks_baseline)
    pval = results[3]
    slope = results[0]; intercept = results[1]; rvalue = [2]
    xmin = min(np.sum(pyract_pop_dev_baseline_z[:, 6:10],axis=1)); xmax = max(np.sum(pyract_pop_dev_baseline_z[:, 6:10],axis=1))  # for plotting 
    ymin = min(first_licks_baseline); ymax = max(first_licks_baseline)  # for plotting 
    
    fig, ax = plt.subplots(figsize=(2.2,2.4))
    ax.scatter(np.sum(pyract_pop_dev_baseline_z[:, 6:10],axis=1), first_licks_baseline, c='grey', ec='none', s=5)
    ax.plot([xmin-.1, xmax+.1], [intercept+(xmin-.1)*slope, intercept+(xmax+.1)*slope], 
            color='k', lw=1)
    ax.set(title='{}\npval={}'.format(recname, round(pval, 5)),
           xlabel='pop. dev (std.)',
           ylabel='t. to 1st-lick (s)')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\{}_v_1st_licks_baseline.png'.format(recname),
                dpi=300)
    plt.close()
    
    
#%% statistics 
recs = list(all_first_lick_z.keys())

slopes = []
all_licks = []; all_devs = []
all_early_licks = []; all_early_devs = []
all_late_licks = []; all_late_devs = []
all_early_devs_avg = []
all_late_devs_avg = []
all_early_devs_prof = []
all_late_devs_prof = []
for key in recs:
    licks = all_first_lick_z[key]
    devs = all_pyract_pop_dev_z[key]
    if len(licks)>10:
        sorted_index = np.argsort(licks)
        early = sorted_index[:5]; late = sorted_index[-5:]
        
        # all_licks.extend(licks)
        # all_devs.extend(devs)
        all_licks.extend(licks[early]); all_devs.extend(devs[early])
        all_licks.extend(licks[late]); all_devs.extend(devs[late])
        all_early_licks.extend(licks[early]) 
        all_early_devs.extend(devs[early])
        all_late_licks.extend(licks[late])
        all_late_devs.extend(devs[late])
        
        all_early_devs_avg.append(np.mean(devs[early]))
        all_late_devs_avg.append(np.mean(devs[late]))
        
        all_early_devs_prof.append(np.mean(all_pyract_pop_dev_profile[key][early], axis=0))
        all_late_devs_prof.append(np.mean(all_pyract_pop_dev_profile[key][late], axis=0))
        
        temp = linregress(licks, devs)
        slopes.append(temp[0])
    

pooled_res = linregress(all_devs, all_licks)
pooled_pval = pooled_res[3]
pooled_slope = pooled_res[0]; pooled_intercept = pooled_res[1]; pooled_rvalue = pooled_res[2]
xmin = min(all_devs); xmax = max(all_devs)  # for plotting 
ymin = min(all_licks); ymax = max(all_licks)  # for plotting 

fig, ax = plt.subplots(figsize=(2.3,2.8))
ax.scatter(all_devs, all_licks, c='grey', s=3)
ax.plot([xmin-.1, xmax+.1], 
        [pooled_intercept+(xmin-.1)*pooled_slope, pooled_intercept+(xmax+.1)*pooled_slope], 
        color='k', lw=1)
ax.set(xlabel='pop. dev. (std.)', 
       ylabel='t. to 1st-licks (s)',
       title='baseline\nr={}\np={}'.format(round(pooled_rvalue,5), round(pooled_pval,5)))
for s in ['top', 'right']: ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()
fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\pooled_baseline.png',
            dpi=300,
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\pooled_baseline.pdf',
            bbox_inches='tight')


#%% mean difference 
early_colour = (.804,.267,.267); late_colour = (.545,0,0)

pf.plot_violin_with_scatter(all_early_devs, all_late_devs, early_colour, late_colour, 
                            paired=False,
                            xticklabels=['early\n1st-lick', 'late\n1st-lick'], ylabel='pop. dev. (std.)',
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\pooled_baseline_violin.png', dpi=300,
                            savepdf=True, savepdfpath=r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\pooled_baseline_violin.pdf')


#%% mean difference after averaging data from each session
pf.plot_violin_with_scatter(all_early_devs_avg, all_late_devs_avg, early_colour, late_colour, 
                            paired=True,
                            xticklabels=['early\n1st-lick', 'late\n1st-lick'], ylabel='pop. dev. (std.)',
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\pooled_baseline_sessavg_violin.png', dpi=300,
                            savepdf=True, savepdfpath=r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only\pooled_baseline_sessavg_violin.pdf')


#%% mean profile 
fig, ax = plt.subplots()

all_mean_early = np.mean(all_early_devs_prof, axis=0)
all_sem_early = sem(all_early_devs_prof, axis=0)
all_mean_late = np.mean(all_late_devs_prof, axis=0)
all_sem_late = sem(all_late_devs_prof, axis=0)

ax.plot(xaxis, all_mean_early, c=early_colour)
ax.fill_between(xaxis, all_mean_early+all_sem_early,
                       all_mean_early-all_sem_early,
                color=early_colour, edgecolor='none', alpha=.1)
ax.plot(xaxis, all_mean_late, c=late_colour)
ax.fill_between(xaxis, all_mean_late+all_sem_late,
                       all_mean_late-all_sem_late,
                color=late_colour, edgecolor='none', alpha=.1)

ax.set(xlim=(-1,4))