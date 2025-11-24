# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:22:43 2024

population deviation poisson 

@author: Dinghao Luo
"""


#%% imports 
import sys
import scipy.io as sio 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from math import log 
from scipy.stats import poisson, zscore, sem, ttest_rel, wilcoxon

if (r'Z:\Dinghao\code_mpfi_dinghao\iutils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf


#%% binning parameter
tot_bins = 16


#%% pre-post ratio dataframe 
df = pd.read_pickle('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.pkl') 


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


#%% main (single-trial act./inh. proportions)
all_pyract_ctrl_pop_dev_z = {}
all_pyract_stim_pop_dev_z = {}
all_pyract_ctrl_pop_dev_prof = {}
all_pyract_stim_pop_dev_prof = {}

for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    rasters = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(recname), 
                           allow_pickle=True).item().values())
    
    # determine if each cell is pyramidal or intern 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    spike_rate = rec_info['firingRate'][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [i for i, intern in enumerate(intern_id) if not intern]
    tot_pyr = sum(pyr_id)
    
    # classify act./inh. neurones 
    pyr_act_ctrl = []; pyr_inh_ctrl = []
    pyr_act_stim = []; pyr_inh_stim = []
    for cluname, row in df.iterrows():
        if cluname.split(' ')[0]==recname:
            clu_ID = int(cluname.split(' ')[1][3:])
            if row['ctrl_pre_post']>=1.25:
                pyr_inh_ctrl.append(clu_ID-2)
            if row['ctrl_pre_post']<=.8:
                pyr_act_ctrl.append(clu_ID-2)  # HPCLC_all_train.py adds 2 to the ID, so we subtracts 2 here 
            if row['stim_pre_post']>=1.25:
                pyr_inh_stim.append(clu_ID-2)
            if row['stim_pre_post']<=.8:
                pyr_act_stim.append(clu_ID-2) 
    
    # behaviour parameters
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]-1
    ctrl_trials = stim_trials+2
    tot_trial = len(stimOn)

    # for each trial we quantify deviation from Poisson
    fig, ax = plt.subplots(figsize=(3,2.5)); xaxis = np.linspace(-3,5,tot_bins+1)[:-1]
    pop_dev_ctrl = []; pop_dev_stim = []
    for trial in ctrl_trials:
        single_dev = np.zeros((len(pyr_act_ctrl), tot_bins))
        cell_counter = 0
        for pyr in pyr_act_ctrl:
            curr_raster = rasters[pyr][trial]
            for tbin, t in enumerate(np.linspace(0,8,tot_bins+1)[:-1]):  # 3 seconds before, 5 seconds after 
                curr_bin = sum(curr_raster[int(t*1250):int((t+1)*1250)])
                single_dev[cell_counter, tbin] = -log(poisson.pmf(curr_bin, spike_rate[pyr]))
            cell_counter+=1
        pop_dev_ctrl.append(np.mean(single_dev, axis=0))
        # ax.plot(xaxis, np.sum(single_dev, axis=0), lw=1, alpha=.1, c='grey')
    for trial in stim_trials:
        single_dev = np.zeros((len(pyr_act_stim), tot_bins))
        cell_counter = 0
        for pyr in pyr_act_stim:
            curr_raster = rasters[pyr][trial]
            for tbin, t in enumerate(np.linspace(0,8,tot_bins+1)[:-1]):  # 3 seconds before, 5 seconds after 
                curr_bin = sum(curr_raster[int(t*1250):int((t+1)*1250)])
                single_dev[cell_counter, tbin] = -log(poisson.pmf(curr_bin, spike_rate[pyr]))
            cell_counter+=1
        pop_dev_stim.append(np.mean(single_dev, axis=0))
        # ax.plot(xaxis, np.sum(single_dev, axis=0), lw=1, alpha=.1, c='steelblue')
        
    pop_dev_ctrl = np.asarray(pop_dev_ctrl)
    pop_dev_stim = np.asarray(pop_dev_stim)
    mean_pop_dev_ctrl = np.mean(pop_dev_ctrl, axis=0)
    sem_pop_dev_ctrl = sem(pop_dev_ctrl, axis=0)
    mean_pop_dev_stim = np.mean(pop_dev_stim, axis=0)
    sem_pop_dev_stim = sem(pop_dev_stim, axis=0)
    
    scale_min_ctrl, scale_max_ctrl = pf.scale_min_max(mean_pop_dev_ctrl, sem_pop_dev_ctrl)
    scale_min_stim, scale_max_stim = pf.scale_min_max(mean_pop_dev_stim, sem_pop_dev_stim)
    
    ax.plot(xaxis, mean_pop_dev_ctrl, lw=2, c='grey')
    ax.fill_between(xaxis, mean_pop_dev_ctrl+sem_pop_dev_ctrl,
                           mean_pop_dev_ctrl-sem_pop_dev_ctrl,
                    color='grey', edgecolor='none', alpha=.1)
    ax.plot(xaxis, mean_pop_dev_stim, lw=2, c='royalblue')
    ax.fill_between(xaxis, mean_pop_dev_stim+sem_pop_dev_stim,
                           mean_pop_dev_stim-sem_pop_dev_stim,
                    color='royalblue', edgecolor='none', alpha=.1)
    
    ax.set(title=recname,
           xlabel='time (s)',
           ylabel='pop. deviation', ylim=(min(scale_min_stim, scale_min_ctrl), max(scale_max_stim, scale_max_ctrl)))
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.tight_layout()
    
    fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only_ctrl_v_stim\{}'.format(recname))
    
    plt.close(fig)
    
    # pop_dev_stack_z = zscore(np.vstack((pop_dev_ctrl, pop_dev_stim)))  # zscore the 2 profiles together
    # pop_dev_ctrl_z = pop_dev_stack_z[:len(ctrl_trials)]
    # pop_dev_stim_z = pop_dev_stack_z[len(ctrl_trials):]
    # sum_pop_dev_ctrl = np.sum(pop_dev_ctrl[:, 6:10], axis=1)
    # sum_pop_dev_stim = np.sum(pop_dev_stim[:, 6:10], axis=1)
    # sum_stack = np.concatenate((sum_pop_dev_ctrl, sum_pop_dev_stim))
    # sum_stack_z = zscore(sum_stack)
    # all_pyract_ctrl_pop_dev_z[recname] = np.sum(pop_dev_ctrl_z[:, 6:10], axis=1)
    # all_pyract_stim_pop_dev_z[recname] = np.sum(pop_dev_stim_z[:, 6:10], axis=1)
    
    all_pyract_ctrl_pop_dev_prof[recname] = pop_dev_ctrl[:,:]
    all_pyract_stim_pop_dev_prof[recname] = pop_dev_stim[:,:]
    
    pf.plot_violin_with_scatter(np.sum(pop_dev_ctrl[:, 6:10], axis=1), 
                                np.sum(pop_dev_stim[:, 6:10], axis=1), 
                                'grey', 'royalblue',
                                xticklabels=['ctrl.', 'stim.'], 
                                ylabel='pop. dev. (std.)',
                                showmeans=True, showmedians=False,
                                title=recname,
                                save=True, 
                                savepath=r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only_ctrl_v_stim\{}_ctrl_stim.png'.format(recname))


#%% filtering out sessions without apparent peaks in deviation 
sess_filter = ['A069r-20230905-02',
               'A069r-20230908-02',
               'A069r-20230912-01',
               'A069r-20230915-02',
               'A071r-20230921-02',
               'A071r-20230928-02']

for key in sess_filter:
    del all_pyract_ctrl_pop_dev_prof[key]
    del all_pyract_stim_pop_dev_prof[key]


#%% save 
np.save(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_pop_dev_pyract_only_ctrl.npy', np.asarray(all_pyract_ctrl_pop_dev_prof),
        allow_pickle=True)
np.save(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_pop_dev_pyract_only_stim.npy', np.asarray(all_pyract_stim_pop_dev_prof),
        allow_pickle=True)


#%% extract data 
recs = list(all_pyract_ctrl_pop_dev_prof.keys())

all_ctrl_devs = []
all_stim_devs = []
all_ctrl_devs_sess = []
all_stim_devs_sess = []
# all_ctrl_devs_dict = {}
# all_stim_devs_dict = {}
for key in recs:
    curr_ctrl = all_pyract_ctrl_pop_dev_prof[key]
    curr_stim = all_pyract_stim_pop_dev_prof[key]
    all_stack_z = zscore(np.vstack((curr_ctrl, curr_stim)), axis=None)
    sec = curr_ctrl.shape[0]
    curr_ctrl_z = all_stack_z[:sec]
    curr_stim_z = all_stack_z[sec:]
    all_ctrl_devs.extend(np.sum(curr_ctrl_z[:, 6:10], axis=1))
    all_stim_devs.extend(np.sum(curr_stim_z[:, 6:10], axis=1))
    all_ctrl_devs_sess.append(np.mean(curr_ctrl_z[:, 6:10]))
    all_stim_devs_sess.append(np.mean(curr_stim_z[:, 6:10]))
    # all_ctrl_devs_dict[key] = np.mean(curr_ctrl_z[:, 6:10])
    # all_stim_devs_dict[key] = np.mean(curr_stim_z[:, 6:10])


#%% statistics
pf.plot_violin_with_scatter(all_ctrl_devs, all_stim_devs, 'grey', 'royalblue',
                            xticklabels=['ctrl', 'stim'], ylabel='pop. dev.',
                            showmeans=True, showmedians=False, paired=True, yscale=True, dpi=300,
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only_ctrl_v_stim\summary.png')

pf.plot_violin_with_scatter(all_ctrl_devs_sess, all_stim_devs_sess, 'grey', 'royalblue',
                            xticklabels=['ctrl', 'stim'], ylabel='pop. dev.',
                            showmeans=True, showmedians=False, paired=True, dpi=300,
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\HPC_all\population_deviation\pyract_only_ctrl_v_stim\summary_sess.png')