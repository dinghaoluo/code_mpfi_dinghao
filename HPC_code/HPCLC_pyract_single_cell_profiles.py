# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:25:24 2024

Run-onset activated cell single cell profile 

@author: Dinghao Luo
"""


#%% imports
import pandas as pd  
import numpy as np 
import scipy.io as sio 
from math import log 
from scipy.stats import sem, poisson
import matplotlib.pyplot as plt 
import sys

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
    
    
#%% pre-post ratio dataframe 
df = pd.read_pickle('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.pkl') 


#%% main
for pathname in pathHPC:
    recname = pathname[-17:]
    if recname=='A063r-20230708-02' or recname=='A063r-20230708-01':  # lick detection problems 
        continue
    print(recname)
    
    trains = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
                          allow_pickle=True).item().values())
    tot_trial = len(trains[0])
    
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
    
    for pyr in pyr_act:
        curr_train = trains[pyr]
        temp = np.zeros((tot_trial, 8*1250))
        for trial in range(tot_trial):
            trial_length = len(curr_train[trial])
            if trial_length<8*1250 and trial_length>0:
                temp[trial, :trial_length] = curr_train[trial][:8*1250]  # use [-3, 5]
            else:
                temp[trial, :] = curr_train[trial][:8*1250]
        mean_prof = np.mean(temp, axis=0)*1250
        sem_prof = sem(temp, axis=0)*1250
        mean_sr = spike_rate[pyr]
        
        fig, ax = plt.subplots(figsize=(2,1.9)); xaxis=np.arange(1250*8)/1250-3
        lnprof, = ax.plot(xaxis, mean_prof, c='royalblue', lw=1)
        ax.fill_between(xaxis, mean_prof+sem_prof,
                               mean_prof-sem_prof,
                        alpha=.1, edgecolor='none', color='royalblue')
        ax.axhline(y=mean_sr, color='grey', alpha=.5, label='mean', linestyle='dashed')
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        scale_min, scale_max = pf.scale_min_max(mean_prof[2500:8750], sem_prof[2500:8750])
        ax.set(xlabel='time (s)', xlim=(-1,4), xticks=[0,2,4],
               ylabel='spike rate (Hz)', ylim=(scale_min, scale_max), 
               title='{} clu{}'.format(recname, pyr))
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.show(fig)
        fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_pyract_profiles\{}_clu{}.png'.format(recname, pyr),
                    dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
        
        # example Poisson probability plot if needed
        deviation = np.zeros((tot_trial, 16))
        for trial in range(tot_trial):
            curr_raster = rasters[pyr][trial]
            for tbin, t in enumerate(np.linspace(0,8,16+1)[:-1]):  # 3 seconds before, 5 seconds after 
                curr_bin = sum(curr_raster[int(t*1250):int((t+1)*1250)])
                coeff=1
                if curr_bin<mean_sr: coeff=-1
                deviation[trial, tbin] = -log(poisson.pmf(curr_bin, mean_sr))*coeff  # polarity achieved with coeff
        
        mean_dev = np.mean(deviation, axis=0)
        sem_dev = sem(deviation, axis=0)
        fig, ax = plt.subplots(figsize=(2,1.9)); xaxis=np.linspace(-2.5,5.5,16+1)[:-1]
        ax.plot(xaxis, mean_dev, c='royalblue', lw=1)
        ax.fill_between(xaxis, mean_dev+sem_dev,
                               mean_dev-sem_dev,
                        color='royalblue', edgecolor='none', alpha=.1)
        scale_min, scale_max = pf.scale_min_max(mean_dev[3:14], sem_dev[3:14])
        ax.set(xlabel='time (s)', xticks=[0,2,4], xlim=(-1,4),
               ylabel='Poisson dev.', ylim=(scale_min, scale_max),
               title='{} clu{}'.format(recname, pyr))
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.show(fig)
        fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_pyract_profiles\{}_clu{}_Poisson_deviation.png'.format(recname, pyr),
                    dpi=300,
                    bbox_inches='tight')
        plt.close(fig)