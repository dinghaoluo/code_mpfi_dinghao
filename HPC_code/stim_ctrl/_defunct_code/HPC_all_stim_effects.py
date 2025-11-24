# -*- coding: utf-8 -*-
"""
Created on Mon May 12 17:25:46 2025

Quantify the effects of LC stimulation on CA1 pyramidal population

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pandas as pd 
import pickle
import sys 
import os 
from scipy.stats import ranksums
import matplotlib.pyplot as plt 

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% load data 
print('loading dataframes...')

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
    )
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 


#%% parameters 
SAMP_FREQ = 1250  # Hz 
PRE = 1  # s 
POST = 4  # s 

BIN_SIZE = 0.4


#%% function 
def bin_raster(raster_trial, bin_size=BIN_SIZE, samp_freq=SAMP_FREQ):
    bin_size_samples = int(bin_size * samp_freq)
    n_bins = len(raster_trial) // bin_size_samples
    trimmed = raster_trial[:n_bins * bin_size_samples]
    binned = trimmed.reshape(n_bins, bin_size_samples).sum(axis=1)
    binned_rate = binned / bin_size
    return binned_rate


#%% dataframe initialisation 
sess = {
    'rectype': [],  # HPCLC, HPCLCterm
    'recname': [],  # Axxxr-202xxxxx-0x
    'n_sig_bins': [],
    'n_act_bins': [],
    'n_inh_bins': [],
    'mean_ctrl': [],
    'mean_stim': [],
    'sem_ctrl': [],
    'sem_stim': [],
    'pvals': [],
    'delta': []
    }
df = pd.DataFrame(sess)


#%% main
for path in paths:
    recname = path[-17:]
    print(f'\n{recname}')
    
    rasters = np.load(
        rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}\{recname}_all_rasters.npy',
        allow_pickle=True
        ).item()
    
    if os.path.exists(
            rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl'
            ):
        with open(
                rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC\{recname}.pkl',
                'rb'
                ) as f:
            beh = pickle.load(f)
    else:
        with open(
                rf'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLCterm\{recname}.pkl',
                'rb'
                ) as f:
            beh = pickle.load(f)
    
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds)
                if cond!='0']
    ctrl_idx = [trial+2 for trial in stim_idx]
    
    # now we exclude trial-pairs where the stim was out of bounds
    pulse_times = beh['pulse_times'][1:]
    run_onsets = beh['run_onsets'][1:len(pulse_times)+1]  # sometimes there is a last trial that does not finish 
    pulse_onsets = [p[0] - r if p else []
                    for p, r in zip(pulse_times, run_onsets)]
    filtered_stim_idx = []
    filtered_ctrl_idx = []
    for stim_i, ctrl_i in zip(stim_idx, ctrl_idx):
        offset = pulse_onsets[stim_i]
        if isinstance(offset, (int, float)) and offset <= SAMP_FREQ:
            filtered_stim_idx.append(stim_i)
            filtered_ctrl_idx.append(ctrl_i)
    print(f'filtered out {len(stim_idx) - len(filtered_stim_idx)} trial pairs with stim_offset > 1 s')
    
    if len(filtered_stim_idx)==0:
        print('no trials left; abort')
        continue
    
    curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    curr_df_pyr_ON = curr_df_pyr[curr_df_pyr['class_ctrl']=='run-onset ON']
    curr_df_pyr_OFF = curr_df_pyr[curr_df_pyr['class_ctrl']=='run-onset OFF']
    pyr_list = curr_df_pyr.index.tolist()
    pyr_ON_list = curr_df_pyr_ON.index.tolist()
    pyr_OFF_list = curr_df_pyr_OFF.index.tolist()
    
    tot_time = SAMP_FREQ * (PRE + POST)
    
    results = []
    for cluname, row in curr_df_pyr.iterrows():
        if cluname in pyr_ON_list:
            clustr = f'{cluname} ON'
        elif cluname in pyr_OFF_list:
            clustr = f'{cluname} OFF'
        else:
            clustr = f'{cluname} other'
        
        # get trains 
        raster = rasters[cluname]
        stim_raster = raster[filtered_stim_idx]
        ctrl_raster = raster[filtered_ctrl_idx]
        
        # bin into 20 ms bins 
        stim_binned = np.stack([
            bin_raster(trial[3750-SAMP_FREQ*PRE:3750+SAMP_FREQ*POST]
                       ) for trial in stim_raster])
        ctrl_binned = np.stack(
            [bin_raster(trial[3750-SAMP_FREQ*PRE:3750+SAMP_FREQ*POST]
                        ) for trial in ctrl_raster])
        
        # mean profiles
        mean_stim = stim_binned.mean(axis=0)
        mean_ctrl = ctrl_binned.mean(axis=0)
        sem_stim = stim_binned.std(axis=0) / np.sqrt(stim_binned.shape[0])
        sem_ctrl = ctrl_binned.std(axis=0) / np.sqrt(ctrl_binned.shape[0])
        
        # bin-by-bin stats
        pvals = []
        delta = mean_stim - mean_ctrl
        for b in range(stim_binned.shape[1]):
            _, p = ranksums(stim_binned[:, b], ctrl_binned[:, b])
            pvals.append(p)
        
        # significance summary
        alpha = 0.05
        sig_bins = np.array(pvals) < alpha
        act_bins = sig_bins & (delta > 0)
        inh_bins = sig_bins & (delta < 0)
        
        df.loc[cluname] = np.array(['HPCLC' if path in rec_list.pathHPCLCopt else 'HPCLCterm',  # rectype
                                    cluname.split(' ')[0],  # recname 
                                    sig_bins.sum(),
                                    act_bins.sum(),
                                    inh_bins.sum(),
                                    mean_ctrl,
                                    mean_stim,
                                    sem_ctrl,
                                    sem_stim,
                                    pvals,
                                    delta
                                        ],
                                    dtype='object')
        
        # plotting 
        fig, axs = plt.subplots(2, 1, figsize=(2.1,3.4))
        fig.suptitle(clustr, fontsize=10)
        
        for line in range(len(filtered_ctrl_idx)):
            axs[0].scatter(np.where(ctrl_raster[line]==1)[0]/SAMP_FREQ-3,
                           [line+1]*int(sum(ctrl_raster[line])),
                           c='grey', ec='none', s=1)
            axs[1].scatter(np.where(stim_raster[line]==1)[0]/SAMP_FREQ-3,
                           [line+1]*int(sum(stim_raster[line])),
                           c='royalblue', ec='none', s=1)
                        
        for i in range(2):
            axs[i].set(xticks=[0,2,4], xlim=(-1, 4),
                       ylabel='trial #')
            for p in ['top', 'right']:
                axs[i].spines[p].set_visible(False)
            
        # only set xlabel for ax 1
        axs[1].set(xlabel='time from run-onset (s)')
        
        # add the bars
        taxis = np.arange(-PRE, POST, BIN_SIZE)
        for i, (t, a, ih) in enumerate(zip(taxis, act_bins, inh_bins)):
            if a:
                axs[1].axvspan(t, t + BIN_SIZE, ymin=1.0, ymax=1.05, color='crimson', lw=3)
            elif ih:
                axs[1].axvspan(t, t + BIN_SIZE, ymin=1.0, ymax=1.05, color='navy', lw=3)

        fig.tight_layout()
        
        # save figure
        if path in rec_list.pathHPCLCopt:
            fig.savefig(
                    rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_rasters\HPC_LC_pyr_stim_effects\{clustr}.png',
                    dpi=300,
                    bbox_inches='tight'
                    )
        if path in rec_list.pathHPCLCtermopt:
            fig.savefig(
                    rf'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_ctrl_stim_rasters\HPC_LCterm_pyr_stim_effects\{clustr}.png',
                    dpi=300,
                    bbox_inches='tight'
                    )

        plt.close(fig)
        

#%% save 
outpath = r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_pyr_stim_effects.pkl'
df.to_pickle(outpath)
print(f'saved to {outpath}')