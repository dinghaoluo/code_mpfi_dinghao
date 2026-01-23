# -*- coding: utf-8 -*-
"""
Created on Thur 24 Aug 16:21:31 2023

plot example tagging pulse response

**unit is converted to ms from original 20000Hz sampling rate**

@author: Dinghao Luo
"""


#%% imports
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from param_to_array import param2array, get_clu

from common import mpl_formatting, gaussian_kernel_unity
mpl_formatting()


#%% paths and parameters 
LC_tagged_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/tagged_only_analysis')

MiceExp_stem = Path('Z:/Dinghao/MiceExp')

# ephys params
SAMP_FREQ   = 20_000  # Hz
GX_SPIKE    = np.arange(-200, 200, 1)
SIGMA_SPIKE = SAMP_FREQ / 1_000

# LC-specific params 
N_CHAN       = 32  # need to change if using other probes
N_SPK_SAMP   = 32  # arbitrary, equals to 1.6ms, default window in kilosort
RD_SAMP_SIZE = 60
N_SWEEPS     = 60
TOT_MS       = 60  # ms (for plotting; so 30 before and 30 after)
TOT_SAMPS    = int(SAMP_FREQ / 1_000 * TOT_MS + 1)  # 1 s = 1000 ms

# precompute Gaussian filter
KERN_SPIKE  = gaussian_kernel_unity(sigma=SIGMA_SPIKE)


#%% load data 
all_tagged_path = LC_tagged_stem / 'LC_all_tagged_info.npy'

all_tagged = np.load(all_tagged_path, allow_pickle=True).item()

# get keys of all tagged cells 
all_tagged_keys = list(all_tagged.keys())


#%% MAIN
all_tagging_latency = {}

for cluname in all_tagged_keys:
    recname    = cluname.split(' ')[0]
    datename   = '-'.join(cluname.split('-')[:2])
    animalname = cluname.split('-')[0][1:]
    
    fullname = MiceExp_stem / f'ANMD{animalname}' / datename / recname
    
    print(f'\n{cluname}')
        
    # load .mat files 
    mat_BTDT_path = fullname / f'{recname}BTDT.mat'
    spInfo_path   = fullname / f'{recname}_DataStructure_mazeSection1_TrialType1_SpInfo_Run0.mat'
    
    mat_BTDT = sio.loadmat(str(mat_BTDT_path))
    spInfo   = sio.loadmat(str(spInfo_path))
    
    behEvents = mat_BTDT['behEventsTdt']
    spike_rate = spInfo['spatialInfoSess'][0][0]['meanFR'][0][0][0]
    
    # load clu and res 
    clu_path = fullname / f'{recname}.clu.1'
    res_path = fullname / f'{recname}.res.1'

    clu = param2array(clu_path)  # load .clu
    res = param2array(res_path)  # load .res
    
    clu = np.delete(clu, 0)  # delete 1st element (noise clusters)
    all_clus = np.delete(np.unique(clu), [0, 1])
    all_clus = np.array([int(x) for x in all_clus])
    all_clus = all_clus[all_clus>=2]
    tot_clus = len(all_clus)
    
    # load spike file
    fspk_path = fullname / f'{recname}.spk.1'
    
    fspk = open(str(fspk_path), 'rb')  # load .spk into a byte bufferedreader

    # tagged
    stim_tp = np.zeros(N_SWEEPS)  # hard-coded for LC stim protocol
    for i in behEvents['stimPulse'][0, 0][-N_SWEEPS:, 4] - 1:  # index 4 is the timepoint readout 
        ind = int(i)
        temp = behEvents['stimPulse'][0, 0][-N_SWEEPS:, :][ind, 0]
        stim_tp[ind] = temp  # time points of each stim 
    
    tagged_spk_index = np.zeros(N_SWEEPS)
    tagged_spk_time  = np.zeros(N_SWEEPS)
    tagging_latency  = np.zeros(N_SWEEPS)
    
    nth_clu  = int(cluname.split('clu')[-1])  # current clu number
    clu_n_id = np.transpose(get_clu(nth_clu, clu))

    # plotting 
    arr = np.zeros([N_SWEEPS, TOT_SAMPS])
    
    fig, axs = plt.subplots(2, 1, figsize=(1.8, 2.7), sharex=True)
        
    axs[0].set(
        title=cluname, 
        ylabel='Sweep #',
        xlim=(-TOT_MS / 2 + 1, TOT_MS / 2 + 1), 
        xticks=[-int(TOT_MS / 2), 0, int(TOT_MS / 2)],
        ylim=(0, N_SWEEPS + 1)
        )
    axs[0].fill_between(
        [0, 5], 
        [0, 0], 
        [N_SWEEPS + 1, N_SWEEPS + 1], 
        color='royalblue', alpha=.5, linewidth=0
        )
    
    for i in range(N_SWEEPS):  # hard-coded
        t_bef  = stim_tp[i] - (TOT_SAMPS - 1) / 2  # 30 ms before stim
        t_stim = stim_tp[i]                        # stim time point
        t_aft  = stim_tp[i] + (TOT_SAMPS - 1) / 2  # stim time point +15ms (Takeuchi et al.)
        
        spks_in_range = [int(float(res[s])-t_stim) for s in clu_n_id 
                         if int(res[s])>=t_bef and int(res[s])<=t_aft]
        
        spks_arr = [int(s + (TOT_SAMPS - 1) / 2) for s in spks_in_range]
        
        arr[i, spks_arr] = 1
        arr[i, :]        = np.convolve(arr[i, :], KERN_SPIKE, mode='same') * 1_000
        
        spks_in_range = [s / 20 for s in spks_in_range]  # / 20 because SAMP_FREQ is 20 * 1000 ms
        
        xlen = len(spks_in_range)
        
        axs[0].scatter(spks_in_range, [i+1] * xlen, s=.3, color='k')
        
    mean_prof = np.mean(arr, axis=0)
    xaxis = np.arange(-(TOT_SAMPS - 1) / 2 - 1, (TOT_SAMPS - 1) / 2) / 20
    axs[1].set(xlim=(-(TOT_MS / 2 + 1), (TOT_MS / 2 + 1)), 
               xticks=[-int(TOT_MS / 2), 0, int(TOT_MS / 2)],
               xlabel='Time from stim. (ms)', 
               ylabel='Spikes')
    axs[1].plot(xaxis, 
                mean_prof, 
                color='k')
        
    for i in range(2):
        for p in ['right', 'top']:
            axs[i].spines[p].set_visible(False)
    
    fig.tight_layout()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            LC_tagged_stem / 'tagging_responses' / f'{cluname}{ext}',
            bbox_inches='tight',
            dpi=300)
    
    plt.close(fig)