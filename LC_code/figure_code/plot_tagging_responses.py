# -*- coding: utf-8 -*-
"""
Created on Thur 24 Aug 16:21:31 2023

plot example tagging pulse response

**unit is converted to ms from original 20000Hz sampling rate**

@author: Dinghao Luo
"""


#%% imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# load tagged data
all_tagged = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
                     allow_pickle=True).item()
all_tagged_keys = list(all_tagged.keys())

samp_freq = 20000  # Hz
gx_spike = np.arange(-200, 200, 1)
sigma_spike = samp_freq/1000

# precompute Gaussian filter
gaus_spike = np.exp(-gx_spike**2 / (2*sigma_spike**2)) / (sigma_spike * np.sqrt(2*np.pi))

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from param_to_array import param2array, get_clu


#%% MAIN
all_tagging_latency = {}

for cluname in all_tagged_keys:
    sessname = cluname[:17]
    datename = cluname[:14]
    animalname = cluname[1:5]
    fullname = 'Z:\Dinghao\MiceExp\ANMD'+animalname+'\\'+datename+'\\'+sessname
    
    print('\nprocessing {}'.format(cluname))
        
    # load .mat
    mat_BTDT = sio.loadmat(fullname+'\\'+sessname+'BTDT.mat')
    behEvents = mat_BTDT['behEventsTdt']
    spInfo = sio.loadmat(fullname+'\\'+sessname+'_DataStructure_mazeSection1_TrialType1_SpInfo_Run0')
    spike_rate = spInfo['spatialInfoSess'][0][0]['meanFR'][0][0][0]
    
    # global vars
    n_chan = 32  # need to change if using other probes
    n_spk_samp = 32  # arbitrary, equals to 1.6ms, default window in kilosort
    rd_samp_size = 60
    
    clu = param2array(fullname+'\\'+sessname+'.clu.1')  # load .clu
    res = param2array(fullname+'\\'+sessname+'.res.1')  # load .res
    
    clu = np.delete(clu, 0)  # delete 1st element (noise clusters)
    all_clus = np.delete(np.unique(clu), [0, 1])
    all_clus = np.array([int(x) for x in all_clus])
    all_clus = all_clus[all_clus>=2]
    tot_clus = len(all_clus)
    
    fspk = open(fullname+'\\'+sessname+'.spk.1', 'rb')  # load .spk into a byte bufferedreader

    # tagged
    stim_tp = np.zeros(60)  # hard-coded for LC stim protocol
    for i in behEvents['stimPulse'][0, 0][-60:, 4]-1:
        ind = int(i)
        temp = behEvents['stimPulse'][0, 0][-60:, :][ind, 0]
        stim_tp[ind] = temp  # time points of each stim 
    
    tagged_spk_index = np.zeros(60)
    tagged_spk_time = np.zeros(60)
    tagging_latency = np.zeros(60)
    
    nth_clu = int(cluname[21:])  # current clu number
    clu_n_id = np.transpose(get_clu(nth_clu, clu))

    arr = np.zeros([60, 1201])
    fig, axs = plt.subplots(2, 1, figsize=(1.8, 2.7))
    for p in ['right', 'top']:
        axs[0].spines[p].set_visible(False)
        axs[1].spines[p].set_visible(False)
        
    axs[0].set(title=cluname, xlabel='time (ms)', ylabel='sweep #',
               xlim=(-31, 31), ylim=(0, 61),
               xticks=[-30, 0, 30])
    axs[0].fill_between([0, 5], [0, 0], [61, 61], color='royalblue', alpha=.5, linewidth=0)
    for i in range(60):  # hard-coded
        t_bef = stim_tp[i]-600  # 30 ms before stim
        t_stim = stim_tp[i]  # stim time point
        t_aft = stim_tp[i]+600  # stim time point +15ms (Takeuchi et al.)
        spks_in_range = [int(float(res[s])-t_stim) for s in clu_n_id if int(res[s])>=t_bef and int(res[s])<=t_aft]
        spks_arr = [s+600 for s in spks_in_range]
        arr[i, spks_arr] = 1
        arr[i, :] = np.convolve(arr[i, :], gaus_spike, mode='same')*1000
        
        spks_in_range = [s/20 for s in spks_in_range]
        xlen = len(spks_in_range)
        axs[0].scatter(spks_in_range, [i+1]*xlen, s=.25, color='k')
        
    mean_prof = np.mean(arr, axis=0)
    xaxis = np.arange(-600,601)/20
    axs[1].set(xlim=(-31, 31), 
               xticks=[-30, 0, 30],
               xlabel='time (ms)', ylabel='spikes')
    axs[1].plot(xaxis, mean_prof, color='k')
        
    fig.tight_layout()
    plt.show()
    
    for ext in ['png', 'pdf']:
        fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all_tagged\tagging_responses\{}.{}'.format(cluname, ext),
                    bbox_inches='tight',
                    dpi=200)
    
    plt.close(fig)