# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:43:47 2022
update:
    25 Jan 2023, input rec_list

saves the average waveforms of all recordings in rec_list, pathLC
@author: Dinghao Luo
"""


#%% imports
import sys
import numpy as np
from random import sample
import scipy.io as sio
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib as plc
import os

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise
from param_to_array import param2array, get_clu

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC

number_eg_spk = 100  # how many example spks to store


#%% main function
def spk_w_sem(fspk, clu, nth_clu):
    
    clu_n_id = [int(x) for x in np.transpose(get_clu(nth_clu, clu))]  # ID of every single spike of clu

    # load spikes (could cut out to be a separate function)
    rnd_sample_size = 1000
    if len(clu_n_id)<rnd_sample_size:
        tot_spks = len(clu_n_id)
    else:
        clu_n_id = sample(clu_n_id, rnd_sample_size)
        tot_spks = len(clu_n_id)
    spks_wfs = np.zeros([tot_spks, n_chan, n_spk_samp])  # wfs for all tagged spikes

    for i in range(tot_spks):  # reading .spk in binary ***might be finicky
        status = fspk.seek(clu_n_id[i]*n_chan*n_spk_samp*2)  # go to correct part of file
        if status == -1:
            raise Exception('Cannot go to the correct part of .spk')
    
        spk = fspk.read(2048)  # 32*32 bts for a spike, 32*32 bts for valence of each point
        spk_fmtd = np.zeros([32, 32])  # spk but formatted as a 32x32 matrix
        for j in range(n_spk_samp):
            for k in range(n_chan):
                spk_fmtd[k, j] = spk[k*2+j*64]
                if spk[k*2+j*64+1] == 255:  # byte following value signifies valence (255 means negative)
                    spk_fmtd[k, j] = spk_fmtd[k, j] - 256  # flip sign, negative values work as subtracting from 256
    
        spks_wfs[i, :, :] = spk_fmtd
    
    # average & max spike waveforms
    av_spks = np.zeros([tot_spks, n_spk_samp])
    max_spks = np.zeros([tot_spks, n_spk_samp])
    
    # added 29 Aug
    eg_spks = np.zeros([number_eg_spk, n_spk_samp])
    ind_eg = np.random.randint(0, tot_spks, number_eg_spk)
    eg_count = 0
    
    for i in range(tot_spks):
        spk_single = np.matrix(spks_wfs[i, :, :])
        spk_diff = np.zeros(n_chan)
        for j in range(n_chan):
            spk_diff[j] = np.amax(spk_single[j, :]) - np.amin(spk_single[j, :])
            spk_max = np.argmax(spk_diff)
        max_spks[i, :] = spk_single[spk_max, :]  # wf w/ highest amplitude
        av_spks[i, :] = spk_single.mean(0)  # wf of averaged amplitude (channels)
        
        if i in ind_eg:
            eg_spks[eg_count, :] = av_spks[i, :]
            eg_count+=1
    
    norm_spks = np.zeros([tot_spks, n_spk_samp])
    for i in range(tot_spks):
        norm_spks[i, :] = normalise(av_spks[i, :])  # normalisation
    
    av_spk = norm_spks.mean(0)  # 1d vector for the average tagged spike wf
    
    # sem calculation
    spk_sem = np.zeros(32)
    
    for i in range(32):
        spk_sem[i] = sem(norm_spks[:, i])
        
    return av_spk, spk_sem, eg_spks;


#%% MAIN
for pathname in pathLC:    
    file_stem = pathname
    loc_A = file_stem.rfind('A')
    file_name = file_stem + '\\' + file_stem[loc_A:loc_A+17]
    
    # only process if there exist not already the .npy
    if os.path.isfile('Z:\Dinghao\code_dinghao\LC_by_sess'+file_name[42:60]+'_avg_spk.npy')==False:
    # if 1==1:
        # header
        print('\n\nProcessing {}'.format(pathname))
    
        # load .mat
        mat_BTDT = sio.loadmat(file_name + 'BTDT.mat')
        behEvents = mat_BTDT['behEventsTdt']
        spInfo = sio.loadmat(file_name + '_DataStructure_mazeSection1_TrialType1_SpInfo_Run0')
        spike_rate = spInfo['spatialInfoSess'][0][0]['meanFR'][0][0][0]
        
        # global vars
        n_chan = 32  # need to change if using other probes
        n_spk_samp = 32  # arbitrary, equals to 1.6ms, default window in kilosort
        
        clu = param2array(file_name + '.clu.1')  # load .clu
        res = param2array(file_name + '.res.1')  # load .res
        
        clu = np.delete(clu, 0)  # delete 1st element (noise clusters)
        all_clus = np.delete(np.unique(clu), [0, 1])
        all_clus = np.array([int(x) for x in all_clus])
        all_clus = all_clus[all_clus>=2]
        tot_clus = len(all_clus)
        
        fspk = open(file_name + '.spk.1', 'rb')  # load .spk into a byte bufferedreader
        
        #---plotting---#
        time_ax = [x*50 for x in range(n_spk_samp)]  # time ticks in *u*s
        
        tot_plots = tot_clus
        col_plots = 4
        
        row_plots = tot_plots // col_plots
        if tot_plots % col_plots != 0:
            row_plots += 1
            
        plc.rcParams['figure.figsize'] = (4*2, row_plots*2.5)
        
        plot_pos = np.arange(1, tot_plots+1)
        
        fig = plt.figure(1)
        
        avg_spk_dict = {}
        avg_sem_dict = {}
        eg_spks_dict = {}
        
        for i in range(tot_clus):
            nth_clu = i + 2
            av_spk, spk_sem, eg_spks = spk_w_sem(fspk, clu, nth_clu)
            
            avg_spk_dict[str(nth_clu)] = av_spk
            avg_sem_dict[str(nth_clu)] = spk_sem
            eg_spks_dict[str(nth_clu)] = eg_spks
            
            ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
            ax.set_title('%s%s' % ('clu ', nth_clu), fontsize = 10)
            ax.plot(time_ax, av_spk)
            ax.fill_between(time_ax, av_spk+spk_sem, av_spk-spk_sem, color='lightblue')
            ax.axis('off')
        
        plt.subplots_adjust(hspace = 0.4)
        plt.show()
        
        # out_directory = r'Z:\Dinghao\code_dinghao\LC_tagged_by_sess'
        # fig.savefig(out_directory + '/' + file_name[42:60] + '_waveforms' + '.png')
        
        np.save('Z:\Dinghao\code_dinghao\LC_by_sess'+file_name[42:60]+'_avg_spk.npy', avg_spk_dict)
        np.save('Z:\Dinghao\code_dinghao\LC_by_sess'+file_name[42:60]+'_avg_sem.npy', avg_sem_dict)
        np.save('Z:\Dinghao\code_dinghao\LC_by_sess'+file_name[42:60]+'_avg_eg_spks.npy', eg_spks_dict)