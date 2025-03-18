# -*- coding: utf-8 -*-
"""
DDDDDDD    EEEEEEE  FFFFFFF  U      U  N NNNNN    CCCCCCC  TTTTTTTTT
D      D   E        F        U      U  NN     N  C             T
D       D  E        F        U      U  N      N  C             T
D       D  EEEEEE   FFFFFF   U      U  N      N  C             T
D      D   E        F         U    U   N      N  C             T
DDDDDDD    EEEEEEE  F          UUUU    N      N   CCCCCCC      T

Created on Mon 10 July 8:22:47 2022

saves the average waveforms of all recordings in rec_list, pathHPCLCopt
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
pathHPC = rec_list.pathHPCLCopt


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
        spk_fmtd = np.zeros([n_chan, 32])  # spk but formatted as a 32x32 matrix
        for j in range(n_spk_samp):
            for k in range(n_chan):
                spk_fmtd[k, j] = spk[k*2+j*64]
                if spk[k*2+j*64+1] == 255:  # byte following value signifies valence (255 means negative)
                    spk_fmtd[k, j] = spk_fmtd[k, j] - 256  # flip sign, negative values work as subtracting from 256
    
        spks_wfs[i, :, :] = spk_fmtd
    
    # average & max spike waveforms
    av_spks = np.zeros([tot_spks, n_spk_samp])
    max_spks = np.zeros([tot_spks, n_spk_samp])
    
    for i in range(tot_spks):
        spk_single = np.matrix(spks_wfs[i, :, :])
        spk_diff = np.zeros(n_chan)
        for j in range(n_chan):
            spk_diff[j] = np.amax(spk_single[j, :]) - np.amin(spk_single[j, :])
            spk_max = np.argmax(spk_diff)
        max_spks[i, :] = spk_single[spk_max, :]  # wf w/ highest amplitude
        av_spks[i, :] = spk_single.mean(0)  # wf of averaged amplitude (channels)
    
    norm_spks = np.zeros([tot_spks, n_spk_samp])
    for i in range(tot_spks):
        norm_spks[i, :] = normalise(av_spks[i, :])  # normalisation
    
    av_spk = norm_spks.mean(0)  # 1d vector for the average tagged spike wf
    
    # sem calculation
    spk_sem = np.zeros(32)
    
    for i in range(32):
        spk_sem[i] = sem(norm_spks[:, i])
        
    return av_spk, spk_sem;


#%% MAIN
for pathname in pathHPC:    
    file_stem = pathname
    loc_A = file_stem.rfind('A')
    file_name = file_stem + '\\' + file_stem[loc_A:loc_A+17]
    
    # only process if there exist not already the .npy
    if os.path.isfile('Z:\Dinghao\code_dinghao\HPC_by_sess'+file_name[42:60]+'_avg_spk.npy')==True:
        # header
        print('\n\nProcessing {}'.format(pathname))
    
        # load .mat
        mat_BTDT = sio.loadmat(file_name + 'BTDT.mat')
        behEvents = mat_BTDT['behEventsTdt']
        spInfo = sio.loadmat(file_name + '_DataStructure_mazeSection1_TrialType1_SpInfo_Run0')
        spike_rate = spInfo['spatialInfoSess'][0][0]['meanFR'][0][0][0]
        
        # global vars
        n_chan = 10  # need to change if using other probes
        n_shank = 6  # same as above
        n_spk_samp = 32  # arbitrary, equals to 1.6ms, default window in kilosort
        
        tot_clus = 0
        avg_spk_dict = {}
        avg_sem_dict = {}
        for i in range(n_shank):
            curr_shank = i+1; print('shank {}...'.format(curr_shank))
            clu = param2array(file_name + '.clu.{}'.format(curr_shank))  # load .clu
            res = param2array(file_name + '.res.{}'.format(curr_shank))  # load .res
            
            clu = np.delete(clu, 0)  # delete 1st element (noise clusters)
            all_clus = np.delete(np.unique(clu), [0, 1])
            all_clus = np.array([int(x) for x in all_clus])
            all_clus = all_clus[all_clus>=2]
            shank_tot_clus = len(all_clus)
            
            fspk = open(file_name + '.spk.{}'.format(curr_shank), 'rb')  # load .spk into a byte bufferedreader
            
            for i in range(shank_tot_clus):
                nth_clu = i + 2
                av_spk, spk_sem = spk_w_sem(fspk, clu, nth_clu)
                
            avg_spk_dict['{} {}'.format(curr_shank, nth_clu)] = av_spk
            avg_sem_dict['{} {}'.format(curr_shank, nth_clu)] = spk_sem
        
        np.save('Z:\Dinghao\code_dinghao\HPC_by_sess'+file_name[42:60]+'_avg_spk.npy', avg_spk_dict)
        np.save('Z:\Dinghao\code_dinghao\HPC_by_sess'+file_name[42:60]+'_avg_sem.npy', avg_sem_dict)