# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:43:47 2022
update:
    25 Jan 2023, input rec_list
    
criterion: line 166, 0.33 and <20 Hz

saves the average and tagged waveforms of all recordings in rec_list, pathLC
@author: Dinghao Luo
"""


#%% imports
import sys
import os
import numpy as np
from random import sample
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as plr
from scipy.stats import sem


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
def spk_w_sem(fspk, clu, nth_clu, spikes_to_load=[]):
    
    clu_n_id = [int(x) for x in np.transpose(get_clu(nth_clu, clu))]  # ID of every single spike of clu
    
    rnd_samp_size = 1000
    if spikes_to_load==[]:
        if len(clu_n_id)<rnd_samp_size:
            tot_spks = len(clu_n_id)
        else:
            clu_n_id = sample(clu_n_id, rnd_samp_size)
            tot_spks = len(clu_n_id)
    else:
        clu_n_id = spikes_to_load

    # load spikes (could cut out to be a separate function)
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
        
        if i in ind_eg and spikes_to_load!=[]:
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
    
    # processing only begins if there exist not *already* the session .npy files
    # if os.path.isfile('Z:\Dinghao\code_dinghao\LC_tagged_by_sess'+file_name[42:60]+'_tagged_spk.npy')==False:
    if 1==1:
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
    
        # tagged
        stim_tp = np.zeros([60, 1])  # hard-coded for LC stim protocol
        if file_stem=='Z:\Dinghao\MiceExp\ANMD060r\A060r-20230530\A060r-20230530-02':
            stim_tp = np.zeros([120, 1]) 
        tag_id = 0
        for i in range(behEvents['stimPulse'][0, 0].shape[0]):
            i = int(i)
            if behEvents['stimPulse'][0, 0][i, 3]<10:  # ~5ms tagged pulses
                # temp = (behEvents['stimPulse'][0, 0][i, 0] 
                #         + (behEvents['stimPulse'][0, 0][i, 1])/10000000)  # pulse time with highest precision
                temp = behEvents['stimPulse'][0, 0][i, 0]
                # temp_s = round(temp/20000, 4)  # f[sampling] = 20kHz
                # print('%s%s%s%s%s%s%s' % ('pulse ', i+1, ': ', temp_s, 's (', temp, ')'))  # print pulse time
                stim_tp[tag_id] = temp
                tag_id += 1
        if tag_id not in [60, 120] : raise Exception('not enough tag pulses (expected 60 or 120)')
        
        tag_rate = np.zeros(tot_clus)
        if_tagged_spks = np.zeros([tot_clus, 60])
        tagged = np.zeros([tot_clus, 2])
        
        for iclu in range(tot_clus):
            nth_clu = iclu + 2
            clu_n_id = np.transpose(get_clu(nth_clu, clu))
            
            tagged[iclu, 0] = nth_clu
            
            for i in range(60):  # hard-coded
                t_0 = stim_tp[i, 0]  # stim time point
                t_1 = stim_tp[i, 0] + 200  # stim time point +10ms (stricter than Takeuchi et al.)
                spks_in_range = filter(lambda x: (int(res[x])>=t_0) and (int(res[x])<=t_1), clu_n_id)
                try:
                    if_tagged_spks[iclu, i] = next(spks_in_range)  # 1st spike in range
                except StopIteration:
                    pass
            tag_rate[iclu] = round(len([x for x in if_tagged_spks[iclu, :] if x > 0])/len(if_tagged_spks[iclu, :]), 3)
            
            # spike rate upper bound added 26 Jan 2023 to filter out non-principal cells 
            if tag_rate[iclu] > .33 and spike_rate[iclu] < 20:
                tagged[iclu, 1] = 1
                print('%s%s%s%s%s' % ('clu ', nth_clu, ' tag rate = ', tag_rate[iclu], ', tagged'))
            else:
                print('%s%s%s%s' % ('clu ', nth_clu, ' tag rate = ', tag_rate[iclu]))
        
        tot_tagged = sum(tagged[:, 1])
        
        tagged_spk_dict = {}
        tagged_sem_dict = {}
        tagged_eg_spk_dict = {}
        for iclu in range(tot_clus):
            if tagged[iclu, 1] == 1:
                tagged_clu = int(tagged[iclu, 0])
                tagged_spikes = if_tagged_spks[tagged_clu-2, :]
                tagged_spikes = [int(x) for x in tagged_spikes]
                tagged_spikes = [spike for spike in tagged_spikes if spike!=0]
                tagged_spk, tagged_sem, tagged_eg_spks = spk_w_sem(fspk, clu, tagged_clu, tagged_spikes)
                
                tagged_spk_dict[str(tagged_clu)] = tagged_spk
                tagged_sem_dict[str(tagged_clu)] = tagged_sem
        
        
        #---plotting---#
        time_ax = [x*50 for x in range(n_spk_samp)]  # time ticks in *u*s
        
        tot_plots = tot_clus
        col_plots = 4
        
        row_plots = tot_plots // col_plots
        if tot_plots % col_plots != 0:
            row_plots += 1
            
        plr.rcParams['figure.figsize'] = (4*2, row_plots*2.5)
        
        plot_pos = np.arange(1, tot_plots+1)
        
        fig = plt.figure(1)
        
        avg_spk_dict = {}
        avg_sem_dict = {}
        avg_eg_spk_dict = {}
        
        for i in range(tot_clus):
            nth_clu = i + 2
            av_spk, spk_sem, eg_spks = spk_w_sem(fspk, clu, nth_clu)
            
            avg_spk_dict[str(nth_clu)] = av_spk
            avg_sem_dict[str(nth_clu)] = spk_sem
            avg_eg_spk_dict[str(nth_clu)] = eg_spks
            
            ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
            ax.set_title('%s%s' % ('clu ', nth_clu), fontsize = 10)
            ax.plot(time_ax, av_spk)
            ax.fill_between(time_ax, av_spk+spk_sem, av_spk-spk_sem, color='lightblue')
            for tgd in list(tagged_spk_dict.keys()):
                tgd_clu = int(tgd)
                if nth_clu == tgd_clu:
                    ax.plot(time_ax, tagged_spk_dict[tgd], color='k')
                    ax.fill_between(time_ax, tagged_spk_dict[tgd]+tagged_sem_dict[tgd],
                                    tagged_spk_dict[tgd]-tagged_sem_dict[tgd], 
                                    color='grey')
            ax.axis('off')
        
        plt.subplots_adjust(hspace = 0.4)
        plt.show()
        
        out_directory = r'Z:\Dinghao\code_dinghao\LC_tagged_by_sess'
        fig.savefig(out_directory + '/' + file_name[42:60] + '_waveforms' + '.png')
        
        np.save('Z:\Dinghao\code_dinghao\LC_tagged_by_sess'+file_name[42:60]+'_tagged_spk.npy', tagged_spk_dict)
        np.save('Z:\Dinghao\code_dinghao\LC_tagged_by_sess'+file_name[42:60]+'_tagged_sem.npy', tagged_sem_dict)
        np.save('Z:\Dinghao\code_dinghao\LC_tagged_by_sess'+file_name[42:60]+'_avg_spk.npy', avg_spk_dict)
        np.save('Z:\Dinghao\code_dinghao\LC_tagged_by_sess'+file_name[42:60]+'_avg_sem.npy', avg_sem_dict)