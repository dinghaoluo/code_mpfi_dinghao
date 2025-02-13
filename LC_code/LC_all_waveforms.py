# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:43:47 2022
update:
    25 Jan 2023, input rec_list
    11 Feb 2025, modified to include all cells
    
criterion: 0.33 response rate and <20 Hz

saves the average and tagged waveforms of all recordings in rec_list, pathLC
@author: Dinghao Luo
"""


#%% imports
import sys
import numpy as np
from random import sample
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import sem

sys.path.append(r'Z:\Dinghao\code_dinghao\common')
from common import normalise, mpl_formatting
from param_to_array import param2array, get_clu
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
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
        
    return av_spk, spk_sem


#%% MAIN
for pathname in pathLC:
    recname = pathname[-17:]
    print(recname)
    
    # processing only begins if there exist not *already* the session .npy files
    # if os.path.isfile('Z:\Dinghao\code_dinghao\LC_tagged_by_sess'+file_name[42:60]+'_tagged_spk.npy')==False:
    if 1==1:
        # header
        print('\n\nProcessing {}'.format(recname))
        
        # load .mat
        mat_BTDT = sio.loadmat(
            r'{}/{}BTDT.mat'
            .format(pathname, recname)
            )
        behEvents = mat_BTDT['behEventsTdt']
        spInfo = sio.loadmat(
            r'{}/{}_DataStructure_mazeSection1_TrialType1_SpInfo_Run0'
            .format(pathname, recname)
            )
        spike_rate = spInfo['spatialInfoSess'][0][0]['meanFR'][0][0][0]
        
        # global vars
        n_chan = 32  # need to change if using other probes
        n_spk_samp = 32  # arbitrary, equals to 1.6ms, default window in kilosort
        
        clu = param2array(r'{}/{}.clu.1'.format(pathname, recname))  # load .clu
        res = param2array(r'{}/{}.res.1'.format(pathname, recname))  # load .res
        
        clu = np.delete(clu, 0)  # delete 1st element (noise clusters)
        all_clus = np.delete(np.unique(clu), [0, 1])
        all_clus = np.array([int(x) for x in all_clus])
        all_clus = all_clus[all_clus>=2]
        tot_clus = len(all_clus)
        
        fspk = open(r'{}/{}.spk.1'.format(pathname, recname), 'rb')  # load .spk into a byte bufferedreader
    
        # tagged
        stim_tp = np.zeros([60, 1])  # hard-coded for LC stim protocol
        if recname=='A060r-20230530-02':
            stim_tp = np.zeros([120, 1]) 
        tag_id = 0
        for i in range(behEvents['stimPulse'][0, 0].shape[0]):
            i = int(i)
            if behEvents['stimPulse'][0, 0][i, 3]<10:  # ~5ms tagged pulses
                temp = behEvents['stimPulse'][0, 0][i, 0]
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
        
        for iclu in range(tot_clus):
            if tagged[iclu, 1]:
                tagged_clu = int(tagged[iclu,0])
                tagged_spikes = if_tagged_spks[tagged_clu-2, :]
                tagged_spikes = [int(x) for x in tagged_spikes]
                tagged_spikes = [spike for spike in tagged_spikes if spike!=0]
                tagged_spk, tagged_sem = spk_w_sem(fspk, clu, tagged_clu, tagged_spikes)
            
                spont_mean, spont_sem = spk_w_sem(fspk, clu, iclu)
                
                fig, axs = plt.subplots(1,2,figsize=(2.1,1.4))
                axs[0].plot(spont_mean, 'k')
                axs[1].plot(tagged_spk)
                
                for i in range(2):
                    for s in ('top', 'right', 'bottom', 'left'):
                        axs[i].spines[s].set_visible(False)
                    axs[i].set(xticks=[], yticks=[])
                
                fig.suptitle(f'{recname} clu{iclu}')
                fig.tight_layout()
                
                for ext in ['.png', '.pdf']:
                    fig.savefig(
                        r'Z:\Dinghao\code_dinghao\LC_ephys\single_cell_waveform\{} clu{} tagged{}'
                        .format(recname, iclu+2, ext))
                
            else:
                spont_mean, spont_sem = spk_w_sem(fspk, clu, iclu)
                tagged_clu = False
            
                fig, ax = plt.subplots(figsize=(1.3, 1.4))
                ax.plot(spont_mean, 'k')
                
                for s in ('top', 'right', 'bottom', 'left'):
                    ax.spines[s].set_visible(False)
                    ax.set(xticks=[], yticks=[])
                
                fig.suptitle(f'{recname} clu{iclu+2}')
                fig.tight_layout()
                
                for ext in ['.png', '.pdf']:
                    fig.savefig(
                        r'Z:\Dinghao\code_dinghao\LC_ephys\single_cell_waveform\{} clu{}{}'
                        .format(recname, iclu+2, ext))