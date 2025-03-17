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
import mat73
import os 
import matplotlib.pyplot as plt
from scipy.stats import sem

sys.path.append(r'Z:\Dinghao\code_dinghao\common')
from common import normalise, mpl_formatting
from param_to_array import param2array, get_clu
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC

# parameters 
NUM_CH = 32  # 32 channels on probe 
NUM_SPK_SAMP = 32  # 32 samples per spikes
NUM_EG_SPK = 100  # how many example spks to store
NUM_RAND_SAMP = 1000  # how many random spikes to read for waveforms 
NUM_TAG_PULSE = 60  # 60 pulses for tagging 
TAG_THRESHOLD = .33


#%% main function
def get_ccgs(pathname: str) -> np.ndarray:
    """
    Retrieves the cross-correlogram (CCG) for a specified cell from a MATLAB file.

    Parameters:
    - pathname (str)

    Returns:
    - np.ndarray: The autocorrelogram for the specified cell, extracted as ACGval[:, cell_index, cell_index]
      from the 'CCGSess' structure in the MATLAB file.
    """
    ccg_file = mat73.loadmat(
        rf'{pathname}\{pathname[-17:]}_DataStructure_mazeSection1_TrialType1_CCG_Ctrl_Run0_mazeSess1.mat'
        )
        
    return ccg_file['CCGSessCtrl']['ccgVal']

def spk_w_sem(fspk, clu, nth_clu, 
              spikes_to_load=None, n_chan=NUM_CH, n_spk_samp=NUM_SPK_SAMP):
    
    # get ID of every single spike of clu
    clu_n_id = [int(x.item()) for x in np.transpose(get_clu(nth_clu, clu))]
    
    rnd_samp_size = NUM_RAND_SAMP
    if spikes_to_load is None:
        if len(clu_n_id)<rnd_samp_size:
            tot_spks = len(clu_n_id)
        else:
            clu_n_id = sample(clu_n_id, rnd_samp_size)
            tot_spks = len(clu_n_id)
    else:
        clu_n_id = spikes_to_load

    # load spikes (could cut out to be a separate function)
    tot_spks = len(clu_n_id)
    spks_wfs = []  # wfs for all spikes

    for i in range(tot_spks):  # reading .spk in binary ***might be finicky
        status = fspk.seek(clu_n_id[i]*n_chan*n_spk_samp*2)  # go to correct part of file
        if status == -1:
            raise Exception('Cannot go to the correct part of .spk')
    
        spk = fspk.read(2048)  # 32*32 bts for a spike, 32*32 bts for valence of each point
        spk_fmtd = np.zeros([n_chan, n_spk_samp])  # spk but formatted as a 32x32 matrix
        for j in range(n_spk_samp):
            for k in range(n_chan):
                try:
                    spk_fmtd[k, j] = spk[k*2+j*64]
                except IndexError:  # index out of range 
                    continue 
                if spk[k*2+j*64+1] == 255:  # byte following value signifies valence (255 means negative)
                    spk_fmtd[k, j] = spk_fmtd[k, j] - 256  # flip sign, negative values work as subtracting from 256
    
        spks_wfs.append(spk_fmtd)
    
    spks_wfs = np.asarray(spks_wfs)
    
    # average & max spike waveforms
    av_spks = np.zeros([tot_spks, n_spk_samp])
    max_spks = np.zeros([tot_spks, n_spk_samp])
    
    for i in range(tot_spks):
        spk_single = spks_wfs[i, :, :]
        spk_diff = np.zeros(n_chan)
        for j in range(n_chan):
            spk_diff[j] = np.amax(spk_single[j, :]) - np.amin(spk_single[j, :])
            spk_max = np.argmax(spk_diff)
        max_spks[i, :] = spk_single[spk_max, :]  # wf w/ highest amplitude
        av_spks[i, :] = np.nanmean(spk_single, axis=0)  # wf of averaged amplitude (channels)
    
    norm_spks = np.zeros([tot_spks, n_spk_samp])
    for i in range(tot_spks):
        norm_spks[i, :] = normalise(av_spks[i, :])  # normalisation
    
    av_spk = norm_spks.mean(0)  # 1d vector for the average tagged spike wf
    
    # sem calculation
    spk_sem = np.zeros(n_spk_samp)
    
    for i in range(n_spk_samp):
        spk_sem[i] = sem(norm_spks[:, i])
        
    return av_spk, spk_sem


#%% MAIN
def main():
    for pathname in paths:
        recname = pathname[-17:]
        print(f'\n\nProcessing {recname}')
        
        sess_folder = rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}'
        os.makedirs(sess_folder, exist_ok=True)
        
        # load .mat
        mat_BTDT = sio.loadmat(
            rf'{pathname}/{recname}BTDT.mat'
            )
        behEvents = mat_BTDT['behEventsTdt']
        spInfo = sio.loadmat(
            rf'{pathname}/{recname}_DataStructure_mazeSection1_TrialType1_SpInfo_Run0'
            )
        spike_rate = spInfo['spatialInfoSess'][0][0]['meanFR'][0][0][0]
        
        clu = param2array(
            rf'{pathname}/{recname}.clu.1'
            )  # load .clu
        res = param2array(
            rf'{pathname}/{recname}.res.1'
            )  # load .res
        
        # all_clu = [int(c) for c in np.unique(clu)
        #            if c not in ('', '0', '1') and
        #            len(np.where(clu==c)[0]) != 1]  
        # # edge case prevention: sometimes (e.g. in A032-20220726-02) a clu only has one spike 
        # tot_clu = len(all_clu)
        
        clu = np.delete(clu, 0)  # delete 1st element (noise clusters)
        all_clu = np.delete(np.unique(clu), [0, 1])
        all_clu = np.array([int(x) for x in all_clu])
        all_clu = all_clu[all_clu>=2]
        tot_clu = len(all_clu)
        
        with open(rf'{pathname}\{recname}.spk.1', 'rb') as fspk:  # load .spk into a byte bufferedreader
        # fspk = open(r'{}/{}.spk.1'.format(pathname, recname), 'rb')  
    
            # tagged
            stim_tp = np.zeros([NUM_TAG_PULSE, 1])  # hard-coded for LC stim protocol
            if recname=='A060r-20230530-02':
                stim_tp = np.zeros([120, 1])  # in this session there was an accidental extra 60 pulses 
            tag_id = 0
            for i in range(behEvents['stimPulse'][0, 0].shape[0]):
                i = int(i)
                if behEvents['stimPulse'][0, 0][i, 3]<10:  # ~5ms tagged pulses
                    temp = behEvents['stimPulse'][0, 0][i, 0]
                    stim_tp[tag_id] = temp
                    tag_id += 1
            if tag_id not in [NUM_TAG_PULSE, 120]:
                raise Exception('not enough tag pulses (expected 60 or 120)')
            
            tag_rate = np.zeros(tot_clu)
            if_tagged_spks = np.zeros([tot_clu, NUM_TAG_PULSE])
            tagged = np.zeros([tot_clu, 2])
            
            for iclu in range(tot_clu):
                nth_clu = iclu + 2
                clu_n_id = [int(x) for x in np.transpose(get_clu(nth_clu, clu))]
                
                tagged[iclu, 0] = nth_clu
                
                for i in range(NUM_TAG_PULSE):  # hard-coded
                    t_0 = stim_tp[i, 0]  # stim time point
                    t_1 = stim_tp[i, 0] + 200  # stim time point +10ms (stricter than Takeuchi et al.)
                    spks_in_range = (
                        x for x in clu_n_id if res[x].strip() and t_0 <= int(res[x]) <= t_1
                        )
                    try:
                        if_tagged_spks[iclu, i] = next(spks_in_range)  # 1st spike in range
                    except StopIteration:
                        pass
                tag_rate[iclu] = round(len([x for x in if_tagged_spks[iclu, :] if x > 0])/len(if_tagged_spks[iclu, :]), 3)
                
                # spike rate upper bound added 26 Jan 2023 to filter out non-principal cells 
                if tag_rate[iclu] > TAG_THRESHOLD and spike_rate[iclu] < 20:
                    tagged[iclu, 1] = 1
                    print('%s%s%s%s%s' % ('clu ', nth_clu, ' tag rate = ', tag_rate[iclu], ', tagged'))
                else:
                    print('%s%s%s%s' % ('clu ', nth_clu, ' tag rate = ', tag_rate[iclu]))
                    
            waveforms = []
            # print('plotting waveforms...')
            for iclu in range(tot_clu):
                if tagged[iclu, 1]:
                    tagged_clu = iclu+2
                    tagged_spikes = if_tagged_spks[tagged_clu-2, :]
                    tagged_spikes = [int(x) for x in tagged_spikes]
                    tagged_spikes = [spike for spike in tagged_spikes if spike!=0]
                    tagged_spk, tagged_sem = spk_w_sem(fspk, clu, tagged_clu, tagged_spikes)
                
                    spont_mean, spont_sem = spk_w_sem(fspk, clu, tagged_clu)  # be careful that here is tagged_clu, which is iclu+2
                    waveforms.append(spont_mean)
                    
                    # fig, axs = plt.subplots(1,2,figsize=(2.1,1.4))
                    # axs[0].plot(spont_mean, 'k')
                    # axs[1].plot(tagged_spk)
                    
                    # for i in range(2):
                    #     for s in ('top', 'right', 'bottom', 'left'):
                    #         axs[i].spines[s].set_visible(False)
                    #     axs[i].set(xticks=[], yticks=[])
                    
                    # fig.suptitle(f'{recname} clu{tagged_clu}')
                    # fig.tight_layout()
                    
                    # for ext in ('.png', '.pdf'):
                    #     fig.savefig(
                    #         r'Z:\Dinghao\code_dinghao\LC_ephys'
                    #         r'\single_cell_waveform'
                    #         rf'\{recname} clu{tagged_clu} tagged{ext}',
                    #         dpi=300,
                    #         bbox_inches='tight')
                    
                else:
                    nontagged_clu = iclu+2  # corresponds to tagged_clu above 
                    
                    spont_mean, spont_sem = spk_w_sem(fspk, clu, nontagged_clu)
                    waveforms.append(spont_mean)
                    
                    # fig, ax = plt.subplots(figsize=(1.3, 1.4))
                    # ax.plot(spont_mean, 'k')
                    
                    # for s in ('top', 'right', 'bottom', 'left'):
                    #     ax.spines[s].set_visible(False)
                    #     ax.set(xticks=[], yticks=[])
                    
                    # fig.suptitle(f'{recname} clu{nontagged_clu}')
                    # fig.tight_layout()
                    
                    # for ext in ('.png', '.pdf'):
                    #     fig.savefig(
                    #         r'Z:\Dinghao\code_dinghao\LC_ephys'
                    #         r'\single_cell_waveform'
                    #         rf'\{recname} clu{nontagged_clu}{ext}',
                    #         dpi=300,
                    #         bbox_inches='tight')
                
        # keys for saving dictionaries 
        keys = [f'{recname} clu{nth_clu}' for nth_clu in range(2, tot_clu+2)]
        
        # save waveforms
        waveforms_dict = {keys[i]: waveforms[i] for i in range(len(keys))}
        np.save(rf'{sess_folder}\{recname}_all_waveforms.npy',
                waveforms_dict)
        
        # get and save ACGs
        CCGs = get_ccgs(pathname)
        print(f'CCGs: {CCGs.shape[1]}')
        print(f'keys: {len(keys)}')
        ACGs_dict = {keys[i]: CCGs[:,i,i] for i in range(len(keys))}
        np.save(rf'{sess_folder}\{recname}_all_ACGs.npy',
                ACGs_dict)
        
        # save tagged identities
        tagged_dict = {keys[i]: int(tagged[i,1]) for i in range(len(keys))}  # tagged[n,0] is just the clu index
        np.save(rf'{sess_folder}\{recname}_all_identities.npy',
                tagged_dict)
        
if __name__ == '__main__':
    main()