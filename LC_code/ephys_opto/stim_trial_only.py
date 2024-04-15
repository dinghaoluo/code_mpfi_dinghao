# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:43:56 2023

stim-trials only, aligned to run or rew

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio

aligned_type = 1  # default to alignedRun

if aligned_type==1:
    info = np.load(
        'Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
        allow_pickle=True).item()
elif aligned_type==2:
    info = np.load(
        'Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info_alignedRew.npy',
        allow_pickle=True).item()


#%% MAIN (which is simply plotting)
# filter out clu's in non-opto recordings
tot_clu = len(info)

for clu in range(tot_clu):
    curr_clu = list(info)[clu]
    curr_clu_name = curr_clu[0]
    
    root = 'Z:\Dinghao\MiceExp'
    fullpath = root+'\ANMD'+curr_clu_name[1:5]+'\\'+curr_clu_name[:14]+'\\'+curr_clu_name[:17]
    beh_info_file = sio.loadmat(fullpath+'\\'+curr_clu_name[:17]+
                                '_DataStructure_mazeSection1_TrialType1_Info.mat')['beh']
    stim_trial = np.squeeze(np.where(beh_info_file['pulseMethod'][0][0][0]!=0))
    
    if stim_trial.size==0:
        del info[curr_clu_name]


# plotting
print('\nplotting heatmaps of stim-trials only...')
tot_plots = tot_clu
col_plots = 5
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

length_for_disp = 13750  # how many samples (in total) to display for imshow
nclu = 0

for clu in range(tot_clu):
    if nclu % 5 == 0:
        print('plotting... {}%'.
              format(np.round(nclu/tot_clu, 2)))
    
    # reshape data for plotting (imshow)
    curr_clu = list(info.items())[clu]
    curr_clu_name = curr_clu[0]
    n_spike = curr_clu[1]
    
    # display only up to length_for_disp
    spike_trunc = np.zeros((len(ind_good_beh), length_for_disp))
    spike_trunc_norm = np.zeros((len(ind_good_beh), length_for_disp))

    j=0
    for trial in ind_good_beh:
        curr_length = len(n_spike[trial])
        if curr_length <= length_for_disp:
            spike_trunc[j, :curr_length] = n_spike[trial][:]
            spike_trunc_norm[j, :curr_length] = normalise(n_spike[trial][:])
        else:
            spike_trunc[j, :] = n_spike[trial][:length_for_disp]
            spike_trunc_norm[j, :] = normalise(n_spike[trial][:length_for_disp])
        j+=1
    
    fig = plt.figure(1, figsize=[4*4, row_plots*2.5]); fig.tight_layout()
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
    ax.set_title(curr_clu_name[-22:], fontsize = 10)
    clu_im = ax.imshow(spike_trunc_norm, aspect='auto', cmap='jet',
                       extent=[-3750, length_for_disp-3750, 
                               spike_trunc_norm.shape[0]+1, 1])
    fig.colorbar(clu_im, ax=ax)
    
    nclu+=1
    
plt.subplots_adjust(hspace = 0.5)
plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
fig.savefig(out_directory + '\\'+'LC_all_tagged_spiketrains_(alignedRew)_nonOpto.png')