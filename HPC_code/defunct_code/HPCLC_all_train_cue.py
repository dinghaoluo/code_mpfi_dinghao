# -*- coding: utf-8 -*-
"""
Created on Wed 20 Dec 16:03:11 2023

pool all cells from all recording sessions aligned to cue

*contains interneurones*

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import sys
import h5py
import mat73
import os

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% MAIN 
for pathname in pathHPC[28:]:
    all_info = {}
    
    recname = pathname[-17:]
    print(recname)
    
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    BehavLFP = mat73.loadmat('{}.mat'.format(pathname+pathname[-18:]+'_BehavElectrDataLFP'))
    Clu = BehavLFP['Clu']
    shank = Clu['shank']
    localClu = Clu['localClu']
    
    spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsCueSpikes']
    
    time_bef = spike_time_file['TimeBef']; time = spike_time_file['Time']
    tot_clu = time.shape[1]
    tot_trial = time.shape[0]  # trial 1 is empty but tot_trial includes it for now
    
    samp_freq = 1250  # Hz
    gx_spike = np.arange(-500, 500, 1)
    sigma_spike = samp_freq/10
    
    # spike reading
    spike_time = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    spike_time_bef = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    for i in range(tot_clu):
        for j in range(1, tot_trial):  # trial 0 is an empty trial
            spike_time[i,j-1] = spike_time_file[time[j,i]][0]
            spike_time_bef[i,j-1] = spike_time_file[time_bef[j,i]][0]
    spike_train_all = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    spike_train_conv = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    conv_spike = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    gaus_spike = [1 / (sigma_spike*np.sqrt(2*np.pi)) * 
                  np.exp(-x**2/(2*sigma_spike**2)) for x in gx_spike]
    for i in range(tot_clu):
        for trial in range(tot_trial-1):
            spikes = np.concatenate([spike_time_bef[i][trial].reshape(-1),
                                     spike_time[i][trial].reshape(-1)])
            spikes = [int(s+3750) for s in spikes if s<-1 or s>1]
            if len(spikes)>0:
                spike_train_trial = np.zeros(spikes[-1]+1)
                spike_train_trial[spikes] = 1
                spike_train_all[i][trial] = spike_train_trial
                spike_train_conv[i][trial] = np.convolve(spike_train_trial, 
                                                         gaus_spike, mode='same')
                # norm_spike[i][trial] = normalise(spike_train_conv[i][trial])
                conv_spike[i][trial] = spike_train_conv[i][trial]
            else:
                conv_spike[i][trial] = np.zeros(12500)  # default to 10 s of emptiness if no spikes in this trial
    
    i = 0
    for clu in range(tot_clu):
        clu_name = '{} clu{} {} {}'.format(pathname[-17:], clu+2, int(shank[clu]), int(localClu[clu]))
        clu_conv_spike = conv_spike[i]
        
        all_info[clu_name] = clu_conv_spike
        i+=1
    
    outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
    if not os.path.exists(outdirroot):
        os.makedirs(outdirroot)
    np.save('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
            all_info)
    print('processed and saved to Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy\n'.format(recname, recname))
    

#%% plot heatmaps for single cells 
# tot_plots = len(all_info)  # total number of clusters
# # col_plots = 10
# # row_plots = tot_plots // col_plots
# # if tot_plots % col_plots != 0:
# #     row_plots += 1
# # plot_pos = np.arange(1, tot_plots+1)

# for i in range(tot_plots):
#     curr_clu_name = list(all_info.items())[i][0]
#     curr_clu = all_info[curr_clu_name]
#     tot_trial = curr_clu.shape[0]
    
#     # put them into a image matrix
#     im_mat = np.zeros((tot_trial, 1250+5000))
#     for trial in range(tot_trial):
#         curr_trial_len = len(curr_clu[trial][2500:8750])
#         if curr_trial_len>=6250:
#             im_mat[trial] = normalise(curr_clu[trial][2500:8750])
#         else:
#             im_mat[trial][:curr_trial_len] = normalise(curr_clu[trial][2500:8750])
    
#     xaxis = np.arange(-1250, 5000)/samp_freq 
    
#     fig, ax = plt.subplots(figsize=(4,4))
#     ax.set_title(curr_clu_name, fontsize = 10)
#     ax.set(xlabel='time (s)')
#     ax.imshow(im_mat,
#               aspect='auto',
#               extent=[-1, 3, 
#                       1, tot_trial])
    
#     plt.show()

#     fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_heatmaps\{}.png'.format(curr_clu_name),
#                 dpi=300,
#                 bbox_inches='tight')


#%% plotting 
# print('plotting average spike rate profiles...')

# micepath = 'Z:\Dinghao\MiceExp\ANMD'

# length_for_disp = 13750  # how many samples (in total) to display
# spike_avg = {}
# spike_sem = {}

# for i in range(len(all_info)):
#     curr_clu = list(all_info.items())[i]
#     curr_clu_name = curr_clu[0]
    
#     # modified 25 Apr 2023, run average only on BASELINE and RECOVERY trials
#     # because it does not make sense to average these 2 subsessions with 
#     # STIMULATION
#     # import opto trial id
#     beh_info_file = sio.loadmat(micepath+curr_clu_name[1:5]+'\\'+curr_clu_name[:14]+
#                                 '\\'+curr_clu_name[:17]+'\\'+curr_clu_name[:17]+
#                                 '_DataStructure_mazeSection1_TrialType1_Info.mat')['beh']
#     stim_trial = np.squeeze(np.where(beh_info_file['pulseMethod'][0][0][0]!=0))
#     if stim_trial.size!=0:  # if there is opto
#         stim_window = [stim_trial[0], stim_trial[-1]]  # first and last stim trials
#         base_clu = np.delete(curr_clu[1], np.arange(stim_window[0], stim_window[1]))
#         n_spike = base_clu
#     else:  # if there is no opto (pulseMethod is all 0 in this case)
#         n_spike = curr_clu[1]
    
#     # display only up to length_for_disp
#     spike_trunc = np.zeros((len(n_spike), length_for_disp))
#     spike_trunc_norm = np.zeros((len(n_spike), length_for_disp))
    
#     for trial in range(n_spike.shape[0]):
#         curr_length = len(n_spike[trial])
#         if curr_length <= length_for_disp:
#             spike_trunc[trial, :curr_length] = n_spike[trial][:]
#             spike_trunc_norm[trial, :curr_length] = normalise(n_spike[trial][:])
#         else:
#             spike_trunc[trial, :] = n_spike[trial][:length_for_disp]
#             spike_trunc_norm[trial, :] = normalise(n_spike[trial][:length_for_disp])
#     spike_avg[curr_clu_name] = np.mean(spike_trunc, 0)*1250
#     spike_sem[curr_clu_name] = sem(spike_trunc, 0)*1250

# HPC_all_avg_sem = {
#     'all avg': spike_avg,
#     'all sem': spike_sem
#     }

# tot_plots = len(all_info)  # total number of clusters
# col_plots = 10
# row_plots = tot_plots // col_plots
# if tot_plots % col_plots != 0:
#     row_plots += 1
# plot_pos = np.arange(1, tot_plots+1)

# fig = plt.figure(1, figsize=[col_plots*4, row_plots*2.85]); fig.tight_layout()

# list_avg = list(spike_avg.values())
# list_sem = list(spike_sem.values())
# for j in range(len(all_info)):
#     curr_clu = list(all_info.items())[j]
#     curr_clu_name = curr_clu[0]
    
#     curr_min = min(list_avg[j][:10000])
#     curr_max = max(list_avg[j][:10000])
    
#     xaxis = np.arange(-1250, 5000)/samp_freq 
    
#     ax = fig.add_subplot(row_plots, col_plots, plot_pos[j])
#     ax.set_title(curr_clu_name[-22:], fontsize = 10)
#     ax.set(xlabel='time (s)', ylabel='spike rate (Hz)',
#            ylim=(curr_min*0.75, curr_max*1.5))
#     clu_avg = ax.plot(xaxis, list_avg[j][2500:8750], color='grey')
#     clu_sem = ax.fill_between(xaxis, list_avg[j][2500:8750]+list_sem[j][2500:8750],
#                                      list_avg[j][2500:8750]-list_sem[j][2500:8750],
#                                      color='grey', alpha=.25)
#     ax.vlines(0, 0, 100, color='grey', alpha=.1)

# plt.subplots_adjust(hspace = 0.5)
# plt.show()

# fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_spikeavg_(alignedRun).png',
#             dpi=300,
#             bbox_inches='tight')


# #%% save 
# np.save('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_avg_sem.npy', 
#         HPC_all_avg_sem)


# #%% heatmap
# fig = plt.figure(1, figsize=[col_plots*4, row_plots*2.85]); fig.tight_layout()

# for j in range(len(all_info)):
#     curr_clu = list(all_info.items())[j]
#     curr_clu_name = curr_clu[0]
#     curr_clu_im = normalise(list_avg[j][2500:8750]).reshape((1, 6250))
    
#     xaxis = np.arange(-1250, 5000)/samp_freq 
    
#     ax = fig.add_subplot(row_plots, col_plots, plot_pos[j])
#     ax.set_title(curr_clu_name[-22:], fontsize = 10) 
#     ax.set(xlabel='time (s)')
#     clu_avg_im = ax.imshow(curr_clu_im, aspect='auto')

# plt.subplots_adjust(hspace = 0.5)
# plt.show()

# fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_spikeavgim_(alignedRun).png',
#             dpi=300,
#             bbox_inches='tight')