# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:09:18 2023

save all rasters into a big .npy file

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import scipy.io as sio
import h5py
import sys

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC

if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')    


#%% create stim rasters 
all_rasters = {}; rasters = {}

for pathname in pathLC:
    print(pathname[-17:])
    
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    
    # check if this is an opto-stim session
    beh_info_file = sio.loadmat(filename+'_Info.mat')['beh']
    stim_trial = np.squeeze(np.where(beh_info_file['pulseMethod'][0][0][0]!=0))
    if stim_trial.size!=0:
        stimtype = beh_info_file['pulseMethod'][0][0][0][stim_trial[0]]
        stimwind = [stim_trial[0], stim_trial[-1]]
    else: 
        stimtype = 'NA'
        stimwind = ['NA', 'NA']
        
    spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']
    
    time_bef = spike_time_file['TimeBef']; time = spike_time_file['Time']
    tot_clu = time.shape[1]
    tot_trial = time.shape[0]  # trial 1 is empty but tot_trial includes it for now
    all_id = list(np.arange(2, tot_clu+2))
    
    # spike reading
    spike_time = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    spike_time_bef = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    for i in range(tot_clu):
        i = int(i)-2
        for j in range(1, tot_trial):  # trial 0 is an empty trial
            spike_time[i,j-1] = spike_time_file[time[j,i]][0]
            spike_time_bef[i,j-1] = spike_time_file[time_bef[j,i]][0]
    
    max_length = 10000  # 80 seconds
    spike_train_all = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    for i in range(tot_clu):
        for trial in range(tot_trial-1):
            clu_id = int(all_id[i])-2
            spikes = np.concatenate([spike_time_bef[clu_id][trial].reshape(-1),
                                     spike_time[clu_id][trial].reshape(-1)])
            spikes = [int(s+3750) for s in spikes]
            spikes = [s for s in spikes if s<max_length]
            spike_train_trial = np.zeros(max_length)
            spike_train_trial[spikes] = 1
            spike_train_all[i][trial] = spike_train_trial
    
    # check tagged cells
    tagged_id_file = np.load('Z:\Dinghao\code_dinghao\LC_tagged_by_sess'+
                             pathname[-18:]+'_tagged_spk.npy',
                             allow_pickle=True).item()
    tag_sess = list(tagged_id_file.keys())
    
    # save into all_rasters
    i = 0
    for clu in all_id:
        cluname = pathname[-17:]+' clu'+str(clu)+' '+str(stimtype)+' '+str(stimwind[0])+' '+str(stimwind[1])
        simp_cluname = pathname[-17:]+' clu'+str(clu)
        if str(clu) in tag_sess:
            cluname += ' tagged'
        all_rasters[cluname] = spike_train_all[i]
        rasters[simp_cluname] = spike_train_all[i]
        i+=1

# at this point all_rasters should have rasters of all tagged cells from only 
# the opto-stim sessions


#%% save all rasters
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters.npy', 
        all_rasters)
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy', 
        rasters)


#%%
# PLEASE DON'T PLOT THIS THING. IT IS TOO BIG TO PLOT AND THERE'S NO POINT IN 
# DOING SO ANYWAYS. BE STUBBORN AT YOUR OWN RISK.

# Dinghao, 24 May 2023


#%% plotting all rasters
# print('\nplotting rasters of all cells in opto-stim session...')
# tot_plots = len(all_rasters)  # how many clu's in total 
# col_plots = 5
# row_plots = tot_plots // col_plots
# if tot_plots % col_plots != 0:
#     row_plots += 1
# plot_pos = np.arange(1, tot_plots+1)
# fig = plt.figure(1, figsize=[5*4, row_plots*4])

# for i in range(tot_plots):
#     curr_clu = list(all_rasters.items())[i]
#     curr_clu_name = curr_clu[0]
#     print('plotting {}'.format(curr_clu_name))
#     curr_raster = curr_clu[1]
    
#     # # import bad beh trial id
#     # root = 'Z:\Dinghao\MiceExp'
#     # fullpath = root+'\ANMD'+curr_clu_name[1:5]+'\\'+curr_clu_name[:14]+'\\'+curr_clu_name[:17]
#     # beh_par_file = sio.loadmat(fullpath+'\\'+curr_clu_name[:17]+
#     #                            '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
#     #                                # -1 to account for MATLAB Python difference
#     # ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
#     #                                  # -1 to account for 0 being an empty trial
#     # ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
#     # ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
#     ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
#     ax.set(xlim=(-3.0, 5.0), xlabel='time (s)',
#                              ylabel='trial #')
#     ax.spines[['right', 'top']].set_visible(False)
    
#     divider1 = curr_clu_name.find(' ', curr_clu_name.find(' ')+1)  # find 2nd space
#     divider2 = divider1 + 2  # 3rd space
#     stimtype = curr_clu_name[divider1+1]  # stimtype after 2nd space
#     if curr_clu_name[-6:]=='tagged':
#         if stimtype=='4':
#             ax.set_title(curr_clu_name[:divider2]+' tagged', fontsize = 10, color='r')
#         elif stimtype=='3':
#             ax.set_title(curr_clu_name[:divider2]+' tagged', fontsize = 10, color='blue')
#         elif stimtype=='2':
#             ax.set_title(curr_clu_name[:divider2]+' tagged', fontsize = 10, color='orange')
#         else:
#             ax.set_title(curr_clu_name[:divider2]+' tagged', fontsize = 10, color='k')
#     else:
#         if stimtype=='4':
#             ax.set_title(curr_clu_name[:divider2], fontsize = 10, color='r')
#         elif stimtype=='3':
#             ax.set_title(curr_clu_name[:divider2], fontsize = 10, color='blue')
#         elif stimtype=='2':
#             ax.set_title(curr_clu_name[:divider2], fontsize = 10, color='orange')
#         else:
#             ax.set_title(curr_clu_name[:divider2], fontsize = 10, color='k')
        
#     tot_trial = curr_raster.shape[0]  # how many trials
#     for trial in range(tot_trial):
#         curr_trial = np.where(curr_raster[trial]==1)[0]
#         curr_trial = [(s-3750)/1250 for s in curr_trial]
#         ax.scatter(curr_trial, [trial+1]*len(curr_trial),
#                    color='grey', s=.35)