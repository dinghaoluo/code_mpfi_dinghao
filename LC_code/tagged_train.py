# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:32:04 2023

plot (all and avg) and save all spike trains for tagged cells

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats import sem
import h5py
import sys 

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise

if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')


#%% MAIN
all_tagged_info = {}

for pathname in pathLC:
    print(pathname[-17:])
    
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    speed_time_file = sio.loadmat(filename + '_alignRun_msess1.mat')
    spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']
    tagged_id_file = np.load('Z:\Dinghao\code_dinghao\LC_tagged_by_sess'+
                             pathname[-18:]+'_tagged_spk.npy',
                             allow_pickle=True).item()
    
    time_bef = spike_time_file['TimeBef']; time = spike_time_file['Time']
    tag_sess = list(tagged_id_file.keys())
    # weird error when reading h5py, need to find source of this 30 Jan 2023
    if filename=='Z:\Dinghao\MiceExp\ANMD032r\A032r-20220726\A032r-20220726-03\A032r-20220726-03_DataStructure_mazeSection1_TrialType1':
        tag_sess = [x for x in tag_sess if x!='8' and x!='9']
    tot_clu = time.shape[1]
    tot_trial = time.shape[0]  # trial 1 is empty but tot_trial includes it for now
    
    samp_freq = 1250  # Hz
    gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
    sigma_speed = 12.5
    gx_spike = np.arange(-500, 500, 1)
    sigma_spike = 125
    
    # speed of all trials
    speed_time_bef = speed_time_file['trialsRun'][0]['speed_MMsecBef'][0][0][1:]
    speed_time = speed_time_file['trialsRun'][0]['speed_MMsec'][0][0][1:]
    gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
                  np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]
    # concatenate bef and after running onset, and convolve with gaus_speed
    speed_time_all = np.empty(shape=speed_time.shape[0], dtype='object')
    for i in range(speed_time.shape[0]):
        bef = speed_time_bef[i]; aft =speed_time[i]
        speed_time_all[i] = np.concatenate([bef, aft])
        speed_time_all[i][speed_time_all[i]<0] = 0
    speed_time_conv = [np.convolve(np.squeeze(single), gaus_speed)[50:-49] 
                       for single in speed_time_all]
    # norm_speed = np.array([normalise(s) for s in speed_time_conv])
    
    # trial length for equal length deployment (use speed trial length)
    trial_length = [trial.shape[0] for trial in speed_time_conv]
    
    # spike reading
    spike_time = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    spike_time_bef = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
    for i in tag_sess:
        i = int(i)-2
        for j in range(1, tot_trial):  # trial 0 is an empty trial
            spike_time[i,j-1] = spike_time_file[time[j,i]][0]
            spike_time_bef[i,j-1] = spike_time_file[time_bef[j,i]][0]
    spike_train_all = np.empty(shape=(len(tag_sess), tot_trial - 1), dtype='object')
    spike_train_conv = np.empty(shape=(len(tag_sess), tot_trial - 1), dtype='object')
    norm_spike = np.empty(shape=(len(tag_sess), tot_trial - 1), dtype='object')
    gaus_spike = [1 / (sigma_spike*np.sqrt(2*np.pi)) * 
                  np.exp(-x**2/(2*sigma_spike**2)) for x in gx_spike]
    for i in range(len(tag_sess)):
        for trial in range(tot_trial-1):
            clu_id = int(tag_sess[i])-2
            spikes = np.concatenate([spike_time_bef[clu_id][trial].reshape(-1),
                                     spike_time[clu_id][trial].reshape(-1)])
            spikes = [int(s+3750) for s in spikes]
            spike_train_trial = np.zeros(trial_length[trial])
            spike_train_trial[spikes] = 1
            spike_train_all[i][trial] = spike_train_trial
            spike_train_conv[i][trial] = np.convolve(spike_train_trial, 
                                                     gaus_spike, mode='same')
            # norm_spike[i][trial] = normalise(spike_train_conv[i][trial])
            norm_spike[i][trial] = spike_train_conv[i][trial]
    
    i = 0
    for clu in tag_sess:
        clu_name = pathname[-17:]+' clu'+clu
        clu_norm_spike = norm_spike[i]
        # clu_norm_speed = norm_speed
        
        # all_tagged_info[clu_name] = np.row_stack((clu_norm_spike,
        #                                           clu_norm_speed))
        all_tagged_info[clu_name] = clu_norm_spike
        i+=1
    
    
#%% save 1
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_info.npy', 
        all_tagged_info)
print('\nprocessed and saved to Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_info.npy')
    

#%% plotting all heatmaps (good trials only)
print('\nplotting heatmaps of all good trials...')
tot_plots = len(all_tagged_info)  # total number of clusters + speed
col_plots = 5
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

max_trial_length = np.zeros(len(all_tagged_info))
length_for_disp = 10000  # how many samples (in total) to display for imshow

for i in range(len(all_tagged_info)):
    # reshape data for plotting (imshow)
    curr_clu = list(all_tagged_info.items())[i]
    curr_clu_name = curr_clu[0]
    n_speed = curr_clu[1][1]
    n_spike = curr_clu[1][0]
    
    # import bad beh trial id
    root = 'Z:\Dinghao\MiceExp'
    fullpath = root+'\ANMD'+curr_clu_name[1:5]+'\\'+curr_clu_name[:14]+'\\'+curr_clu_name[:17]
    beh_par_file = sio.loadmat(fullpath+'\\'+curr_clu_name[:17]+
                               '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference
    ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                     # -1 to account for 0 being an empty trial
    ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
    # display only up to length_for_disp
    spike_trunc = np.zeros((len(ind_good_beh), length_for_disp))
    spike_trunc_norm = np.zeros((len(ind_good_beh), length_for_disp))
    
    j=0
    for trial in ind_good_beh:
        # curr_length = len(n_speed[trial])
        # if curr_length <= length_for_disp:
        #     spike_trunc[j, :curr_length] = n_spike[trial][:]
        #     spike_trunc_norm[j, :curr_length] = normalise(n_spike[trial][:])
        # else:
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
    
plt.subplots_adjust(hspace = 0.5)
plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
fig.savefig(out_directory + '\\'+'LC_all_tagged_spiketrains_(alignedRun).png')


#%% plotting all heatmaps (good and non-stim trials only)
print('\nplotting heatmaps of all good and non-opto trials...')
tot_plots = len(all_tagged_info)  # total number of clusters
col_plots = 5
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

max_trial_length = np.zeros(len(all_tagged_info))
length_for_disp = 13750  # how many samples (in total) to display for imshow

for i in range(len(all_tagged_info)):
    # reshape data for plotting (imshow)
    curr_clu = list(all_tagged_info.items())[i]
    curr_clu_name = curr_clu[0]
    n_spike = curr_clu[1]
    
    # import bad beh trial id
    root = 'Z:\Dinghao\MiceExp'
    fullpath = root+'\ANMD'+curr_clu_name[1:5]+'\\'+curr_clu_name[:14]+'\\'+curr_clu_name[:17]
    beh_par_file = sio.loadmat(fullpath+'\\'+curr_clu_name[:17]+
                               '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference
    ind_bad_beh = np.squeeze(np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1)
                                     # -1 to account for 0 being an empty trial
    ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    
    # import opto trial id
    beh_info_file = sio.loadmat(fullpath+'\\'+curr_clu_name[:17]+
                                '_DataStructure_mazeSection1_TrialType1_Info.mat')['beh']
    stim_trial = np.squeeze(np.where(beh_info_file['pulseMethod'][0][0][0]!=0))
    for i in range(len(stim_trial)):
        trial = stim_trial[i]
        if beh_info_file['pulseMethod'][0][0][0][trial]==4:
            stim_trial[i]+=1  # if rew-stim, don't show the next trial
            
    # concat and delete 
    del_trial = np.concatenate((ind_bad_beh, stim_trial))
    del_trial = np.unique(del_trial)
    ind_good_beh = np.delete(ind_good_beh, del_trial)
    
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
    
plt.subplots_adjust(hspace = 0.5)
plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
fig.savefig(out_directory + '\\'+'LC_all_tagged_spiketrains_(alignedRun)_nonOpto.png')


#%% extract averaged firing from all_train
spike_avg = {}
spike_sem = {}

all_avg_sem = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_avg_sem.npy',
                      allow_pickle=True).item()
all_avg = all_avg_sem['all avg']; all_sem = all_avg_sem['all sem']
tot_clu = len(all_avg)
all_clu_name = list(all_avg.keys())
tag_clu_name = list(all_tagged_info.keys())

for clu in range(tot_clu):
    if all_clu_name[clu] in tag_clu_name:
        spike_avg[all_clu_name[clu]] = all_avg[all_clu_name[clu]]
        spike_sem[all_clu_name[clu]] = all_sem[all_clu_name[clu]]


#%% save 2
# GOOD AND NON-OPTO TRIALS ONLY
LC_all_tagged_avg_sem = {
    'all tagged avg': spike_avg,
    'all tagged sem': spike_sem}
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_avg_sem.npy', 
        LC_all_tagged_avg_sem)


#%% plot avg spike rate profiles (good trials only)
print('\nplotting avg spike rate profiles (good trials only)...')

avg_list = list(spike_avg.values())
sem_list = list(spike_sem.values())
for j in range(len(all_tagged_info)):
    curr_clu = list(all_tagged_info.items())[j]
    curr_clu_name = curr_clu[0]
    curr_max = max(avg_list[j][:10000])

    fig = plt.figure(1, figsize=[4*4, row_plots*2.5]); fig.tight_layout()
    xaxis = np.arange(-1250, 5000, 1)/samp_freq 

    ax = fig.add_subplot(row_plots, col_plots, plot_pos[j])
    ax.set_title(curr_clu_name[-22:], fontsize = 10)
    ax.set(xlabel='time (s)', ylabel='spike rate (Hz)',
           ylim=(0, curr_max*1.5))
    clu_avg = ax.plot(xaxis, avg_list[j][2500:8750], color='grey')
    clu_sem = ax.fill_between(xaxis, avg_list[j][2500:8750]+sem_list[j][2500:8750],
                                     avg_list[j][2500:8750]-sem_list[j][2500:8750],
                                     color='grey', alpha=.25)
    ax.vlines(0, 0, 20, color='grey', alpha=.1)

plt.subplots_adjust(hspace = 0.5)
plt.show()

fig.savefig(out_directory + '\\'+'LC_all_tagged_spikeavg_(alignedRun)_grey.png')


#%% heatmap 
# print('\nplotting avg spike rate heatmap (pooled)...')

# clust = list(np.load('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_clustered_bc.npy',
#                      allow_pickle=True).item().values())

# pooled = np.zeros([len(all_tagged_info), 10000])
# ordered = np.zeros([len(all_tagged_info), 10000])
# ordered_corr = np.zeros([len(all_tagged_info), 10000])

# fig, ax = plt.subplots()

# for i in range(len(all_tagged_info)):
#     pooled[i,:] = normalise(avg_list[i][:10000])
# def myclust(e):
#     return clust[e]
# bcind = list(range(len(all_tagged_info)))
# bcind.sort(key=myclust)
# for i in range(len(all_tagged_info)):
#     ordered[i,:] = pooled[bcind[i],:]

# corr = np.corrcoef(ordered[:,2500:5000])
# corr_f_order = corr[0,:]
# def mycorr(e):
#     return corr_f_order[e]
# corrind = list(range(len(all_tagged_info)))
# corrind.sort(key=mycorr)
# for i in range(len(all_tagged_info)):
#     ordered_corr[i,:] = ordered[corrind[i],:]
    
# ax.imshow(ordered_corr, aspect='auto',
#           extent=[-3, 5, len(corrind), 0])
# ax.set(title='all tagged LC units',
#        xlabel='time (s)',
#        ylabel='cell #')

# fig.savefig(out_directory + '\\'+'LC_all_tagged_spikeavgheat_(alignedRun).png')


#%% correlation matrix 
# print('\nplotting correlation matrix...')

# corr_f_disp = np.corrcoef(ordered_corr[:,2500:5000])

# fig, ax = plt.subplots()

# ax.imshow(corr_f_disp)

# plt.show()

# fig.savefig(out_directory + '\\'+'LC_all_tagged_spikeavgcorr_(alignedRun).png')