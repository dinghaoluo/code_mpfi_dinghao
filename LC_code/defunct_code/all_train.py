# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:22:33 2023

pool all cells from all recording sessions with tagged cells

*contains interneurones*

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats import sem
import sys
import h5py

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% MAIN 
all_info = {}

for pathname in pathLC:
    print(pathname[-17:])
    
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    speed_time_file = sio.loadmat(filename + '_alignRun_msess1.mat')
    spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']
    
    time_bef = spike_time_file['TimeBef']; time = spike_time_file['Time']
    # weird error when reading h5py, need to find source of this 30 Jan 2023
    # if filename=='Z:\Dinghao\MiceExp\ANMD032r\A032r-20220726\A032r-20220726-03\A032r-20220726-03_DataStructure_mazeSection1_TrialType1':
    #     tag_sess = [x for x in tag_sess if x!='8' and x!='9']
    tot_clu = time.shape[1]
    tot_trial = time.shape[0]  # trial 1 is empty but tot_trial includes it for now
    
    samp_freq = 1250  # Hz
    gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
    sigma_speed = samp_freq/100
    gx_spike = np.arange(-500, 500, 1)
    sigma_spike = samp_freq/10
    
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
            spikes = [int(s+3750) for s in spikes]
            spike_train_trial = np.zeros(trial_length[trial])
            spike_train_trial[spikes] = 1
            spike_train_all[i][trial] = spike_train_trial
            spike_train_conv[i][trial] = np.convolve(spike_train_trial, 
                                                     gaus_spike, mode='same')
            # norm_spike[i][trial] = normalise(spike_train_conv[i][trial])
            conv_spike[i][trial] = spike_train_conv[i][trial]
    
    i = 0
    for clu in range(tot_clu):
        clu_name = pathname[-17:]+' clu'+str(clu+2)
        clu_conv_spike = conv_spike[i]
        
        all_info[clu_name] = clu_conv_spike
        i+=1
    
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_info.npy', 
        all_info)
print('\nprocessed and saved to Z:\Dinghao\code_dinghao\LC_all_clu\LC_all_info.npy')
    

#%% plotting 
print('plotting average spike rate profiles for all cells in tagged sessions...')

micepath = 'Z:\Dinghao\MiceExp\ANMD'

length_for_disp = 13750  # how many samples (in total) to display
spike_avg = {}
spike_sem = {}

for i in range(len(all_info)):
    curr_clu = list(all_info.items())[i]
    curr_clu_name = curr_clu[0]
    
    # modified 25 Apr 2023, run average only on BASELINE and RECOVERY trials
    # because it does not make sense to average these 2 subsessions with 
    # STIMULATION, even for Dbh- cells
    # import opto trial id
    beh_info_file = sio.loadmat(micepath+curr_clu_name[1:5]+'\\'+curr_clu_name[:14]+
                                '\\'+curr_clu_name[:17]+'\\'+curr_clu_name[:17]+
                                '_DataStructure_mazeSection1_TrialType1_Info.mat')['beh']
    stim_trial = np.squeeze(np.where(beh_info_file['pulseMethod'][0][0][0]!=0))
    if stim_trial.size!=0:  # if there is opto
        stim_window = [stim_trial[0], stim_trial[-1]]  # first and last stim trials
        base_clu = np.delete(curr_clu[1], np.arange(stim_window[0], stim_window[1]))
        n_spike = base_clu
    else:  # if there is no opto (pulseMethod is all 0 in this case)
        n_spike = curr_clu[1]
    
    # display only up to length_for_disp
    spike_trunc = np.zeros((len(n_spike), length_for_disp))
    spike_trunc_norm = np.zeros((len(n_spike), length_for_disp))
    
    for trial in range(n_spike.shape[0]):
        curr_length = len(n_spike[trial])
        if curr_length <= length_for_disp:
            spike_trunc[trial, :curr_length] = n_spike[trial][:]
            spike_trunc_norm[trial, :curr_length] = normalise(n_spike[trial][:])
        else:
            spike_trunc[trial, :] = n_spike[trial][:length_for_disp]
            spike_trunc_norm[trial, :] = normalise(n_spike[trial][:length_for_disp])
    spike_avg[curr_clu_name] = np.mean(spike_trunc, 0)*1250
    spike_sem[curr_clu_name] = sem(spike_trunc, 0)*1250

LC_all_avg_sem = {
    'all avg': spike_avg,
    'all sem': spike_sem
    }

tot_plots = len(all_info)  # total number of clusters
col_plots = 10
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1, figsize=[col_plots*4, row_plots*2.85]); fig.tight_layout()

list_avg = list(spike_avg.values())
list_sem = list(spike_sem.values())
for j in range(len(all_info)):
    curr_clu = list(all_info.items())[j]
    curr_clu_name = curr_clu[0]
    
    curr_min = min(list_avg[j][:10000])
    curr_max = max(list_avg[j][:10000])
    
    xaxis = np.arange(-1250, 5000)/samp_freq 
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[j])
    ax.set_title(curr_clu_name[-22:], fontsize = 10)
    ax.set(xlabel='time (s)', ylabel='spike rate (Hz)',
           ylim=(curr_min*0.75, curr_max*1.5))
    clu_avg = ax.plot(xaxis, list_avg[j][2500:8750], color='grey')
    clu_sem = ax.fill_between(xaxis, list_avg[j][2500:8750]+list_sem[j][2500:8750],
                                     list_avg[j][2500:8750]-list_sem[j][2500:8750],
                                     color='grey', alpha=.25)
    ax.vlines(0, 0, 100, color='grey', alpha=.1)

plt.subplots_adjust(hspace = 0.5)
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_spikeavg_(alignedRun).png',
            dpi=300,
            bbox_inches='tight')


#%% save 
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_avg_sem.npy', 
        LC_all_avg_sem)


#%% single-cell heatmap
fig = plt.figure(1, figsize=[col_plots*4, row_plots*2.85]); fig.tight_layout()

for j in range(len(all_info)):
    curr_clu = list(all_info.items())[j]
    curr_clu_name = curr_clu[0]
    curr_clu_im = normalise(list_avg[j][2500:8750]).reshape((1, 6250))
    
    xaxis = np.arange(-1250, 5000)/samp_freq 
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[j])
    ax.set_title(curr_clu_name[-22:], fontsize = 10)
    ax.set(xlabel='time (s)')
    clu_avg_im = ax.imshow(curr_clu_im, aspect='auto')

plt.subplots_adjust(hspace = 0.5)
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_spikeavgim_(alignedRun).png',
            dpi=300,
            bbox_inches='tight')