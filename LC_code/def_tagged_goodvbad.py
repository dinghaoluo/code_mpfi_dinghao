# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:20:55 2023

LC: visual and statistical comparison between good and bad trial bursts 

bad trial parameters 30 Jan 2023 (in getBehParameters()):
    totRunLenT > 13 |
    numRun > 10 |
    totStopLenT > 2 |
    rewarded == -1

@author: Dinghao Luo
"""

#%% imports
import os
import sys
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt 
import scipy.io as sio

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list, single_unit
pathLC = rec_list.pathLC

if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')


#%% MAIN
all_tagged_train = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
                           allow_pickle=True).item()
all_good = {}; all_good_sem = {}
all_bad = {}; all_bad_sem = {}
max_length = 13750  # max length for trial analysis

for pathname in pathLC:
    sessname = pathname[-17:]
    
    # import bad beh trial numbers
    beh_par_file = sio.loadmat(pathname+pathname[-18:]+
                               '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference
    ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                     # -1 to account for 0 being an empty trial
    ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
    # import tagged cell spike trains from all_tagged_train
    if len(ind_bad_beh) >= 15:  # 15 bad trials as least, prevents contam.
        for name in list(all_tagged_train.keys()):
            if name[:17] == sessname:
                curr_tagged = all_tagged_train[name][0,:]  # train of current clu
                curr_good = np.zeros([len(ind_good_beh), max_length])
                curr_bad = np.zeros([len(ind_bad_beh), max_length])
                for i in range(len(ind_good_beh)):
                    curr_length = len(curr_tagged[ind_good_beh[i]])
                    curr_good[i, :curr_length] = curr_tagged[ind_good_beh[i]][:max_length]
                for i in range(len(ind_bad_beh)):
                    curr_length = len(curr_tagged[ind_bad_beh[i]])
                    curr_bad[i, :curr_length] = curr_tagged[ind_bad_beh[i]][:max_length]
                all_good[name] = np.mean(curr_good, axis=0)
                all_good_sem[name] = sem(curr_good, axis=0)
                all_bad[name] = np.mean(curr_bad, axis=0)
                all_bad_sem[name] = sem(curr_bad, axis=0)

all_g_avg = []
all_b_avg = []
for clu in list(all_good.items()):
    all_g_avg.append(clu[1])
for clu in list(all_bad.items()):
    all_b_avg.append(clu[1])
all_g_sem = sem(all_g_avg, axis=0)
all_g_avg = np.mean(all_g_avg, axis=0)
all_b_sem = sem(all_b_avg, axis=0)
all_b_avg = np.mean(all_b_avg, axis=0)


#%% plotting
# print('\nplotting avg good vs bad spike trains...')
# tot_plots = len(all_good)  # total number of clusters
# col_plots = 5
# row_plots = tot_plots // col_plots
# if tot_plots % col_plots != 0:
#     row_plots += 1
# plot_pos = np.arange(1, tot_plots+1)

# fig = plt.figure(1, figsize=[5*4, row_plots*2.5]); fig.tight_layout()
# xaxis = np.arange(-3750, 10000, 1)/1250 

# for i in range(tot_plots):
#     curr_clu_good = list(all_good.items())[i]
#     curr_clu_name = curr_clu_good[0]
#     curr_good_avg = curr_clu_good[1]
#     curr_good_sem = all_good_sem[curr_clu_name]
#     curr_bad_avg = all_bad[curr_clu_name]
#     curr_bad_sem = all_bad[curr_clu_name]
    
#     ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
#     ax.set_title(curr_clu_name[-22:], fontsize = 10)
#     ax.set(ylim=(0, np.max(curr_good_avg)*1.5))
#     good_avg = ax.plot(xaxis, curr_good_avg, color='seagreen')
#     good_sem = ax.fill_between(xaxis, curr_good_avg+curr_good_sem,
#                                       curr_good_avg-curr_good_sem,
#                                       color='springgreen')
#     bad_avg = ax.plot(xaxis, curr_bad_avg, color='firebrick', alpha=.3)
#     # bad_sem = ax.fill_between(xaxis, curr_bad_avg+curr_bad_sem,
#     #                                  curr_bad_avg-curr_bad_sem,
#     #                                  color='lightcoral')
#     ax.vlines(0, 0, 1, color='grey', alpha=.25)

# plt.subplots_adjust(hspace = 0.5)
# plt.show()

# out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
# if not os.path.exists(out_directory):
#     os.makedirs(out_directory)
# fig.savefig(out_directory + '\\'+'LC_tagged_goodvbad_(alignedRun).png')


#%% avg profile
# print('\nplotting avg good vs bad averaged spike trains...')

# fig, ax = plt.subplots(figsize=(6,4))
# avg_good_ln, = ax.plot(xaxis, all_g_avg, color='darkgreen')
# avg_bad_ln, = ax.plot(xaxis, all_b_avg, color='darkred', alpha=.3)
# ax.fill_between(xaxis, all_g_avg+all_g_sem, all_g_avg-all_g_sem,
#                  color='mediumseagreen', alpha=.2)
# ax.fill_between(xaxis, all_b_avg+all_b_sem, all_b_avg-all_b_sem,
#                  color='indianred', alpha=.2)
# ax.set(title='avg spiking profile, good v bad trials',
#        ylim=(0, np.max(all_g_avg)*1.5),
#        ylabel='spike rate (au)',
#        xlabel='time (s)')
# ax.legend([avg_good_ln, avg_bad_ln], ['good trials', 'bad trials'])

# fig.savefig(out_directory + '\\'+'LC_tagged_goodvbad_(alignedRun)_avg.png',
#             dpi=300,
#             bbox_inches='tight')


#%% plotting
print('\nplotting avg onset-bursting good vs bad spike trains...')
tot_plots = len(all_good)  # total number of clusters
col_plots = 5
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1, figsize=[5*4, row_plots*2.5]); fig.tight_layout()
xaxis = np.arange(-3750, 10000, 1)/1250 

for i in range(tot_plots):
    curr_clu_good = list(all_good.items())[i]
    curr_clu_name = curr_clu_good[0]
    curr_good_avg = curr_clu_good[1]
    curr_good_sem = all_good_sem[curr_clu_name]
    curr_bad_avg = all_bad[curr_clu_name]
    curr_bad_sem = all_bad[curr_clu_name]
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
    peak = single_unit.neu_peak_detection(curr_good_avg)
    if peak==True:
        ax.set_title(curr_clu_name[-22:], color='r', fontsize = 10)
    else:
        ax.set_title(curr_clu_name[-22:], fontsize = 10)
    ax.set(ylim=(0, np.max(curr_good_avg)*1250*1.5),
           ylabel='spike rate (Hz)',
           xlabel='time (s)')
    good_avg = ax.plot(xaxis, curr_good_avg*1250, color='seagreen')
    good_sem = ax.fill_between(xaxis, curr_good_avg*1250+curr_good_sem*1250,
                                      curr_good_avg*1250-curr_good_sem*1250,
                                      color='springgreen')
    bad_avg = ax.plot(xaxis, curr_bad_avg*1250, color='firebrick', alpha=.3)
    # bad_sem = ax.fill_between(xaxis, curr_bad_avg+curr_bad_sem,
    #                                  curr_bad_avg-curr_bad_sem,
    #                                  color='lightcoral')
    ax.vlines(0, 0, 20, color='grey', alpha=.25)

plt.subplots_adjust(hspace = 0.5)
plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
if not os.path.exists(out_directory):
    os.makedirs(out_directory)
fig.savefig(out_directory + '\\'+'LC_tagged_goodvbad_(alignedRun).png')


#%% avg profile for onset-bursting clus
print('\nplotting avg onset-bursting good vs bad averaged spike trains...')

all_g_burst_avg = []
all_b_burst_avg = []
for clu in list(all_good.items()):
    peak = single_unit.neu_peak_detection(clu[1])
    if peak==True:
        all_g_burst_avg.append(clu[1])
        all_b_burst_avg.append(all_bad[clu[0]])
all_g_burst_sem = sem(all_g_burst_avg, axis=0)
all_g_burst_avg = np.mean(all_g_burst_avg, axis=0)
all_b_burst_sem = sem(all_b_burst_avg, axis=0)
all_b_burst_avg = np.mean(all_b_burst_avg, axis=0)

fig, ax = plt.subplots(figsize=(6,4))
avg_good_ln, = ax.plot(xaxis, all_g_burst_avg*1250, color='darkgreen')
avg_bad_ln, = ax.plot(xaxis, all_b_burst_avg*1250, color='darkred', alpha=.3)
ax.fill_between(xaxis, all_g_burst_avg*1250+all_g_burst_sem*1250, 
                       all_g_burst_avg*1250-all_g_burst_sem*1250,
                       color='mediumseagreen', alpha=.2)
ax.fill_between(xaxis, all_b_burst_avg*1250+all_b_burst_sem*1250, 
                       all_b_burst_avg*1250-all_b_burst_sem*1250,
                       color='indianred', alpha=.2)
ax.set(title='avg spiking profile (onset-bursting), good v bad trials',
       ylim=(0, np.max(all_g_avg)*1250*1.5),
       ylabel='spike rate (Hz)',
       xlabel='time (s)')
ax.legend([avg_good_ln, avg_bad_ln], 
          ['good trials (burst)', 'bad trials (burst)'])

fig.savefig(out_directory + '\\'+'LC_tagged_goodvbad_(alignedRun)_avg_burst.png',
            dpi=300,
            bbox_inches='tight')