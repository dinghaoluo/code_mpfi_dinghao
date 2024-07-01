# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:20:55 2023

LC: visual and statistical comparison, broad v narrow
params for ttahp:
    broad >= 600usec
    narrow < 600usec

@author: Dinghao Luo
"""

#%% imports
import os
import sys
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt 

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC

if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')


#%% MAIN
all_tagged_train = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
                           allow_pickle=True).item()
all_ttahp = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_ttahp.npy',
                    allow_pickle=True).item()
all_brd = {}; all_brd_sem = {}
all_narr = {}; all_narr_sem = {}
max_length = 13750  # max length for trial analysis

for pathname in pathLC:
    sessname = pathname[-17:]
    
    # import tagged cell spike trains from all_tagged_train
    for name in list(all_tagged_train.keys()):
        if name[:17] == sessname:
            curr_tagged = all_tagged_train[name][0,:]  # train of current clu
            curr_all = np.zeros([len(curr_tagged), max_length])
            for trial in range(len(curr_tagged)):
                curr_length = len(curr_tagged[trial])
                curr_all[trial, :curr_length] = curr_tagged[trial][:max_length]
            
            if all_ttahp[name] < 600:
                all_narr[name] = np.mean(curr_all, axis=0)
                all_narr_sem[name] = sem(curr_all, axis=0)
            else:
                all_brd[name] = np.mean(curr_all, axis=0)
                all_brd_sem[name] = sem(curr_all, axis=0)

all_brd_avg = []
all_narr_avg = []
for clu in list(all_brd.items()):
    all_brd_avg.append(clu[1])
for clu in list(all_narr.items()):
    all_narr_avg.append(clu[1])
all_brd_avg_sem = sem(all_brd_avg, axis=0)
all_brd_avg = np.mean(all_brd_avg, axis=0)
all_narr_avg_sem = sem(all_narr_avg, axis=0)
all_narr_avg = np.mean(all_narr_avg, axis=0)


#%% plotting
print('\nplotting avg narrow and broad spike trains...')
tot_plots = len(all_ttahp)  # total number of clusters
tot_brd = len(all_brd); tot_narr = len(all_narr)
col_plots = 5
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1, figsize=[5*4, row_plots*2.5]); fig.tight_layout()
xaxis = np.arange(-3750, 10000, 1)/1250 

for i in range(tot_narr):
    curr_clu = list(all_narr.items())[i]
    curr_clu_name = curr_clu[0]
    curr_clu_avg = curr_clu[1]
    curr_clu_sem = all_narr_sem[curr_clu_name]
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
    ax.set_title(curr_clu_name[-22:], fontsize = 10)
    ax.set(ylim=(0, np.max(curr_clu_avg)*1250*1.5))
    narr_avg = ax.plot(xaxis, curr_clu_avg*1250, color='royalblue')
    narr_sem = ax.fill_between(xaxis, (curr_clu_avg+curr_clu_sem)*1250,
                                      (curr_clu_avg-curr_clu_sem)*1250,
                                      color='lightsteelblue')
    ax.vlines(0, 0, 10, color='grey', alpha=.1)

for i in range(tot_brd):
    curr_clu = list(all_brd.items())[i]
    curr_clu_name = curr_clu[0]
    curr_clu_avg = curr_clu[1]
    curr_clu_sem = all_brd_sem[curr_clu_name]
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i+tot_narr])
    ax.set_title(curr_clu_name[-22:], fontsize = 10)
    ax.set(ylim=(0, np.max(curr_clu_avg)*1250*1.5))
    narr_avg = ax.plot(xaxis, curr_clu_avg, color='tan')
    narr_sem = ax.fill_between(xaxis, (curr_clu_avg+curr_clu_sem)*1250,
                                      (curr_clu_avg-curr_clu_sem)*1250,
                                      color='moccasin')
    ax.vlines(0, 0, 1, color='grey', alpha=.1)

plt.subplots_adjust(hspace = 0.5)
plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
if not os.path.exists(out_directory):
    os.makedirs(out_directory)
fig.savefig(out_directory + '\\'+'LC_tagged_narrvbrd_(alignedRun).png')


#%% avg profile
print('\nplotting avg narrow v broad averaged spike trains...')

fig, ax = plt.subplots(figsize=(6, 4))
avg_brd_ln, = ax.plot(xaxis, all_brd_avg*1250, color='tan')
avg_narr_ln, = ax.plot(xaxis, all_narr_avg*1250, color='royalblue')
ax.fill_between(xaxis, (all_brd_avg+all_brd_avg_sem)*1250,
                       (all_brd_avg-all_brd_avg_sem)*1250,
                       color='moccasin', alpha=.1)
ax.fill_between(xaxis, (all_narr_avg+all_narr_avg_sem)*1250,
                       (all_narr_avg-all_narr_avg_sem)*1250,
                       color='lightsteelblue', alpha=.1)
ax.vlines(0, 0, 10, color='grey', alpha=.1)
ax.set(title='avg spiking profile, narrow v broad units',
       ylim=(0, np.max(all_narr_avg)*1250*1.5),
       ylabel='spike rate (Hz)',
       xlabel='time (s)')
ax.legend([avg_brd_ln, avg_narr_ln], ['broad units', 'narrow units'])

fig.savefig(out_directory + '\\'+'LC_tagged_narrvbrd_(alignedRun)_avg.png',
            dpi=300,
            bbox_inches='tight')