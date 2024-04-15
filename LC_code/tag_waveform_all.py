# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:25:09 2023

SUMMARISE ALL TAGGED CELL WAVEFORMS

@author: Dinghao Luo
"""


#%% imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plc
import sys

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% MAIN
# dictionary containing all tagged cells:
#   keys include pathnames to recordings, a space, and cell number
#   values are 4x32 numpy arrays:
#       [0,:]: avg waveform
#       [1,:]: avg waveform sem
#       [2,:]: tag waveform
#       [3,:]: tag waveform sem
all_tagged = {}

# addresses of all prepped files (avg and tagged waveforms and sem's)
if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')
rootfolder = 'Z:\Dinghao\code_dinghao\LC_tagged_by_sess'

for pathname in pathLC:
    sessname = pathname[-17:]
    dict_stem = rootfolder+'\\'+sessname
    
    spk = np.load(dict_stem+'_avg_spk.npy', allow_pickle=True).item()
    sem = np.load(dict_stem+'_avg_sem.npy', allow_pickle=True).item()
    tag_spk = np.load(dict_stem+'_tagged_spk.npy', allow_pickle=True).item()
    tag_sem = np.load(dict_stem+'_tagged_sem.npy', allow_pickle=True).item()
    
    tag_clus = list(tag_spk.keys())  # id's of all tagged clu(s)
    tag_avg_spk = {}  # stores tagged cells' avg waveforms
    tag_avg_sem = {}
    
    for clu in tag_clus:
        tag_avg_spk = spk[clu]
        tag_avg_sem = sem[clu]
        
        packed = np.row_stack((tag_avg_spk, 
                                   tag_avg_sem, 
                                   tag_spk[clu],
                                   tag_sem[clu]))
        
        clu_id_full = sessname+' clu'+clu
        all_tagged[clu_id_full] = packed
    
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\\'+'LC_all_waveforms.npy', 
        all_tagged)

# plotting
tot_plots = len(all_tagged)
col_plots = 8

row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
    
plc.rcParams['figure.figsize'] = (8*2, row_plots*2.5)

plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1)

time_ax = np.arange(32)

avg_spk_dict = {}
avg_sem_dict = {}

all_tagged_list = list(all_tagged.items())

for i in range(len(all_tagged)):
    clu_name = all_tagged_list[i][0]
    
    spk = all_tagged_list[i][1][0]
    spk_norm = normalise(spk)
    scaling_factor = spk_norm[0]/spk[0]
    sem = all_tagged_list[i][1][1] * scaling_factor
    tag_spk = all_tagged_list[i][1][2]
    tag_spk_norm = normalise(tag_spk)
    scaling_factor = tag_spk_norm[0]/tag_spk[0]
    tag_sem = all_tagged_list[i][1][3] * scaling_factor
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
    ax.set_title(clu_name, fontsize = 8, color='grey')
    ax.plot(time_ax, spk_norm)
    ax.fill_between(time_ax, spk_norm+sem, spk_norm-sem, color='lightblue')
    ax.axis('off')

plt.subplots_adjust(hspace = 0.4)
plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
fig.savefig(out_directory+'\\'+'LC_all_waveforms_avg.png')


fig = plt.figure(1)

for i in range(len(all_tagged)):
    clu_name = all_tagged_list[i][0]
    
    spk = all_tagged_list[i][1][0]
    spk_norm = normalise(spk)
    scaling_factor = spk_norm[0]/spk[0]
    sem = all_tagged_list[i][1][1] * scaling_factor
    tag_spk = all_tagged_list[i][1][2]
    tag_spk_norm = normalise(tag_spk)
    scaling_factor = tag_spk_norm[0]/tag_spk[0]
    tag_sem = all_tagged_list[i][1][3] * scaling_factor
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
    ax.set_title(clu_name, fontsize = 8, color='grey')
    ax.plot(time_ax, tag_spk_norm, color='k')
    ax.fill_between(time_ax, tag_spk_norm+tag_sem, tag_spk_norm-tag_sem, 
                    color='grey')
    ax.axis('off')

plt.subplots_adjust(hspace = 0.4)
plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
fig.savefig(out_directory+'\\'+'LC_all_waveforms_tagged.png')