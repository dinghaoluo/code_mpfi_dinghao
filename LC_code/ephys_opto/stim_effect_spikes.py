# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:43:01 2023

plot stim-effect
already tested on hChR2-Dbh animals with prot[20Hz, 1s, d.c. 50%~75%]

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from read_clu import get_clu_20kHz
from param_to_array import param2array


#%% MAIN 
# read spike time (20kHz) for tagged clu
filename = 'Z:\Dinghao\MiceExp\ANMD056r\A056r-20230417\A056r-20230417-04\A056r-20230417-04'
clu = param2array(filename + '.clu.1')  # load .clu
res = param2array(filename + '.res.1')  # load .res
t_spike = get_clu_20kHz(5, res, clu)  # timestamps in 20kHz for spikes 

# get stim time
stim = sio.loadmat('Z:/Dinghao/MiceExp/ANMD056r/A056r-20230417/A056r-20230417-04/A056r-20230417-04BTDT.mat')['behEventsTdt']['stimOn'][0][0]
tot_stim = stim.shape[0]  # total number of stim trains (with tagging stims)
n_per_stim = int(stim[0, 3])  # how many pulses per train
dur_per_stim = stim[0, 5]  # how much time a pulse actually lasts for
dur_per_stim_full = stim[0, 6]  # divided by duty cycle
dur_stim = n_per_stim * dur_per_stim_full  # duration of each stim train
t_stim = []  # timestamps for stimulation on in 20kHz
for s in range(tot_stim):
    if stim[s, 5] >= 10:
        t_stim.append(stim[s, 0])  # admit stim timestamp only for opto stims
tot_optstim = len(t_stim)

# get all spikes within stim on window
window_radius = dur_stim * 20  # this is in 20kHz
all_stim = []
for s in range(tot_optstim):
    window = [t_stim[s]-window_radius, t_stim[s]+window_radius]
    spikes = [x for x in t_spike if x>=window[0] and x<=window[1]]
    spikes-=t_stim[s]  # normalise everything using t_stim as 0
    spikes/=20  # convert into ms
    all_stim.append(spikes)


#%% plotting 
print('plotting stimulation effects...')
fig, ax = plt.subplots(figsize=(4, 2))
ax.set(xlim=(-1100, 1100),
       xlabel='time (ms)', ylabel='norm. avg. spikes')
for p in ['right', 'top']:
    ax.spines[p].set_visible(False)

# single stim effects
v_loc = 1.05  # for plotting multiple stim trains
for s in range(tot_optstim):
    ax.scatter(all_stim[s], [v_loc]*len(all_stim[s]), s=.3, color='k')
    v_loc+=.05
ax.set(ylim=(0, v_loc))

# histogram 
bins = np.linspace(-1000, 1000, 41)
concat_stim = []
for s in all_stim:
    concat_stim.extend(s)
histogram, bin_edges = np.histogram(concat_stim, bins=bins)
histogram = histogram/max(histogram)
ax.bar(bin_edges[:-1]+25, histogram, width=50, 
       color='grey', edgecolor='k', linewidth=.5)

# stimulation pulses
s_on = np.linspace(0, 1000, n_per_stim+1)
ax.bar(s_on[:-1]+dur_per_stim/2, 5, color='b', alpha=.3, width=dur_per_stim)