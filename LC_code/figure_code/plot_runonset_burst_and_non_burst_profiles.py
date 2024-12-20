# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:49:27 2024

plot the mean spiking profiles RO and non-RO LC cells 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load spike train 
profile_run = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_info.npy',
                      allow_pickle=True).item()


#%% get cell lists for different classes 
cell_prop = pd.read_pickle(r'Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')
tag_list = [clu for clu in cell_prop.index if cell_prop['tagged'][clu]]
put_list = [clu for clu in cell_prop.index if cell_prop['putative'][clu]]
pooled_list = tag_list+put_list

RO_list = [clu for clu in cell_prop.index if clu in pooled_list and cell_prop['peakness'][clu]]
nonRO_list = [clu for clu in cell_prop.index if clu in pooled_list and not cell_prop['peakness'][clu]]


#%% main 
RO = np.zeros((len(RO_list), 1250*5))
nonRO = np.zeros((len(pooled_list)-len(RO_list), 1250*5))

for i, cluname in enumerate(RO_list):
    profile = profile_run[cluname]
    profile_trunc = np.zeros((len(profile), 5*1250))
    for j, trial in enumerate(profile):
        if len(trial)>2500+5*1250:
            profile_trunc[j,:] = profile[j][2500:2500+5*1250]
        else:
            profile_trunc[j,:len(trial)-2500] = profile[j][2500:]
    RO[i,:] = np.nanmean(profile_trunc, axis=0)

for i, cluname in enumerate(nonRO_list):
    profile = profile_run[cluname]
    profile_trunc = np.zeros((len(profile), 5*1250))
    for j, trial in enumerate(profile):
        if len(trial)>2500+5*1250:
            profile_trunc[j,:] = profile[j][2500:2500+5*1250]
        else:
            profile_trunc[j,:len(trial)-2500] = profile[j][2500:]
    nonRO[i,:] = np.nanmean(profile_trunc, axis=0)
    
RO_mean = np.nanmean(RO, axis=0)*1250
RO_sem = sem(RO, axis=0)*1250
nonRO_mean = np.nanmean(nonRO, axis=0)*1250
nonRO_sem = sem(nonRO, axis=0)*1250


#%% plotting
xaxis = np.arange(-1250, 1250*4)/1250

fig, ax = plt.subplots(figsize=(2,1.2))

ax.plot(xaxis, RO_mean, color='k')
ax.fill_between(xaxis, RO_mean-RO_sem,
                       RO_mean+RO_sem,
                color='k', edgecolor='none', alpha=.25)
ax.set(xlabel='time from run-onset (s)',
       ylabel='spike rate (Hz)',
       title='RO-peaking Dbh+')
for s in ['top','right']: ax.spines[s].set_visible(False)

for ext in ['png', 'pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\RO_peaking_mean_prof.{}'.format(ext),
                dpi=200, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(2,1.2))

ax.plot(xaxis, nonRO_mean, color='grey')
ax.fill_between(xaxis, nonRO_mean-nonRO_sem,
                       nonRO_mean+nonRO_sem,
                color='grey', edgecolor='none', alpha=.25)
ax.set(xlabel='time from run-onset (s)',
       ylabel='spike rate (Hz)',
       title='non-RO-peaking Dbh+')
for s in ['top','right']: ax.spines[s].set_visible(False)

for ext in ['png', 'pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\non_RO_peaking_mean_prof.{}'.format(ext),
                dpi=200, bbox_inches='tight')