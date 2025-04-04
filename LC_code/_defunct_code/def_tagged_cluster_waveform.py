# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:54:49 2023

plot and analyse waveforms of the 3 clusters (all tagged cells)

@author: Dinghao Luo
"""

#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as plc
import sys

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise

## load waveform .npy file 
waveforms = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
                    allow_pickle=True).item()

## load clustering result .npy file 
clusters = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_tagged_clustered_fromall.npy',
                   allow_pickle=True).item()
cluster1 = clusters['cluster 1']
cluster2 = clusters['cluster 2']
cluster3 = clusters['cluster 3']


#%% functions
def get_waveform(cluname):
    '''
    return waveform avg and sem as tuple
    '''
    return waveforms[cluname][0,:], waveforms[cluname][1,:]


#%% MAIN 
cluster1_wf = []; cluster2_wf = []; cluster3_wf = []
cluster1_sem = []; cluster2_sem = []; cluster3_sem = []

for cluname in list(waveforms.keys()):
    wf, sem = get_waveform(cluname)
    if cluname in cluster1:
        cluster1_wf.append(wf)
        cluster1_sem.append(sem)
    elif cluname in cluster2:
        cluster2_wf.append(wf)
        cluster2_sem.append(sem)
    elif cluname in cluster3:
        cluster3_wf.append(wf)
        cluster3_sem.append(sem)
        

#%% plotting individual 
tot_plots = len(cluster1)+len(cluster2)+len(cluster3)
col_plots = 8

row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
    
plc.rcParams['figure.figsize'] = (8*2, row_plots*2.5)

plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1)

xaxis = np.arange(32)

for clu in range(len(cluster1_wf)):
    cluname = cluster1[clu]
    wf = cluster1_wf[clu]
    wf_norm = normalise(wf)
    scaling_factor = wf_norm[0]/wf[0]
    sem = cluster1_sem[clu]*scaling_factor
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[clu])
    ax.set_title(cluname, fontsize = 8, color='darkorange')
    ax.plot(xaxis, wf_norm, 'darkorange')
    ax.fill_between(xaxis, wf_norm+sem, wf_norm-sem, 
                    color='darkorange', alpha=.1)
    ax.axis('off')
    
for clu in range(len(cluster2_wf)):
    cluname = cluster2[clu]
    wf = cluster2_wf[clu]
    wf_norm = normalise(wf)
    scaling_factor = wf_norm[0]/wf[0]
    sem = cluster2_sem[clu]*scaling_factor
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[clu+len(cluster1_wf)])
    ax.set_title(cluname, fontsize = 8, color='limegreen')
    ax.plot(xaxis, wf_norm, 'limegreen')
    ax.fill_between(xaxis, wf_norm+sem, wf_norm-sem, 
                    color='limegreen', alpha=.1)
    ax.axis('off')

for clu in range(len(cluster3_wf)):
    cluname = cluster3[clu]
    wf = cluster3_wf[clu]
    wf_norm = normalise(wf)
    scaling_factor = wf_norm[0]/wf[0]
    sem = cluster3_sem[clu]*scaling_factor
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[clu+len(cluster1_wf)+len(cluster2_wf)])
    ax.set_title(cluname, fontsize = 8, color='crimson')
    ax.plot(xaxis, wf_norm, 'crimson')
    ax.fill_between(xaxis, wf_norm+sem, wf_norm-sem, 
                    color='crimson', alpha=.1)
    ax.axis('off')
    
plt.show()

#%% save figure
fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_waveforms_by_clusters_individual.png',
            dpi=300,
            bbox_inches='tight',
            transparent=False)


#%% plotting average
fig, axs = plt.subplots(1, 3, figsize=(9, 3))

c1_mean = np.mean(cluster1_wf, axis=0)
c1_std = np.std(cluster1_wf, axis=0)
c2_mean = np.mean(cluster2_wf, axis=0)
c2_std = np.std(cluster2_wf, axis=0)
c3_mean = np.mean(cluster3_wf, axis=0)
c3_std = np.std(cluster3_wf, axis=0)

axs[0].plot(c1_mean, color='darkorange')
axs[0].fill_between(xaxis, c1_mean+c1_std, c1_mean-c1_std, color='darkorange',
                    alpha=.1)
axs[1].plot(c2_mean, color='limegreen')
axs[1].fill_between(xaxis, c2_mean+c2_std, c2_mean-c2_std, color='limegreen',
                    alpha=.1)
axs[2].plot(c3_mean, color='crimson')
axs[2].fill_between(xaxis, c3_mean+c3_std, c3_mean-c3_std, color='crimson',
                    alpha=.1)

fig.tight_layout()

#%% save figure
fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_waveforms_by_clusters.png',
            dpi=300,
            bbox_inches='tight',
            transparent=False)