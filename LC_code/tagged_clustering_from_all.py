# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:41:46 2023

Use hierarchical clustering from all cells (tagged sessions) to cluster tagged 
    cells

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt
import sys
from scipy.stats import sem

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% data wrangling
# put averaged activity of units into at list of lists, each list being a
# firing rate vector, bin=1/1250s
clstr_all = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_clustered_hierarchical_centroid.npy',
                    allow_pickle=True).item()['clustering result']
leaves = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_clustered_hierarchical_centroid.npy',
                    allow_pickle=True).item()['leaves']

tagged_avg = np.load('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_avg_sem.npy',
                     allow_pickle=True).item()['all tagged avg']
tagged_clu_names = list(tagged_avg.keys())

list_clstr_all = list(clstr_all.keys())
clstr_tagged_sr = []; clstr_tagged_sr_name = [];
for i in range(len(leaves)):
    curr_ind = leaves[i]
    curr_clu_name = list_clstr_all[curr_ind]
    if curr_clu_name in tagged_clu_names:
        clstr_tagged_sr.append(tagged_avg[curr_clu_name])
        clstr_tagged_sr_name.append(curr_clu_name)


#%% order based on hierarchical leaves and plot heatmap
print('plotting heatmap aligned to leaves (tagged cells only)...')
heatmap_mat = np.zeros((len(clstr_tagged_sr), 6250))
for i in range(len(clstr_tagged_sr)):
    heatmap_mat[i,:] = normalise(clstr_tagged_sr[i][2500:8750])

fig, ax = plt.subplots(figsize=(6,4))
ax.set(title='average spiking profiles of all tagged Dbh+ cells',
       xlabel='time (s)', ylabel='cell ID')

ax.imshow(heatmap_mat, aspect='auto',
          extent=[-1, 4, len(clstr_tagged_sr), 0])

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_dendro_heatmap.png',
            dpi=300)


#%% data wrangling
clusters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_clustered_hierarchical_centroid.npy',
                    allow_pickle=True).item()['clusters']

cluster1_tagged = [clu for clu in clusters['cluster 1'] if clu in tagged_clu_names]
cluster2_tagged = [clu for clu in clusters['cluster 2'] if clu in tagged_clu_names]
cluster3_tagged = [clu for clu in clusters['cluster 3'] if clu in tagged_clu_names]


#%% plot average activity of all groups, coloured by dendrogram colours 
print('plotting averaged activity of groups...')

group1_sr = [tagged_avg[clu] for clu in cluster1_tagged]
group2_sr = [tagged_avg[clu] for clu in cluster2_tagged]
group3_sr = [tagged_avg[clu] for clu in cluster3_tagged]  
group1_mean = np.mean(group1_sr, axis=0)
group1_sem = sem(group1_sr, axis=0)
group2_mean = np.mean(group2_sr, axis=0)
group2_sem = sem(group2_sr, axis=0)
group3_mean = np.mean(group3_sr, axis=0)
group3_sem = sem(group3_sr, axis=0)
group_tot = [len(group1_sr), len(group2_sr), len(group3_sr)]

fig, axs = plt.subplots(1, 3, figsize=(13, 4)); fig.tight_layout(pad=3)
for plot in range(3):
    axs[plot].set(xlabel='time (s)', ylabel='spike rate (Hz)',
                  title='cluster {}, n={}'.format(plot+1, group_tot[plot]))

xaxis = np.arange(-1250, 5000)/1250

axs[0].plot(xaxis, group1_mean[2500:8750], color='darkorange')
axs[0].fill_between(xaxis, 
                    group1_mean[2500:8750]+group1_sem[2500:8750],
                    group1_mean[2500:8750]-group1_sem[2500:8750],
                    color='darkorange', alpha=.1)
axs[1].plot(xaxis, group2_mean[2500:8750], color='limegreen')
axs[1].fill_between(xaxis, 
                    group2_mean[2500:8750]+group2_sem[2500:8750],
                    group2_mean[2500:8750]-group2_sem[2500:8750],
                    color='limegreen', alpha=.1)
axs[2].plot(xaxis, group3_mean[2500:8750], color='crimson')
axs[2].fill_between(xaxis, 
                    group3_mean[2500:8750]+group3_sem[2500:8750],
                    group3_mean[2500:8750]-group3_sem[2500:8750],
                    color='crimson', alpha=.1)

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_clustering_hierarchical_groupavg.png',
            dpi=300,
            bbox_inches='tight',
            transparent=True)


# #%% save
# tagged_clstr_dict = {
#     'cluster 1': [],
#     'cluster 2': [],
#     'cluster 3': []
#     }
# for i in range(len(clstr_tagged_sr)):
#     if list_clstr_tagged[i] == 1:
#         tagged_clstr_dict['cluster 1'].append(list_clstr_tagged_keys[i])
#     elif list_clstr_tagged[i] == 2:
#         tagged_clstr_dict['cluster 2'].append(list_clstr_tagged_keys[i])
#     else:
#         tagged_clstr_dict['cluster 3'].append(list_clstr_tagged_keys[i])
        
# np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_clustered_fromall.npy', 
#         tagged_clstr_dict)


#%% histogram for frequencies of occurrence
print('plotting barplot for freq of occ...')

fig, ax = plt.subplots(figsize=(3, 6))
ax.set(title='clusters (LC general population)',
       ylabel='proportion of cells',
       ylim=(0, .7))
ax.spines[['right', 'top']].set_visible(False)

tot_clu = len(clstr_tagged_sr)
freq1 = len(cluster1_tagged)/tot_clu
freq2 = len(cluster2_tagged)/tot_clu
freq3 = len(cluster3_tagged)/tot_clu
# freq4 = len(group4)/tot_clu

labels = []
for i in range(1, 4):
    labels.append('cluster {}'.format(i))

ax.bar(labels,
       [freq1, freq2, freq3],
       color=['white']*3,
       edgecolor=['darkorange', 'limegreen', 'violet'],
       linewidth=2)

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_clustering_hierarchicalfromall_freqocc.png',
            dpi=300,
            bbox_inches='tight')