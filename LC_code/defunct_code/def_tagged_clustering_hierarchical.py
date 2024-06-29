# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:41:46 2023

Use hierarchical clustering to cluster tagged LC units

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
import sys
from scipy.stats import sem

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% data wrangling
# put averaged activity of units into at list of lists, each list being a
# firing rate vector, bin=1/1250s
sr_file = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_avg_sem.npy',
                  allow_pickle=True).item()
list_avg_sr = list(sr_file['all tagged avg'].values())
key_avg_sr = list(sr_file['all tagged avg'].keys())

avg_sr_list = [list(normalise(x[2500:5000])) for x in list_avg_sr]


#%% hierarchical clustering
print('conducting hierarchical clustering...')
clustering_method = 'centroid'
clstr = linkage(avg_sr_list, method=clustering_method)

fig, ax = plt.subplots(figsize=(10,4))
ax.set(title='hierarchical clustering for all tagged Dbh+ cells',
       xlabel='cell ID', ylabel='depth')
ax.axis('off')

threshold = 15  # depth to cut the dendrogram

# clustered and cut
all_clstr = fcluster(clstr, t=threshold, criterion='distance')
leaves = leaves_list(clstr)

# plot dendrogram with cutting threshold
hierarchy.set_link_color_palette(['darkorange', 'limegreen', 'crimson'])
dendrogram = dendrogram(clstr, color_threshold=threshold)
# ax.hlines(30, 0, 1000, color='grey')

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_dendro_{}.png'.
            format(clustering_method),
            dpi=300)

# save into .npy
hier_clu = {}
for i in range(len(list_avg_sr)):
    curr_clu_name = key_avg_sr[i]
    hier_clu[curr_clu_name] = str(all_clstr[i])

np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_clustered_hierarchical_{}.npy'.
        format(clustering_method), 
        hier_clu)


#%% order based on hierarchical leaves and plot heatmap
print('plotting heatmap aligned to leaves...')
heatmap_mat = np.zeros((len(list_avg_sr), 6250))
for i in range(len(list_avg_sr)):
    heatmap_mat[i,:] = normalise(list_avg_sr[leaves[i]][2500:8750])

fig, ax = plt.subplots(figsize=(6,4))
ax.set(title='average spiking profiles of all tagged Dbh+ cells',
       xlabel='time (s)', ylabel='cell ID')

ax.imshow(heatmap_mat, aspect='auto',
          extent=[-1, 4, len(list_avg_sr), 0])

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_dendro_heatmap.png',
            dpi=300)


#%% plot average activity of all groups, coloured by dendrogram colours 
print('plotting averaged activity of groups...')

list_hier = list(hier_clu.values())
group1 = []
group2 = []
group3 = []

for i in range(len(list_avg_sr)):
    if list_hier[i] == '1':
        group1.append(list_avg_sr[i])
    if list_hier[i] == '3':
        group2.append(list_avg_sr[i])
    if list_hier[i] == '4':
        group3.append(list_avg_sr[i])
        
group1_mean = np.mean(group1, axis=0)
group1_sem = sem(group1, axis=0)
group2_mean = np.mean(group2, axis=0)
group2_sem = sem(group2, axis=0)
group3_mean = np.mean(group3, axis=0)
group3_sem = sem(group3, axis=0)

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for plot in range(3):
    axs[plot].set(xlabel='time (s)', ylabel='spike rate (Hz)',
                  title='cluster {}'.format(plot+1))

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
            bbox_inches='tight')