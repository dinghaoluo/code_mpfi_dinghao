# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:41:46 2023

Use hierarchical clustering on all LC units from tagged sessions

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
import sys
from scipy.stats import sem
import pandas as pd

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% data wrangling
# put averaged activity of units into at list of lists, each list being a
# firing rate vector, bin=1/1250s
sr_file = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_avg_sem.npy',
                  allow_pickle=True).item()

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
putative_keys = []
for cell in cell_prop.index:
    pt = cell_prop['putative'][cell]  # putative
    
    if pt:
        putative_keys.append(cell)

avg_sr_list = []; avg_im_list = []
for key in putative_keys:
    avg_sr_list.append(list(normalise(sr_file['all avg'][key][2500:7500])))
    avg_im_list.append(list(normalise(sr_file['all avg'][key])))


#%% hierarchical clustering
print('conducting hierarchical clustering on ALL cells from tagged sessions...')
clustering_method = 'centroid'
clstr = linkage(avg_sr_list, method=clustering_method)

fig, ax = plt.subplots(figsize=(10,4))
ax.set(title='hierarchical clustering for all cells from tagged sessions',
       xlabel='cell ID', ylabel='depth')
ax.axis('off')

threshold = 22  # depth to cut the dendrogram

# clustered and cut
all_clstr = fcluster(clstr, t=threshold, criterion='distance')
leaves = leaves_list(clstr)

# plot dendrogram with cutting threshold
hierarchy.set_link_color_palette(['darkorange', 'grey', 'limegreen', 'violet'])
dendro = dendrogram(clstr, color_threshold=threshold)
# ax.hlines(30, 0, 1000, color='grey')

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_dendro_{}.png'.
            format(clustering_method),
            dpi=300)


#%% save into a dict
hier_clu = {}
for i in range(len(avg_sr_list)):
    curr_clu_name = putative_keys[i]
    hier_clu[curr_clu_name] = str(all_clstr[i])

hier_clu_with_leaves = {
    'clustering result': hier_clu,
    'leaves': leaves
    }


#%% order based on hierarchical leaves and plot heatmap
print('plotting heatmap aligned to leaves...')
heatmap_mat = np.zeros((len(avg_sr_list), 6250))
for i in range(len(avg_sr_list)):
    heatmap_mat[i,:] = normalise(avg_im_list[leaves[i]][2500:8750])

fig, ax = plt.subplots(figsize=(6,4))
ax.set(title='all LC cells',
       xlabel='time (s)', ylabel='cell ID')

hm = ax.imshow(heatmap_mat, aspect='auto',
               extent=[-1, 4, len(avg_sr_list), 0])
plt.colorbar(hm)

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_dendro_heatmap.png',
            dpi=300,
            bbox_inches='tight')


#%% plot average activity of all groups, coloured by dendrogram colours 
print('plotting averaged activity of groups...')

list_hier = list(hier_clu.values())
group1 = []  # run-onset bursting
cluster1 = []
group2 = []  # pre-ro inhibition
cluster2 = []
group3 = []  # reward ramping
cluster3 = []
group4 = []  # post run-onset dip

for i in range(len(avg_sr_list)):
    if list_hier[i] == '1':
        group1.append(avg_sr_list[i])
        cluster1.append(avg_sr_list[i])
    if list_hier[i] == '3':
        group2.append(avg_sr_list[i])
        cluster2.append(avg_sr_list[i])
    if list_hier[i] == '4':
        group3.append(avg_sr_list[i])
        cluster3.append(avg_sr_list[i])
        
group1_mean = np.mean(group1, axis=0)
group1_sem = sem(group1, axis=0)
group2_mean = np.mean(group2, axis=0)
group2_sem = sem(group2, axis=0)
group3_mean = np.mean(group3, axis=0)
group3_sem = sem(group3, axis=0)
group_tot = [len(group1), len(group2), len(group3)]

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
axs[2].plot(xaxis, group3_mean[2500:8750], color='violet')
axs[2].fill_between(xaxis, 
                    group3_mean[2500:8750]+group3_sem[2500:8750],
                    group3_mean[2500:8750]-group3_sem[2500:8750],
                    color='violet', alpha=.1)

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_clustering_hierarchical_groupavg.png',
            dpi=300,
            bbox_inches='tight')


#%% save into .npy
# keys of all the clusters
hier_clu_with_leaves['clusters'] = {
    'cluster 1': cluster1,
    'cluster 2': cluster2,
    'cluster 3': cluster3}

np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_clustered_hierarchical_{}.npy'.
        format(clustering_method), 
        hier_clu_with_leaves)


#%% histogram for frequencies of occurrence
print('plotting barplot for freq of occ...')

fig, ax = plt.subplots(figsize=(3, 6))
ax.set(title='clusters (putative Dbh+)',
       ylabel='proportion of cells',
       ylim=(0, .7))
ax.spines[['right', 'top']].set_visible(False)

tot_clu = len(list_hier)
freq1 = len(group1)/tot_clu
freq2 = len(group2)/tot_clu
freq3 = len(group3)/tot_clu
# freq4 = len(group4)/tot_clu

labels = []
for i in range(1, 4):
    labels.append('cluster {}'.format(i))

ax.bar(labels,
       [freq1, freq2, freq3],
       color=['white']*3,
       edgecolor=['darkorange', 'limegreen', 'violet'],
       linewidth=2)

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_clustering_hierarchical_freqocc.png',
            dpi=300,
            bbox_inches='tight')