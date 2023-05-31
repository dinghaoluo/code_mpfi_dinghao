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

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% data wrangling
# put averaged activity of units into at list of lists, each list being a
# firing rate vector, bin=1/1250s
sr_file = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_avg_sem.npy',
                  allow_pickle=True).item()
list_avg_sr = list(sr_file['all avg'].values())
key_avg_sr = list(sr_file['all avg'].keys())

# avg_sr_list = [list(normalise(x[2500:5000])) for x in list_avg_sr]
avg_sr_list = [list(normalise(x[2500:7500])) for x in list_avg_sr]


#%% hierarchical clustering
print('conducting hierarchical clustering on ALL cells from tagged sessions...')
clustering_method = 'centroid'
clstr = linkage(avg_sr_list, method=clustering_method)

fig, ax = plt.subplots(figsize=(10,4))
ax.set(title='hierarchical clustering for all cells from tagged sessions',
       xlabel='cell ID', ylabel='depth')
ax.axis('off')

threshold = 18.1  # depth to cut the dendrogram

# clustered and cut
all_clstr = fcluster(clstr, t=threshold, criterion='distance')
leaves = leaves_list(clstr)

# plot dendrogram with cutting threshold
hierarchy.set_link_color_palette(['grey', 'darkorange', 'grey', 'limegreen', 'grey', 'violet', 'grey', 'grey'])
dendrogram = dendrogram(clstr, color_threshold=threshold)
# ax.hlines(30, 0, 1000, color='grey')

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_dendro_{}.png'.
            format(clustering_method),
            dpi=300)


#%% save into a dict
hier_clu = {}
for i in range(len(list_avg_sr)):
    curr_clu_name = key_avg_sr[i]
    hier_clu[curr_clu_name] = str(all_clstr[i])

hier_clu_with_leaves = {
    'clustering result': hier_clu,
    'leaves': leaves
    }


#%% order based on hierarchical leaves and plot heatmap
print('plotting heatmap aligned to leaves...')
heatmap_mat = np.zeros((len(list_avg_sr), 6250))
for i in range(len(list_avg_sr)):
    heatmap_mat[i,:] = normalise(list_avg_sr[leaves[i]][2500:8750])

fig, ax = plt.subplots(figsize=(6,4))
ax.set(title='all LC cells',
       xlabel='time (s)', ylabel='cell ID')

hm = ax.imshow(heatmap_mat, aspect='auto',
               extent=[-1, 4, len(list_avg_sr), 0])
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

for i in range(len(list_avg_sr)):
    if list_hier[i] == '2':
        group1.append(list_avg_sr[i])
        cluster1.append(key_avg_sr[i])
    if list_hier[i] == '4':
        group2.append(list_avg_sr[i])
        cluster2.append(key_avg_sr[i])
    if list_hier[i] == '10':
        group3.append(list_avg_sr[i])
        cluster3.append(key_avg_sr[i])
        
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

fig, ax = plt.subplots(figsize=(4, 12))
ax.set(title='clusters (LC general population)',
       ylabel='proportion of cells')
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