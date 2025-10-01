# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:50:11 2023
Modified on 26 Feb 2025 to install wrapper and clean up data structure loading

perform UMAP on ACG's from the general population to identify putative Dbh+ 
    neurones

@author: Dinghao Luo
"""


#%% imports 
import umap
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans

from common import normalise, mpl_formatting, gaussian_kernel_unity
mpl_formatting()

import rec_list
paths = rec_list.pathLC


#%% parameters 
colour_tag = (70/255, 101/255, 175/255)
colour_put = (101/255, 82/255, 163/255)


#%% main 
keys, identities, acgs = [], [], []  # at this point there is only tagged and non-tagged in identities 

for path in paths:
    recname = path[-17:]
    
    curr_identities = np.load(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}\{recname}_all_identities.npy',
        allow_pickle=True
        ).item()
    curr_acgs = np.load(
        rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}\{recname}_all_ACGs.npy',
        allow_pickle=True
        ).item()
    
    keys.extend(curr_identities.keys())
    identities.extend([int(i) for i in curr_identities.values()])
    acgs.extend(curr_acgs.values())
    
tagged_idx, non_tagged_idx = [i for i, val in enumerate(identities) if val==1], [i for i, val in enumerate(identities) if val==0]


## ACG UMAP
gaussian = gaussian_kernel_unity(sigma=2)

# smooth the acgs
acg_arr = np.array([normalise(np.convolve(acg, gaussian, mode='same')[9800:10000]) for acg in acgs])

reducer = umap.UMAP(metric='cosine',
                    min_dist=0.0,
                    n_neighbors=30,
                    random_state=666)

scaled_acg_arr = StandardScaler().fit_transform(acg_arr)

acg_embedding = reducer.fit_transform(scaled_acg_arr)

tagged_med = np.median(acg_embedding[tagged_idx,:], axis=0)
min_dist = np.zeros(len(acgs)); dist2mean = np.zeros(len(acgs))
for idx in non_tagged_idx:
    coord = acg_embedding[idx,:]
    mind = 100  # initialisation
    for j in tagged_idx:
        tg_coord = acg_embedding[j,:]
        new_mind = np.sqrt((coord[0]-tg_coord[0])**2+(coord[1]-tg_coord[1])**2)
        if new_mind<mind:
            mind = new_mind
    min_dist[idx] = mind
    dist2mean[idx] = np.sqrt((coord[0]-tagged_med[0])**2+(coord[1]-tagged_med[1])**2)
    
dist2mean_norm = normalise(np.concatenate((dist2mean, np.array([max(dist2mean)*1.1]))))
dist2mean_cmap = mpl.colormaps['ocean'](dist2mean_norm)[:len(acgs)]


## plotting
fig, ax = plt.subplots(figsize=(3,2.6))
umap_scatter = ax.scatter(acg_embedding[:, 0], acg_embedding[:, 1], s=10,
                          c=dist2mean_cmap, alpha=.75, ec='none')
umap_scatter_tagged = ax.scatter(acg_embedding[tagged_idx, 0], acg_embedding[tagged_idx, 1], 
                                 s=10, color=colour_tag, ec='none')
umap_scatter_tgcom = ax.scatter(tagged_med[0], tagged_med[1],
                                s=20, color='darkred')
ax.legend([umap_scatter_tagged, umap_scatter_tgcom], ['tgd. $\it{Dbh}$+', 'tgd. $\it{Dbh}$+ CoM'], 
          frameon=False, loc='upper left', fontsize=8)

xlower = min(acg_embedding[:,0])-1.2
xupper = max(acg_embedding[:,0])+1.2
ylower = min(acg_embedding[:,1])-.8
yupper = max(acg_embedding[:,1])+.8
ax.set(title='UMAP embedding of ACGs',
       xlabel='1st dim.', ylabel='2nd dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\UMAP\LC_all_UMAP_distance_to_tagged_med{ext}',
                dpi=300,
                bbox_inches='tight')


## same figure but with greyscale colours
fig, ax = plt.subplots(figsize=(3,2.6))
umap_scatter_grey = ax.scatter(acg_embedding[:,0], acg_embedding[:,1], s=10,
                               c='grey', alpha=.5, ec='none')
umap_scatter_tg_grey = ax.scatter(acg_embedding[tagged_idx, 0], acg_embedding[tagged_idx,1],
                                  s=10, c=colour_tag, ec='none')
ax.legend([umap_scatter_tg_grey, umap_scatter_grey], ['tagged', 'non-tagged'],
          frameon=False, loc='upper left', fontsize=7)
ax.set(title='UMAP embedding of ACGs',
       xlabel='1st dim.', ylabel='2nd dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\UMAP\LC_all_UMAP_greyscale{ext}',
                dpi=300,
                bbox_inches='tight')


## k-means 
reducer_10 = umap.UMAP(metric='cosine',
                       min_dist=0.0,
                       n_neighbors=30,
                       n_components=2,  # identify 2 clusters
                       random_state=666)
acg_embedding_10 = reducer_10.fit_transform(scaled_acg_arr)

kmeans = KMeans(n_clusters=2)
kmeans.fit(acg_embedding_10)

labels = kmeans.labels_

# IMPORTANT: post-hoc relabelling based on tagged to ensure that the
#   cluster labels stay the same across runs (1 for tagged/putative and 
#   0 for other)
if labels[tagged_idx[0]] == 0:  # if tagged (and thus putative) is 0
    labels = [1-x for x in labels]  # flip 1's and 0's

labels_dict = {keys[i]: labels[i] for i in range(len(keys))}

np.save(
        r'Z:\Dinghao\code_dinghao\LC_ephys\UMAP\LC_all_UMAP_kmeans.npy',
        labels_dict
        )


## plotting
fig, ax = plt.subplots(figsize=(3,2.6))

colours = []
for i in labels:
    if i==1:
        colours.append('k')
    else:
        colours.append('grey')

ax.scatter(acg_embedding[:,0], acg_embedding[:,1], c=colours, s=10, ec='none', alpha=.5)

c1 = ax.scatter([], [], c='k', s=10, ec='none')
c2 = ax.scatter([], [], c='grey', s=10, ec='none')
ax.legend([c1, c2], ['cluster 1', 'cluster 2'], frameon=False, fontsize=8)

ax.set(title='UMAP embedding of ACGs',
       xlabel='1st dim.', ylabel='2nd dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\UMAP\LC_all_UMAP_kmeans_binary{ext}',
                dpi=300,
                bbox_inches='tight')


## replot with categorised data 
fig, ax = plt.subplots(figsize=(3,2.6))

colours = []
for i in labels:
    if i==1:
        colours.append(colour_put)
    else:
        colours.append('grey')

ax.scatter(acg_embedding[:,0], acg_embedding[:,1], 
           c=colours, s=10, ec='k', linewidth=.35, alpha=1)
ax.scatter(acg_embedding[tagged_idx, 0], acg_embedding[tagged_idx, 1],
           s=10, c=colour_tag, ec='k', linewidth=.35, alpha=1)

# for legend
ntgcolor = ax.scatter([], [], 
                      s=10, ec='k', c='grey', linewidth=.35, alpha=1)
tgcolor = ax.scatter([], [], 
                     s=10, ec='k', c=colour_tag, linewidth=.35, alpha=1)
ptcolor = ax.scatter([], [], 
                     s=10, ec='k', c=colour_put, linewidth=.35, alpha=1)

ax.legend([tgcolor, ptcolor, ntgcolor], 
          ['tagged Dbh+', 'putative Dbh+', 'putative Dbh-'], 
          frameon=False, fontsize=8)

ax.set(title='UMAP embedding of ACGs',
       xlabel='1st dim.', ylabel='2nd dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_ephys\UMAP'
                rf'\LC_all_UMAP_kmeans_categorised{ext}',
                dpi=300,
                bbox_inches='tight')


# ## below is for interactive plotting 
# # imports 
# import umap.plot as umap_plot  # to avoid shadowing global import of umap
# import pandas as pd 

# # mapper for plotting 
# mapper = reducer.fit(scaled_acg_arr)

# umap_plot.connectivity(mapper, show_points=True, width=1400, height=1200)
# umap_plot.connectivity(mapper, edge_bundling='hammer', width=1400, height=1200)
# umap_plot.diagnostic(mapper, diagnostic_type='pca')
# umap_plot.diagnostic(mapper, diagnostic_type='vq', width=1400, height=1400)
# umap_plot.diagnostic(mapper, diagnostic_type='local_dim', width=1400, height=1200)
# umap_plot.diagnostic(mapper, diagnostic_type='neighborhood', width=1400, height=1200)

# # interactive (text only)
# hover_dict = {key: identities[idx]
#               for idx, key in enumerate(keys)}
# hover_arr = np.array([*hover_dict.values()])

# hover_data = pd.DataFrame({'index': np.arange(len(keys)),
#                            'label': keys})
# hover_data['item'] = hover_data.label.map(hover_dict)

# umap.plot.output_file(
#     r'Z:\Dinghao\code_dinghao\LC_ephys\UMAP\interactive_UMAP.html'
#     )

# p = umap.plot.interactive(mapper, labels=hover_arr, hover_data=hover_data, point_size=10)
# umap.plot.show(p)