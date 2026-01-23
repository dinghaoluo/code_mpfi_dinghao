# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:50:11 2023
Modified on 26 Feb 2025 to install wrapper and clean up data structure loading

perform UMAP on ACG's from the general population to identify putative Dbh+ 
    neurones

@author: Dinghao Luo
"""


#%% imports 
from pathlib import Path 

import umap
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans

from common import normalise, mpl_formatting, gaussian_kernel_unity
mpl_formatting()

from common import colour_putative, colour_tagged, colour_other

import rec_list
paths = rec_list.pathLC


#%% parameters and path stems 
all_sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
UMAP_stem     = Path('Z:/Dinghao/code_dinghao/LC_ephys/UMAP')


#%% main 
keys, identities, ACGs = [], [], []  # at this point there is only tagged and non-tagged in identities 

for path in paths:
    recname = Path(path).name
    
    # identities 
    curr_identities_path = all_sess_stem / recname / f'{recname}_all_identities.npy'
    curr_identities = np.load(curr_identities_path, allow_pickle=True).item()
    
    # ACGs
    curr_ACGs_path = all_sess_stem / recname / f'{recname}_all_ACGs.npy'
    curr_ACGs = np.load(curr_ACGs_path, allow_pickle=True).item()
    
    keys.extend(curr_identities.keys())
    identities.extend([int(i) for i in curr_identities.values()])
    ACGs.extend(curr_ACGs.values())
    
tagged_idx, non_tagged_idx = ([i for i, val in enumerate(identities) if val==1], 
                              [i for i, val in enumerate(identities) if val==0])


# ----------------
# ACG UMAP starts 
# ----------------
# create Gaussian kernel for smoothing 
Gaussian = gaussian_kernel_unity(sigma=2)

# smooth the ACGs (only need one side)
ACG_arr = np.array(
    [normalise(np.convolve(ACG, Gaussian, mode='same')[9800:10000]) for ACG in ACGs]
    )

# initiate reducer 
reducer = umap.UMAP(metric='cosine',
                    min_dist=0.0,
                    n_neighbors=30,
                    random_state=666)

# standard-scale the array first 
scaled_ACG_arr = StandardScaler().fit_transform(ACG_arr)

# actual embedding 
ACG_embedding = reducer.fit_transform(scaled_ACG_arr)
# --------------
# ACG UMAP ends 
# --------------


# get coördinates of centres
tagged_med = np.median(ACG_embedding[tagged_idx,:], axis=0)

min_dist   = np.zeros(len(ACGs))
dist2mean  = np.zeros(len(ACGs))
for idx in non_tagged_idx:
    coörd = ACG_embedding[idx,:]
    
    mind = 100  # initialisation
    for j in tagged_idx:
        tg_coörd = ACG_embedding[j,:]
        new_mind = np.sqrt((coörd[0]-tg_coörd[0])**2 + (coörd[1]-tg_coörd[1])**2)
        if new_mind<mind:
            mind = new_mind
    
    min_dist[idx]  = mind
    dist2mean[idx] = np.sqrt((coörd[0]-tagged_med[0])**2 + (coörd[1]-tagged_med[1])**2)
    
dist2mean_norm = normalise(np.concatenate((dist2mean, np.array([max(dist2mean)*1.1]))))
dist2mean_cmap = mpl.colormaps['ocean'](dist2mean_norm)[:len(ACGs)]


#%% plotting
fig, ax = plt.subplots(figsize=(3,2.6))
umap_scatter        = ax.scatter(ACG_embedding[:, 0], ACG_embedding[:, 1], s=10,
                                 c=dist2mean_cmap, alpha=.75, ec='none')
umap_scatter_tagged = ax.scatter(ACG_embedding[tagged_idx, 0], ACG_embedding[tagged_idx, 1], 
                                 s=10, color=colour_tagged, ec='none')
umap_scatter_tgcom  = ax.scatter(tagged_med[0], tagged_med[1],
                                 s=20, color='darkblue')
ax.legend([umap_scatter_tagged, umap_scatter_tgcom], 
          ['Tgd. Dbh+', 'Tgd. Dbh+ CoM'], 
          frameon=False, loc='upper left', fontsize=8)

xlower = min(ACG_embedding[:,0]) - 1.2
xupper = max(ACG_embedding[:,0]) + 1.2
ylower = min(ACG_embedding[:,1]) - .8
yupper = max(ACG_embedding[:,1]) + .8
ax.set(title='UMAP embedding of ACGs',
       xlabel='First dim.', ylabel='Second dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(
        UMAP_stem / f'LC_all_UMAP_distance_to_tagged_med{ext}',
        dpi=300,
        bbox_inches='tight'
        )


# k-means
reducer_10 = umap.UMAP(metric='cosine',
                       min_dist=0.0,
                       n_neighbors=30,
                       n_components=2,  # identify 2 clusters
                       random_state=666)
ACG_embedding_10 = reducer_10.fit_transform(scaled_ACG_arr)

kmeans = KMeans(n_clusters=2)
kmeans.fit(ACG_embedding_10)

labels = kmeans.labels_

# IMPORTANT: post-hoc relabelling based on tagged to ensure that the
#   cluster labels stay the same across runs (1 for tagged/putative and 
#   0 for other)
if labels[tagged_idx[0]] == 0:  # if tagged (and thus putative) is 0
    labels = [1-x for x in labels]  # flip 1's and 0's

labels_dict = {keys[i]: labels[i] for i in range(len(keys))}

np.save(UMAP_stem / 'LC_all_UMAP_kmeans.npy', labels_dict)


## plotting
fig, ax = plt.subplots(figsize=(3,2.6))

colours = []
for i in labels:
    if i==1:
        colours.append('k')
    else:
        colours.append('grey')

ax.scatter(ACG_embedding[:,0], ACG_embedding[:,1], c=colours, s=10, ec='none', alpha=.5)

c1 = ax.scatter([], [], c='k', s=10, ec='none')
c2 = ax.scatter([], [], c='grey', s=10, ec='none')
ax.legend([c1, c2], ['cluster 1', 'cluster 2'], frameon=False, fontsize=8)

ax.set(title='UMAP embedding of ACGs',
       xlabel='First dim.', ylabel='Second dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(UMAP_stem / f'LC_all_UMAP_kmeans_binary{ext}',
                dpi=300,
                bbox_inches='tight')


#%% replot with categorised data 
fig, ax = plt.subplots(figsize=(2.4,2.15))

colours = []
for i in labels:
    if i==1:
        colours.append(colour_putative)
    else:
        colours.append(colour_other)

ax.scatter(ACG_embedding[:,0], ACG_embedding[:,1], 
           c=colours, s=8, ec='k', linewidth=.5, alpha=1)
ax.scatter(ACG_embedding[tagged_idx, 0], ACG_embedding[tagged_idx, 1],
           s=8, c=colour_tagged, ec='k', linewidth=.5, alpha=1)

# for legend
ntgcolor = ax.scatter([], [], 
                      s=10, ec='k', c=colour_other, linewidth=.5, alpha=1)
tgcolor = ax.scatter([], [], 
                     s=10, ec='k', c=colour_tagged, linewidth=.5, alpha=1)
ptcolor = ax.scatter([], [], 
                     s=10, ec='k', c=colour_putative, linewidth=.5, alpha=1)

ax.legend([tgcolor, ptcolor, ntgcolor], 
          ['Tagged Dbh+', 'Putative Dbh+', 'Putative Dbh-'], 
          frameon=False, fontsize=8)

ax.set(title='UMAP embedding of ACGs',
       xlabel='First dim.', ylabel='Second dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(UMAP_stem / f'LC_all_UMAP_kmeans_categorised{ext}',
                dpi=300,
                bbox_inches='tight')