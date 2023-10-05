# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:50:11 2023

perform UMAP on ACG's from the general population to identify putative Dbh+ 
    neurones

@author: Dinghao Luo
"""


#%% interactive plotting
import umap.plot
# from bokeh.plotting import figure, show, output_notebook
# from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
# from bokeh.palettes import Spectral10
import pandas as pd 


#%% imports 
import umap
import sys
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% load data 
# acgs = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_acg.npy',
#                allow_pickle=True).item()
acgs = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_acg_baseline.npy',
               allow_pickle=True).item()
all_keys = list(acgs.keys())

tag_list = list(np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
                allow_pickle=True).item().keys())
tagged = np.array([int(clu in tag_list) for clu in list(acgs.keys())])
tagged_ind = np.where(tagged==1)[0]
non_tagged_ind = np.where(tagged==0)[0]


#%% main - waveform UMAP 
# waveform_arr = np.array([wf[0,:] for wf in list(waveforms.values())])

# reducer = umap.UMAP()

# scaled_waveform_arr = StandardScaler().fit_transform(waveform_arr)

# wf_embedding = reducer.fit_transform(scaled_waveform_arr)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.scatter(wf_embedding[:, 0], wf_embedding[:, 1], s=2,
#            c=[sns.color_palette()[x] for x in tagged])
# ax.set(title='UMAP projection of waveformss of LC/peri-LC cells')


#%% main - acg UMAP
x = np.arange(-10, 10, 1)
sigma = 2
gaussian = [1 / (sigma*np.sqrt(2*np.pi)) * 
              np.exp(-t**2/(2*sigma**2)) for t in x]

# smooth the acgs
acg_arr = np.array([normalise(np.convolve(acg, gaussian, mode='same')[9800:10000]) for acg in list(acgs.values())])
# acg_arr = np.array([np.convolve(acg, gaussian, mode='same')[9800:10000] for acg in list(acgs.values())])

reducer = umap.UMAP(metric='cosine',
                    min_dist=0.0,
                    n_neighbors=30,
                    random_state=100)

scaled_acg_arr = StandardScaler().fit_transform(acg_arr)

acg_embedding = reducer.fit_transform(scaled_acg_arr)

tagged_med = np.median(acg_embedding[tagged_ind,:], axis=0)
min_dist = np.zeros(len(acgs)); dist2mean = np.zeros(len(acgs))
for ind in non_tagged_ind:
    coord = acg_embedding[ind,:]
    mind = 100  # initialisation
    for j in tagged_ind:
        tg_coord = acg_embedding[j,:]
        new_mind = np.sqrt((coord[0]-tg_coord[0])**2+(coord[1]-tg_coord[1])**2)
        if new_mind<mind:
            mind = new_mind
    min_dist[ind] = mind
    dist2mean[ind] = np.sqrt((coord[0]-tagged_med[0])**2+(coord[1]-tagged_med[1])**2)

min_dist_norm = normalise(np.concatenate((min_dist, np.array([max(min_dist)*1.1]))))
min_dist_cmap = mpl.colormaps['ocean'](min_dist_norm)[:len(acgs)]

dist2mean_norm = normalise(np.concatenate((dist2mean, np.array([max(dist2mean)*1.1]))))
dist2mean_cmap = mpl.colormaps['ocean'](dist2mean_norm)[:len(acgs)]

fig, ax = plt.subplots(figsize=(3.5,3))
umap_scatter = ax.scatter(acg_embedding[:, 0], acg_embedding[:, 1], s=5,
                          c=dist2mean_cmap, alpha=.75, ec='none')
umap_scatter_tagged = ax.scatter(acg_embedding[tagged_ind, 0], acg_embedding[tagged_ind, 1], 
                                 s=5, color='orange', ec='none')
umap_scatter_tgcom = ax.scatter(tagged_med[0], tagged_med[1],
                                s=20, color='darkred')
ax.legend([umap_scatter_tagged, umap_scatter_tgcom], ['tgd. $\it{Dbh}$+', 'tgd. $\it{Dbh}$+ CoM'], 
          frameon=False, loc='upper left', fontsize=8)

xlower = min(acg_embedding[:,0])-.5
xupper = max(acg_embedding[:,0])+.5
ylower = min(acg_embedding[:,1])-.5
yupper = max(acg_embedding[:,1])+.5
ax.set(title='UMAP embedding of ACGs',
       xlabel='1st dim.', ylabel='2nd dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()
plt.show()
# fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_UMAP_acg.png',
#             dpi=300,
#             bbox_inches='tight',
#             transparent=False)

plt.close(fig)


#%% same figure but with greyscale colours
fig, ax = plt.subplots(figsize=(3.5,3))
umap_scatter_grey = ax.scatter(acg_embedding[:,0], acg_embedding[:,1], s=10,
                               c='grey', alpha=.5, ec='none')
umap_scatter_tg_grey = ax.scatter(acg_embedding[tagged_ind, 0], acg_embedding[tagged_ind,1],
                                  s=10, c='royalblue', ec='none')
ax.legend([umap_scatter_tg_grey, umap_scatter_grey], ['tagged', 'non-tagged'],
          frameon=False, loc='upper left', fontsize=7)
ax.set(title='UMAP embedding of ACGs',
       xlabel='1st dim.', ylabel='2nd dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()
plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_UMAP_acg_grey.png',
            dpi=500,
            bbox_inches='tight',
            transparent=False)

plt.close(fig)


#%% colorbar
# a = np.array([[0,1]])
# fig, ax = plt.subplots(figsize=(.25, 6))
# pl.figure(figsize=(.25, 6))
# img = pl.imshow(a, cmap='ocean')
# pl.gca().set_visible(False)
# cax = pl.axes([0.1, 0.2, 0.8, 0.6])
# pl.colorbar(orientation='vertical', cax=cax)

# plt.show()
# fig.savefig('Z:\Dinghao\code_dinghao\LC_all\colourbar_ocean.png',
#             dpi=300,
#             bbox_inches='tight',
#             transparent=False)


#%% save
# min_dist_dict = {}; dist2mean_dict = {}
# for i, key in enumerate(list(acgs.keys())):
#     min_dist_dict[key] = min_dist[i]
#     dist2mean_dict[key] = dist2mean[i]

# np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_UMAP_min_dist.npy',
#         min_dist_dict)
# np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_UMAP_dist2mean.npy',
#         dist2mean_dict)


#%% try k-means 
reducer_10 = umap.UMAP(metric='cosine',
                       min_dist=0.0,
                       n_neighbors=30,
                       n_components=2,
                       random_state=100)
acg_embedding_10 = reducer_10.fit_transform(scaled_acg_arr)

kmeans = KMeans(n_clusters=2)
kmeans.fit(acg_embedding_10)

labels = kmeans.labels_

np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_UMAP_kmeans.npy',
        labels)

fig, ax = plt.subplots(figsize=(3.5,3))

colours = []
for i in labels:
    if i==0:
        colours.append('k')
    else:
        colours.append('grey')

ax.scatter(acg_embedding[:,0], acg_embedding[:,1], c=colours, s=10, ec='none')

c1 = ax.scatter([], [], c='k', s=10, ec='none')
c2 = ax.scatter([], [], c='grey', s=10, ec='none')
ax.legend([c1, c2], ['cluster 1', 'cluster 2'], frameon=False, fontsize=7)

ax.set(title='UMAP embedding of ACGs',
       xlabel='1st dim.', ylabel='2nd dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()
plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_UMAP_acg_kmeans.png',
            dpi=500,
            bbox_inches='tight',
            transparent=False)


#%% replot with categorised data 
fig, ax = plt.subplots(figsize=(3.5,3))

colours = []
for i in labels:
    if i==0:
        colours.append('orange')
    else:
        colours.append('grey')

ax.scatter(acg_embedding[:,0], acg_embedding[:,1], c=colours, s=10, ec='none')
ax.scatter(acg_embedding[tagged_ind, 0], acg_embedding[tagged_ind,1],
                    s=10, c='royalblue', ec='none')

ntggrey = ax.scatter([], [], s=10, c='grey', ec='none')
tgblue = ax.scatter([], [], s=10, c='royalblue', ec='none')
ptorange = ax.scatter([], [], s=10, c='orange', ec='none')

ax.legend([ntggrey, tgblue, ptorange], ['putative Dbh-', 'tagged Dbh+', 'putative Dbh+'], frameon=False, fontsize=7)

ax.set(title='UMAP embedding of ACGs',
       xlabel='1st dim.', ylabel='2nd dim.',
       xticks=[], yticks=[],
       xlim=(xlower, xupper), ylim=(ylower, yupper))

fig.tight_layout()
plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_UMAP_acg_kmeans_categorised.png',
            dpi=500,
            bbox_inches='tight',
            transparent=False)


#%% mapper for plotting 
mapper = reducer.fit(scaled_acg_arr)


#%% diagnostics 
umap.plot.connectivity(mapper, show_points=True, width=1400, height=1200)
umap.plot.connectivity(mapper, edge_bundling='hammer', width=1400, height=1200)
umap.plot.diagnostic(mapper, diagnostic_type='pca')
umap.plot.diagnostic(mapper, diagnostic_type='vq', width=1400, height=1400)
local_dims = umap.plot.diagnostic(mapper, diagnostic_type='local_dim', width=1400, height=1200)
umap.plot.diagnostic(mapper, diagnostic_type='neighborhood', width=1400, height=1200)


#%% interactive (text only)
hover_dict = {}
for key in all_keys:
    if key in tag_list:
        hover_dict[key] = 'tagged'
    else:
        hover_dict[key] = 'not tagged'
hover_arr = np.array(list(hover_dict.values()))

hover_data = pd.DataFrame({'index': np.arange(len(all_keys)),
                            'label': all_keys})
hover_data['item'] = hover_data.label.map(hover_dict)

umap.plot.output_file(r'Z:\Dinghao\code_dinghao\LC_all\interactive_UMAP.html')

p = umap.plot.interactive(mapper, labels=hover_arr, hover_data=hover_data, point_size=10)
umap.plot.show(p)