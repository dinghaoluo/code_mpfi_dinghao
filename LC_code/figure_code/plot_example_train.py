# -*- coding: utf-8 -*-
"""
Created on Fri 29 Sep 16:15:50 2023

plot example spike train to show ACG calc

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt


#%% load example raster
raster = np.load('Z:/Dinghao/code_dinghao/HPC_all/HPC_all_rasters_npy_simp/A063r-20230706-01.npy',
                 allow_pickle=True).item()

first_train = list(raster.values())[0][0]

spikes = np.where(first_train==1)[0]


#%% plot 
fig, ax = plt.subplots(figsize=(6,2))

for s in spikes:
    ax.plot([s, s], [0, 1], color='k', linewidth=0.5)

ax.set(ylim=(-10, 10))

fig.tight_layout()

plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egtrain.png',
            dpi=300,
            bbox_inches='tight')


#%% histogram 
fig, ax = plt.subplots(figsize=(2,1))

for p in ['top','right']:
    ax.spines[p].set_visible(False)
for p in ['bottom','left']:
    ax.spines[p].set_linewidth(1)
    
ax.set_xticks([])
ax.set_yticks([])

ax.hist(spikes[:10], bins=10, edgecolor='k', linewidth=1)
ax.set(xlim=(0, 3000))

fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egtrain_hist.png',
            dpi=500,
            bbox_inches='tight')


#%% histogram shifted
fig, ax = plt.subplots(figsize=(2,1))

for p in ['top','right']:
    ax.spines[p].set_visible(False)
for p in ['bottom','left']:
    ax.spines[p].set_linewidth(1)
    
ax.set_xticks([])
ax.set_yticks([])

ax.hist([s+500 for s in spikes][:10], bins=10, edgecolor='k', linewidth=1, alpha=.5)
ax.set(xlim=(0,3000))

fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egtrain_hist_shifted.png',
            dpi=500,
            bbox_inches='tight')