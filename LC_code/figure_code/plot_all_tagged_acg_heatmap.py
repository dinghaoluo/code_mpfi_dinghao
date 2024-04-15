# -*- coding: utf-8 -*-
"""
Created on Wed 6 Sep 17:32:59 2023

all tagged ACG heatmap

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
import pandas as pd 
import sys

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% load data 
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')
tagged_keys = []; putative_keys = []; non_keys = []
for cell in cell_prop.index:
    tg = cell_prop['tagged'][cell]  # if tagged 
    pt = cell_prop['putative'][cell]  # if putative
    
    if tg:
        tagged_keys.append(cell)
    if pt:
        putative_keys.append(cell)
    if not tg and not pt:
        non_keys.append(cell)
    
        
acgs = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_acg_baseline.npy',
               allow_pickle=True).item()


#%% smoothing Gaussian kernel 
x = np.arange(-10, 10, 1)  # ms 
sigma = 2  # ms 
gaussian = [1 / (sigma*np.sqrt(2*np.pi)) * 
            np.exp(-t**2/(2*sigma**2)) for t in x]


#%% main (tagged)
max_pt = {}  # argmax for average for all cells
for cluname in tagged_keys:
    max_pt[cluname] = np.mean(acgs[cluname][9950:10050])

def sort_helper(x):
    return max_pt[x]
tagged_keys = sorted(tagged_keys, key=sort_helper)

acg_mat = np.zeros((len(tagged_keys), 399))  # hard-coded because what's the point

for i, cluname in enumerate(tagged_keys):
    acg = acgs[cluname][9800:10201]
    acg_scaled = normalise(np.convolve(acg, gaussian, mode='same')[2:])
    
    acg_mat[i,:] = acg_scaled


#%% plot (tagged)
fig, ax = plt.subplots(figsize=(4,4))

image = ax.imshow(acg_mat, aspect='auto', extent=[-200,200,len(tagged_keys)+1,1])

ax.set(xlabel='lag (ms)', ylabel='cell #')
plt.colorbar(image)

fig.suptitle('ACGs of tagged $\it{Dbh}$+ cells')

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_ACG_heatmap.png',
         dpi=500,
         bbox_inches='tight')

plt.close(fig)


#%% main (putative Dbh+)
max_pt_put = {}  # argmax for average for all cells
for cluname in putative_keys:
    max_pt_put[cluname] = np.mean(acgs[cluname][9950:10050])

def sort_helper_put(x):
    return max_pt_put[x]
putative_keys = sorted(putative_keys, key=sort_helper_put)

acg_mat_put = np.zeros((len(putative_keys), 399))  # hard-coded because what's the point

for i, cluname in enumerate(putative_keys):
    acg = acgs[cluname][9800:10201]
    acg_scaled = normalise(np.convolve(acg, gaussian, mode='same')[2:])
    
    acg_mat_put[i,:] = acg_scaled


#%% plot (putative Dbh-)
fig, ax = plt.subplots(figsize=(4,4))

image = ax.imshow(acg_mat_put, aspect='auto', extent=[-200,200,len(putative_keys)+1,1])

ax.set(xlabel='lag (ms)', ylabel='cell #')
plt.colorbar(image)

fig.suptitle('ACGs of putative $\it{Dbh}$+ cells')

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_putative_Dbh_ACG_heatmap.png',
            dpi=500,
            bbox_inches='tight')

plt.close(fig)


#%% main (putative Dbh-)
max_pt_non = {}  # argmax for average for all cells
for cluname in non_keys:
    max_pt_non[cluname] = np.mean(acgs[cluname][9950:10050])

def sort_helper_non(x):
    return max_pt_non[x]
non_keys = sorted(non_keys, key=sort_helper_non)

acg_mat_non = np.zeros((len(non_keys), 399))  # hard-coded because what's the point

for i, cluname in enumerate(non_keys):
    acg = acgs[cluname][9800:10201]
    acg_scaled = normalise(np.convolve(acg, gaussian, mode='same')[2:])
    
    acg_mat_non[i,:] = acg_scaled


#%% plot (putative Dbh-)
fig, ax = plt.subplots(figsize=(4,4))

image = ax.imshow(acg_mat_non, aspect='auto', extent=[-200,200,len(non_keys)+1,1])

ax.set(xlabel='lag (ms)', ylabel='cell #')
plt.colorbar(image)

fig.suptitle('ACGs of putative $\it{Dbh}$- cells')

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_putative_Dbhneg_ACG_heatmap.png',
            dpi=500,
            bbox_inches='tight')

plt.close(fig)