# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:17:04 2023

Plot heatmap of average firing profiles of RO-peaking cells based on argmax

@author: Dinghao Luo
"""


#%% imports 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% load data
avg_profile = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_avg_sem.npy',
                      allow_pickle=True).item()['all avg']

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
RO_peaking_keys = []
for cell in cell_prop.index:
    pt = cell_prop['putative'][cell]  # putative
    up = cell_prop['peakness'][cell]  # peakness
    
    if pt and up:
        RO_peaking_keys.append(cell)


#%% data wrangling
max_pt = {}  # argmax for average for all cells
for key in RO_peaking_keys:
    max_pt[key] = np.argmax(avg_profile[key])

def sort_helper(x):
    return max_pt[x]
RO_peaking_keys = sorted(RO_peaking_keys, key=sort_helper)

im_matrix = np.zeros((len(RO_peaking_keys), 1250*8))
for i, key in enumerate(RO_peaking_keys):
    im_matrix[i, :] = normalise(avg_profile[key][:1250*8])


#%% plotting 
fig, ax = plt.subplots(figsize=(7, 5))
ax.set(title='putative Dbh+ cells',
       xlabel='time (s)',
       ylabel='cell #')

im_ordered = ax.imshow(im_matrix, aspect='auto',
                       extent=[-3, 5, len(RO_peaking_keys), 0])
plt.colorbar(im_ordered)

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_putDbh_ROpeaking_ordered_heatmap.png',
            dpi=300,
            bbox_inches='tight',
            transparent=True)


#%% specify RO peaking tagged Dbh cells
tagged_keys = []
for cell in cell_prop.index:
    tg = cell_prop['tagged'][cell]  # tagged
    up = cell_prop['peakness'][cell]  # peakness
    
    if tg and up:
        tagged_keys.append(cell)
        
        
#%% data wrangling
tagged_max_pt = {}  # argmax for average for all cells
for key in tagged_keys:
    tagged_max_pt[key] = np.argmax(avg_profile[key])

def tagged_sort_helper(x):
    return tagged_max_pt[x]
tagged_keys = sorted(tagged_keys, key=tagged_sort_helper)

tagged_im_matrix = np.zeros((len(tagged_keys), 1250*8))
for i, key in enumerate(tagged_keys):
    tagged_im_matrix[i, :] = normalise(avg_profile[key][:1250*8])


#%% plotting 
fig, ax = plt.subplots(figsize=(7, 5))
ax.set(title='tagged Dbh+ cells',
       xlabel='time (s)',
       ylabel='cell #')

tagged_im_ordered = ax.imshow(tagged_im_matrix, aspect='auto',
                              extent=[-3, 5, len(tagged_keys), 0])
plt.colorbar(tagged_im_ordered)

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_ROpeaking_ordered_heatmap.png',
            dpi=300,
            bbox_inches='tight',
            transparent=True)