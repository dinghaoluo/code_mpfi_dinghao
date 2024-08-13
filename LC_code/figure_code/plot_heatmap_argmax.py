# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:17:04 2023
Modified on Sat 13 Apr 11:00:14 2024: changed heatmap cmap 

Plot heatmap of average firing profiles based on argmax

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

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load data
avg_profile = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_avg_sem.npy',
                      allow_pickle=True).item()['all avg']

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
putative_keys = []
for cell in cell_prop.index:
    pt = cell_prop['putative'][cell]  # putative
    
    if pt:
        putative_keys.append(cell)


#%% data wrangling
max_pt = {}  # argmax for average for all cells
for key in putative_keys:
    max_pt[key] = np.argmax(avg_profile[key])

def sort_helper(x):
    return max_pt[x]
putative_keys = sorted(putative_keys, key=sort_helper)

im_matrix = np.zeros((len(putative_keys), 1250*8))
for i, key in enumerate(putative_keys):
    im_matrix[i, :] = normalise(avg_profile[key][:1250*8])


#%% plotting 
fig, ax = plt.subplots(figsize=(4, 3.8))
ax.set(xlabel='time (s)',
       ylabel='cell #')
fig.suptitle('putative Dbh+ cells')

im_ordered = ax.imshow(im_matrix, aspect='auto',
                       extent=[-3, 5, len(putative_keys), 0])
plt.colorbar(im_ordered, shrink=.5)

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_putDbh_ordered_heatmap.pdf',
            bbox_inches='tight')

plt.close(fig)


#%% specify RO peaking tagged Dbh cells
tagged_keys = []
for cell in cell_prop.index:
    tg = cell_prop['tagged'][cell]  # putative
    
    if tg:
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
fig, ax = plt.subplots(figsize=(4, 3.8))
ax.set(xlabel='time (s)',
       ylabel='cell #')
fig.suptitle('tagged Dbh+ cells')

tagged_im_ordered = ax.imshow(tagged_im_matrix, aspect='auto',
                              extent=[-3, 5, len(tagged_keys), 0])
plt.colorbar(tagged_im_ordered, shrink=.5)

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_ordered_heatmap.pdf',
            bbox_inches='tight')

plt.close(fig)


#%% all tagged and putative
keys = []
for cell in cell_prop.index:
    tg = cell_prop['tagged'][cell]
    pt = cell_prop['putative'][cell]
    
    if tg or pt:
        keys.append(cell)
        
        
#%% data wrangling
max_pt = {}  # argmax for average for all cells
for key in keys:
    max_pt[key] = np.argmax(avg_profile[key])

def sort_helper(x):
    return max_pt[x]
keys = sorted(keys, key=sort_helper)

im_matrix = np.zeros((len(keys), 1250*5))
for i, key in enumerate(keys):
    im_matrix[i, :] = normalise(avg_profile[key][2500:1250*4+3750])


#%% plotting 
fig, ax = plt.subplots(figsize=(2.6,2.3))
ax.set(xlabel='time (s)',
       ylabel='cell #')
fig.suptitle('Dbh+ cells')

im_ordered = ax.imshow(im_matrix, aspect='auto',
                       extent=[-1, 4, 1, len(keys)], cmap='Greys')
plt.colorbar(im_ordered, shrink=.5, ticks=[0,1], label='norm. spike rate')

ax.set(yticks=[1, 100, 200],
       xticks=[0,2,4])

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_pooled_ordered_heatmap.png',
            dpi=500, bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\paper\figures\figure_1_orderedLC.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_pooled_ordered_heatmap.pdf',
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\paper\figures\figure_1_orderedLC.pdf',
            bbox_inches='tight')

plt.close(fig)