# -*- coding: utf-8 -*-
"""
Created on Mon June 12 16:39:23 2023

assistant plots for UMAP interactive image prep

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


#%% load files 
waveforms = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_waveforms.npy',
                    allow_pickle=True).item()

rasters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()
    
# ACGs = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_acg.npy', 
#                allow_pickle=True).item()
ACGs = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_acg_baseline.npy', 
               allow_pickle=True).item()

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
putative_keys = []; tagged_keys = []
for cell in cell_prop.index:
    pt = cell_prop['putative'][cell]  # putative
    tg = cell_prop['tagged'][cell]  # tagged
    
    if pt:
        putative_keys.append(cell)
    if tg:
        tagged_keys.append(cell)


#%% function
def get_waveform(cluname):
    return waveforms[cluname][0,:], waveforms[cluname][1,:]

def get_raster(cluname):
    return rasters[cluname]

def get_ACG(cluname):
    return ACGs[cluname]


#%% extract info and set parameters 
all_clu = list(rasters.keys())

ACGrange = 500  # millisecond
ACGmidt = (len(list(ACGs.values())[0])-1)/2  # midpoint of ACG's
t1 = int(ACGmidt-ACGrange); t2 = int(ACGmidt+ACGrange)


#%% plotting loop
for cluname in all_clu:
    print('plotting {}'.format(cluname))
    colour = 'grey'
    title = cluname
    
    wf, wfsem = get_waveform(cluname)
    raster = get_raster(cluname)
    acg = get_ACG(cluname)
    
    fig, axs = plt.subplot_mosaic('AB;CC', figsize=(8,8))
    fig.suptitle(title, color=colour)
    
    # waveform plot 
    for p in ['left', 'right', 'top']:
        axs['A'].spines[p].set_visible(False)
    axs['A'].set_yticks([])
    xaxis = np.arange(len(wf))
    axs['A'].plot(wf, colour)
    axs['A'].fill_between(xaxis, wf+wfsem, wf-wfsem,
                          color=colour, alpha=.25)
    
    # raster plot
    for p in ['right', 'top']:
        axs['B'].spines[p].set_visible(False)
    axs['B'].set(xlabel='time (s)', ylabel='trial #')
    tot_trial = raster.shape[0]  # how many trials
    for trial in range(tot_trial):
        curr_trial = np.where(raster[trial]==1)[0]
        curr_trial = [(s-3750)/1250 for s in curr_trial]
        axs['B'].scatter(curr_trial, [trial+1]*len(curr_trial),
                         color='grey', s=.35)
    
    # ACG plot 
    for p in ['right', 'top', 'left']:
        axs['C'].spines[p].set_visible(False)
    axs['C'].set(xlabel='time (ms)')
    axs['C'].set_yticks([])
    axs['C'].set_yticklabels([])
    xaxis = np.arange(-ACGrange, ACGrange)
    axs['C'].plot(xaxis, acg[t1:t2], colour)
        
    # save 
    fig.tight_layout()
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\UMAP_interactive_images\{}.png'.format(title),
                dpi=300,
                bbox_inches='tight',
                transparent=False)