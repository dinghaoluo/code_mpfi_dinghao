# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:04:39 2023

plot waveforms, CCG's and rasters of all LC cells

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import sys 
import pandas as pd
 
if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% load files 
waveforms = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_waveforms.npy',
                    allow_pickle=True).item()

rasters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()
    
# ACGs = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_acg.npy', 
#                allow_pickle=True).item()
ACGs = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_acg_baseline.npy', 
               allow_pickle=True).item()

tag_list = list(np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
                allow_pickle=True).item().keys())

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% get putative 
putative_keys = []
for cell in cell_prop.index:
    pt = cell_prop['putative'][cell]  # putative
    
    if pt:
        putative_keys.append(cell)


#%% function
def get_waveform(cluname):
    return waveforms[cluname][0,:], waveforms[cluname][1,:]

def get_raster(cluname):
    return rasters[cluname]

def get_ACG(cluname):
    return ACGs[cluname]


#%% extract info and set parameters 
all_clu = list(rasters.keys())

ACGrange = 200  # millisecond
ACGmidt = (len(list(ACGs.values())[0])-1)/2  # midpoint of ACG's
t1 = int(ACGmidt-ACGrange); t2 = int(ACGmidt+ACGrange)


#%% plotting loop
for cluname in all_clu[24:27]:
    print('plotting {}'.format(cluname))
    
    if cluname in tag_list:
        colour = 'k'
        colourACG = 'royalblue'
        title = cluname + '_tagged'
    elif cluname in putative_keys:
        colour = 'k'
        colourACG = 'orange'
        title = cluname + '_putative'
    else:
        colour = 'grey'
        colourACG = 'grey'
        title = cluname
    
    wf, wfsem = get_waveform(cluname)
    raster = get_raster(cluname)
    acg = get_ACG(cluname)
    
    fig, axs = plt.subplot_mosaic('AB;CC', figsize=(4,5))
    fig.suptitle(title, color=colour)
    
    # waveform plot 
    for p in ['left', 'right', 'top']:
        axs['A'].spines[p].set_visible(False)
    axs['A'].set_yticks([])
    xaxis = np.arange(len(wf))
    axs['A'].plot(wf, colourACG)
    axs['A'].fill_between(xaxis, wf+wfsem, wf-wfsem,
                          color=colourACG, alpha=.25)
    
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
    x = np.arange(-ACGrange, ACGrange, 1)
    sigma = 2
    gaussian = [1 / (sigma*np.sqrt(2*np.pi)) * 
                np.exp(-t**2/(2*sigma**2)) for t in x]
    for p in ['right', 'top']:
        axs['C'].spines[p].set_visible(False)
    axs['C'].set(xlabel='lag (ms)', ylabel='norm. correlation',
                 xlim=(-195, 195), ylim=(0,1),
                 xticks=[-150,-50,50,150], yticks=[0,.5,1])
    xaxis = np.arange(-ACGrange, ACGrange)
    axs['C'].plot(xaxis, normalise(np.convolve(acg[t1:t2], gaussian, mode='same')), 
                  colourACG)
    axs['C'].fill_between(xaxis, 
                          [0]*ACGrange*2, 
                          normalise(np.convolve(acg[t1:t2], gaussian, mode='same')),
                          color=colourACG, alpha=.3)
        
    # save 
    fig.tight_layout()
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_property_plots\{}.pdf'.format(title),
                dpi=300,
                bbox_inches='tight')
    
    plt.close(fig)