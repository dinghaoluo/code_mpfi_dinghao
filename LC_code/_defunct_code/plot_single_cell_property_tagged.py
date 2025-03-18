# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:04:39 2023

plot waveforms, CCG's and rasters of tagged LC cells

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
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

tag_list = list(np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
                allow_pickle=True).item().keys())


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
    if cluname in tag_list:
        print('plotting {}'.format(cluname))
        
        colour = 'k'
        title = cluname + '_tagged'
        
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
        for p in ['right', 'top']:
            axs['C'].spines[p].set_visible(False)
        axs['C'].set(xlabel='time (ms)', ylabel='spike rate (Hz)')
        xaxis = np.arange(-ACGrange, ACGrange)
        axs['C'].plot(xaxis, acg[t1:t2], colour)
            
        # save 
        fig.tight_layout()
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\single_cell_property_plots\{}.png'.format(title),
                    dpi=300,
                    bbox_inches='tight',
                    transparent=False)