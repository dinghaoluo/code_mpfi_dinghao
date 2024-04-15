# -*- coding: utf-8 -*-
"""
Created on Mon 25 Sep 15:54:47 2023

plot single cell raster comparisons (alignment check)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 

# plotting parameters 
xaxis = np.arange(-1250, 5000)/1250
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load raster files 
rasters_run = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_run.npy',
                      allow_pickle=True).item()
rasters_rew = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_rew.npy',
                      allow_pickle=True).item()
rasters_cue = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_cue.npy',
                      allow_pickle=True).item()


#%% plotting
clu_list = list(rasters_run.keys())

for clu in clu_list[9:10]:
    print(clu)
    run = rasters_run[clu]
    rew = rasters_rew[clu]
    cue = rasters_cue[clu]
    tot_trial = run.shape[0]
    
    fig, axs = plt.subplots(1,3, figsize=(8, 2))
    
    for i in range(3):
        axs[i].set(xlim=(-1.05, 4.05), ylim=(-1, tot_trial+1))
        for p in ['top', 'right']:
            axs[i].spines[p].set_visible(False)
    axs[0].set_title('run')
    axs[1].set_title('rew')
    axs[2].set_title('cue')
    fig.suptitle(clu)
    
    for trial in range(tot_trial):
        spikes_run = [s/1250 for s in run[trial] if s>-1250 and s<5000]  # -1 ~ 4 s
        spike_count_run = len(spikes_run)
        axs[0].scatter(spikes_run, [trial+1]*spike_count_run, color='grey', s=1.5, ec='none')
        
        spikes_rew = [s/1250 for s in rew[trial] if s>-1250 and s<5000]
        spike_count_rew = len(spikes_rew)
        axs[1].scatter(spikes_rew, [trial+1]*spike_count_rew, color='grey', s=1.5, ec='none')
        
        spikes_cue = [s/1250 for s in cue[trial] if s>-1250 and s<5000]
        spike_count_cue = len(spikes_cue)
        axs[2].scatter(spikes_cue, [trial+1]*spike_count_cue, color='grey', s=1.5, ec='none')
        
    fig.tight_layout()
    plt.show()
    
    # fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_alignment_compare\{}.png'.format(clu),
    #             dpi=500,
    #             bbox_inches='tight')
    fig.savefig(r'Z:\Dinghao\paper\figures\figure_1_egLC.pdf',
                dpi=500,
                bbox_inches='tight')
    
    plt.close()