# -*- coding: utf-8 -*-
"""
Created on Mon 30 Sep 17:43:30 2024

plot single cell rasters aligned to run-onsets, with smoothed SR curve overlaid on top

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load raster files 
rasters_run = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_run.npy',
                      allow_pickle=True).item()
profile_run = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_info.npy',
                      allow_pickle=True).item()


#%% plotting
clu_list = list(rasters_run.keys())
xaxis = np.arange(1250*5)/1250-1

for clu in clu_list:
    print(clu)
    raster = rasters_run[clu]
    profile = profile_run[clu]
    tot_trial = raster.shape[0]
    
    fig, ax = plt.subplots(figsize=(2.3, 1.55))

    ax.set(xlim=(-1, 4), xlabel='time (s)', xticks=[0, 2, 4],
           ylim=(-1, tot_trial+1), ylabel='trial #',
           title='{}\nrun-onset aligned'.format(clu))
    axt = ax.twinx(); axt.set(ylabel='spike rate\n(Hz)')
    ax.spines['top'].set_visible(False)
    axt.spines['top'].set_visible(False)
    
    profile_trunc = np.zeros((tot_trial, 5*1250))  # truncated trial profiles 
    for trial in range(tot_trial):
        spikes_run = [s/1250 for s in raster[trial] if s>-1250 and s<5000]  # -1 ~ 4 s
        spike_count_run = len(spikes_run)
        ax.scatter(spikes_run, [trial+1]*spike_count_run, color='grey', s=1, ec='none')
        
        trial_length = len(profile[trial])
        if trial_length>2500+5*1250:
            profile_trunc[trial,:] = profile[trial][2500:2500+5*1250]
        else:
            profile_trunc[trial,:trial_length-2500] = profile[trial][2500:]
    
    mean_profile = np.nanmean(profile_trunc, axis=0)*1250
    axt.plot(xaxis, mean_profile, color='k')
        
    fig.tight_layout()
    plt.show()
    
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_w_curve\{}.png'.format(clu),
                dpi=200, bbox_inches='tight')
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_w_curve\{}.pdf'.format(clu),
                bbox_inches='tight')
    
    plt.close()