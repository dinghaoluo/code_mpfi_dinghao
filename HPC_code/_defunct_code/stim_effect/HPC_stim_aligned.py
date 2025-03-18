# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:24:17 2023

Stim-aligned spiking profile 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
from scipy.stats import sem

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt

rasters = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters.npy',
                  allow_pickle=True).item()

plot_single_cell = True


#%% functions 
def conv_raster(rastarr):
    """
    Parameters
    ----------
    rastarr : raster array, trials x time bins 

    Returns
    -------
    conv_arr : raster array after convolution
    """
    gx_spike = np.arange(-500, 500, 1)
    sigma_spike = 1250/10
    gaus_spike = [1 / (sigma_spike*np.sqrt(2*np.pi)) * 
                  np.exp(-x**2/(2*sigma_spike**2)) for x in gx_spike]
    
    tot_trial = rastarr.shape[0]  # how many trials
    conv_arr = np.zeros((tot_trial, 6250))
    for trial in range(tot_trial):
        curr_trial = rastarr[trial]
        curr_trial = np.convolve(curr_trial, gaus_spike,
                                 mode='same')*1250
        conv_arr[trial] = curr_trial[2500:8750]

    return conv_arr  


#%% extract spikes aligned (only run after all_rasters)
for pathname in pathHPC:
    sessname = pathname[-17:]
    
    for clu in list(rasters.items()):
        if clu[0][:17] == sessname:
            clu_name = clu[0]
            raster = conv_raster(rasters[clu_name])
            tot_trial = raster.shape[0]
            
            divider1 = clu_name.find(' ', clu_name.find(' ')+1)  # find 2nd space
            divider2 = divider1 + 2  # 3rd space
            stimtype = clu_name[divider1+1]  # stimtype after 2nd space
            stimwind = clu_name[divider2+1:]
            stim_divider = stimwind.find(' ')
            stimstart = int(stimwind[:stim_divider])
            stimend = int(stimwind[stim_divider+1:])+1
            
            # ALL START FROM 0
            stim_trials = np.arange(stimstart, stimend, 3)-1
            non_stim_trials = np.delete(np.arange(0, tot_trial), stim_trials)
            
            stim_profiles = raster[stim_trials]
            stim_avg = np.mean(stim_profiles, axis=0)
            stim_sem = sem(stim_profiles, axis=0)
            
            non_stim_profiles = raster[non_stim_trials]
            non_stim_avg = np.mean(non_stim_profiles, axis=0)
            non_stim_sem = sem(non_stim_profiles, axis=0)
            
            # plot for visual inspection
            if plot_single_cell:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                
                xaxis = np.arange(-1250, 5000)/1250
                
                axs[0].plot(xaxis,
                            non_stim_avg,
                            color='grey')
                axs[0].fill_between(xaxis,
                                    non_stim_avg+non_stim_sem,
                                    non_stim_avg-non_stim_sem,
                                    color='grey', alpha=.1)
                axs[0].set(title='non-stim',
                           xlabel='time (s)',
                           ylabel='spike rate (Hz)')
                
                axs[1].plot(xaxis,
                            stim_avg,
                            color='royalblue')
                axs[1].fill_between(xaxis,
                                    stim_avg+stim_sem,
                                    stim_avg-stim_sem,
                                    color='royalblue', alpha=.1)
                axs[1].set(title='stim',
                           xlabel='time (s)',
                           ylabel='spike rate (Hz)')
                
                fig.suptitle(clu_name[:divider1])
                
                fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\stim_effect_{}\{}.png'.format(stimtype, clu_name[:divider1]),
                            dpi=300,
                            bbox_inches='tight')