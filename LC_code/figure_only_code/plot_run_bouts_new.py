# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:33:49 2023

plot run bouts 

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.io as sio
import sys
import mat73

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% gaussian kernel for speed smoothing 
samp_freq = 1250  # Hz
gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
sigma_speed = samp_freq/100
gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
              np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]


#%% MAIN
file_name = 'A029r-20220623-03'
file_path = r'Z:\Dinghao\MiceExp\ANMD'+file_name[1:5]+'\\'+file_name[:14]+'\\'+file_name[:17]+'\\'+file_name[:17]
run_bout_path = r'Z:\Dinghao\code_dinghao\run_bouts'
run_bout_path = run_bout_path+'\\'+file_name[:17]+'_run_bouts_py.csv'
behave_lfp_path = file_path+'_BehavElectrDataLFP.mat'
alignRun_path = file_path+'_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'

run_bout_table = pd.read_csv(run_bout_path)
beh_lfp = mat73.loadmat(behave_lfp_path)
tracks = beh_lfp['Track']; laps = beh_lfp['Laps']
trialsRun = sio.loadmat(alignRun_path)['trialsRun'][0][0]
    

#%% read variables for plotting 
lickLfp = laps['lickLfpInd']
lickLfp_flat = []
for trial in range(len(lickLfp)):
    for i in range(len(lickLfp[trial][0])):
        lickLfp_flat.append(int(lickLfp[trial][0][i]))
lickLfp_flat = np.array(lickLfp_flat)
speed_MMsec = tracks['speed_MMsecAll']
for tbin in range(len(speed_MMsec)):
    if speed_MMsec[tbin]<0:
        speed_MMsec[tbin] = (speed_MMsec[tbin-1]+speed_MMsec[tbin+1])/2
speed_MMsec = np.convolve(speed_MMsec, gaus_speed, mode='same')/10  # /10 for cm
startLfpInd = trialsRun['startLfpInd'][0]
endLfpInd = trialsRun['endLfpInd'][0]


#%% plotting 
save_path_base = r'Z:\Dinghao\code_dinghao\run_bouts\fsa_run_bouts_plots_python\\'

for t in np.arange(2, len(endLfpInd)-2, 3):
# for t in np.arange(2, 10, 3):
    lfp_indices_t = np.arange(startLfpInd[t]-1250, min(endLfpInd[t+2]+1250, len(speed_MMsec)))
    lap_start = lfp_indices_t[0]
    
    fig, ax = plt.subplots(figsize=(len(lfp_indices_t)/1500, 3))
    ax.set(xlabel='time (s)', ylabel='speed (cm/s)',
           ylim=(0, 1.1*max(speed_MMsec[lfp_indices_t])),
           xlim=(0, len(lfp_indices_t)/samp_freq),
           title='trials {} to {}'.format(t-1, t+1))
    for p in ['right', 'top']:
        ax.spines[p].set_visible(False)
    
    xaxis = np.arange(0, len(lfp_indices_t))/samp_freq
    ax.plot(xaxis, speed_MMsec[lfp_indices_t])
    
    # plot starts
    startLfpInd_t = startLfpInd[np.in1d(startLfpInd, lfp_indices_t)]
    ax.vlines((startLfpInd_t-lap_start)/samp_freq, 0, 200, 'r', linestyle='dashed')
    
    # plot run bout starts
    run_bout_t = run_bout_table.iloc[:,1][np.in1d(run_bout_table.iloc[:,1], lfp_indices_t)]
    ax.vlines((run_bout_t-lap_start)/samp_freq, 0, 200, 'g', linestyle='dashed')
    
    # plot licks
    licks_t = lickLfp_flat[np.in1d(lickLfp_flat, lfp_indices_t)]
    ylim = 1.1*max(speed_MMsec[lfp_indices_t])
    ax.vlines((licks_t-lap_start)/samp_freq, ylim, ylim*.95, 'magenta')
    
    # save fig 
    if len(run_bout_t)>0:
        save_path = save_path_base+'trials {} to {} w rb.png'.format(t-1, t+1)
        fig.savefig(save_path,
                    dpi=300,
                    bbox_inches='tight')
    else:
        save_path = save_path_base+'trials {} to {}.png'.format(t-1, t+1)
        fig.savefig(save_path,
                    dpi=300,
                    bbox_inches='tight')