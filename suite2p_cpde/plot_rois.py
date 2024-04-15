# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:57:58 2024

function to plot traces for .stat files (Suite2p)

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os

import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if ('Z:/Dinghao/code_dinghao/common' in sys.path) == False:
    sys.path.append('Z:/Dinghao/code_dinghao/common')
from common import normalise


#%% function 
def moving_average(arr, window_size=3):
    
    frame = 0
    moving_averages = []
    while frame < len(arr)-window_size+1:
        window_average = np.sum(arr[frame:frame+window_size]) / window_size
        moving_averages.append(window_average)
        frame+=1
        
    return moving_averages


#%% plotting function 
def plot_rois(filename, perfig=10, frames=1000):
    
    """
    filename: str, address of F.npy
    
    perfig: int, how many rois per plot
    """
    
    f = np.load(filename)
    
    tot_rois = f.shape[0]
    xaxis = np.arange(frames-2)/10  # 30 fps, smoothed with bin=3
    tot_figs = tot_rois // perfig
    # if tot_figs % perfig != 0:
    #     tot_figs += 1

    roi_counter = 0
    for i in range(tot_figs):
        fig, ax = plt.subplots(perfig, 1, figsize=(10, 20))
        for j in range(perfig):
            curr_roi = f[roi_counter][:frames]
            curr_roi = normalise(curr_roi)
            curr_roi = moving_average(curr_roi)
        
            ax[j].plot(xaxis, curr_roi)
            ax[j].set(xlabel='time (s)', title='roi {}'.format(roi_counter+1))
            
            roi_counter+=1
            
        plt.subplots_adjust(hspace=1.3)
        plt.show()
        
        fig.savefig('{}/traces/{}.png'.format(filename[:-6], roi_counter+1),
                    dpi=300,
                    bbox_inches='tight')


#%% main 
# can manually use this script to input recording name
filelist = [
    'Z:/Nico/AC918-20231017_02/K_4_p_2_decay_time_5/mock_suite2p/F.npy']

# filename = 'Z:/Nico/AC918-20231017_02/K_4_gSig_3/mock_suite2p/F.npy'

for filename in filelist:
    output_path = '{}/traces'.format(filename[:-6])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    plot_rois(filename)