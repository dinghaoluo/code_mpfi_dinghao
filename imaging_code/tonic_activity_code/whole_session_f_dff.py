# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:18:22 2024

plots F and dF/F of single ROIs throughout the session for long-timescale changes

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os

if (r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import smooth_convolve

# plotting parameters
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% recording list
if r'Z:\Dinghao\code_dinghao' not in sys.path:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE


#%% main loop
for path in pathGRABNE:
    recname = path[-17:]
    print(recname)
    
    outpath = r'{}_roi_extract\tonic'.format(path)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    roi_traces = np.load(r'{}/processed/suite2p/plane0/F.npy'.format(path),
                         allow_pickle=True)
    roi_traces = ipf.filter_outlier(roi_traces)
    # roi_traces_dff = ipf.calculate_dFF(roi_traces)  # dFF
    tot_roi = roi_traces.shape[0]
    tot_frame = roi_traces.shape[1]
    
    xaxis = np.arange(tot_frame)/30
    
    p_per_plot = 10
    n_plot = int(np.ceil(tot_roi/p_per_plot))  # ROIs per plot
    
    
    # raw
    curr_roi = 0
    roi_range = [1, 10]
    for plot in range(n_plot):
        fig = plt.figure(1, figsize=(24,13))
        
        for p in range(p_per_plot):
            ax = fig.add_subplot(10,1, p+1)
            ax.plot(xaxis, roi_traces[curr_roi], c='darkgreen', lw=.5)
            ax.set(xlim=(0,xaxis[-1]), 
                   title='ROI {}'.format(curr_roi+1))
            
            curr_roi+=1
            if curr_roi>=tot_roi:  # halt plotting if all ROIs have been plotted 
                break
            
        ax.set(xlabel='time (s)')
        fig.tight_layout()
        plt.show(fig)
        
        fig.savefig(r'{}\rois_{}_{}.png'.format(outpath, roi_range[0], roi_range[1]),
                    dpi=120, bbox_inches='tight')
        fig.savefig(r'{}\rois_{}_{}.pdf'.format(outpath, roi_range[0], roi_range[1]),
                    bbox_inches='tight')
        
        roi_range = [r+10 for r in roi_range]
        if roi_range[1]>tot_roi:
            roi_range[1] = tot_roi
        
    # smoothed
    curr_roi = 0
    roi_range = [1, 10]
    for plot in range(n_plot):
        fig = plt.figure(1, figsize=(25,12.5))
        
        for p in range(p_per_plot):         
            ax = fig.add_subplot(10,1, p+1)
            ax.plot(xaxis, smooth_convolve(roi_traces[curr_roi],sigma=10), c='darkgreen', lw=.5)
            ax.set(xlim=(0,xaxis[-1]), 
                   title='ROI {}'.format(curr_roi+1))
            
            curr_roi+=1
            if curr_roi>=tot_roi:  # halt plotting if all ROIs have been plotted 
                break
        
        ax.set(xlabel='time (s)')
        fig.tight_layout()
        plt.show(fig)
        
        fig.savefig(r'{}\rois_{}_{}_smoothed.png'.format(outpath, roi_range[0], roi_range[1]),
                    dpi=120, bbox_inches='tight')
        fig.savefig(r'{}\rois_{}_{}_smoothed.pdf'.format(outpath, roi_range[0], roi_range[1]),
                    bbox_inches='tight')
        
        roi_range = [r+10 for r in roi_range]
        if roi_range[1]>tot_roi:
            roi_range[1] = tot_roi