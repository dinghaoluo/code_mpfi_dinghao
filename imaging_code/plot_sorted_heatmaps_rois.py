# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:32:01 2024

plot run-onset- and reward-aligned sorted heatmaps of each session 

@author: Dinghao Luo
"""


#%% imports 
import sys 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if (r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import smooth_convolve

if r'Z:\Dinghao\code_dinghao' not in sys.path:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE


#%% load behav data 
behaviour = pd.read_pickle('Z:/Dinghao/code_dinghao/behaviour/all_GRABNE_sessions.pkl')


#%% main loop
for path in pathGRABNE:  # start from the first good recording animal (A093i)
    recname = path[-17:]
    print(recname)
    
    roi_traces = np.load(r'{}/processed/suite2p/plane0/F.npy'.format(path),
                         allow_pickle=True)
    roi_traces_dff = ipf.calculate_dFF(roi_traces)  # dFF
    tot_roi = roi_traces_dff.shape[0]
    tot_frame = roi_traces_dff.shape[1]
    
    behav = behaviour.loc[recname]
    
    # calculate mean trace for each grid 
    run_trace_means = []
    rew_trace_means = []
    for g in range(tot_roi):
        curr_trace = roi_traces_dff[g,:] 
        
        # run-onset aligned
        run_traces = []
        rfs = behav['run_onset_frames']
        for f in rfs:
            if f!=-1 and f-30>=0 and f+4*30<=tot_frame:  # -1 means the run-onset is out of range for the frames
                run_traces.append(smooth_convolve(curr_trace[f-30:f+4*30]))  # 5 seconds in total 
        run_trace_means.append(np.mean(run_traces, axis=0))
        
        # rew aligned
        rew_traces = []
        rewfs = behav['pump_frames']
        for f in rewfs:
            if f!=-1 and f-30>=0 and f+4*30<=tot_frame:  # -1 means the run-onset is out of range for the frames
                rew_traces.append(smooth_convolve(curr_trace[f-30:f+4*30]))  # 5 seconds in total 
        rew_trace_means.append(np.mean(rew_traces, axis=0))
        
    run_curr_max_pt = {}  # argmax for all mean trace 
    for g in range(tot_roi):
        run_curr_max_pt[g] = np.argmax(run_trace_means[g])
    def helper(x):
        return run_curr_max_pt[x]
    run_ord_ind = sorted(np.arange(tot_roi), key=helper)  # ordered indices
    run_trace_means_ordered = np.zeros((tot_roi, 5*30))  # ordered heatmap
    for i, grid in enumerate(run_ord_ind): 
        run_trace_means_ordered[i,:] = run_trace_means[grid]
        
    rew_curr_max_pt = {}  # argmax for all mean trace 
    for g in range(tot_roi):
        rew_curr_max_pt[g] = np.argmax(rew_trace_means[g])
    def helper(x):
        return rew_curr_max_pt[x]
    rew_ord_ind = sorted(np.arange(tot_roi), key=helper)  # ordered indices
    rew_trace_means_ordered = np.zeros((tot_roi, 5*30))  # ordered heatmap
    for i, grid in enumerate(rew_ord_ind): 
        rew_trace_means_ordered[i,:] = rew_trace_means[grid]
    
    # plot run
    fig, axs = plt.subplots(1,2, figsize=(5.5,3.3))
    
    axs[0].imshow(run_trace_means, aspect='auto', extent=[-1,4,0,tot_roi], cmap='viridis', interpolation='none')
    axs[1].imshow(run_trace_means_ordered, aspect='auto', extent=[-1,4,0,tot_roi], cmap='viridis', interpolation='none')
    
    axs[0].set(title='mean traces\n(ROIs, RO-aligned)')
    axs[1].set(title='mean traces\n(ROIs, RO-aligned, sorted)')
    for p in [0,1]:
        axs[p].set(xlabel='time (s)', ylabel='grid #')
    
    fig.suptitle(recname)
    
    fig.tight_layout()
    plt.show(fig)
    
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_heatmaps_RO_aligned_rois\{}.png'.format(recname),
                dpi=500, bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_heatmaps_RO_aligned_rois\{}.pdf'.format(recname),
                bbox_inches='tight')
    
    plt.close(fig)
    
    # plot rew
    fig, axs = plt.subplots(1,2, figsize=(5.5,3.3))
    
    axs[0].imshow(rew_trace_means, aspect='auto', extent=[-1,4,0,tot_roi], cmap='viridis', interpolation='none')
    axs[1].imshow(rew_trace_means_ordered, aspect='auto', extent=[-1,4,0,tot_roi], cmap='viridis', interpolation='none')
    
    axs[0].set(title='mean traces\n(grid, rew-aligned)')
    axs[1].set(title='mean traces\n(grid, rew-aligned, sorted)')
    for p in [0,1]:
        axs[p].set(xlabel='time (s)', ylabel='grid #')
    
    fig.suptitle(recname)
    
    fig.tight_layout()
    plt.show(fig)
    
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_heatmaps_rew_aligned_rois\{}.png'.format(recname),
                dpi=120, bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_heatmaps_rew_aligned_rois\{}.pdf'.format(recname),
                bbox_inches='tight')
    
    