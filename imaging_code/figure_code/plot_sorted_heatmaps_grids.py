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


#%% containers
sensitive_run = []  # proportions of run-onset-sensitive grids
sensitive_rew = []


#%% main loop
for path in pathGRABNE[-2:]:
    recname = path[-17:]
    print(recname)
    
    grid_traces = np.load(r'{}_grid_extract/grid_traces_31.npy'.format(path),
                          allow_pickle=True)
    grid_traces_dff = ipf.calculate_dFF(grid_traces)  # dFF
    tot_grid = grid_traces_dff.shape[0]
    tot_frame = grid_traces_dff.shape[1]
    
    behav = behaviour.loc[recname]
    
    # calculate mean trace for each grid 
    run_trace_means = []
    rew_trace_means = []
    for g in range(tot_grid):
        curr_trace = grid_traces_dff[g,:] 
        
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
    for g in range(tot_grid):
        run_curr_max_pt[g] = np.argmax(run_trace_means[g])
    def helper(x):
        return run_curr_max_pt[x]
    run_ord_ind = sorted(np.arange(tot_grid), key=helper)  # ordered indices
    run_trace_means_ordered = np.zeros((tot_grid, 5*30))  # ordered heatmap
    for i, grid in enumerate(run_ord_ind): 
        run_trace_means_ordered[i,:] = run_trace_means[grid]
        
    rew_curr_max_pt = {}  # argmax for all mean trace 
    for g in range(tot_grid):
        rew_curr_max_pt[g] = np.argmax(rew_trace_means[g])
    def helper(x):
        return rew_curr_max_pt[x]
    rew_ord_ind = sorted(np.arange(tot_grid), key=helper)  # ordered indices
    rew_trace_means_ordered = np.zeros((tot_grid, 5*30))  # ordered heatmap
    for i, grid in enumerate(rew_ord_ind): 
        rew_trace_means_ordered[i,:] = rew_trace_means[grid]
    
    # sensitivity
    run_list = list(run_curr_max_pt.values())
    rew_list = list(rew_curr_max_pt.values())
    sensitive_run.append(sum(i>15 and i<45 for i in run_list))
    sensitive_rew.append(sum(i>15 and i<45 for i in rew_list))
    
    # plot run
    fig, axs = plt.subplots(1,2, figsize=(5.5,3.3))
    
    axs[0].imshow(run_trace_means, aspect='auto', extent=[-1,4,0,tot_grid], cmap='viridis', interpolation='none')
    axs[1].imshow(run_trace_means_ordered, aspect='auto', extent=[-1,4,0,tot_grid], cmap='viridis', interpolation='none')
    
    axs[0].set(title='mean traces\n(grid, RO-aligned)')
    axs[1].set(title='mean traces\n(grid, RO-aligned, sorted)')
    for p in [0,1]:
        axs[p].set(xlabel='time (s)', ylabel='grid #')
    
    fig.suptitle(recname)
    
    fig.tight_layout()
    plt.show(fig)
    
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_heatmaps\sorted_heatmaps_RO_aligned_grids_31s\{}.png'.format(recname),
                dpi=500, bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_heatmaps\sorted_heatmaps_RO_aligned_grids_31\{}.pdf'.format(recname),
                bbox_inches='tight')
    
    plt.close(fig)
    
    # plot rew
    fig, axs = plt.subplots(1,2, figsize=(5.5,3.3))
    
    axs[0].imshow(rew_trace_means, aspect='auto', extent=[-1,4,0,tot_grid], cmap='viridis', interpolation='none')
    axs[1].imshow(rew_trace_means_ordered, aspect='auto', extent=[-1,4,0,tot_grid], cmap='viridis', interpolation='none')
    
    axs[0].set(title='mean traces\n(grid, rew-aligned)')
    axs[1].set(title='mean traces\n(grid, rew-aligned, sorted)')
    for p in [0,1]:
        axs[p].set(xlabel='time (s)', ylabel='grid #')
    
    fig.suptitle(recname)
    
    fig.tight_layout()
    plt.show(fig)
    
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_heatmaps\sorted_heatmaps_rew_aligned_grids_31\{}.png'.format(recname),
                dpi=120, bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_heatmaps\sorted_heatmaps_rew_aligned_grids_31\{}.pdf'.format(recname),
                bbox_inches='tight')
    

#%% calculation 
run_s_093 = [s/256 for s in sensitive_run[:27]]
rew_s_093 = [s/256 for s in sensitive_rew[:27]]
li_093 = behaviour['lick_selectivity'][:27]
run_s_094 = [s/256 for s in sensitive_run[27:]]
rew_s_094 = [s/256 for s in sensitive_rew[27:]]
li_094 = behaviour['lick_selectivity'][27:-2]


#%% plot A093
xaxis = np.arange(1, 27+1)

fig, ax = plt.subplots(figsize=(3.25, 1.8))
gl, = ax.plot(xaxis, run_s_093, color='royalblue')

ax2 = ax.twinx()
li, = ax2.plot(xaxis, li_093, color='orchid')

ax.legend([gl, li], ['% grids', 'lick sel.'], frameon=False, loc='lower right', fontsize=7)

ax.set(xlabel='session #', 
       ylabel='% RO-grids', ylim=(.1, .75),
       title='ANMD093i, all sess.')
ax2.set(ylabel='lick sel.', ylim=(.45, 1))

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\within_subject\A093i_RO_lick_sel.png',
            dpi=120, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\within_subject\A093i_RO_lick_sel.pdf',
            bbox_inches='tight')

from scipy.stats import pearsonr
r = pearsonr(run_s_093, li_093)


#%% plot A094
xaxis = np.arange(1, 45-27+1)

fig, ax = plt.subplots(figsize=(2.5,1.25))
ax.plot(xaxis, run_s_094)

ax2 = ax.twinx()
ax2.plot(xaxis, li_094, color='orange')

ax.set(xlabel='session #',
       ylabel='RO-grid count',
       title='ANMD094i, all sess.')
ax2.set(ylabel='lick select.')