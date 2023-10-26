# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:03:02 2023

Plot cue-start difference for emphasising this point

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import sys

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% MAIN 
for pathname in pathLC:
    sessname = pathname[-17:]
    print(sessname)
    
    filename = pathname+pathname[-18:]+'_DataStructure_mazeSection1_TrialType1'
    # import bad beh trial indices
    behPar = sio.loadmat(filename+'_behPar_msess1.mat')
    # -1 to account for MATLAB Python difference
    ind_bad_beh = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    # -1 to account for 0 being an empty trial
    ind_good_beh = np.arange(behPar['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
    # import cue and run lfps
    alignRun = sio.loadmat(filename+'_alignRun_msess1.mat')
    alignCue = sio.loadmat(filename+'_alignCue_msess1.mat')
    
    cues = alignCue['trialsCue']['startLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    diffs = starts - cues
    tot_trial = ind_good_beh
    
    # plot
    fig, ax = plt.subplots(figsize=(5,2))
    ax.set(title='aligned to cues',
           xlabel='time (s)', ylabel='trial #',
           xlim=(-1, 5))
    for p in ['top','right']:
        ax.spines[p].set_visible(False)
    for p in ['left','bottom']:
        ax.spines[p].set_linewidth(1)
    
    row_count = 1
    for trial in ind_good_beh:
        cueline, = ax.plot([0, 0],
                           [row_count, row_count+1],
                           linewidth=2, color='skyblue')
        startline, = ax.plot([(diffs[trial]/1250)+.5, (diffs[trial]/1250)+.5],
                             [row_count, row_count+1],
                             linewidth=2, color='red')
        row_count+=1
    
    ax.legend([cueline, startline], ['cue', 'run-onset'], frameon=False)
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\cues_v_starts\{}.png'.format(sessname),
                dpi=300,
                bbox_inches='tight')