# -*- coding: utf-8 -*-
"""
Created on Sat 11 Nov 13:26:32 2023

read dLight imaging data and plot trial-by-trial heatmaps

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import mat73
import scipy.io as sio
import sys
import os

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
sessdLight = rec_list.sessdLight


#%% parameters 
mode = 'axons_v2.0'


#%% function
if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% main 
for sessname in sessdLight:
    print(sessname)
    
    animal=sessname[:4]; date=sessname[:-3]; sess=sessname[-2:]
    
    # channel 1
    chan = 'Channel1'
    c1_pathname = r'Z:\Jingyu\2P_Recording\{}\{}\{}\{}\{}\{}'.format(animal,
                                                                     date, 
                                                                     sess, 
                                                                     mode,
                                                                     chan, 
                                                                     '{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(sessname))
    c1 = mat73.loadmat(c1_pathname)
    trialsRun = c1['trialsRun']
    
    # note to self: the data structure in the imaging mat struct is different,
    # in that each array is a trial, and within each trial the data are 
    # in the shape of cell ID x frames. So we need to read each cell's traces
    # with a for loop. I don't know the rationale behind this structure, but
    # I agree that it is exceptionally annoying.
    
    # extract signals 
    dFF = trialsRun['dFFGF'][1:]  # skip the first empty trial
    tot_trial = len(dFF)
    tot_clu = dFF[1].shape[1]
    
    mat_dFF = np.zeros((tot_clu, tot_trial, 500+500*4))  # 1 s bef, 4 s after
    for trial in range(tot_trial):
        curr_trial = dFF[trial]
        trial_length = curr_trial.shape[0]
        for clu in range(tot_clu):
            if trial_length>=(500*3+500*4):
                mat_dFF[clu, trial, :] = curr_trial[500*2:500*3+500*4, clu]
            if trial_length<(500*3+500*4):
                mat_dFF[clu, trial, :trial_length-500*2] = curr_trial[500*2:, clu]
    
    outdirroot = 'Z:\Dinghao\code_dinghao\dLight\{}'.format(sessname)
    if not os.path.exists(outdirroot):
        os.makedirs(outdirroot)
    outdir = '{}\heatmaps'.format(outdirroot)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    for clu in range(tot_clu):
        fig, ax = plt.subplots(figsize=(4,3))
        ax.imshow(mat_dFF[clu,:,:], aspect='auto', extent=([-1, 4, 1, tot_trial+1]))
        ax.set(title='ROI{}'.format(clu+1))
    
        fig.savefig('{}\{}'.format(outdir, 'ROI{}'.format(clu+1)),
                    dpi=500,
                    bbox_inches='tight')
        
        plt.close(fig)