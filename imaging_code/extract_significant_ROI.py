# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 18:20:50 2024

pick out ROIs that have significant activity 
- criteria:
    1. exceeds 99% shuffled and
    2. does not correlate with neuropil 
    in the 2 seconds following the aligned landmark

@author: Dinghao Luo
"""


#%% imports 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.stats import pearsonr

# import util functions 
if (r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_utility_functions as iuf
import imaging_pipeline_functions as ipf

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import smooth_convolve

if r'Z:\Dinghao\code_dinghao' not in sys.path:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE


#%% statistic parameters 
alpha=.01
sig_window = 2  # in seconds 


#%% dataframe to contain all results 
profiles = {'sig_act_roi': []
            }

df = pd.DataFrame(profiles)


#%% main 
for rec_path in pathGRABNE:
    recname = rec_path[-17:]
    
    ext_path = rec_path+r'_roi_extract'
    
    run_aligned = np.load(ext_path+r'\suite2pROI_run_dFF_aligned.npy', allow_pickle=True)
    run_aligned_neu = np.load(ext_path+r'\suite2pROI_run_dFF_aligned_neu.npy', allow_pickle=True)
    
    rew_aligned = np.load(ext_path+r'\suite2pROI_rew_dFF_aligned.npy', allow_pickle=True)
    rew_aligned_neu = np.load(ext_path+r'\suite2pROI_rew_dFF_aligned_neu.npy', allow_pickle=True)
    
    tot_roi, tot_trial, tot_time = run_aligned.shape
    
    signif_act_roi = []
    
    for r in range(tot_roi):
        curr_run_aligned = run_aligned[r,:,:]
        curr_run_aligned_neu = run_aligned_neu[r,:,:]
        
        # check for nan 
        if ipf.sum_mat(np.isnan(curr_run_aligned))!=0: continue
        
        # smoothing?
        for t in range(tot_trial):
            curr_run_aligned[t,:] = smooth_convolve(curr_run_aligned[t,:])
            curr_run_aligned_neu[t,:] = smooth_convolve(curr_run_aligned_neu[t,:])
        
        curr_run_mean = np.mean(curr_run_aligned, axis=0)
        curr_run_mean_neu = np.mean(curr_run_aligned_neu, axis=0)
        
        shuf, shuf_95, shuf_5 = iuf.circ_shuffle(curr_run_aligned, alpha=alpha, num_shuf=1000)
        
        # test for data-shuff significance by looking at the 2 seconds after RO
        sig_95up = [f for f, [v, vs] in enumerate(zip(curr_run_mean[:30*sig_window], shuf_95[:30*sig_window])) if v>vs]
        sig_5down = [f for f, [v, vs] in enumerate(zip(curr_run_mean[:30*sig_window], shuf_5[:30*sig_window])) if v<vs]
        
        # test for correlation
        rval, pval = pearsonr(curr_run_mean, curr_run_mean_neu)
        
        # plot for inspection 
        xaxis = np.arange(30*5)/30-1
        fig, ax = plt.subplots(figsize=(2,1))
        ax.plot(xaxis, shuf, color='grey', linewidth=1)
        ax.fill_between(xaxis, shuf_95,
                               shuf_5, 
                               color='grey', edgecolor='none', alpha=.5)
        ax.plot(xaxis, curr_run_mean, color='darkgreen', linewidth=1)
        ax.set(xlabel='time (s)',
               ylabel='dFF',
               title='roi {}\nr={} pval={}'.format(r, round(rval,3), round(pval,3)))
        
        if pval>=alpha and (len(sig_95up)>30 or len(sig_5down)>30):
            signif_act_roi.append(r)
            
    df.loc[recname] = np.asarray([signif_act_roi
                                  ],
                                 dtype='object')