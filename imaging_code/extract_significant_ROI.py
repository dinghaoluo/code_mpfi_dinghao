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
from scipy.stats import pearsonr, sem

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


#%% plot trace?
plot_trace = True


#%% read dataframe and get indices 
df = pd.read_pickle(r'Z:\Dinghao\code_dinghao\GRABNE\significant_activity_roi.pkl') 
processed_recs = list(df.index.values)


#%% statistic parameters 
alpha=.01
sig_window = 2  # in seconds 


#%% dataframe to contain all results 
profiles = {'sig_act_roi_run': [],
            'sig_act_roi_rew': []
            }

df = pd.DataFrame(profiles, dtype=object)


#%% main 
for rec_path in pathGRABNE:
    recname = rec_path[-17:]
    
    # if processed, skip
    if recname in processed_recs: 
        print(recname+' already processed; skipped')
        continue
    else:
        print(recname)
    
    ext_path = rec_path+r'_roi_extract'
    
    run_aligned = np.load(ext_path+r'\suite2pROI_run_dFF_aligned.npy', allow_pickle=True)
    run_aligned_neu = np.load(ext_path+r'\suite2pROI_run_dFF_aligned_neu.npy', allow_pickle=True)
    
    rew_aligned = np.load(ext_path+r'\suite2pROI_rew_dFF_aligned.npy', allow_pickle=True)
    rew_aligned_neu = np.load(ext_path+r'\suite2pROI_rew_dFF_aligned_neu.npy', allow_pickle=True)
    
    tot_roi, tot_trial_run, tot_time = run_aligned.shape
    tot_trial_rew = rew_aligned.shape[1]
    
    signif_act_roi_run = []
    signif_act_roi_rew = []
    
    if plot_trace:
        sig_means_run = []
        sig_sems_run = [] 
        sig_shuf_run = []
        sig_shuf_95_run = []
        sig_shuf_5_run = []
        sig_means_rew = []
        sig_sems_rew = [] 
        sig_shuf_rew = []
        sig_shuf_95_rew = []
        sig_shuf_5_rew = []
    
    for r in range(tot_roi):
        curr_run_aligned = run_aligned[r,:,:]
        curr_run_aligned_neu = run_aligned_neu[r,:,:]
        curr_rew_aligned = rew_aligned[r,:,:]
        curr_rew_aligned_neu = rew_aligned_neu[r,:,:]
        
        # check for nan 
        if ipf.sum_mat(np.isnan(curr_run_aligned))!=0 or ipf.sum_mat(np.isnan(curr_rew_aligned))!=0: continue
        
        # smoothing?
        for t in range(tot_trial_run):
            curr_run_aligned[t,:] = smooth_convolve(curr_run_aligned[t,:])
            curr_run_aligned_neu[t,:] = smooth_convolve(curr_run_aligned_neu[t,:])
        for t in range(tot_trial_rew):
            curr_rew_aligned[t,:] = smooth_convolve(curr_rew_aligned[t,:])
            curr_rew_aligned_neu[t,:] = smooth_convolve(curr_rew_aligned_neu[t,:])
        
        curr_run_mean = np.mean(curr_run_aligned, axis=0)
        curr_run_mean_neu = np.mean(curr_run_aligned_neu, axis=0)
        curr_rew_mean = np.mean(curr_rew_aligned, axis=0)
        curr_rew_mean_neu = np.mean(curr_rew_aligned_neu, axis=0)
        
        shuf_run, shuf_95_run, shuf_5_run = iuf.circ_shuffle(curr_run_aligned, alpha=alpha, num_shuf=500)
        shuf_rew, shuf_95_rew, shuf_5_rew = iuf.circ_shuffle(curr_rew_aligned, alpha=alpha, num_shuf=500)
        
        # test for data-shuff significance by looking at the 2 seconds after RO
        sig_95up_run = [f for f, [v, vs] in enumerate(zip(curr_run_mean[:30*sig_window], shuf_95_run[:30*sig_window])) if v>vs]
        sig_5down_run = [f for f, [v, vs] in enumerate(zip(curr_run_mean[:30*sig_window], shuf_5_run[:30*sig_window])) if v<vs]
        sig_95up_rew = [f for f, [v, vs] in enumerate(zip(curr_rew_mean[:30*sig_window], shuf_95_rew[:30*sig_window])) if v>vs]
        sig_5down_rew = [f for f, [v, vs] in enumerate(zip(curr_rew_mean[:30*sig_window], shuf_5_rew[:30*sig_window])) if v<vs]
        
        # test for correlation
        rval_run, pval_run = pearsonr(curr_run_mean, curr_run_mean_neu)
        rval_rew, pval_rew = pearsonr(curr_rew_mean, curr_rew_mean_neu)
        
        if pval_run>=.05 and (len(sig_95up_run)>30 or len(sig_5down_run)>30):  # not using 0.01 here, since we want to filter out these ROIs as much as possible 
            signif_act_roi_run.append(r)
        if pval_rew>=.05 and (len(sig_95up_rew)>30 or len(sig_5down_rew)>30):
            signif_act_roi_rew.append(r)

        if plot_trace:
            sig_means_run.append(curr_run_mean)
            sig_sems_run.append(sem(curr_run_aligned, axis=0))
            sig_shuf_run.append(shuf_run)
            sig_shuf_95_run.append(shuf_95_run)
            sig_shuf_5_run.append(shuf_5_run)
            sig_means_rew.append(curr_rew_mean)
            sig_sems_rew.append(sem(curr_rew_aligned, axis=0))
            sig_shuf_rew.append(shuf_rew)
            sig_shuf_95_rew.append(shuf_95_rew)
            sig_shuf_5_rew.append(shuf_5_rew)
    
    tot_sig_roi_run = len(signif_act_roi_run)
    tot_sig_roi_rew = len(signif_act_roi_rew)
            
    if plot_trace:
        n_row = int(tot_sig_roi_run/5)
        if n_row==0:
            n_col = tot_sig_roi_rew
            n_row = 1
        else:
            n_col = int(np.ceil(tot_sig_roi_run/n_row))
        xaxis = (np.arange(5*30)-30)/30
        fig = plt.figure(1, figsize=(n_col*3, n_row*2.1))
        for p, r in enumerate(signif_act_roi_run):
            ax = fig.add_subplot(n_row, n_col, p+1)
            ax.set(xlim=(-1,4), xlabel='time (s)', xticks=[0,2,4],
                   ylabel='dFF', title='roi {}'.format(r))
            ax.plot(xaxis, sig_shuf_run[p], color='grey', linewidth=.8)
            ax.fill_between(xaxis, sig_shuf_95_run[p],
                                   sig_shuf_5_run[p],
                            color='grey', edgecolor='none', alpha=.2)
            ax.plot(xaxis, sig_means_run[p], color='darkgreen', linewidth=.8)
            ax.fill_between(xaxis, sig_means_run[p]+sig_sems_run[p],
                                   sig_means_run[p]-sig_sems_run[p],
                            color='darkgreen', edgecolor='none', alpha=.2)
            ax.axvspan(0, 0, color='burlywood', alpha=.5, linestyle='dashed', linewidth=1)
        fig.suptitle('run_aligned_sig_act_only')
        fig.tight_layout()
        fig.savefig('{}/suite2pROI_avgdFF_run_aligned_sig_roi_only.png'.format(ext_path),
                    dpi=120,
                    bbox_inches='tight')
        plt.close(fig)
        
        n_row = int(tot_sig_roi_rew/5)  # recalculate due to how rew and run aligned trial numbers do not match, Dinghao, 13 Aug
        if n_row==0:
            n_col = tot_sig_roi_rew
            n_row = 1
        else:
            n_col = int(np.ceil(tot_sig_roi_rew/n_row))
        fig = plt.figure(1, figsize=(n_col*3, n_row*2.1))
        for p, r in enumerate(signif_act_roi_rew):
            ax = fig.add_subplot(n_row, n_col, p+1)
            ax.set(xlim=(-1,4), xlabel='time (s)', xticks=[0,2,4],
                   ylabel='dFF', title='roi {}'.format(r))
            ax.plot(xaxis, sig_shuf_rew[p], color='grey', linewidth=.8)
            ax.fill_between(xaxis, sig_shuf_95_rew[p],
                                   sig_shuf_5_rew[p],
                            color='grey', edgecolor='none', alpha=.2)
            ax.plot(xaxis, sig_means_rew[p], color='darkgreen', linewidth=.8)
            ax.fill_between(xaxis, sig_means_rew[p]+sig_sems_rew[p],
                                   sig_means_rew[p]-sig_sems_rew[p],
                            color='darkgreen', edgecolor='none', alpha=.2)
            ax.axvspan(0, 0, color='burlywood', alpha=.5, linestyle='dashed', linewidth=1)
        fig.suptitle('rew_aligned_sig_act_only')
        fig.tight_layout()
        fig.savefig('{}/suite2pROI_avgdFF_rew_aligned_sig_roi_only.png'.format(ext_path),
                    dpi=120,
                    bbox_inches='tight')
        plt.close(fig)
    
    df.loc[recname] = pd.Series([signif_act_roi_run, signif_act_roi_rew],
                                index=['sig_act_roi_run', 'sig_act_roi_rew'])
    

#%% save dataframe 
df.to_csv(r'Z:\Dinghao\code_dinghao\GRABNE\significant_activity_roi.csv')
df.to_pickle(r'Z:\Dinghao\code_dinghao\GRABNE\significant_activity_roi.pkl')