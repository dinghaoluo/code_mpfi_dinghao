# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:31:12 2023

speed and spike rate correlation

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr, wilcoxon
import pandas as pd
import scipy.io as sio


#%% load data 
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% read cell tags 
clu_list = list(cell_prop.index)

tag_list = []; put_list = []
tag_rop_list = []; put_rop_list = []
for clu in cell_prop.index:
    tg = cell_prop['tagged'][clu]
    pt = cell_prop['putative'][clu]
    rop = cell_prop['peakness'][clu]
    
    if tg:
        tag_list.append(clu)
        if rop:
            tag_rop_list.append(clu)
    if pt:
        put_list.append(clu)
        if rop:
            put_rop_list.append(clu)


#%% MAIN
for cluname in clu_list:
    # get velocity file
    sessname = cluname[:17]
    datename = cluname[:14]
    animalname = 'ANMD{}'.format(cluname[1:5])
    path = 'Z:\Dinghao\MiceExp\{}\{}\{}\{}'.format(animalname, datename, sessname, sessname)
    speed_time_file = sio.loadmat('{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(path))
    
    # filter shape
    samp_freq = 1250  # Hz
    gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
    sigma_speed = samp_freq/100  # 10 ms sigma
    gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
                  np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]
    
    # concatenate before and after
    speed_time_bef = speed_time_file['trialsRun'][0]['speed_MMsecBef'][0][0][1:]
    speed_time = speed_time_file['trialsRun'][0]['speed_MMsec'][0][0][1:]
    speed_time_all = np.empty(shape=speed_time.shape[0], dtype='object')
    for i in range(speed_time.shape[0]):
        bef = speed_time_bef[i]; aft =speed_time[i]
        speed_time_all[i] = np.concatenate([bef, aft])
        speed_time_all[i][speed_time_all[i]<0] = 0
    speed_time_conv = [np.convolve(np.squeeze(trial), gaus_speed, mode='same') 
                       for trial in speed_time_all]
    
    rate = all_train[cluname]
    tot_trial = len(rate)
    
    num_shufs = 100
    
    fig, axs = plt.subplot_mosaic('A;A;B', figsize=(4,6))
    fig.suptitle(cluname)
    
    corr = []; corr_shuf = []; diff = []
    for trial in range(tot_trial):
        corr.append(pearsonr(rate[trial], speed[trial])[0])
        
        rate_shuf = np.zeros([num_shufs, len(rate[trial])])
        for n in range(num_shufs):
            rand_shift = np.random.randint(1, len(rate[trial])+1)
            rate_shuf[n, :] = np.roll(rate[trial], -rand_shift)
        rate_shuf_mean = np.mean(rate_shuf, axis=0)
        corr_shuf.append(pearsonr(rate_shuf_mean, speed[trial])[0])
        diff.append(corr[trial]-corr_shuf[trial])
        
        axs['A'].scatter(corr[trial], trial, s=3, marker='^', color='darkblue')
        axs['A'].scatter(corr_shuf[trial], trial, s=2, color='grey')
        if diff[trial]>0:
            axs['A'].plot([corr[trial], corr_shuf[trial]],
                          [trial, trial], linewidth=1, color='darkblue', alpha=.5)
        else:
            axs['A'].plot([corr[trial], corr_shuf[trial]],
                          [trial, trial], linewidth=1, color='grey', alpha=.5)
        
    rt = axs['A'].scatter([],[],s=3,color='darkblue')
    sf = axs['A'].scatter([],[],s=3,color='grey',alpha=.5)
    axs['A'].legend((rt, sf), ('corr.', 'shuf. corr.'), fontsize=8)
    
    axs['A'].set(ylabel='trial #', xlabel='r')
    for p in ['top', 'right']:
        axs['A'].spines[p].set_visible(False)
    
    corr_mean = np.mean(corr); corr_shuf_mean = np.mean(corr_shuf)
    pval = wilcoxon(corr, corr_shuf)[1]
    axs['B'].scatter(corr_mean, 1, s=50, marker='^', color='darkblue')
    axs['B'].scatter(corr_shuf_mean, 1, s=50, color='grey')
    if corr_mean>corr_shuf_mean:
        axs['B'].plot([corr_mean, corr_shuf_mean], [1, 1], 
                      linewidth=3, color='darkblue', alpha=.5)
    else:
        axs['B'].plot([corr_mean, corr_shuf_mean], [1, 1], 
                      linewidth=3, color='grey', alpha=.5)
    
    axs['B'].set(title='p = {}'.format(round(pval, 4)))
    for p in ['top', 'right', 'left']:
        axs['B'].spines[p].set_visible(False)
    axs['B'].set_yticklabels([])
    axs['B'].set_yticks([])
    
    fig.tight_layout()
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\single_cell_correlation_speed_rate\{}_tagged.png'.format(cluname),
                dpi=500,
                bbox_inches='tight',
                transparent=False)