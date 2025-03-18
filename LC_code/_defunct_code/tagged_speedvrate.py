# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:31:12 2023

speed and firing rate correlation

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr, wilcoxon


#%% load
tagged_all_info = np.load('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_info.npy',
                          allow_pickle=True).item()
tag_list = list(tagged_all_info.keys())


#%% MAIN
for cluname in tag_list:
    rate = tagged_all_info[cluname][0]
    speed = tagged_all_info[cluname][1]
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
                dpi=300,
                bbox_inches='tight',
                transparent=False)