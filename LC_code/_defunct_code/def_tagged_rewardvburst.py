# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:29:45 2023

Does the burst have anything to do with licking?

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd
from scipy.stats import ttest_rel

rasters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
tag_list = []
for cell in cell_prop.index:
    tg = cell_prop['tagged'][cell]  # putative
    
    if tg:
        tag_list.append(cell)


#%% MAIN 
count_sensitive = 0

for cluname in tag_list:
    print(cluname)
    raster = rasters[cluname]
    train = all_train[cluname]
    
    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    alignRun = sio.loadmat(filename)
    
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    next_starts = [s for s in starts[1:]]
    next_starts = [s-starts[i] for i, s in enumerate(next_starts)]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:
            pumps[trial] = [20000]
    pumps = [p[0] for p in pumps]  # step out; no idea why list in the first place
        
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] if l-starts[trial] > 1000]
        if len(lk)==0:
            first_licks.append(10000)
        else:
            first_licks.extend(lk[0]-starts[trial])

    temp = list(np.arange(tot_trial))
    pumps_ordered, temp_ordered = zip(*sorted(zip(pumps, temp)))

    # plotting
    fig, axs = plt.subplot_mosaic('AAB',figsize=(10,5))    
    axs['A'].set(title=cluname,
                 xlabel='time (s)', ylabel='trial # by rewards',
                 xlim=(-3, 8))
    for p in ['top', 'right']:
        axs['A'].spines[p].set_visible(False)

    pre_rate = []; post_rate = []
    for trial in range(tot_trial):
        curr_raster = raster[temp_ordered[trial]]
        curr_train = train[temp_ordered[trial]]
        window = [pumps_ordered[trial]+3750-625, pumps_ordered[trial]+3750, pumps_ordered[trial]+3750+625]
        pre_rate.append(sum(curr_train[window[0]:window[1]])/2)
        post_rate.append(sum(curr_train[window[1]:window[2]])/2)
        
        curr_trial = np.where(raster[temp_ordered[trial]]==1)[0]
        curr_trial = [(s-3750)/1250 for s in curr_trial]
        axs['A'].scatter(curr_trial, [trial+1]*len(curr_trial),
                         color='grey', s=.35)
        axs['A'].plot([pumps_ordered[trial]/1250, pumps_ordered[trial]/1250],
                      [trial, trial+1],
                      linewidth=2, color='darkgreen')
        # ax.plot([first_licks[temp_ordered[trial]]/1250, pumps[temp_ordered[trial]]/1250],
        #         [trial, trial+1],
        #         linewidth=2, color='darkgreen', alpha=.35)
    for trial in range(tot_trial):
        if temp_ordered[trial]<tot_trial-1:  # there is no next start for the last trial
            axs['A'].plot([next_starts[temp_ordered[trial]]/1250, next_starts[temp_ordered[trial]]/1250],
                          [trial, trial+1],
                          linewidth=2, color='red', alpha=.5)
 
    pp, = axs['A'].plot([],[],color='darkgreen',label='rew.')
    st, = axs['A'].plot([],[],color='red',alpha=.5,label='n.s.')
    axs['A'].legend(handles=[pp, st])
    
    # t-test and pre-post comp.
    t_res = ttest_rel(a=pre_rate, b=post_rate)
    pval = t_res[1]
    if pval<0.05:
        count_sensitive+=1
    for p in ['top', 'right', 'bottom']:
        axs['B'].spines[p].set_visible(False)
    axs['B'].set_xticklabels(['pre', 'post'], minor=False)
    axs['B'].set(ylabel='spike rate (Hz)',
                 title='pre- v post-rew. p={}'.format(round(pval, 5)))

    bp = axs['B'].boxplot([pre_rate, post_rate],
                          positions=[.5, 1],
                          patch_artist=True,
                          notch='True')
    colors = ['coral', 'darkcyan']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    bp['fliers'][0].set(marker ='v',
                    color ='#e7298a',
                    markersize=2,
                    alpha=0.5)
    bp['fliers'][1].set(marker ='o',
                    color ='#e7298a',
                    markersize=2,
                    alpha=0.5)
    for median in bp['medians']:
        median.set(color='darkred',
                   linewidth=1)
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\single_cell_raster_by_rewards\{}_tagged.png'.format(cluname),
                dpi=300,
                bbox_inches='tight')