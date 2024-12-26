# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:29:45 2023

Order trials based on first lick time

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats import ttest_rel

all_info = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_info.npy',
                    allow_pickle=True).item()
clu_list = list(all_info.keys())


#%% MAIN 
noStim = input('Get rid of stim trials? (Y/N) (for plotting purposes... etc. etc.)\n')

count_sensitive = 0

for cluname in clu_list:
    if cluname[:17]=='A063r-20230708-01' or cluname[:17]=='A063r-20230708-02':  # lick detection problem
        continue
    
    print(cluname)
    
    sessname = cluname[:17]
    rasters = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(sessname),
                      allow_pickle=True).item()
    
    raster = rasters[cluname]
    train = all_info[cluname]
    
    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    alignRun = sio.loadmat(filename)
    
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:
            pumps[trial] = 20000
    
    behParf = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    behPar = sio.loadmat(behParf)
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stimOn_ind = np.where(stimOn!=0)[0]+1
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] if l-starts[trial] > 1000]
        if len(lk)==0:
            first_licks.append(10000)
        else:
            first_licks.extend(lk[0]-starts[trial])

    temp = list(np.arange(tot_trial))
    licks_ordered, temp_ordered = zip(*sorted(zip(first_licks, temp)))
    
    if noStim=='Y' or noStim=='y':
        temp_ordered = [t for t in temp_ordered if t not in stimOn_ind]
        tot_trial = len(temp_ordered)  # reset tot_trial if noStim

    # plotting
    fig, axs = plt.subplot_mosaic('AAB',figsize=(10,5))
    axs['A'].set(title=cluname,
                 xlabel='time (s)', ylabel='trial # by first licks',
                 xlim=(-3, 8))
    for p in ['top', 'right']:
        axs['A'].spines[p].set_visible(False)

    pre_rate = []; post_rate = []
    for trial in range(tot_trial):
        curr_raster = raster[temp_ordered[trial]]
        curr_train = train[temp_ordered[trial]]
        window = [licks_ordered[trial]+3750-625, licks_ordered[trial]+3750, licks_ordered[trial]+3750+625]
        pre_rate.append(sum(curr_train[window[0]:window[1]])/2)
        post_rate.append(sum(curr_train[window[1]:window[2]])/2)
        
        curr_trial = np.where(raster[temp_ordered[trial]]==1)[0]
        curr_trial = [(s-3750)/1250 for s in curr_trial]
        
        c = 'grey'
        calpha = .1
        if (noStim=='N' or noStim=='n') and stimOn[temp_ordered[trial]]==1:
            c = 'red'
            calpha = 1.0
        
        axs['A'].scatter(curr_trial, [trial+1]*len(curr_trial),
                         color=c, s=.35)
        axs['A'].plot([licks_ordered[trial]/1250, licks_ordered[trial]/1250],
                      [trial, trial+1],
                      linewidth=2, color='darkred', alpha=calpha)
        axs['A'].plot([pumps[temp_ordered[trial]]/1250, pumps[temp_ordered[trial]]/1250],
                      [trial, trial+1],
                      linewidth=2, color='darkgreen', alpha=.35)
     
    fl, = axs['A'].plot([],[],color='darkred',label='1st licks')
    pp, = axs['A'].plot([],[],color='darkgreen',alpha=.35,label='rew.')
    axs['A'].legend(handles=[fl, pp])
    
    
    # t-test and pre-post comp.
    t_res = ttest_rel(a=pre_rate, b=post_rate)
    pval = t_res[1]
    if pval<0.05:
        count_sensitive+=1
    for p in ['top', 'right', 'bottom']:
        axs['B'].spines[p].set_visible(False)
    axs['B'].set_xticklabels(['pre', 'post'], minor=False)
    axs['B'].set(ylabel='spike rate (Hz)',
                 title='pre- v post-first-lick p={}'.format(round(pval, 5)))

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
        
    if noStim=='Y' or noStim=='y':
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\single_cell_raster_by_first_licks_noStim\{}.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')
    else:
        fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\single_cell_raster_by_first_licks\{}.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')