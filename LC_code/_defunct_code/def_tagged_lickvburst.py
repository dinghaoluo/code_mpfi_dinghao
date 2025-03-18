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
from scipy.stats import ttest_rel, wilcoxon

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
noStim = input('Get rid of stim trials? (Y/N) (for plotting purposes... etc. etc.)\n')

count_sensitive = 0

for cluname in tag_list:
    print(cluname)
    raster = rasters[cluname]
    train = all_train[cluname]
    
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
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\single_cell_raster_by_first_licks_noStim\{}_tagged.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')
    else:
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\single_cell_raster_by_first_licks\{}_tagged.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')
    

#%% percentage responsive to first licks
print('{}% of tagged cells are responsive to first licks.'.format(count_sensitive/len(tag_list)))


#%% population analysis of cluster 1 cells
clusters = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_clustered_hierarchical_centroid.npy',
                   allow_pickle=True).item()['clusters']
cluster1 = clusters['cluster 1']

sess_list = [
    'A045r-20221207-02',
    'A049r-20230120-04',
    'A056r-20230418-02',
    'A056r-20230421-03']

window = [3750-625, 3750+625]  # window for spike summation
early_all = []; late_all = []

for sessname in sess_list:
    print(sessname)
    
    early_sess = []; late_sess = []
    
    early_sum = 0; late_sum = 0
    for cluname in tag_list:
        if cluname[:17]==sessname and cluname in cluster1:
            raster = rasters[cluname]
            
            filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
            alignRun = sio.loadmat(filename)
            
            licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
            starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
            tot_trial = licks.shape[0]
                
            first_licks = []
            for trial in range(tot_trial):
                lk = [l for l in licks[trial] if l-starts[trial] > 1000]
                if len(lk)==0:
                    first_licks.append(10000)
                else:
                    first_licks.extend(lk[0]-starts[trial])

            temp = list(np.arange(tot_trial))
            licks_ordered, temp_ordered = zip(*sorted(zip(first_licks, temp)))
            
            for trial in range(10, 20):
                curr_raster = raster[temp_ordered[trial]]
                early_sum += sum(curr_raster[window[0]:window[1]])
            for trial in range(tot_trial-20, tot_trial-10):
                curr_raster = raster[temp_ordered[trial]]
                late_sum += sum(curr_raster[window[0]:window[1]])
            early_sess.append(early_sum)
            late_sess.append(late_sum)
    
    early_all.append(early_sess)
    late_all.append(late_sess)
    
    early_all_mean = [np.mean(ls)/10 for ls in early_all]
    late_all_mean = [np.mean(ls)/10 for ls in late_all]
    

#%% plot 
fig, ax = plt.subplots(figsize=(4,4))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticklabels(['early', 'late'], minor=False)

pval = wilcoxon(early_all_mean, late_all_mean)[1]
ax.set(ylabel='population spike rate (Hz)',
       title='early v late lick trials p={}'.format(round(pval, 3)),
       ylim=(8, 23))

bp = ax.boxplot([early_all_mean, late_all_mean],
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
    
ax.scatter([[.5]*4, [1]*4], [early_all_mean, late_all_mean], zorder=2,
           s=15, color='grey', edgecolor='k', alpha=.5)
ax.plot([[.5]*4, [1]*4], [early_all_mean, late_all_mean], zorder=2,
        color='grey', alpha=.5)

fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_population_earlyvlate.png',
            dpi=300,
            bbox_inches='tight')