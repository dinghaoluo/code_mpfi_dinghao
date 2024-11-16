# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:29:45 2023

Does the RO-peak have anything to do with licking?

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd
from scipy.stats import ttest_rel, ranksums, wilcoxon

# plotting parameters
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

xaxis = np.arange(6*1250)/1250-1


#%% load data 
rasters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
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
noStim = 'Y'
window = [3750-313, 3750+313]

for cluname in clu_list:
    if cluname==clu_list[171]:
        continue
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
    bad_beh_ind = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] if l-starts[trial] > 1250]  # only if the animal does not lick in the first second (carry-over licks)
        if len(lk)==0:
            first_licks.append(10000)
        else:
            first_licks.extend(lk[0]-starts[trial])

    temp = list(np.arange(tot_trial))
    licks_ordered, temp_ordered = zip(*sorted(zip(first_licks, temp)))
    
    # pick out early and late trials for plotting, Dinghao 18 Sept 2024
    early_trials = []; late_trials = []
    for i in range(30):
        if temp_ordered[i] not in bad_beh_ind and temp_ordered[i] not in stimOn_ind:
            if len(early_trials)<10:
                early_trials.append(temp_ordered[i])
        if temp_ordered[-(i+1)] not in bad_beh_ind and temp_ordered[-(i+1)] not in stimOn_ind:
            if len(late_trials)<10:
                late_trials.append(temp_ordered[-(i+1)])

    # plotting
    fig, axs = plt.subplots(2, 1, figsize=(3,2.5))
    
    early_prof = np.zeros((10, 1250*6))
    late_prof = np.zeros((10, 1250*6))
    
    for i, trial in enumerate(early_trials):
        curr_raster = raster[trial]
        early_prof[i, :len(train[trial][2500:2500+6*1250])] = train[trial][2500:2500+6*1250]*500
        curr_trial = np.where(curr_raster==1)[0]
        curr_trial = [(s-3750)/1250 for s in curr_trial if s>2500]  # starts from -1 s 
        
        c = 'grey'
        calpha = 0.7
        dotsize = 0.35
        
        axs[1].scatter(curr_trial, [i+1]*len(curr_trial),
                       color=c, alpha=calpha, s=dotsize)
        axs[1].plot([first_licks[trial]/1250, first_licks[trial]/1250],
                    [i, i+1],
                    linewidth=2, color='orchid')
        # axs['A'].plot([pumps[temp_ordered[trial]]/1250, pumps[temp_ordered[trial]]/1250],
        #               [trial, trial+1],
        #               linewidth=2, color='darkgreen')
        
    for i, trial in enumerate(reversed(late_trials)):
        curr_raster = raster[trial]
        late_prof[i, :len(train[trial][2500:2500+6*1250])] = train[trial][2500:2500+6*1250]*500
        curr_trial = np.where(curr_raster==1)[0]
        curr_trial = [(s-3750)/1250 for s in curr_trial if s>2500]  # starts from -1 s 
        
        c = 'grey'
        calpha = 0.7
        dotsize = 0.35
        
        axs[0].scatter(curr_trial, [i+1]*len(curr_trial),
                       color=c, alpha=calpha, s=dotsize)
        axs[0].plot([first_licks[trial]/1250, first_licks[trial]/1250],
                    [i, i+1],
                    linewidth=2, color='orchid')
        # axs['A'].plot([pumps[temp_ordered[trial]]/1250, pumps[temp_ordered[trial]]/1250],
        #               [trial, trial+1],
        #               linewidth=2, color='darkgreen')
        
    
    e_mean = np.mean(early_prof, axis=0)
    l_mean = np.mean(late_prof, axis=0)
    max_y = max(max(e_mean), max(l_mean))
    min_y = min(min(e_mean), min(l_mean))
    
    axt1 = axs[1].twinx()
    axt1.plot(xaxis, np.mean(early_prof, axis=0), color='k')
    axt1.set(ylabel='spike rate (Hz)',
             ylim=(min_y, max_y))
    
    axt0 = axs[0].twinx()
    axt0.plot(xaxis, np.mean(late_prof, axis=0), color='k')
    axt0.set(ylabel='spike rate (Hz)',
             ylim=(min_y, max_y))
    
    for i in range(2):
        axs[i].set(xlabel='time from run-onset (s)', ylabel='trial #',
                   yticks=[1, 10], xticks=[0, 2, 4],
                   xlim=(-1, 5))
        for p in ['top', 'right']:
            axs[i].spines[p].set_visible(False)
    
    fig.suptitle(cluname)
    
    fig.tight_layout()
    plt.show()
    
    if cluname in tag_list:
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks_earlyvlate_only\{}_tagged.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks_earlyvlate_only\{}_tagged.pdf'.format(cluname),
                    bbox_inches='tight')
    elif cluname in put_list:
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks_earlyvlate_only\{}_putative.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks_earlyvlate_only\{}_putative.pdf'.format(cluname),
                    bbox_inches='tight')
    else:
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks_earlyvlate_only\{}.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks_earlyvlate_only\{}.pdf'.format(cluname),
                    bbox_inches='tight')
    
    plt.close(fig)