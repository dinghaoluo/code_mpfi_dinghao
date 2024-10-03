# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:29:45 2023
Modified on Tue 24 Sept 2024 

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
noStim = 'N'

for cluname in clu_list:
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
    
    if noStim=='Y' or noStim=='y':
        temp_ordered = [t for t in temp_ordered if t not in stimOn_ind]
        tot_trial = len(temp_ordered)  # reset tot_trial if noStim

    suffix = ' '
    if cluname in tag_list: suffix = ' tagged Dbh+'
    if cluname in put_list: suffix = ' putative Dbh+'
    clutitle = cluname + suffix

    # plotting
    fig, ax = plt.subplots(figsize=(3,2.2))
    ax.set(xticks=[0, 2, 4], xlim=(-1, 6), xlabel='time (s)', 
           yticks=[1, 50, 100], ylabel='trial # by first licks',
           title=clutitle)
    for p in ['top', 'right']:
        ax.spines[p].set_visible(False)

    line_counter = 0  # for counting scatter plot lines 
    for trial in range(tot_trial):
        if temp_ordered[trial] in bad_beh_ind:
            continue
        curr_raster = raster[temp_ordered[trial]]
        curr_train = train[temp_ordered[trial]]
        
        curr_trial = np.where(raster[temp_ordered[trial]]==1)[0]
        curr_trial = [(s-3750)/1250 for s in curr_trial if s>2500]  # starts from -1 s 
        
        c = 'grey'
        calpha = 0.7
        dotsize = 1
        if (noStim=='N' or noStim=='n') and stimOn[temp_ordered[trial]]==1:
            c = 'royalblue'
            calpha = 1.0
            dotsize = 2
        
        ax.scatter(curr_trial, [line_counter+1]*len(curr_trial),
                   color=c, alpha=calpha, s=dotsize)
        ax.plot([licks_ordered[trial]/1250, licks_ordered[trial]/1250],
                [line_counter, line_counter+1],
                linewidth=2, color='orchid')
        # ax.plot([pumps[temp_ordered[trial]]/1250, pumps[temp_ordered[trial]]/1250],
        #          [trial, trial+1],
        #          linewidth=2, color='darkgreen')
        
        line_counter+=1
     
    fl, = ax.plot([],[],color='orchid',label='1st licks')
    # pp, = ax.plot([],[],color='darkgreen',alpha=.35,label='rew.')
    # ax.legend(handles=[fl, pp], frameon=False, fontsize=8)
    
    # plt.grid(False)
    plt.show()
    
    if cluname in tag_list:
        fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks\raster_only\{}_tagged.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')
        fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks\raster_only\{}_tagged.pdf'.format(cluname),
                    bbox_inches='tight')
    elif cluname in put_list:
        fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks\raster_only\{}_putative.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')
        fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks\raster_only\{}_putative.pdf'.format(cluname),
                    bbox_inches='tight')
    else:
        fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks\raster_only\{}.png'.format(cluname),
                    dpi=300,
                    bbox_inches='tight')
        fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_by_first_licks\raster_only\{}.pdf'.format(cluname),
                    bbox_inches='tight')
    
    # plt.close(fig)
    

#%% figure code, to plot density of stim trials 
# import seaborn as sns

# density = []
# for trial in temp_ordered:
#     if stimOn[trial]==1:
#         density.append(1)
#     else:
#         density.append(0)
# density_ind = np.where(np.array(density)==1)
# density_ind = [0-s for s in density_ind]

# fig, ax = plt.subplots(figsize=(10,2))

# for p in ['top','right','left','bottom']:
#     ax.spines[p].set_visible(False)
# ax.set(yticks=[]); ax.set(xticks=[])

# ax.hist(density_ind, bins=24, edgecolor='k', color='royalblue', linewidth=3)

# sns.set_style('whitegrid')
# sns.kdeplot(density_ind, bw=0.5)

# fig.tight_layout()
# plt.show()

# fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\eg_session_stimdensity_hist.png',
#             dpi=500,
#             bbox_inches='tight')