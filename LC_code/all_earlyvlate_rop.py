# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:37:16 2023

loop over all cells for early v late trials

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
import pandas as pd 
import sys 
import scipy.io as sio
from scipy.stats import wilcoxon 


#%% load data 
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')

rasters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()
trains = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_info.npy',
                 allow_pickle=True).item()

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% obtain structure 
clu_list = list(cell_prop.index)


#%% parameters for processing 
window = [3750-313, 3750+313]  # window for spike summation, half a sec around run onsets


#%% main loop 
for pathname in pathLC[36:37]:
    sessname = pathname[-17:]
    
    for cluname in clu_list:
        # if cluname[:17]==sessname:
        if cluname=='A067r-20230821-02 clu20':
            print(cluname)
            train = trains[cluname]
            raster = rasters[cluname]
            early = []  # 10 spike rates 
            late = []  # 10 spike rates 
            
            filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
            alignRun = sio.loadmat(filename)
            
            licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
            starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
            tot_trial = licks.shape[0]
            
            behParf = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
            behPar = sio.loadmat(behParf)
            stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
            stimOn_ind = np.where(stimOn!=0)[0]-1
            bad_beh_ind = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
            
            first_licks = []
            for trial in range(tot_trial):
                lk = [l for l in licks[trial] if l-starts[trial] > 1250]  # only if the animal does not lick in the first second (carry-over licks)
                
                if len(lk)==0:
                    first_licks.append(10000)
                else:
                    first_licks.extend(lk[0]-starts[trial])

            # sort trials by first lick time
            temp = list(np.arange(tot_trial))
            licks_ordered, temp_ordered = zip(*sorted(zip(first_licks, temp)))
            
            early_trials = []
            late_trials = []
            
            for trial in temp_ordered[:40]:
                if trial not in stimOn_ind:
                    early_trials.append(trial)
                if len(early_trials)>=20:
                    break
            for trial in reversed(temp_ordered[:40]):
                if trial not in stimOn_ind:
                    late_trials.append(trial)
                if len(late_trials)>=20:
                    break
            
            early_prof = []
            late_prof = []
            
            for trial in early_trials:
                curr_raster = raster[trial]
                curr_train = train[trial]
                early.append(sum(curr_raster[window[0]:window[1]]))
                early_prof.append(curr_train[2500:5000])
            for trial in late_trials:
                curr_raster = raster[trial]
                curr_train = train[trial]
                late.append(sum(curr_raster[window[0]:window[1]]))
                late_prof.append(curr_train[2500:5000])
                
            fig, ax = plt.subplots(figsize=(4,3))
            xaxis = np.arange(2500)/1250-1
            early, = ax.plot(xaxis, np.mean(early_prof, axis=0)*1250, color='red')
            late, = ax.plot(xaxis, np.mean(late_prof, axis=0)*1250, color='darkred')
            ax.legend([early, late], ['early', 'late'], frameon=False)
            ax.set(xlabel='time (s)', ylabel='spike rate (Hz)',
                   xticks=[-1, 0, 1], yticks=[3, 5, 7])
    
            # fig, ax = plt.subplots(figsize=(4,4))
        
            for p in ['top', 'right']:
                ax.spines[p].set_visible(False)
            # ax.set_xticklabels(['early', 'late'], minor=False)
        
            # pval = wilcoxon(early, late)[1]
            # ax.set(ylabel='avg. spike rate (Hz)',
            #        title='early v late lick trials p={}'.format(round(pval, 3)))
        
            # bp = ax.boxplot([early, late],
            #                 positions=[0.5, 1.5],
            #                 patch_artist=True,
            #                 notch='True')
            # colors = ['coral', 'darkcyan']
            # for patch, color in zip(bp['boxes'], colors):
            #     patch.set_facecolor(color)
            # bp['fliers'][0].set(marker ='v',
            #                 color ='#e7298a',
            #                 markersize=2,
            #                 alpha=0.5)
            # bp['fliers'][1].set(marker ='o',
            #                 color ='#e7298a',
            #                 markersize=2,
            #                 alpha=0.5)
            # for median in bp['medians']:
            #     median.set(color='darkred',
            #                linewidth=1)
                
            # ax.scatter([[.85]*20, [1.85]*20], [early, late], zorder=2,
            #            s=15, color='grey', edgecolor='k', alpha=.5)
            # ax.plot([[.85]*20, [1.85]*20], [early, late], zorder=2,
            #         color='grey', alpha=.5)
            
            # tg = cell_prop['tagged'][cluname]
            # pt = cell_prop['putative'][cluname]
            # pk = cell_prop['peakness'][cluname]
            
            # fig.suptitle('{} pk{} tg{} pt{}'.format(cluname, int(pk), int(tg), int(pt)))
        
            # fig.tight_layout()
            # plt.show()
        
            # fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_early_v_late\{}.png'.format(cluname),
            #             dpi=500,
            #             bbox_inches='tight')
        
            # plt.close(fig)