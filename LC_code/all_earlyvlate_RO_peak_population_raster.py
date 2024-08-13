# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:30:12 2023

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial' 
import sys 
import pandas as pd 
import scipy.io as sio
from scipy.stats import wilcoxon, ttest_rel, sem


#%% load data 
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')

rasters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% obtain structure
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


#%% parameters for processing 
window = [3750-313, 3750+313]  # window for spike summation, half a sec around run onsets


#%% population analysis of RO peaking cells
sess_list = [
    'A045r-20221207-02',
    
    'A049r-20230120-04',
    
    'A062r-20230626-01',
    'A062r-20230626-02',
    'A062r-20230629-01',
    'A062r-20230629-02',
    
    'A065r-20230726-01',
    'A065r-20230727-01',
    'A065r-20230728-01',
    'A065r-20230728-02',
    'A065r-20230729-01',
    'A065r-20230801-01',
    
    'A067r-20230821-01',
    'A067r-20230821-02',
    'A067r-20230823-01',
    'A067r-20230823-02',
    'A067r-20230824-01',
    'A067r-20230824-02',
    'A067r-20230825-01',
    'A067r-20230825-02']

early_all_tagged = []; late_all_tagged = []
early_all_putative = []; late_all_putative = []
early_all_pooled = []; late_all_pooled = []

for sessname in sess_list:
    print(sessname)
    
    print('tagged...')
    early_sess = []; late_sess = []
    early_sum = 0; late_sum = 0
    for cluname in tag_rop_list:
        if cluname[:17]==sessname:
            raster = rasters[cluname]
            
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

            temp = list(np.arange(tot_trial))
            licks_ordered, temp_ordered = zip(*sorted(zip(first_licks, temp)))
            
            early_trials = []
            late_trials = []
            
            for trial in temp_ordered[:50]:
                if trial not in stimOn_ind:
                    early_trials.append(trial)
                if len(early_trials)>=20:
                    break
            for trial in reversed(temp_ordered[:50]):
                if trial not in stimOn_ind:
                    late_trials.append(trial)
                if len(late_trials)>=20:
                    break
            
            for trial in early_trials:
                curr_raster = raster[trial]
                early_sum += sum(curr_raster[window[0]:window[1]])
            for trial in late_trials:
                curr_raster = raster[trial]
                late_sum += sum(curr_raster[window[0]:window[1]])
            early_sess.append(early_sum)
            late_sess.append(late_sum)
    
    # early sess and late sess now have all rop cells in this session
    early_all_tagged.append(early_sess)  # list of lists  
    late_all_tagged.append(late_sess)  # same as above
    
    print('putative...')
    early_sess = []; late_sess = []
    early_sum = 0; late_sum = 0
    for cluname in put_rop_list:
        if cluname[:17]==sessname:
            raster = rasters[cluname]
            
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

            temp = list(np.arange(tot_trial))
            licks_ordered, temp_ordered = zip(*sorted(zip(first_licks, temp)))
            
            early_trials = []
            late_trials = []
            
            for trial in temp_ordered[:50]:
                if trial not in stimOn_ind:
                    early_trials.append(trial)
                if len(early_trials)>=20:
                    break
            for trial in reversed(temp_ordered[:50]):
                if trial not in stimOn_ind:
                    late_trials.append(trial)
                if len(late_trials)>=20:
                    break
            
            for trial in early_trials:
                curr_raster = raster[trial]
                early_sum += sum(curr_raster[window[0]:window[1]])
            for trial in late_trials:
                curr_raster = raster[trial]
                late_sum += sum(curr_raster[window[0]:window[1]])
            early_sess.append(early_sum)
            late_sess.append(late_sum)
    
    # early sess and late sess now have all rop cells in this session
    early_all_putative.append(early_sess)  # list of lists  
    late_all_putative.append(late_sess)  # same as above
    
    print('pooled...')
    early_sess = []; late_sess = []
    early_sum = 0; late_sum = 0
    pooled_list = put_rop_list + tag_rop_list
    for cluname in pooled_list:
        if cluname[:17]==sessname:
            raster = rasters[cluname]
            
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

            temp = list(np.arange(tot_trial))
            licks_ordered, temp_ordered = zip(*sorted(zip(first_licks, temp)))
            
            early_trials = []
            late_trials = []
            
            for trial in temp_ordered[:50]:
                if trial not in stimOn_ind:
                    early_trials.append(trial)
                if len(early_trials)>=20:
                    break
            for trial in reversed(temp_ordered[:50]):
                if trial not in stimOn_ind:
                    late_trials.append(trial)
                if len(late_trials)>=20:
                    break
            
            for trial in early_trials:
                curr_raster = raster[trial]
                early_sum += sum(curr_raster[window[0]:window[1]])
            for trial in late_trials:
                curr_raster = raster[trial]
                late_sum += sum(curr_raster[window[0]:window[1]])
            early_sess.append(early_sum)
            late_sess.append(late_sum)
    
    # early sess and late sess now have all rop cells in this session
    early_all_pooled.append(early_sess)  # list of lists  
    late_all_pooled.append(late_sess)  # same as above
    
del_tagged = []; del_putative = []; del_pooled = []
for sess in range(len(sess_list)):  # get rid of sessions with fewer than 2 tagged cells
    if len(early_all_tagged[sess])<2:
        del_tagged.append(sess)
    if len(early_all_putative[sess])<2:
        del_putative.append(sess)
    if len(early_all_pooled[sess])<2:
        del_pooled.append(sess)

early_all_tagged = [s for i, s in enumerate(early_all_tagged) if i not in del_tagged]
late_all_tagged = [s for i, s in enumerate(late_all_tagged) if i not in del_tagged]
early_all_putative = [s for i, s in enumerate(early_all_putative) if i not in del_putative]
late_all_putative = [s for i, s in enumerate(late_all_putative) if i not in del_putative]
early_all_pooled = [s for i, s in enumerate(early_all_pooled) if i not in del_pooled]
late_all_pooled = [s for i, s in enumerate(late_all_pooled) if i not in del_pooled]

early_all_tagged_mean = [np.nanmean(ls)/10 for ls in early_all_tagged]  # average for each session 
late_all_tagged_mean = [np.nanmean(ls)/10 for ls in late_all_tagged]  # same as above 

early_all_putative_mean = [np.nanmean(ls)/10 for ls in early_all_putative]  # average for each session 
late_all_putative_mean = [np.nanmean(ls)/10 for ls in late_all_putative]  # same as above 

early_all_pooled_mean = [np.nanmean(ls)/10 for ls in early_all_pooled]  # average for each session 
late_all_pooled_mean = [np.nanmean(ls)/10 for ls in late_all_pooled]  # same as above 
    

#%% plot 
fig, ax = plt.subplots(figsize=(4,4))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticklabels(['early', 'late'], minor=False)

pval = wilcoxon(early_all_tagged_mean, late_all_tagged_mean)[1]
ax.set(ylabel='population spike rate (Hz)',
       title='early v late lick trials p={}'.format(round(pval, 3)))

bp = ax.boxplot([early_all_tagged_mean, late_all_tagged_mean],
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
    
ax.scatter([[.5]*len(early_all_tagged), [1]*len(early_all_tagged)], [early_all_tagged_mean, late_all_tagged_mean], zorder=2,
           s=15, color='grey', edgecolor='k', alpha=.5)
ax.plot([[.5]*len(early_all_tagged), [1]*len(early_all_tagged)], [early_all_tagged_mean, late_all_tagged_mean], zorder=2,
        color='grey', alpha=.5)

fig.tight_layout()
plt.show()

# fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_ROpeak_population_earlyvlate_tagged.png',
#             dpi=500,
#             bbox_inches='tight')

plt.close(fig)


#%% plot putative
fig, ax = plt.subplots(figsize=(4,4))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticks([1, 2])
ax.set_xticklabels(['early', 'late'], minor=False)

pval = wilcoxon(early_all_putative_mean, late_all_putative_mean)[1]
ax.set(ylabel='population spike rate (Hz)',
       title='early v late lick trials p={}'.format(round(pval, 3)))

bp = ax.bar([1, 2], [np.mean(early_all_putative_mean), np.mean(late_all_putative_mean)], width=.15)
    
ax.scatter([[1.25]*len(early_all_putative), [1.75]*len(early_all_putative)], [early_all_putative_mean, late_all_putative_mean], zorder=2,
           s=15, color='grey', edgecolor='k', alpha=.5)
ax.plot([[1.25]*len(early_all_putative), [1.75]*len(early_all_putative)], [early_all_putative_mean, late_all_putative_mean], zorder=2,
        color='grey', alpha=.5)

ax.set(xlim=(0.75, 2.25))

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_putative_ROpeak_population_earlyvlate_putative.png',
            dpi=500,
            bbox_inches='tight')

plt.close(fig)


#%% statistics
pval = wilcoxon(early_all_pooled_mean, late_all_pooled_mean)[1]


#%% plot all (logged)
# log everything 
fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([early_all_pooled_mean, late_all_pooled_mean],
                   showmeans=True, showextrema=False)
vp['bodies'][0].set_color('darkred')
vp['bodies'][1].set_color('darkred')
for i in [0,1]:
    vp['bodies'][i].set_edgecolor('k')
    vp['bodies'][i].set_alpha(i*0.2+0.4)
vp['cmeans'].set(color='darkred')
    
ax.scatter([[1]*len(early_all_pooled), [2]*len(early_all_pooled)], [early_all_pooled_mean, late_all_pooled_mean], zorder=2,
           s=3, color='grey', edgecolor='none', alpha=1)
ax.plot([[1]*len(early_all_pooled), [2]*len(early_all_pooled)], [early_all_pooled_mean, late_all_pooled_mean], lw=1, zorder=2,
        color='grey', alpha=.5)

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.spines['left'].set_linewidth(1)

ax.set(xlim=(0.5, 2.5),
       xticks=[1,2], xticklabels=['early', 'late'],
       yticks=[5,10,15,20],
       ylabel='log pop. spike rate (Hz)',
       title='early v late lick trials\n wilcp={}'.format(round(pval, 6)))

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_pooled_ROpeak_population_earlyvlate.png',
            bbox_inches='tight',
            dpi=500)
fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_pooled_ROpeak_population_earlyvlate.pdf',
            bbox_inches='tight')
fig.savefig(r'Z:/Dinghao/paper/figures/figure_2_early_v_late_pop_spike_rate.pdf',
            bbox_inches='tight')

plt.close(fig)


#%% plot all (normalised to early)
diff_pooled = []
for i in range(len(early_all_pooled_mean)):
    diff_pooled.append(late_all_pooled_mean[i]-early_all_pooled_mean[i])

fig, ax = plt.subplots(figsize=(3,4))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.set_xticks([1, 2])
# ax.set_yticks([1, 2, 3])
ax.set_xticklabels(['early', 'late'], minor=False)

# statistics
pval = wilcoxon(early_all_pooled_mean, late_all_pooled_mean)[1]
ax.set(ylabel='Δ pop. spike rate (Hz)',
       title='early v late lick trials p={}'.format(round(pval, 3)))

# bar = ax.bar([1, 2], [0, np.mean(diff_pooled)], yerr=[0, sem(diff_pooled)], capsize=5,
#             color=['gainsboro', 'dimgrey'], edgecolor=['k','k'], width=.35)
bp = ax.boxplot([[0]*len(early_all_pooled), diff_pooled],
           positions=[1, 2],
           patch_artist=True)
colors = ['gainsboro', 'dimgrey']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
bp['fliers'][0].set(marker ='o',
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
    
ax.scatter([[1]*len(early_all_pooled), [2]*len(early_all_pooled)], [[0]*len(early_all_pooled), diff_pooled], zorder=2,
           s=15, color='grey', edgecolor='k', alpha=.5)
ax.plot([[1]*len(early_all_pooled), [2]*len(early_all_pooled)], [[0]*len(early_all_pooled), diff_pooled], zorder=2,
        color='grey', alpha=.5)

ax.set(xlim=(0.5, 2.5))

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_pooled_ROpeak_population_earlyvlate_delta.pdf',
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_pooled_ROpeak_population_earlyvlate_delta.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)