# -*- coding: utf-8 -*-
"""
Created based on all_raster_lick_ordered.py, on Thu Oct 17 16:44:45 2024

align rasters to 1st-licks

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd
from scipy.stats import ttest_rel, ranksums, wilcoxon, sem

# plotting parameters
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load data 
all_rasters = np.load(r'Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                      allow_pickle=True).item()
all_train = np.load(r'Z:\Dinghao\code_dinghao\LC_all\LC_all_info.npy',
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


#%% timer 
import time
start = time.time()


#%% shuffle function 
def cir_shuf(conv_aligned_spike_array, length=6*1250):
    """
    Parameters
    ----------
    conv_aligned_spike_array : numpy array
        smoothed spike array of one clu in this sessios, aligned to 1st-licks.
    length : int, optional
        the number of time bins in the input array per row.

    Returns
    -------
        the flattened array containing every trial in this session each shuffled once.
    """
    tot_trial = conv_aligned_spike_array.shape[0]
    trial_shuf_array = np.zeros([tot_trial, length])
    for trial in range(tot_trial):
        rand_shift = np.random.randint(1, length/2)
        trial_shuf_array[trial,:] = np.roll(conv_aligned_spike_array[trial], -rand_shift)
    return np.mean(trial_shuf_array, axis=0)

def bootstrap_ratio(conv_aligned_spike_array, bootstraps=500, length=6*1250):
    """
    Parameters
    ----------
    conv_aligned_spike_array : numpy array
        smoothed spike array of one clu in this sessios, aligned to 1st-licks.
    bootstraps : int, optional
        the number of times we want to run the bootstrapping. The default is 500.
    length : int, optional
        the number of time bins in the input array per row.

    Returns
    -------
        the percentage thresholds for the bootstrapping result.
    """
    shuf_ratio = np.zeros(bootstraps)
    for shuffle in range(bootstraps):
        shuf_result = cir_shuf(conv_aligned_spike_array, length)
        shuf_ratio[shuffle] = np.sum(shuf_result[3750:3750+1250])/np.sum(shuf_result[3750-1250:3750])
    return np.percentile(shuf_ratio, [99.9, 99, 95, 50, 5, 1, .1], axis=0)


#%% MAIN 
xaxis = np.arange(6*1250)/1250-3  # in seconds 

lick_sensitive = []
lick_sensitive_type = []
lick_sensitive_signif = []

for cluname in clu_list:
    print(cluname)
    raster = all_rasters[cluname]
    train = all_train[cluname]
    
    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    alignRun = sio.loadmat(filename)
    
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]
    
    behParf = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    behPar = sio.loadmat(behParf)
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stimOn_ind = np.where(stimOn!=0)[0]+1
    bad_beh_ind = list(np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1)
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] if l-starts[trial] > 1250]  # only if the animal does not lick in the first second (carry-over licks)
        if len(lk)==0:
            first_licks.append(np.NAN)
            bad_beh_ind.append(trial)  # just in case...
        else:
            first_licks.extend(lk[0]-starts[trial])

    # plotting
    fig = plt.figure(figsize=(3.5, 1.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1])  # make axs[1] narrower
    
    # define the subplots using the GridSpec layout
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    trial_list = [t for t in np.arange(tot_trial) if t not in bad_beh_ind and t not in stimOn_ind]  # do not mind the bad and/or stim trials
    tot_trial = len(trial_list)

    ratio = []; ratio_shuf_95 = []; ratio_shuf_5 = []
    aligned_prof = np.zeros((tot_trial, 1250*6))
    aligned_raster = np.zeros((tot_trial, 1250*6))
    for i, trial in enumerate(trial_list):
        curr_raster = raster[trial]
        curr_train = train[trial]
        curr_lick = first_licks[trial]
        
        # process aligned_prof
        end_idx = curr_lick+3750+3750
        if len(curr_train) < end_idx:
            if len(curr_train)+6*1250 < end_idx:
                aligned_prof[i, :] = np.zeros(6*1250)
            else:
                aligned_prof[i, :] = np.pad(curr_train[curr_lick+3750-3750:curr_lick+3750+3750], (0, end_idx-len(curr_train)))
        else:
            aligned_prof[i, :] = curr_train[curr_lick+3750-3750:curr_lick+3750+3750]
            
        # similarly, process aligned_raster
        if len(curr_raster) < end_idx:
            if len(curr_raster)+6*1250 < end_idx:
                aligned_raster[i, :] = np.zeros(6*1250)
            else:
                aligned_raster[i, :] = np.pad(curr_raster[curr_lick+3750-3750:curr_lick+3750+3750], (0, end_idx-len(curr_raster)))
        else:
            aligned_raster[i, :] = curr_raster[curr_lick+3750-3750:curr_lick+3750+3750]
        
        curr_trial = [(s-3750)/1250 for s in np.where(curr_raster[curr_lick+3750-3750:curr_lick+3750+3750]==1)[0]]
        axs[0].scatter(curr_trial, [i+1]*len(curr_trial),
                         color='grey', alpha=.25, s=1)
    
    # top--.001, .01, .05, mean, .05, .01, .001--bottom; 7 values in total
    shuf_ratios = bootstrap_ratio(aligned_prof)
     
    aligned_prof_mean = np.nanmean(aligned_prof, axis=0)*1250
    aligned_prof_sem = sem(aligned_prof, axis=0)*1250
    true_ratio = np.sum(aligned_prof_mean[3750:3750+1250])/np.sum(aligned_prof_mean[3750-1250:3750])
    
    if true_ratio>=shuf_ratios[2]: 
        lick_sensitive.append(True)
        lick_sensitive_type.append('ON')
        suffix = ' lick-ON'
        for i, ratio in enumerate(shuf_ratios[:3]):  # iterate till mean
            if true_ratio>ratio:
                break
        if i==0: signif = '***'
        if i==1: signif = '**'
        if i==2: signif = '*'
    elif true_ratio<=shuf_ratios[-3]:
        lick_sensitive.append(True)
        lick_sensitive_type.append('OFF')
        suffix = ' lick-OFF'
        for i, ratio in enumerate(reversed(shuf_ratios[4:])):
            if true_ratio<ratio:
                break
        if i==0: signif = '***'
        if i==1: signif = '**'
        if i==2: signif = '*'
    else:
        lick_sensitive.append(False)
        lick_sensitive_type.append(np.NAN)
        suffix = ''
        signif = 'n.s.'
    
    lick_sensitive_signif.append(signif)
    
    axs[0].set(xlabel='time to 1st-lick (s)', xlim=(-3, 3), xticks=[-3, 0, 3],
               ylabel='trial #',
               title=cluname+suffix)
    axs[0].title.set_fontsize(10)

    ax_twin = axs[0].twinx()
    ax_twin.plot(xaxis, aligned_prof_mean, color='k')
    ax_twin.fill_between(xaxis, aligned_prof_mean+aligned_prof_sem,
                                aligned_prof_mean-aligned_prof_sem,
                         color='k', alpha=.25, edgecolor='none')
    ax_twin.set(ylabel='spike rate (Hz)')
    axs[1].plot([-1,1],[shuf_ratios[2], shuf_ratios[2]], color='grey')
    axs[1].plot([-1,1],[shuf_ratios[3], shuf_ratios[3]], color='grey')
    axs[1].plot([-1,1],[shuf_ratios[-3], shuf_ratios[-3]], color='grey')
    axs[1].plot([-1,1],[true_ratio, true_ratio], color='red')
    axs[1].set(xlim=(-2,2), xticks=[], xticklabels=[],
               ylabel='post-pre ratio',
               title=signif)
    for s in ['top', 'right', 'bottom']: 
        axs[1].spines[s].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    ax_twin.spines['top'].set_visible(False)
    
    fig.tight_layout()
    plt.show()
    
    tp = ''
    if cluname in tag_list: tp = ' tgd'
    if cluname in put_list: tp = ' put'
    
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_1st_lick_aligned\{}'.format(cluname+tp+suffix),
                dpi=200)


#%% timer 
passed_time = time.time() - start
print(passed_time)


#%% save to dataframe
cell_prop = cell_prop.assign(lick_sensitive_shuf=pd.Series(lick_sensitive).values)
cell_prop = cell_prop.assign(lick_sensitive_shuf_type=pd.Series(lick_sensitive_type).values)
cell_prop = cell_prop.assign(lick_sensitive_shuf_signif=pd.Series(lick_sensitive_signif).values)

cell_prop.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')