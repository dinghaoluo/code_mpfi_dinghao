# -*- coding: utf-8 -*-
"""
Created based on all_raster_lick_ordered.py, on Thu Oct 17 16:44:45 2024

align rasters to 1st-licks

@author: Dinghao Luo
"""


#%% imports 
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd
import sys
from tqdm import tqdm

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% GPU acceleration
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
except ModuleNotFoundError:
    print('CuPy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
except Exception as e:
    # catch any other unexpected errors and print a general message
    print(f'An error occurred: {e}')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    # we are assuming that there is only 1 GPU device
    xp = cp
    import numpy as np  # for loading npy files that contain objects (dicts in this case)
    from common import sem_gpu as sem
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print(f'GPU-acceleration with {str(name)} and cupy')
else:
    import numpy as np
    from scipy.stats import sem
    xp = np
    print('GPU-acceleartion unavailable')


#%% load data 
all_rasters = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_rasters_simp_name.npy',
    allow_pickle=True
    ).item()
all_train = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_info.npy',
    allow_pickle=True
    ).item()

cell_prop = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_single_cell_properties.pkl'
    )


#%% specify RO peaking putative Dbh cells
clulist = list(cell_prop.index)

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
from time import time 
start = time()


#%% parameters 
xaxis = np.arange(6*1250)/1250-3  # in seconds, since sampling freq is 1250 Hz 


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
    trial_shuf_array = xp.zeros([tot_trial, length])
    for trial in range(tot_trial):
        rand_shift = xp.random.randint(1, length/2)
        trial_shuf_array[trial,:] = xp.roll(conv_aligned_spike_array[trial], -rand_shift)
    return xp.mean(trial_shuf_array, axis=0)

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
    shuf_ratio = xp.zeros(bootstraps)
    for shuffle in tqdm(range(bootstraps), desc='bootstrap-shuffling'):
        shuf_result = cir_shuf(conv_aligned_spike_array, length)
        shuf_ratio[shuffle] = xp.sum(shuf_result[3750:3750+1250])/xp.sum(shuf_result[3750-1250:3750])
    return xp.percentile(shuf_ratio, [99.9, 99, 95, 50, 5, 1, .1], axis=0)


#%% MAIN 
lick_sensitive = []
lick_sensitive_type = []
lick_sensitive_signif = []

for cluname in clulist:
    print(cluname)
    raster = all_rasters[cluname]
    train = all_train[cluname]
    
    alignRun = sio.loadmat(
        r'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
        .format(
            cluname[1:5], cluname[:14], cluname[:17], cluname[:17]
            )
        )
    
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]
    
    behPar = sio.loadmat(
        r'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
        .format(
            cluname[1:5], cluname[:14], cluname[:17], cluname[:17]
            )
        )
    
    stimOn = xp.asarray(behPar['behPar']['stimOn'][0][0][0][1:])
    stimOn_ind = xp.where(stimOn!=0)[0]+1
    
    bad_beh = xp.asarray(behPar['behPar'][0]['indTrBadBeh'][0][0])
    bad_beh_ind = list(xp.where(bad_beh==1)[1]-1) if bad_beh.sum()!=0 else []
    
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

    trial_list = [t for t in np.arange(tot_trial) 
                  if t not in bad_beh_ind 
                  and t not in stimOn_ind]  # do not mind the bad and/or stim trials
    tot_trial = len(trial_list)

    ratio = []; ratio_shuf_95 = []; ratio_shuf_5 = []
    aligned_prof = xp.zeros((tot_trial, 1250*6))
    aligned_raster = xp.zeros((tot_trial, 1250*6))
    for i, trial in enumerate(trial_list):
        curr_raster = xp.asarray(raster[trial])
        curr_train = xp.asarray(train[trial])
        
        curr_lick = first_licks[trial]  # this is used as indices so we don't need it in VRAM (if GPU_AVAILABLE)
        
        # process aligned_prof
        end_idx = curr_lick+3750+3750  # the second +3750 is to deal with the bef_time (relative to run) of 3 s
        if len(curr_train) < end_idx:
            if len(curr_train)+6*1250 < end_idx:
                aligned_prof[i, :] = xp.zeros(6*1250)  # no spikes 3 seconds around first lick
            else:
                aligned_prof[i, :] = xp.pad(
                    curr_train[curr_lick+3750-3750:curr_lick+3750+3750], 
                    (0, end_idx-len(curr_train))
                    )
        else:
            aligned_prof[i, :] = curr_train[curr_lick+3750-3750:curr_lick+3750+3750]
            
        # similarly, process aligned_raster
        if len(curr_raster) < end_idx:
            if len(curr_raster)+6*1250 < end_idx:
                aligned_raster[i, :] = xp.zeros(6*1250)
            else:
                aligned_raster[i, :] = xp.pad(curr_raster[curr_lick+3750-3750:curr_lick+3750+3750], (0, end_idx-len(curr_raster)))
        else:
            aligned_raster[i, :] = curr_raster[curr_lick+3750-3750:curr_lick+3750+3750]
        
        curr_trial = [(s-3750)/1250 
                      for s 
                      in xp.where(curr_raster[curr_lick+3750-3750:curr_lick+3750+3750]==1)[0]]
        
        # for plotting matpltolib doesn't take cupy arrays 
        # also because this is the last place we are using curr_trial, we can 
        #   simply replace it in-place
        if GPU_AVAILABLE: curr_trial = [spike.get() for spike in curr_trial]
        axs[0].scatter(curr_trial, [i+1]*len(curr_trial),
                         color='grey', alpha=.25, s=.6)
    
    # top--.001, .01, .05, mean, .05, .01, .001--bottom; 7 values in total
    shuf_ratios = bootstrap_ratio(aligned_prof)
     
    aligned_prof_mean = xp.nanmean(aligned_prof, axis=0)*1250
    aligned_prof_sem = sem(aligned_prof, axis=0)*1250
    true_ratio = xp.sum(aligned_prof_mean[3750:3750+1250])/xp.sum(aligned_prof_mean[3750-1250:3750])
    
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
        lick_sensitive_type.append(xp.NAN)
        suffix = ''
        signif = 'n.s.'
    
    lick_sensitive_signif.append(signif)
    
    axs[0].set(xlabel='time to 1st lick (s)', xlim=(-3, 3), xticks=[-3, 0, 3],
               ylabel='trial #',
               title=cluname+suffix)
    axs[0].title.set_fontsize(10)

    ax_twin = axs[0].twinx()
    
    # similar to above where curr_trial was converted in-place, here we can 
    #   convert aligned_prof etc. in-place 
    if GPU_AVAILABLE: 
        aligned_prof_mean = aligned_prof_mean.get()
        aligned_prof_sem = aligned_prof_sem.get()
        shuf_ratios = [ratio.get() for ratio in shuf_ratios]
        true_ratio = true_ratio.get()
    
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
    
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\LC_ephys\lick_sensitivity\single_cell_first_lick_aligned_rasters\{}'
        .format(
            cluname+tp+suffix
            ),
        dpi=200
        )


#%% timer 
print(time() - start)


#%% save to dataframe
cell_prop = cell_prop.assign(lick_sensitive_shuf=pd.Series(lick_sensitive).values)
cell_prop = cell_prop.assign(lick_sensitive_shuf_type=pd.Series(lick_sensitive_type).values)
cell_prop = cell_prop.assign(lick_sensitive_shuf_signif=pd.Series(lick_sensitive_signif).values)

cell_prop.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')