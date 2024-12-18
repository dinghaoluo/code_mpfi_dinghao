# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:26:49 2023

compiling cell properties into a dataframe

@author: Dinghao Luo
"""


#%% imports 
import os
import pandas as pd
import scipy.io as sio
import sys
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr

# peak detection functions
sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import peak_detection_functions as pdf  # once i imported this as pd... stupid
import alignment_functions as af
from common import gaussian_kernel_unity


#%% GPU acceleration
# it turned out that all of the computation here can be done faster with CPU
# therefore i removed all the GPU acceleration stuff 
import numpy as np
from scipy.stats import sem


#%% load data 
print('loading data...')
all_rasters = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_rasters_simp_name.npy',
    allow_pickle=True
    ).item()
all_trains = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_info.npy',
    allow_pickle=True
    ).item()
tagged_waveforms = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\tagged_only_analysis\LC_all_waveforms.npy',
    allow_pickle=True
    ).item()
kmeans = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_UMAP_kmeans.npy',
    allow_pickle=True
    )  # this is for putative cell classification

clulist = list(all_rasters.keys())
taglist = list(tagged_waveforms.keys())


#%% parameters
global loadnew  # if a new session starts 
global spike_rate_file, beh_file, align_run_file, speed_trunc
global first_licks, trial_list, tot_trial_good_nonstim

global samp_freq
samp_freq = 1250

xaxis = np.arange(6 * samp_freq) / samp_freq - 3  # in seconds, since sampling freq is 1250 Hz 


#%% functions 
def compute_lick_sensitivity(cluname, 
                             trains, 
                             rasters, 
                             identity,
                             around=6,
                             bootstrap=1000,
                             GPU_AVAILABLE=False):
    global loadnew, align_run_file, beh_file, first_licks, trial_list, tot_trial_good_nonstim
    if loadnew:
        licks = align_run_file['trialsRun']['lickLfpInd'][0][0][0][1:]
        starts = align_run_file['trialsRun']['startLfpInd'][0][0][0][1:]
        tot_trial = licks.shape[0]
    
        stim_on = np.asarray(beh_file['behPar']['stimOn'][0][0][0][1:])
        stim_idx = np.where(stim_on!=0)[0]+1
    
        bad = np.asarray(beh_file['behPar'][0]['indTrBadBeh'][0][0])
        bad_idx = list(np.where(bad==1)[1]-1) if bad.sum()!=0 else []
    
        first_licks = []
        for trial in range(tot_trial):
            lk = [l for l in licks[trial] if l-starts[trial] > samp_freq]  # only if the animal does not lick in the first second (carry-over licks)
            if len(lk)==0:
                first_licks.append(np.NAN)
                bad_idx.append(trial)  # just in case...
            else:
                first_licks.extend(lk[0]-starts[trial])
        
        trial_list = [t for t in np.arange(tot_trial) 
                      if t not in bad_idx 
                      and t not in stim_idx]  # do not mind the bad and/or stim trials
        tot_trial_good_nonstim = len(trial_list)

    # plotting
    fig = plt.figure(figsize=(3.5, 1.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1])  # make axs[1] narrower
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    
    ratio = []
    aligned_prof = np.zeros((tot_trial_good_nonstim, samp_freq * around))
    aligned_raster = np.zeros((tot_trial_good_nonstim, samp_freq * around))
    for i, trial in enumerate(trial_list):
        raster = np.asarray(rasters[trial])
        train = np.asarray(trains[trial])
        
        first_lick = first_licks[trial]
        
        # process aligned_prof
        end_idx = first_lick + 3*samp_freq + 3*samp_freq  # the second +3750 is to deal with the bef_time (relative to run) of 3 s
        if len(train) < end_idx:
            if len(train) + around * samp_freq < end_idx:
                aligned_prof[i, :] = np.zeros(around * samp_freq)  # no spikes 3 seconds around first lick
            else:
                aligned_prof[i, :] = np.pad(
                    train[first_lick:first_lick + 6 * samp_freq], 
                    (0, end_idx-len(train))
                    )
        else:
            aligned_prof[i, :] = train[first_lick:first_lick + 6 * samp_freq]
            
        # similarly, process aligned_raster
        if len(raster) < end_idx:
            if len(raster) + around * samp_freq < end_idx:
                aligned_raster[i, :] = np.zeros(around * samp_freq)
            else:
                aligned_raster[i, :] = np.pad(
                    raster[first_lick:first_lick + 6 * samp_freq],
                    (0, end_idx-len(raster))
                    )
        else:
            aligned_raster[i, :] = raster[first_lick:first_lick + 6 * samp_freq]
        
        curr_trial = [(s - 3 * samp_freq) / samp_freq 
                      for s 
                      in np.where(raster[first_lick:first_lick + 6 * samp_freq]==1)[0]]
        
        axs[0].scatter(curr_trial, [i+1]*len(curr_trial),
                       color='grey', alpha=.25, s=.6)
    
    # top--.001, .01, .05, mean, .05, .01, .001--bottom; 7 values in total
    shuf_ratios = af.bootstrap_ratio(
        aligned_prof,
        bootstrap=bootstrap,
        GPU_AVAILABLE=GPU_AVAILABLE
        )
     
    aligned_prof_mean = np.nanmean(aligned_prof, axis=0)*1250
    aligned_prof_sem = sem(aligned_prof, axis=0)*1250
    true_ratio = np.sum(aligned_prof_mean[3750:3750+1250])/np.sum(aligned_prof_mean[3750-1250:3750])
    
    if true_ratio>=shuf_ratios[2]: 
        lick_sensitive = True
        lick_sensitive_type = 'ON'
        suffix = ' lick-ON'
        for i, ratio in enumerate(shuf_ratios[:3]):  # iterate till mean
            if true_ratio>ratio:
                break
        if i==0: lick_sensitive_signif = '***'
        if i==1: lick_sensitive_signif = '**'
        if i==2: lick_sensitive_signif = '*'
    elif true_ratio<=shuf_ratios[-3]:
        lick_sensitive = True
        lick_sensitive_type = 'OFF'
        suffix = ' lick-OFF'
        for i, ratio in enumerate(reversed(shuf_ratios[4:])):
            if true_ratio<ratio:
                break
        if i==0: lick_sensitive_signif = '***'
        if i==1: lick_sensitive_signif = '**'
        if i==2: lick_sensitive_signif = '*'
    else:
        lick_sensitive = False
        lick_sensitive_type = np.NAN
        suffix = 'unresponsive'
        lick_sensitive_signif = 'n.s.'
        
    axs[0].set(xlabel='time to 1st lick (s)', xlim=(-3, 3), xticks=[-3, 0, 3],
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
               title=lick_sensitive_signif)
    for s in ['top', 'right', 'bottom']: 
        axs[1].spines[s].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    ax_twin.spines['top'].set_visible(False)
    
    fig.tight_layout()
    plt.show()
    
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\LC_ephys\lick_sensitivity\single_cell_first_lick_aligned_rasters\{}'
        .format(f'{cluname} {identity} {suffix}'),
        dpi=200
        )
    
    return lick_sensitive, lick_sensitive_type, lick_sensitive_signif

def compute_speed_rate_corr(cluname, trains):
    """
    calculates the correlation between speed and spike rate for a single cell

    parameters:
    - cluname (str): identifier for the cell
    - trains (list): list of spike train arrays (trials x timepoints)

    returns:
    - tuple: (mean correlation coefficient, mean p-value)
    """
    global loadnew, speed_trunc, align_run_file
    if loadnew:
        align_run_file = sio.loadmat(
            r'Z:\Dinghao\MiceExp\ANMD{}\A{}\A{}\A{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
            .format(
                cluname[1:5], cluname[1:14], cluname[1:17], cluname[1:17]
                )
            )
        
        # speed of all trials
        speed_time_bef = align_run_file['trialsRun'][0]['speed_MMsecBef'][0][0][1:]
        speed_time = align_run_file['trialsRun'][0]['speed_MMsec'][0][0][1:]
        
        # concatenate bef and after running onset, and convolve with gaus_speed
        speed_time_all = np.empty(shape=speed_time.shape[0], dtype='object')
        for i in range(speed_time.shape[0]):
            bef = speed_time_bef[i]; aft = speed_time[i]
            speed_time_all[i] = np.concatenate([bef, aft])
            speed_time_all[i][speed_time_all[i]<0] = 0
        gaus_speed = gaussian_kernel_unity(samp_freq / 100)
        speed_time_conv = [np.convolve(np.squeeze(trial), gaus_speed, mode='same')
                           for trial in speed_time_all]
        
        speed_trunc = np.zeros((len(speed_time_conv), 5 * samp_freq))
        for trial, speeds in enumerate(speed_time_conv):
            if len(speeds) > (3 + 4) * samp_freq:  # 3 s bef, 4 s after
                speed_trunc[trial,:] = speeds[2 * samp_freq: 7 * samp_freq]
            else:
                speed_trunc[trial, :len(speeds) - 2 * samp_freq] = speeds[2 * samp_freq:]
    
    train_trunc = np.zeros((len(trains), 5 * samp_freq))
    for trial, train in enumerate(trains):
        if len(train) > (3 + 4) * samp_freq:
            train_trunc[trial,:] = train[2 * samp_freq: 7 * samp_freq]
        else:
            train_trunc[trial, :len(train) - 2 * samp_freq] = train[2 * samp_freq:]
    
    rate_speed_corr = np.zeros((len(trains), 2))
    for trial in range(len(trains)):
        rate_speed_corr[trial,:] = list(pearsonr(speed_trunc[trial], train_trunc[trial]))
        
    # r, p values
    return np.mean(rate_speed_corr[:,0]), np.mean(rate_speed_corr[:,1])

def get_identity(cluname: str, 
                 taglist: list, 
                 kmeans: list, 
                 spike_rates: float) -> str:
    """
    Determines the identity of a cell based on tagging, clustering, and its spike rate.

    Parameters:
    - cluname (str): The unique identifier for the cell.
    - taglist (list): List of tagged cell identifiers.
    - kmeans (list): KMeans clustering results, where '1' signifies non-putative.
    - spike_rate (float): The cell's spike rate.

    Returns:
    - str: The identity of the cell as 'tagged', 'putative', or 'other'.
    """
    if cluname in taglist:
        return 'tagged'
    for i, e in enumerate(kmeans):  # only executes if the cell is not tagged 
        if e==1:
            return 'other'
        elif spike_rate>=10:  # filters out cells with a high spike rate
            return 'other'
        else:
            return 'putative'

def get_spike_rate(cluname: str) -> float:
    """
    Retrieves the spike rate for a given cell from the corresponding data file.

    Parameters:
    - cluname (str): The unique identifier for the cell, in the format "Axxxr-202xxxxx-0x clu xx x".

    Returns:
    - float: The spike rate of the cell.
    """
    global loadnew, spike_rate_file
    if loadnew:  # if a new session starts
        spike_rate_file = sio.loadmat(
            r'Z:\Dinghao\MiceExp\ANMD{}\{}\{}\{}_DataStructure_mazeSection1_TrialType1_FR_Ctrl_Run0_mazeSess1'
            .format(
                cluname[1:5], cluname[:14], cluname[:17], cluname[:17]
                )
            )
    spike_rate_ctrl = spike_rate_file['mFRStructSessCtrl']['mFR'][0][0][0]
    clu_id = int(cluname.split('clu')[1])
    return spike_rate_ctrl[clu_id-2]

def get_first_stim_idx(cluname: str) -> int:
    """
    Retrieves the index of the first stimulation trial for a given cell.

    Parameters:
    - cluname (str): The unique identifier for the cell, in the format "Axxxr-202xxxxx-0x clu xx x".

    Returns:
    - int: The index of the first stimulation trial, or -1 if no stimulation trial exists.
    """
    global loadnew, beh_file
    if loadnew:  # if a new session starts
        beh_file = sio.loadmat(
            r'Z:\Dinghao\MiceExp\ANMD{}\{}\{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1'
            .format(
                cluname[1:5], cluname[:14], cluname[:17], cluname[:17]
                )
            )
    trial_types = beh_file['behPar']['stimOn'][0][0][0]
    return next((trial for trial, stim_on 
                 in enumerate(trial_types) 
                 if stim_on),
                -1)


#%% dataframe initialisation/loading
fname = r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_properties.pkl'
if os.path.exists(fname):
    df = pd.read_pickle(fname)
    print(f'df loaded from {fname}')
    processed_sess = df.index.tolist()
else:
    sess = {
        'sessname': [],
        'identity': [],
        'spike_rate': [],
        'run_onset_peak': [],
        'speed_rate_r': [],
        'speed_rate_p': [],
        'lick_sensitive': [],
        'lick_sensitive_type': [],
        'lick_sensitive_signif': []
        }
    df = pd.DataFrame(sess)


#%% main loop 
sessname = ''

for cluname in clulist[:1]:
    print(cluname)
    trains = all_trains[cluname]
    rasters = all_rasters[cluname]
    
    if cluname[:17]==sessname:
        sessname = cluname[:17]
        loadnew = False
    else:  # if a new session starts, load new files
        loadnew = True
    
    # spike rate 
    spike_rate = get_spike_rate(cluname)
    
    # cell identity ('tagged', 'putative' or 'other')
    identity = get_identity(cluname, taglist, kmeans, spike_rate)
    
    # we don't need the single-trial parameters, but we do need the first stim
    # to identify the baseline for run-onset burst detection
    first_stim = get_first_stim_idx(cluname)
    
    # run-onset burst detection 
    peak, mean_prof, shuf_prof = pdf.peak_detection(
        trains, 
        bootstrap=1000)
    pdf.plot_peak_v_shuf(cluname, mean_prof, shuf_prof, peak,
                         savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\peak_detection\{}'
                         .format(f'{cluname} {identity} {peak}')
                         )  # plot the detected peaks and save to ...
    
    # rate-speed correlation
    speed_rate_r, speed_rate_p = compute_speed_rate_corr(cluname, trains)
    
    # lick alignment 
    lick_sensitive, lick_sensitivity_type, lick_sensitivyt_signif = compute_lick_sensitivity(
        cluname,
        trains, 
        rasters,
        identity,
        bootstrap=1000
        )

    df.loc[cluname] = np.array(
        [sessname,
         identity,
         spike_rate,
         peak,
         speed_rate_r,
         speed_rate_p,
         lick_sensitive,
         lick_sensitivity_type,
         lick_sensitivyt_signif],
        dtype='object'
        )
        
#%% save dataframe 
df.to_pickle(r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_properties.pkl')
print('\ndataframe saved')