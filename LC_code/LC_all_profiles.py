# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:26:49 2023

compiling cell properties into a dataframe

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import pandas as pd
import scipy.io as sio
import sys
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr, sem

# peak detection functions
sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import peak_detection_functions as pdf  # once i imported this as pd... stupid
import alignment_functions as af
from common import gaussian_kernel_unity

# rec list 
sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC


#%% GPU acceleration
# it turned out that all of the computation here can be done faster with CPU
# therefore i removed all the GPU acceleration stuff 
# import numpy as np
# from scipy.stats import sem
'''
the above is not true: we simply did not parallelise the operations enough--
    now with GPU acceleration running 1 cell takes 2 seconds, compared to the 
    half a minute before, as it should be
'''
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
    from common import sem_gpu as sem
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print(f'GPU-acceleration with {str(name)} and cupy')
else:
    xp = np
    from scipy.stats import sem
    print('GPU-acceleration unavailable')


#%% functions 
def compute_lick_info(beh_file: np.ndarray,
                      align_run_file: np.ndarray,
                      samp_freq=1250) -> tuple:
    """
    extracts lick timing information and filters trials based on behaviour.

    parameters:
    - beh_file (np.ndarray): behavioural data file containing trial parameters
    - align_run_file (np.ndarray): file containing aligned running and licking data
    - samp_freq (int): sampling frequency in Hz, default 1250

    returns:
    - tuple:
        1. int: total number of good non-stimulation trials
        2. list: list of indices for good non-stimulation trials
        3. xp.ndarray: end indices of lick-aligned trials
        4. xp.ndarray: first lick times relative to trial start
    """
    licks = align_run_file['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = align_run_file['trialsRun']['startLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]

    stim_on = np.asarray(beh_file['behPar']['stimOn'][0][0][0][1:])
    stim_idx = np.where(stim_on!=0)[0]+1

    bad = np.asarray(beh_file['behPar'][0]['indTrBadBeh'][0][0])
    bad_idx = list(np.where(bad==1)[0]-1) if bad.sum()!=0 else []  # was np.where(bad==1)[1]-1 previously for some reason

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
    
    # prepare for vectorisation
    first_licks = xp.asarray([first_licks[trial] for trial in trial_list])
    end_indices = first_licks + 3 * samp_freq + 3 * samp_freq  # the second +3*samp_freq is to deal with the bef_time (relative to run) of 3 s
    
    return tot_trial_good_nonstim, trial_list, end_indices, first_licks

def compute_lick_sensitivity(cluname: str,
                             trains: xp.ndarray, 
                             rasters: xp.ndarray, 
                             identity: str,
                             tot_trial_good_nonstim: int,
                             trial_list: list,
                             end_indices: xp.ndarray,
                             first_licks: xp.ndarray,
                             samp_freq=1250,
                             around=6,
                             bootstrap=1000,
                             GPU_AVAILABLE=False) -> tuple:
    """
    computes the lick sensitivity of a cell based on spike trains and rasters.

    parameters:
    - cluname (str): identifier for the cell
    - trains (xp.ndarray): spike train data (trials x timepoints)
    - rasters (xp.ndarray): raster data (trials x timepoints)
    - identity (str): identity of the cell ('tagged', 'putative', etc.)
    - tot_trial_good_nonstim (int): number of good non-stimulation trials
    - trial_list (list): indices of good non-stimulation trials
    - end_indices (xp.ndarray): end indices of lick-aligned trials
    - first_licks (xp.ndarray): first lick times relative to trial start
    - samp_freq (int): sampling frequency in Hz, default 1250
    - around (int): time window in seconds, default 6
    - bootstrap (int): number of bootstrap iterations, default 1000
    - GPU_AVAILABLE (bool): if true, uses GPU for calculations, default false

    returns:
    - tuple:
        1. bool: whether the cell is lick-sensitive
        2. str: type of sensitivity ('ON', 'OFF', or unresponsive)
        3. str: significance of sensitivity ('***', '**', '*', or 'n.s.')
    """
    ratio = []    
    aligned_prof = xp.zeros((tot_trial_good_nonstim, samp_freq * around))
    aligned_rasters = xp.zeros((tot_trial_good_nonstim, samp_freq * around))
    for i, trial in enumerate(trial_list):
        train = xp.asarray(trains[trial])
        raster = xp.asarray(rasters[trial])
        if end_indices[i] <= len(train):
            aligned_prof[i, :] = train[first_licks[i]:first_licks[i]+6*samp_freq]
        else:
            aligned_prof[i, :] = xp.pad(
                train[first_licks[i]:],
                (0, 6 * samp_freq - len(train[first_licks[i]:])),
                mode='constant'
                )
        if end_indices[i] <= len(raster):
            aligned_rasters[i, :] = raster[first_licks[i]:first_licks[i]+6*samp_freq]
        else:
            aligned_rasters[i, :] = xp.pad(
                raster[first_licks[i]:],
                (0, 6 * samp_freq - len(raster[first_licks[i]:])),
                mode='constant'
                )
    
    # top--.001, .01, .05, mean, .05, .01, .001--bottom; 7 values in total
    shuf_ratios = af.bootstrap_ratio(
        aligned_prof,
        bootstrap=bootstrap,
        GPU_AVAILABLE=GPU_AVAILABLE
        )
     
    aligned_prof_mean = xp.nanmean(aligned_prof, axis=0)
    aligned_prof_sem = sem(aligned_prof, axis=0)
    true_ratio = xp.sum(
        aligned_prof_mean[3*samp_freq : 3*samp_freq+1*samp_freq]
        ) / xp.sum(
            aligned_prof_mean[3*samp_freq-1*samp_freq : 3*samp_freq]
            )
    
    if true_ratio>=shuf_ratios[2]: 
        lick_sensitive = True
        lick_sensitive_type = 'ON'
        suffix = 'lick-ON'
        for i, ratio in enumerate(shuf_ratios[:3]):  # iterate till mean
            if true_ratio>ratio:
                break
        if i==0: lick_sensitive_signif = '***'
        if i==1: lick_sensitive_signif = '**'
        if i==2: lick_sensitive_signif = '*'
    elif true_ratio<=shuf_ratios[-3]:
        lick_sensitive = True
        lick_sensitive_type = 'OFF'
        suffix = 'lick-OFF'
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

    # compute spike time for raster plots
    raster_arr = [
        (raster.nonzero()[0]-3*samp_freq)/samp_freq
        for raster in aligned_rasters]
    
    plot_lick_sensitivity(
        cluname,
        suffix,
        identity,
        lick_sensitive_signif,
        raster_arr, 
        aligned_prof_mean, 
        aligned_prof_sem, 
        shuf_ratios, 
        true_ratio,
        GPU_AVAILABLE=GPU_AVAILABLE)
    
    return lick_sensitive, lick_sensitive_type, lick_sensitive_signif

def compute_speeds(align_run_file: xp.ndarray, 
                   samp_freq=1250) -> xp.ndarray:
    """
    compute the speed of all trials and apply gaussian smoothing.

    parameters:
    - align_run_file: the file containing aligned running data.
    - samp_freq: the sampling frequency in Hz, default is 1250.

    returns:
    - an array of truncated and smoothed speed values for each trial.
    """
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
            
    return speed_trunc

def compute_speed_rate_corr(cluname: str, 
                            trains: xp.ndarray, 
                            speed_trunc: xp.ndarray,
                            samp_freq=1250) -> tuple:
    """
    calculate the correlation between running speed and spike rate for a single cell.

    parameters:
    - cluname: identifier for the cell.
    - trains: spike train data (trials x timepoints).
    - speed_trunc: truncated and smoothed speed data (trials x timepoints).
    - samp_freq: sampling frequency in Hz, default is 1250.

    returns:
    - tuple: (mean correlation coefficient, mean p-value).
    """
    train_trunc = np.zeros((len(trains), 5 * samp_freq))
    for trial, train in enumerate(trains):
        if len(train) > (3 + 4) * samp_freq:
            train_trunc[trial,:] = train[2 * samp_freq: 7 * samp_freq]
        else:
            train_trunc[trial, :len(train) - 2 * samp_freq] = train[2 * samp_freq:]
    
    rate_speed_corr = np.zeros((len(trains), 2))
    for trial in range(len(trains)):
        rate_speed_corr[trial,:] = pearsonr(speed_trunc[trial], train_trunc[trial])
        
    # r, p values
    return np.mean(rate_speed_corr[:,0]), np.mean(rate_speed_corr[:,1])

def get_acg(cluname: str,
            acg_dict: dict) -> np.ndarray:
    return acg_dict[cluname]

def get_first_stim_idx(cluname: str,
                       beh_file: xp.ndarray) -> int:
    """
    retrieve the index of the first stimulation trial for a given cell.

    parameters:
    - cluname: unique identifier for the cell, in the format "Axxxr-202xxxxx-0x clu xx x".
    - beh_file: behavioural data file containing stimulation trial information.

    returns:
    - int: index of the first stimulation trial, or -1 if no stimulation trial exists.
    """
    trial_types = beh_file['behPar']['stimOn'][0][0][0]
    return next((trial for trial, stim_on 
                 in enumerate(trial_types) 
                 if stim_on),
                -1)

def get_identity(cluname: str, 
                 tagged_keys: list, 
                 kmeans: dict, 
                 spike_rate: float) -> str:
    """
    determine the identity of a cell based on tagging, clustering, and its spike rate.

    parameters:
    - cluname: the unique identifier for the cell.
    - tagged_keys: list of tagged cell identifiers.
    - kmeans: kmeans clustering results, where '1' signifies non-putative.
    - spike_rate: the cell's spike rate.

    returns:
    - the identity of the cell as 'tagged', 'putative', or 'other'.
    """
    if cluname in tagged_keys:
        return 'tagged'
    else:
        if kmeans[cluname]==0:
            return 'other'
        elif kmeans[cluname]==1:
            if spike_rate<=10:  # filters out cells with a high spike rate
                return 'putative'
            else:
                return 'other'

def get_spike_rate(cluname: str, 
                   spike_rate_file: np.ndarray) -> float:
    spike_rate_ctrl = spike_rate_file['mFRStructSessCtrl']['mFR'][0][0][0]
    clu_id = int(cluname.split('clu')[1])
    return spike_rate_ctrl[clu_id-2]

def get_trial_matrix(trains, trialtype_idx, max_samples=1250*8):
    """
    get the trial matrix for a given cluster and trial type indices.

    parameters:
    - trains: list of spike trains for all clusters.
    - trialtype_idx: list of trial indices to include.
    - max_samples: maximum number of samples per trial.
    - clu: cluster identifier.

    returns:
    - temp_matrix: matrix of spike trains for the specified trials and cluster.
    """
    if len(trialtype_idx)==0:  # if there is no trial in the list 
        return xp.nan
    
    temp_matrix = xp.zeros((len(trialtype_idx), max_samples))
    
    trial_lengths = [len(trains[t]) 
                     if isinstance(trains[t], (list, np.ndarray)) 
                     else 0  # if there is nothing in this trial then trial_length=0
                     for t in trialtype_idx]
    
    for idx, trial in enumerate(trialtype_idx):
        if 0 < trial_lengths[idx] < max_samples:
            temp_matrix[idx, :trial_lengths[idx]] = xp.array(trains[trial][:])
        elif trial_lengths[idx] > 0:
            temp_matrix[idx, :] = xp.array(trains[trial][:max_samples])
    return temp_matrix

def get_trialtype_idx(stim_conds: list) -> tuple:
    """
    retrieves indices for baseline, stimulation, and control trials.
    
    parameters:
    - stim_conds (list): list of stimulation conditions for each trial.
    
    returns:
    - tuple:
        1. list: indices for baseline trials.
        2. list: indices for stimulation trials.
        3. list: indices for control trials.
    """
    stim_idx = [trial for trial, cond in enumerate(stim_conds)
                if cond!='0']
    
    if not stim_idx:
        return list(range(len(stim_conds))), [], []  # if no stim trials
    else:
        return (
            list(range(stim_idx[0])), 
            stim_idx, 
            [idx+2 for idx in stim_idx]  # stim_idx+2 are indices of control trials
            )
    
def get_waveform(cluname: str,
                 waveforms: dict) -> np.ndarray:
    return waveforms[cluname]

def plot_lick_sensitivity(cluname: str,
                          suffix: str,
                          identity: str,
                          lick_sensitive_signif: str,
                          aligned_rasters: xp.ndarray, 
                          aligned_prof_mean: xp.ndarray,
                          aligned_prof_sem: xp.ndarray,
                          shuf_ratios: xp.ndarray,
                          true_ratio: float,
                          samp_freq=1250,
                          GPU_AVAILABLE=False) -> None:
    """
    plot the spike rate and lick-aligned raster for a given cell.
    
    parameters:
    - cluname: identifier for the cell.
    - suffix: additional label for the plot title and filename.
    - identity: classification of the cell (e.g., 'tagged', 'putative').
    - lick_sensitive_signif: significance label for lick sensitivity.
    - aligned_rasters: spike times aligned to the first lick (trials x spike times).
    - aligned_prof_mean: mean spike rate profile aligned to licks.
    - aligned_prof_sem: standard error of the mean for the spike rate profile.
    - shuf_ratios: shuffled post/pre spike rate ratios.
    - true_ratio: actual post/pre spike rate ratio.
    - samp_freq: sampling frequency in Hz, default is 1250.
    - GPU_AVAILABLE: if true, transfers GPU arrays to CPU before plotting.
    
    returns:
    - none
    """
    if GPU_AVAILABLE:
        aligned_prof_mean = aligned_prof_mean.get()
        aligned_prof_sem = aligned_prof_sem.get()
        aligned_rasters = [arr.get() if isinstance(arr, cp.ndarray) else arr for arr in aligned_rasters]
        shuf_ratios = [ratio.get() for ratio in shuf_ratios]
        true_ratio = true_ratio.get()
    
    xaxis = np.arange(6 * samp_freq) / samp_freq - 3  # in seconds, since sampling freq is 1250 Hz 
    
    fig = plt.figure(figsize=(3.5, 1.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1])  # make axs[1] narrower
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    
    for i, trial in enumerate(aligned_rasters):
        axs[0].scatter(trial, [i+1]*len(trial),
                       color='grey', alpha=.25, s=.6)
        
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
    
    idstring = f'{cluname} {identity} {suffix}'
    for ext in ['.png', '.pdf']:
        fig.savefig(
            r'Z:\Dinghao\code_dinghao\LC_ephys\lick_sensitivity'
            rf'\rasters_first_lick_aligned\{idstring}{ext}',
            dpi=300,
            bbox_inches='tight'
            )


#%% main loop 
def main(fname=r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl',
         max_sample=1250*8,
         samp_freq=1250):
    
    ## dataframe initialisation/loading
    fname = fname
    sess = {
        'sessname': [],
        'identity': [],
        'spike_rate': [],
        'acg': [],
        'waveform': [],
        'run_onset_peak': [],
        'speed_rate_r': [],
        'speed_rate_p': [],
        'lick_sensitive': [],
        'lick_sensitive_type': [],
        'lick_sensitive_signif': [],
        'mean_profile': [],
        'sem_profile': [],
        'baseline_mean': [],
        'baseline_sem': [],
        'stim_mean': [],
        'stim_sem': [],
        'ctrl_mean': [],
        'ctrl_sem': []
        }
    df = pd.DataFrame(sess)
    
    # load behaviour file
    behaviour = pd.read_pickle(
        r'Z:/Dinghao/code_dinghao/behaviour/all_LC_sessions.pkl'
        )
    
    # load UMAP results
    kmeans = np.load(
        r'Z:\Dinghao\code_dinghao\LC_ephys\UMAP\LC_all_UMAP_kmeans.npy',
        allow_pickle=True
        ).item()  # this is for putative cell classification
    
    for path in paths:
        recname = path[-17:]
        print(f'\nprocessing {recname}...')
        
        # get trial types 
        beh = behaviour.loc[recname]
        stim_conds = [trial[15] for trial in beh['trial_statements']][1:]  # index 15 is the stim condition
        (
            baseline_idx,
            stim_idx,
            ctrl_idx
        ) = get_trialtype_idx(
            stim_conds
            )
        
        # load data 
        all_trains = np.load(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions'
            rf'\{recname}\{recname}_all_trains.npy',
            allow_pickle=True
            ).item()
        all_rasters = np.load(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions'
            rf'\{recname}\{recname}_all_rasters.npy',
            allow_pickle=True
            ).item()
        all_identities = np.load(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions'
            rf'\{recname}\{recname}_all_identities.npy',
            allow_pickle=True
            ).item()
        all_acgs = np.load(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions'
            rf'\{recname}\{recname}_all_ACGs.npy',
            allow_pickle=True
            ).item()
        all_waveforms = np.load(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions'
            rf'\{recname}\{recname}_all_waveforms.npy',
            allow_pickle=True
            ).item()
        spike_rate_file = sio.loadmat(
            rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:14]}'
            rf'\{recname}\{recname}'
            '_DataStructure_mazeSection1_TrialType1_FR_Ctrl_Run0_mazeSess1.mat'
            )
        beh_file = sio.loadmat(
            rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:14]}'
            rf'\{recname}\{recname}'
            '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
            )
        align_run_file = sio.loadmat(
            rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:14]}'
            rf'\{recname}\{recname}'
            '_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
            )
        speeds = compute_speeds(align_run_file)
        
        # get list of clunames 
        keys = [*all_trains]
        
        # get list of tagged cell cluname(s)
        tagged_keys = [clu for clu in all_identities
                       if all_identities[clu]==1]
        
        for cluname in keys:
            print(f'{cluname}')
            trains = all_trains[cluname]
            rasters = all_rasters[cluname]
            
            # get waveform and ACG
            waveform = get_waveform(cluname, all_waveforms)
            acg = get_acg(cluname, all_acgs)
            
            # get mean and sem spiking profiles 
            mean_profile = xp.mean(
                xp.array([train[:max_sample] for train in trains]),
                axis=0
                )
            sem_profile = sem(
                xp.array([train[:max_sample] for train in trains]),
                axis=0
                )
            baseline_mean = xp.mean(
                get_trial_matrix(trains, baseline_idx),
                axis=0
                )
            baseline_sem = sem(
                get_trial_matrix(trains, baseline_idx),
                axis=0
                )
            if GPU_AVAILABLE:
                mean_profile = mean_profile.get()
                sem_profile = sem_profile.get()
                baseline_mean = baseline_mean.get()
                baseline_sem = baseline_sem.get()
            
            if stim_idx:
                stim_mean = xp.mean(
                    get_trial_matrix(trains, stim_idx),
                    axis=0
                    )
                stim_sem = sem(
                    get_trial_matrix(trains, stim_idx),
                    axis=0
                    )
                ctrl_mean = xp.mean(
                    get_trial_matrix(trains, ctrl_idx),
                    axis=0
                    )
                ctrl_sem = sem(
                    get_trial_matrix(trains, ctrl_idx),
                    axis=0
                    )
                if GPU_AVAILABLE:
                    stim_mean = stim_mean.get()
                    stim_sem = stim_sem.get()
                    ctrl_mean = ctrl_mean.get()
                    ctrl_sem = ctrl_sem.get()
            else:
                stim_mean, ctrl_mean, stim_sem, ctrl_sem = [], [], [], []
        
            # spike rate 
            spike_rate = get_spike_rate(cluname, spike_rate_file)
        
            # cell identity ('tagged', 'putative' or 'other')
            identity = get_identity(cluname, tagged_keys, kmeans, spike_rate)
        
            # we don't need the single-trial parameters, but we do need the first stim
            # to identify the baseline for run-onset burst detection
            first_stim = get_first_stim_idx(cluname, beh_file)
        
            # run-onset burst detection 
            peak, mean_prof, shuf_prof = pdf.peak_detection(
                trains,
                first_stim=first_stim,
                around=4,  # check baseline on a higher threshold
                peak_width=2,  # 2 seconds to include some of the slightly offset peaks 
                bootstrap=5000,
                GPU_AVAILABLE=GPU_AVAILABLE)
            
            pdf.plot_peak_v_shuf(
                cluname, mean_prof, shuf_prof, peak,
                peak_width=2,
                savepath=r'Z:\Dinghao\code_dinghao\LC_ephys'
                         rf'\peak_detection\{cluname} {identity} {peak}'
                )  # plot the detected peaks and save to ...
            
            # rate-speed correlation
            (
                speed_rate_r, 
                speed_rate_p
            ) = compute_speed_rate_corr(cluname, 
                                        trains, 
                                        speeds)

            # lick alignment 
            (
                tot_trial_good_nonstim, 
                trial_list, 
                end_indices, 
                first_licks
            ) = compute_lick_info(
                beh_file, 
                align_run_file
                )
                
            (
                lick_sensitive, 
                lick_sensitivity_type, 
                lick_sensitivity_signif
            ) = compute_lick_sensitivity(
                cluname,
                trains, 
                rasters,
                identity,
                tot_trial_good_nonstim,
                trial_list,
                end_indices,
                first_licks,
                bootstrap=5000,
                GPU_AVAILABLE=GPU_AVAILABLE
                )
            
            # put into dataframe 
            df.loc[cluname] = np.array(
                [recname,
                 identity,
                 spike_rate,
                 acg,
                 waveform,
                 peak,
                 speed_rate_r,
                 speed_rate_p,
                 lick_sensitive,
                 lick_sensitivity_type,
                 lick_sensitivity_signif,
                 mean_profile,
                 sem_profile,
                 baseline_mean,
                 baseline_sem,
                 stim_mean,
                 stim_sem,
                 ctrl_mean,
                 ctrl_sem],
                dtype='object'
                )
            
    ## save dataframe 
    df.to_pickle(fname)
    print('\ndataframe saved')
    
if __name__ == '__main__':
    main()