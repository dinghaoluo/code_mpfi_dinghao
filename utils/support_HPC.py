# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:14:04 2025

support functions for HPC scripts to reduce cluttering 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import scipy.io as sio 
import mat73
import pandas as pd 


#%% functions 
def calculate_occupancy(speeds, dt, distance_bins):
    """
    calculate occupancy in spatial bins based on speed and time bins.

    parameters:
    - speeds: array of speed values (in cm/s) at each time bin.
    - dt: time interval for each bin (in seconds).
    - distance_bins: array defining the edges of spatial bins (in cm).

    returns:
    - occupancy: array of time spent in each spatial bin (in seconds).
    """
    # convert to array first 
    speeds = np.asarray(speeds)
    
    # cumulative distance travelled at each time step
    cumulative_distance = np.cumsum(speeds * dt)

    # assign each cumulative distance to a spatial bin
    bin_indices = np.digitize(cumulative_distance, distance_bins) - 1

    # compute occupancy by summing time intervals (dt) for each bin
    occupancy = np.bincount(bin_indices, minlength=len(distance_bins)) * dt

    return occupancy

def classify_run_onset_activation_ratio(train, 
                                        run_onset_activated_thres,
                                        run_onset_inhibited_thres,
                                        run_onset_bin=3750,
                                        SAMP_FREQ=1250):
    """
    classify run-onset modulation by comparing pre- and post-run firing rates.

    parameters:
    - train: 1d array of firing rates or activity trace over time.
    - run_onset_activated_thres: threshold below which the ratio is considered 'ON'.
    - run_onset_inhibited_thres: threshold above which the ratio is considered 'OFF'.
    - run_onset_bin: timepoint index marking the run onset, default is 3750.
    - SAMP_FREQ: sampling frequency in Hz, default is 1250.

    returns:
    - ratio: float, ratio of pre- to post-onset activity.
    - ratiotype: string label of modulation category ('run-onset ON', 'run-onset OFF', or 'run-onset unresponsive').
    """    
    pre = np.nanmean(train[int(run_onset_bin-SAMP_FREQ*1.5):int(run_onset_bin-SAMP_FREQ*.5)])
    post = np.nanmean(train[int(run_onset_bin+SAMP_FREQ*.5):int(run_onset_bin+SAMP_FREQ*1.5)])
    ratio = pre/post
    if ratio < run_onset_activated_thres:
        ratiotype = 'run-onset ON'
    elif ratio > run_onset_inhibited_thres:
        ratiotype = 'run-onset OFF'
    else:
        ratiotype = 'run-onset unresponsive'
        
    return ratio, ratiotype

def compute_modulation_index(ctrl, 
                             stim, 
                             span=4,
                             run_onset_bin=3750,
                             SAMP_FREQ=1250):
    """
    compute the modulation index (MI) between control and stimulation activity over a specified time window.

    parameters:
    - ctrl: 1d array of control condition data (e.g., firing rates).
    - stim: 1d array of stimulation condition data.
    - span: duration of the analysis window in seconds after run onset (default = 4).
    - run_onset_bin: index marking run onset timepoint (default = 3750).
    - SAMP_FREQ: sampling frequency in Hz (default = 1250).

    returns:
    - MI_full: float, modulation index over the full window.
    - MI_early: float, modulation index over the first half of the window.
    - MI_late: float, modulation index over the second half of the window.
    """
    # first calculate MI over the full trial ({span} seconds)
    ctrl_full = np.nanmean(ctrl[run_onset_bin:run_onset_bin+SAMP_FREQ*span])
    stim_full = np.nanmean(stim[run_onset_bin:run_onset_bin+SAMP_FREQ*span])
    MI_full = (stim_full-ctrl_full) / (stim_full+ctrl_full)
    
    # demarcation 
    demarc = int(run_onset_bin+SAMP_FREQ*span/2)
    
    # next, early MI and late MI
    ctrl_early = np.nanmean(ctrl[run_onset_bin:demarc])
    stim_early = np.nanmean(stim[run_onset_bin:demarc])
    MI_early = (stim_early-ctrl_early) / (stim_early+ctrl_early)
    ctrl_late = np.nanmean(ctrl[demarc:run_onset_bin+SAMP_FREQ*span])
    stim_late = np.nanmean(stim[demarc:run_onset_bin+SAMP_FREQ*span])
    MI_late = (stim_late-ctrl_late) / (stim_late+ctrl_late)
    
    return MI_full, MI_early, MI_late

def compute_modulation_index_shuf(ctrl_matrix, 
                                  stim_matrix,
                                  span=4,
                                  bootstrap=100,
                                  run_onset_bin=3750,
                                  SAMP_FREQ=1250):
    """
    compute the modulation index (MI) after randomly shuffling trial identities between control and stimulation groups.

    parameters:
    - ctrl_matrix: 2d array (trials × timepoints) of control condition data.
    - stim_matrix: 2d array (trials × timepoints) of stimulation condition data.
    - span: duration of the analysis window in seconds after run onset (default = 4).
    - bootstrap: number of random shuffles to perform (default = 100).
    - run_onset_bin: index marking run onset timepoint (default = 3750).
    - SAMP_FREQ: sampling frequency in Hz (default = 1250).

    returns:
    - MI_full: float, modulation index over the full window (shuffled data).
    - MI_early: float, modulation index over the first half of the window.
    - MI_late: float, modulation index over the second half of the window.
    """
    ctrl_matrix = np.asarray(ctrl_matrix)
    stim_matrix = np.asarray(stim_matrix)
    tot_trials = ctrl_matrix.shape[0]
    
    pooled_matrix = np.vstack((ctrl_matrix, stim_matrix))
    pooled_idx = np.arange(ctrl_matrix.shape[0]+stim_matrix.shape[0])
    shuf_ctrl_idx = np.zeros((bootstrap, tot_trials), dtype=int)
    shuf_stim_idx = np.zeros((bootstrap, tot_trials), dtype=int)
    for i in range(bootstrap):  # shuffle n times
        shuf = np.random.permutation(pooled_idx)
        shuf_ctrl_idx[i, :] = shuf[:tot_trials]
        shuf_stim_idx[i, :] = shuf[tot_trials:]
    shuf_ctrl_mean = pooled_matrix[shuf_ctrl_idx, :].mean(axis=0).mean(axis=0)
    shuf_stim_mean = pooled_matrix[shuf_stim_idx, :].mean(axis=0).mean(axis=0)
    
    # first calculate MI over the full trial ({span} seconds)
    ctrl_full = np.nanmean(shuf_ctrl_mean[run_onset_bin:run_onset_bin+SAMP_FREQ*span])
    stim_full = np.nanmean(shuf_stim_mean[run_onset_bin:run_onset_bin+SAMP_FREQ*span])
    MI_full = (stim_full-ctrl_full) / (stim_full+ctrl_full)
    
    # demarcation 
    demarc = int(run_onset_bin+SAMP_FREQ*span/2)
    
    # next, early MI and late MI
    ctrl_early = np.nanmean(shuf_ctrl_mean[run_onset_bin:demarc])
    stim_early = np.nanmean(shuf_stim_mean[run_onset_bin:demarc])
    MI_early = (stim_early-ctrl_early) / (stim_early+ctrl_early)
    ctrl_late = np.nanmean(shuf_ctrl_mean[demarc:run_onset_bin+SAMP_FREQ*span])
    stim_late = np.nanmean(shuf_stim_mean[demarc:run_onset_bin+SAMP_FREQ*span])
    MI_late = (stim_late-ctrl_late) / (stim_late+ctrl_late)
    
    return MI_full, MI_early, MI_late  # this is shuffled 

def compute_spatial_information(spike_counts, occupancy,
                                GPU_AVAILABLE=False):
    """
    compute Skaggs spatial information for a single neuron.

    parameters:
    - spike_counts: array of spike counts per spatial bin.
    - occupancy: array of time spent in each spatial bin (in seconds).

    returns:
    - spatial_info: spatial information in bits per spike.
    """
    # ensure spike_counts and occupancy are np arrays
    if GPU_AVAILABLE:
        occupancy = occupancy.get()
    
    # handle cases with zero occupancy
    valid_bins = occupancy > 0
    
    spike_counts = spike_counts[valid_bins]
    occupancy = occupancy[valid_bins]
    
    # compute probability of occupancy (p(x))
    total_time = np.sum(occupancy)
    p_x = occupancy / total_time

    # compute firing rate per bin (λ(x))
    lambda_x = spike_counts / occupancy

    # compute overall mean firing rate (λ)
    lambda_bar = np.sum(lambda_x * p_x)

    # compute spatial information (Skaggs formula)
    with np.errstate(divide='ignore', invalid='ignore'):
        spatial_info = np.nansum(
            p_x * (lambda_x / lambda_bar) * np.log2(lambda_x / lambda_bar)
        )
    
    return spatial_info

def compute_temporal_information(spike_times, bin_size_steps):
    """
    compute temporal information for a single neuron sampled at 1250 Hz.

    parameters:
    - spike_times: array of spike times (in 1/1250 seconds steps).
    - bin_size_steps: size of each temporal bin (in steps of 1/1250 seconds).

    returns:
    - temporal_info: temporal information in bits per spike.
    """
    # ensure spike_times is a np array 
    spike_times = np.asarray(spike_times)
    
    # define temporal bins
    total_steps = len(spike_times)
    num_bins = total_steps // bin_size_steps
    bin_edges = np.arange(0, total_steps + bin_size_steps, bin_size_steps)

    # compute spike counts per bin
    spike_counts, _ = np.histogram(spike_times, bins=bin_edges)

    # probability of occupancy per bin (uniform for equal bin sizes)
    p_t = np.ones(num_bins) / num_bins

    # compute firing rate per bin
    lambda_t = spike_counts / (bin_size_steps / 1250)  # convert bin size to seconds

    # overall mean firing rate
    lambda_bar = np.sum(lambda_t * p_t)

    # compute temporal information
    with np.errstate(divide='ignore', invalid='ignore'):
        temporal_info = np.nansum(
            p_t * (lambda_t / lambda_bar) * np.log2(lambda_t / lambda_bar)
        )
    
    return temporal_info

def compute_trial_by_trial_variability(train):
    """
    compute trial-by-trial variability for a neuron's spike trains.

    parameters:
    - train: list of numpy arrays, each representing the firing vector of a trial.

    returns:
    - variability_median: variability as 1 - median of pairwise correlations.
    """
    if len(train)==0: 
        return np.nan
    
    # threshold each trial by the maximum length of a trial 
    max_length = max([len(v) for v in train])
    train = [v[:max_length] for v in train]
    
    # compute correlation matrix 
    num_trials = len(train)
    corr_matrix = np.full((num_trials, num_trials), np.nan)  # initialise
    
    for i in range(num_trials):
        for j in range(i + 1, num_trials):  # only compute upper triangular
            if np.nanstd(train[i]) == 0 or np.nanstd(train[j]) == 0:
                corr_matrix[i, j] = np.nan
            else:
                corr_matrix[i, j] = np.corrcoef(train[i], train[j])[0, 1]

    # extract upper triangular correlations
    corr_values = corr_matrix[np.triu_indices(num_trials, k=1)]

    # compute variability metrics
    variability_median = 1 - np.nanmedian(corr_values)  # median-based variability

    return variability_median

def get_cell_info(info_filename):
    """
    load cell type identities and spike rates from a MATLAB info file.

    parameters:
    - info_filename: path to the MATLAB .mat file containing cell information.

    returns:
    - cell_identities: list of strings ('pyr' or 'int') indicating cell types.
    - spike_rates: 1d array of spike rates for all cells.
    """
    info = sio.loadmat(str(info_filename))
    # rec_info = info['rec'][0][0]
    autocorr = info['autoCorr'][0][0]
    
    # spike_rates = rec_info['firingRate'][0]
    
    # use the pipeline-calculated FR 
    filestem = str(info_filename).split('_Info')[0]
    spike_rates = sio.loadmat(
        rf'{filestem}_FR_Run1.mat'
        )['mFRStruct']['mFR'][0][0].flatten()
    
    is_pyr = autocorr['isPyrneuron'][0]
    cell_identities = ['putative_pyr' if i else 'int' for i in is_pyr]
    
    return cell_identities, spike_rates

def get_good_bad_idx(beh_series):
    """
    get indices of good and bad trials based on behaviour quality.

    parameters:
    - beh_series: behaviour data series containing trial quality information.

    returns:
    - good_idx: list of indices for good trials.
    - bad_idx: list of indices for bad trials.
    """
    bad_trial_map = beh_series['bad_trials']
    
    # trial 1 is empty and not included in the spike train
    good_idx = [trial-1 for trial, quality in enumerate(bad_trial_map) if not quality and trial>0]
    bad_idx = [trial-1 for trial, quality in enumerate(bad_trial_map) if quality and trial>0]
    
    return good_idx, bad_idx

def get_good_bad_idx_MATLAB(pathname, sess=1):
    """
    extract indices of good and bad trials from a MATLAB behavioural parameter file.

    parameters:
    - pathname: full path to the session folder containing the MATLAB behaviour file.

    returns:
    - good_idx_matlab: list of indices (0-based) for trials marked as good.
    - bad_idx_matlab: list of indices (0-based) for trials marked as bad.
    """
    recname = pathname.split('\\')[-1]
    beh_parameter_file = sio.loadmat(
       rf'{pathname}\{recname}_DataStructure_mazeSection1_TrialType1_behPar_msess{sess}.mat'
       )
    
    # same as the previous function
    bad_idx_matlab = [trial-1 for trial, quality 
                      in enumerate(beh_parameter_file['behPar'][0]['indTrBadBeh'][0][0])
                      if quality and trial>0]
    good_idx_matlab = [trial-1 for trial, quality 
                       in enumerate(beh_parameter_file['behPar'][0]['indTrBadBeh'][0][0])
                       if not quality and trial>0]
    
    return good_idx_matlab, bad_idx_matlab

def get_place_cell_idx(classification_filename):
    """
    get indices of place cells identified by the MATLAB pipeline.

    parameters:
    - classification_filename: path to the MATLAB classification file.

    returns:
    - place_cell_idx: array of indices for place cells.
    """
    # get indices of place cells identifies by the MATLAB pipeline
    # -1 because the indices were 1-indexed, whereas get_pyr_info() uses 0-indexing
    return sio.loadmat(str(classification_filename))['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]-1

def get_relative_depth(pathname):
    depth_struct = sio.loadmat(str(pathname))['depthNeu'][0]
    return depth_struct['relDepthNeu'][0][0]  # a list 

def get_trial_matrix(trains, trialtype_idx, max_samples, clu):
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
        return np.nan
    
    temp_matrix = np.zeros((len(trialtype_idx), max_samples))
    for idx, trial in enumerate(trialtype_idx):
        try:
            trial_length = len(trains[clu][trial])
        except TypeError:
            trial_length = 0
        if 0 < trial_length < max_samples:
            temp_matrix[idx, :trial_length] = np.asarray(trains[clu][trial][:])
        elif trial_length > 0:
            temp_matrix[idx, :] = np.asarray(trains[clu][trial][:max_samples])
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

def get_trialtype_idx_MATLAB(beh_filename):
    """
    get indices for baseline, stimulation, and control trials.

    parameters:
    - beh_filename: path to the MATLAB behaviour file.

    returns:
    - baseline_idx: indices for baseline trials.
    - stim_idx: indices for stimulation trials.
    - ctrl_idx: indices for control trials.
    """
    behPar = sio.loadmat(str(beh_filename))
    max_length = len(behPar['behPar']['stimOn'][0][0][0]) - 1
    
    stim_idx = np.where(behPar['behPar']['stimOn'][0][0][0]!=0)[0]
    
    try:
        baseline_idx = np.arange(1, stim_idx[0])
        ctrl_idx = stim_idx + 2
        
        ctrl_mask = ctrl_idx < max_length
        ctrl_idx = ctrl_idx[ctrl_mask]
    
        return baseline_idx, stim_idx, ctrl_idx  # stim_idx+2 are indices of control trials

    except IndexError:
        return np.arange(1, len(behPar['behPar']['stimOn'][0][0][0])), [], []  # if no stim trials

def load_beh_series(df_filename, recname):
    return pd.read_pickle(df_filename).loc[recname]

def load_speeds(beh_series):
    speed_times = beh_series['speed_times_aligned']
    new_speed_times = []
    for trial in range(1, len(speed_times)):  # trial 1 is empty and not included in spike trains
        curr_speed_times = speed_times[trial]
        curr_aligned = [s[1] for s in curr_speed_times]
        new_speed_times.append(curr_aligned)
    return new_speed_times
    
def load_train(npy_filename):
    npy_file = np.load(npy_filename, allow_pickle=True).item()
    return list(npy_file.keys()), list(npy_file.values())

def load_dist_spike_array(dist_filename):
    dist_mat = mat73.loadmat(str(dist_filename))['filteredSpikeDistArrayRun']
    trains_dist = []  # use a list to mirror the structure of trains.npy
    for clu in range(len(dist_mat)):
        trains_dist.append(dist_mat[clu][1:])  # trial 1 is empty
    return trains_dist