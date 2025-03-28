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
                                        run_onset_inhibited_thres):
    """
    classify run-onset activation ratio based on pre- and post-run periods.

    parameters:
    - train: array of firing rates over time.
    - run_onset_activated_thres: threshold for classifying activation.
    - run_onset_inhibited_thres: threshold for classifying inhibition.
    - samp_freq: sampling frequency in Hz, default is 1250.
    - run_onset_bin: bin marking the run onset, default is 3750.

    returns:
    - ratio: pre/post activation ratio.
    - ratiotype: string indicating the activation class ('ON', 'OFF', 'unresponsive').
    """
    global samp_freq, run_onset_bin 
    
    pre = np.nanmean(train[int(run_onset_bin-samp_freq*1.5):int(run_onset_bin-samp_freq*.5)])
    post = np.nanmean(train[int(run_onset_bin+samp_freq*.5):int(run_onset_bin+samp_freq*1.5)])
    ratio = pre/post
    if ratio < run_onset_activated_thres:
        ratiotype = 'run-onset ON'
    elif ratio > run_onset_inhibited_thres:
        ratiotype = 'run-onset OFF'
    else:
        ratiotype = 'run-onset unresponsive'
        
    return ratio, ratiotype

def compute_modulation_index(ctrl, stim, 
                             span=4):
    """
    compute the modulation index (MI) between control and stimulation data.

    parameters:
    - ctrl: array of control data (e.g., firing rates).
    - stim: array of stimulation data (e.g., firing rates).
    - span: duration of the analysis window (in seconds), default is 4.

    returns:
    - MI_full: modulation index computed over the full span.
    - MI_early: modulation index computed for the early half of the span.
    - MI_late: modulation index computed for the late half of the span.
    """
    global samp_freq, run_onset_bin
    
    # first calculate MI over the full trial ({span} seconds)
    ctrl_full = np.nanmean(ctrl[run_onset_bin:run_onset_bin+samp_freq*span])
    stim_full = np.nanmean(stim[run_onset_bin:run_onset_bin+samp_freq*span])
    MI_full = (stim_full-ctrl_full) / (stim_full+ctrl_full)
    
    # demarcation 
    demarc = int(run_onset_bin+samp_freq*span/2)
    
    # next, early MI and late MI
    ctrl_early = np.nanmean(ctrl[run_onset_bin:demarc])
    stim_early = np.nanmean(stim[run_onset_bin:demarc])
    MI_early = (stim_early-ctrl_early) / (stim_early+ctrl_early)
    ctrl_late = np.nanmean(ctrl[demarc:run_onset_bin+samp_freq*span])
    stim_late = np.nanmean(stim[demarc:run_onset_bin+samp_freq*span])
    MI_late = (stim_late-ctrl_late) / (stim_late+ctrl_late)
    
    return MI_full, MI_early, MI_late

def compute_modulation_index_shuf(ctrl_matrix, stim_matrix,
                                  span=4,
                                  bootstrap=100):
    """
    compute the shuffled modulation index (MI) between control and stimulation data.

    parameters:
    - ctrl_matrix: matrix of control data, where rows represent trials and columns represent time points.
    - stim_matrix: matrix of stimulation data, where rows represent trials and columns represent time points.
    - span: duration of the analysis window (in seconds), default is 4.
    - bootstrap: number of bootstrap iterations for shuffling, default is 100.

    returns:
    - MI_full: modulation index computed over the full span for shuffled data.
    - MI_early: modulation index computed for the early half of the span for shuffled data.
    - MI_late: modulation index computed for the late half of the span for shuffled data.
    """
    global samp_freq, run_onset_bin
    
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
    ctrl_full = np.nanmean(shuf_ctrl_mean[run_onset_bin:run_onset_bin+samp_freq*span])
    stim_full = np.nanmean(shuf_stim_mean[run_onset_bin:run_onset_bin+samp_freq*span])
    MI_full = (stim_full-ctrl_full) / (stim_full+ctrl_full)
    
    # demarcation 
    demarc = int(run_onset_bin+samp_freq*span/2)
    
    # next, early MI and late MI
    ctrl_early = np.nanmean(shuf_ctrl_mean[run_onset_bin:demarc])
    stim_early = np.nanmean(shuf_stim_mean[run_onset_bin:demarc])
    MI_early = (stim_early-ctrl_early) / (stim_early+ctrl_early)
    ctrl_late = np.nanmean(shuf_ctrl_mean[demarc:run_onset_bin+samp_freq*span])
    stim_late = np.nanmean(shuf_stim_mean[demarc:run_onset_bin+samp_freq*span])
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
    get cell identities (pyramidal or interneuron) and their spike rates from the info file.

    parameters:
    - info_filename: path to the MATLAB info file.

    returns:
    - cell_identities: array indicating cell types (True for interneurons, False for pyramidal cells).
    - spike_rates: array of spike rates for all cells.
    """
    rec_info =  sio.loadmat(info_filename)['rec'][0][0]
    cell_identities = rec_info['isIntern'][0]
    spike_rates = rec_info['firingRate'][0]
    
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

def get_good_bad_idx_MATLAB(beh_series, pathname):
    """
    get indices of good and bad trials based on MATLAB pipeline.

    parameters:
    - beh_series: behaviour data series containing MATLAB trial quality information.

    returns:
    - good_idx_matlab: list of indices for good trials.
    - bad_idx_matlab: list of indices for bad trials.
    """
    beh_parameter_file = sio.loadmat(
        f'{pathname}{pathname[-18:]}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
        )
    
    # same as the previous function
    bad_idx_matlab = [trial-1 for trial, quality 
                      in enumerate(beh_parameter_file['behPar'][0]['indTrBadBeh'][0][0])
                      if quality and trial>0]
    good_idx_matlab = [trial-1 for trial, quality 
                       in enumerate(beh_parameter_file['behPar'][0]['indTrBadBeh'][0][0])
                       if not quality and trial>0]
    
    return good_idx_matlab, bad_idx_matlab

def get_relative_depth(pathname):
    depth_struct = sio.loadmat(pathname)['depthNeu'][0]
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
    return sio.loadmat(classification_filename)['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]-1

def get_trialtype_idx(beh_filename):
    """
    get indices for baseline, stimulation, and control trials.

    parameters:
    - beh_filename: path to the MATLAB behaviour file.

    returns:
    - baseline_idx: indices for baseline trials.
    - stim_idx: indices for stimulation trials.
    - ctrl_idx: indices for control trials.
    """
    behPar = sio.loadmat(beh_filename)
    stim_idx = np.where(behPar['behPar']['stimOn'][0][0][0]!=0)[0]
    
    if len(stim_idx)>0:
        return np.arange(1, stim_idx[0]), stim_idx, stim_idx+2  # stim_idx+2 are indices of control trials
    else:
        return np.arange(1, len(behPar['behPar']['stimOn'][0][0][0])), [], []  # if no stim trials

def load_beh_series(df_filename, recname):
    return pd.read_pickle(df_filename).loc[recname]

def load_speeds(beh_series):
    speed_times = beh_series['speed_times']
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
    dist_mat = mat73.loadmat(dist_filename)['filteredSpikeDistArrayRun']
    trains_dist = []  # use a list to mirror the structure of trains.npy
    for clu in range(len(dist_mat)):
        trains_dist.append(dist_mat[clu][1:])  # trial 1 is empty
    return trains_dist