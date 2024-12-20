# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:34:28 2024
Modified on Tue 10 Dec 2024

genearte profiles for all pyramidal cells in hippocampus recordings, 
    segregated into baseline, ctrl and stim trials 

@author: Dinghao Luo
"""


#%% imports 
import mat73
import scipy.io as sio
from tqdm import tqdm 
from time import time 
import sys 
import os 
import pandas as pd

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% dataframe initialisation/loading
fname = r'Z:\Dinghao\code_dinghao\HPC_all\HPC_all_pyr_profiles.pkl'
fname = ''
if os.path.exists(fname):
    df = pd.read_pickle(fname)
    print(f'df loaded from {fname}')
    processed_sess = df.index.tolist()
else:
    processed_sess = []
    sess = {
        'rectype': [],  # HPCLC, HPCLCterm
        'recname': [],  # Axxxr-202xxxxx-0x
        'spike_rate': [],  # in Hz
        'place_cell': [],  # booleon
        'pre_post': [],  # post/pre ([.5:1.5]/[-1.5:-.5])
        'pre_post_stim': [],  # in stim trials 
        'pre_post_ctrl': [],  # in stim-ctrl trials 
        'class': [],  # run-onset activated/inhibited/unresponsive
        'class_stim': [],
        'class_ctrl': [],
        'var': [],  # trial-by-trial variability in firing
        'var_stim': [],
        'var_ctrl': [],
        'SI': [],  # spatial information
        'SI_stim': [],
        'SI_ctrl': [],
        'TI': [],  # temporal information 
        'TI_stim': [],
        'TI_ctrl': [],
        'prof_mean': [],  # mean firing profile
        'prof_sem': [],
        'prof_stim_mean': [],
        'prof_stim_sem': [],
        'prof_ctrl_mean': [],
        'prof_ctrl_sem': [],
        'prof_good_mean': [],
        'prof_good_sem': [],
        'prof_bad_mean': [],
        'prof_bad_sem': [],
        'prof_good_mean_matlab': [],
        'prof_good_sem_matlab': [],
        'prof_bad_mean_matlab': [],
        'prof_bad_sem_matlab': [],
        'MI': [],  # modulation index
        'MI_early': [],
        'MI_late': [],
        'MI_shuf': [],
        'MI_shuf_early': [],
        'MI_shuf_late': []
        }
    df = pd.DataFrame(sess)



#%% load paths to recordings 
sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathHPCLCopt + rec_list.pathHPCLCtermopt


#%% parameters
global samp_freq, run_onset_bin

# behaviour 
track_length = 200  # in cm
bin_size = 0.1  # in cm
run_onset_bin = 3750  # in bins 

# ephys 
samp_freq = 1250  # in Hz
max_time = 10  # collect (for each trial) a maximum of 10 s of spiking-profile
max_samples = samp_freq * max_time

# pre_post ratio thresholds 
run_onset_activated_thres = 0.80
run_onset_inhibited_thres = 1.25


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
    import numpy as np
    from common import sem_gpu as sem
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print(f'GPU-acceleration with {str(name)} and cupy')
else:
    import numpy as np 
    from scipy.stats import sem 
    xp = np
    print('GPU-acceleartion unavailable')
    
'''
So far it seems that there is nothing to parallelise in this script for signi-
ficant improvement in performance, so let's just stick to CPU computation
'''
print('not using GPU acceleration due to inefficiency')
xp = np  # negate CuPy assignment
GPU_AVAILABLE = False
from scipy.stats import sem


#%% functions 
def calculate_occupancy(speeds, dt, distance_bins):
    """
    Calculate occupancy in spatial bins based on speed and time bins.

    Parameters:
    - speed: array of speed values (in cm/s) at each time bin.
    - dt: time interval for each bin (in seconds).
    - distance_bins: array defining the edges of spatial bins (in cm).

    Returns:
    - occupancy: array of time spent in each spatial bin (in seconds).
    """
    # convert to array first 
    speeds = xp.asarray(speeds)
    
    # cumulative distance travelled at each time step
    cumulative_distance = xp.cumsum(speeds * dt)

    # assign each cumulative distance to a spatial bin
    bin_indices = xp.digitize(cumulative_distance, distance_bins) - 1

    # compute occupancy by summing time intervals (dt) for each bin
    occupancy = xp.bincount(bin_indices, minlength=len(distance_bins)) * dt

    return occupancy

def classify_run_onset_activation_ratio(train, 
                                        run_onset_activated_thres,
                                        run_onset_inhibited_thres, 
                                        samp_freq=1250, run_onset_bin=3750):
    pre = xp.nanmean(train[int(run_onset_bin-samp_freq*1.5):int(run_onset_bin-samp_freq*.5)])
    post = xp.nanmean(train[int(run_onset_bin+samp_freq*.5):int(run_onset_bin+samp_freq*1.5)])
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
    global samp_freq, run_onset_bin
    
    # first calculate MI over the full trial ({span} seconds)
    ctrl_full = xp.nanmean(ctrl[run_onset_bin:run_onset_bin+samp_freq*span])
    stim_full = xp.nanmean(stim[run_onset_bin:run_onset_bin+samp_freq*span])
    MI_full = (stim_full-ctrl_full) / (stim_full+ctrl_full)
    
    # demarcation 
    demarc = int(run_onset_bin+samp_freq*span/2)
    
    # next, early MI and late MI
    ctrl_early = xp.nanmean(ctrl[run_onset_bin:demarc])
    stim_early = xp.nanmean(stim[run_onset_bin:demarc])
    MI_early = (stim_early-ctrl_early) / (stim_early+ctrl_early)
    ctrl_late = xp.nanmean(ctrl[demarc:run_onset_bin+samp_freq*span])
    stim_late = xp.nanmean(stim[demarc:run_onset_bin+samp_freq*span])
    MI_late = (stim_late-ctrl_late) / (stim_late+ctrl_late)
    
    return MI_full, MI_early, MI_late

def compute_modulation_index_shuf(ctrl_matrix, stim_matrix,
                                  span=4,
                                  bootstrap=100):
    global samp_freq, run_onset_bin
    pooled_matrix = xp.vstack((ctrl_matrix, stim_matrix))
    pooled_idx = xp.arange(ctrl_matrix.shape[0]+stim_matrix.shape[0])
    shuf_ctrl_idx = xp.zeros((bootstrap, len(ctrl_idx)), dtype=int)
    shuf_stim_idx = xp.zeros((bootstrap, len(ctrl_idx)), dtype=int)
    for i in range(bootstrap):  # shuffle n times
        shuf = xp.random.permutation(pooled_idx)
        shuf_ctrl_idx[i, :] = shuf[:len(ctrl_idx)]
        shuf_stim_idx[i, :] = shuf[len(ctrl_idx):]
    shuf_ctrl_mean = pooled_matrix[shuf_ctrl_idx, :].mean(axis=0).mean(axis=0)
    shuf_stim_mean = pooled_matrix[shuf_stim_idx, :].mean(axis=0).mean(axis=0)
    
    # first calculate MI over the full trial ({span} seconds)
    ctrl_full = xp.nanmean(shuf_ctrl_mean[run_onset_bin:run_onset_bin+samp_freq*span])
    stim_full = xp.nanmean(shuf_stim_mean[run_onset_bin:run_onset_bin+samp_freq*span])
    MI_full = (stim_full-ctrl_full) / (stim_full+ctrl_full)
    
    # demarcation 
    demarc = int(run_onset_bin+samp_freq*span/2)
    
    # next, early MI and late MI
    ctrl_early = xp.nanmean(shuf_ctrl_mean[run_onset_bin:demarc])
    stim_early = xp.nanmean(shuf_stim_mean[run_onset_bin:demarc])
    MI_early = (stim_early-ctrl_early) / (stim_early+ctrl_early)
    ctrl_late = xp.nanmean(shuf_ctrl_mean[demarc:run_onset_bin+samp_freq*span])
    stim_late = xp.nanmean(shuf_stim_mean[demarc:run_onset_bin+samp_freq*span])
    MI_late = (stim_late-ctrl_late) / (stim_late+ctrl_late)
    
    return MI_full, MI_early, MI_late  # this is shuffled 

def compute_spatial_information(spike_counts, occupancy):
    """
    Compute Skaggs spatial information for a single neuron.
    
    Parameters:
    - spike_counts: array of spike counts per spatial bin.
    - occupancy: array of time spent in each spatial bin (in seconds).

    Returns:
    - spatial_info: Spatial information in bits per spike.
    """
    # ensure spike_counts and occupancy are xp arrays
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
    Compute temporal information for a single neuron sampled at 1250 Hz.

    Parameters:
    - spike_times: array of spike times (in 1/1250 seconds steps).
    - bin_size_steps: size of each temporal bin (in steps of 1/1250 seconds).

    Returns:
    - temporal_info: Temporal information in bits per spike.
    """
    # ensure spike_times is a xp array 
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
    Compute trial-by-trial variability for a neuron's spike trains.
    
    Parameters:
    - train: list of numpy arrays, each representing the firing vector of a trial.

    Returns:
    - variability_median: Variability as 1 - median of pairwise correlations.
    """
    if len(train)==0: 
        return np.nan
    
    # threshold each trial by the maximum length of a trial 
    max_length = max([len(v) for v in train])
    train = [v[:max_length] for v in train]
    
    # compute correlation matrix 
    num_trials = len(train)
    corr_matrix = xp.full((num_trials, num_trials), xp.nan)  # initialise
    
    for i in range(num_trials):
        for j in range(i + 1, num_trials):  # only compute upper triangular
            if xp.nanstd(train[i]) == 0 or xp.nanstd(train[j]) == 0:
                corr_matrix[i, j] = np.nan
            else:
                corr_matrix[i, j] = xp.corrcoef(train[i], train[j])[0, 1]

    # extract upper triangular correlations
    corr_values = corr_matrix[xp.triu_indices(num_trials, k=1)]

    # compute variability metrics
    variability_median = 1 - xp.nanmedian(corr_values)  # median-based variability

    return variability_median

def get_good_bad_idx(beh_series):
    bad_trial_map = beh_series['bad_trials']
    good_idx = [trial for trial, quality in enumerate(bad_trial_map) if not quality]
    bad_idx = [trial for trial, quality in enumerate(bad_trial_map) if quality]
    
    return good_idx, bad_idx

def get_good_bad_idx_MATLAB(beh_series):
    beh_parameter_file = sio.loadmat(
        f'{pathname}{pathname[-18:]}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
        )    
    bad_idx_matlab = [trial for trial, quality 
                      in enumerate(beh_parameter_file['behPar'][0]['indTrBadBeh'][0][0])
                      if quality]
    good_idx_matlab = [trial for trial, quality 
                       in enumerate(beh_parameter_file['behPar'][0]['indTrBadBeh'][0][0])
                       if not quality]
    
    return good_idx_matlab, bad_idx_matlab

def get_trial_matrix(trains, trialtype_idx, max_samples, clu):
    if len(trialtype_idx)==0:  # if there is no trial in the list 
        return np.nan
    
    temp_matrix = xp.zeros((len(trialtype_idx), max_samples))
    for idx, trial in enumerate(trialtype_idx):
        try:
            trial_length = len(trains[clu][trial])
        except TypeError:
            trial_length = 0
        if 0 < trial_length < max_samples:
            temp_matrix[idx, :trial_length] = xp.asarray(trains[clu][trial][:])*samp_freq  # *samp_freq to convert to Hz
        elif trial_length > 0:
            temp_matrix[idx, :] = xp.asarray(trains[clu][trial][:max_samples])*samp_freq
    return temp_matrix

def get_place_cell_idx(classification_filename):
    # get indices of place cells identifies by the MATLAB pipeline
    # -1 because the indices were 1-indexed, whereas get_pyr_info() uses 0-indexing
    return sio.loadmat(classification_filename)['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]-1

def get_pyr_info(info_filename):
    rec_info =  sio.loadmat(info_filename)['rec'][0][0]
    pyr_idx = [i for i, clu in enumerate(rec_info['isIntern'][0]) if clu==False]
    spike_rate = [rec_info['firingRate'][0][i] for i in pyr_idx]
    
    return pyr_idx, spike_rate

def get_trialtype_idx(beh_filename):
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
    for trial in range(len(speed_times)):
        curr_speed_times = speed_times[trial]
        curr_aligned = [s[1] for s in curr_speed_times]
        new_speed_times.append(curr_aligned)
    return new_speed_times
    
def load_train(npy_filename):
    npy_file = np.load(npy_filename, allow_pickle=True).item()
    return list(npy_file.keys()), list(npy_file.values())

def load_dist_spike_array(dist_filename):
    return mat73.loadmat(dist_filename)['filteredSpikeDistArrayRun']


#%% MAIN
for pathname in paths:
    recname = pathname[-17:]
    print(recname)
    
    t0 = time()
    
    if pathname in rec_list.pathHPCLCopt:
        prefix = 'HPCLC'
    elif pathname in rec_list.pathHPCLCtermopt:
        prefix = 'HPCLCterm'
    
    # load beh dataframe 
    beh_df = load_beh_series(
        r'Z:\Dinghao\code_dinghao\behaviour\all_{}_sessions.pkl'
        .format(prefix), recname
        )
    speeds = load_speeds(beh_df)
    good_idx, bad_idx = get_good_bad_idx(beh_df)
    
    # import bad beh trial indices from MATLAB pipeline 
    good_idx_matlab, bad_idx_matlab = get_good_bad_idx_MATLAB(pathname)
    
    # calculate occupancy
    distance_bins = xp.arange(0, track_length + bin_size, bin_size)
    occupancy = [calculate_occupancy(s, dt=.02, distance_bins=distance_bins) for 
                 s in speeds]
    
    # load spike trains as a list
    clu_list, trains = load_train(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{}\HPC_all_info_{}.npy'.
        format(recname, recname)
        )
    
    # load spike trains (in distance) as a list
    trains_dist = load_dist_spike_array(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_convSpikesDistAligned_msess1_Run0.mat'.
        format(pathname, recname))
    
    # get pyr_idx and corresponding spike rates
    pyr_idx, pyr_spike_rate = get_pyr_info(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.
        format(pathname, recname))
    
    # get place cell indices 
    place_cell_idx = get_place_cell_idx(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.
        format(pathname, recname))
    
    # behaviour parameters
    baseline_idx, stim_idx, ctrl_idx = get_trialtype_idx(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.
        format(pathname, recname)
        )

    # iterate over all pyramidal cells 
    print('collecting profiles')
    for i, pyr in tqdm(enumerate(pyr_idx)):
        # spike-profile matrix
        baseline_matrix = get_trial_matrix(trains, baseline_idx, max_samples, pyr)
        ctrl_matrix = get_trial_matrix(trains, ctrl_idx, max_samples, pyr)
        stim_matrix = get_trial_matrix(trains, stim_idx, max_samples, pyr)
        
        # mean profiles
        baseline_mean = xp.nanmean(baseline_matrix, axis=0)
        ctrl_mean = xp.nanmean(ctrl_matrix, axis=0)
        stim_mean = xp.nanmean(stim_matrix, axis=0)
        
        # sem profiles 
        baseline_sem = sem(baseline_matrix, axis=0)
        ctrl_sem = sem(ctrl_matrix, axis=0)
        stim_sem = sem(stim_matrix, axis=0)
        
        # pre-post ratio calculation
        baseline_run_onset_ratio, baseline_run_onset_ratiotype = classify_run_onset_activation_ratio(
            baseline_mean, run_onset_activated_thres, run_onset_inhibited_thres)
        ctrl_run_onset_ratio, ctrl_run_onset_ratiotype = classify_run_onset_activation_ratio(
            ctrl_mean, run_onset_activated_thres, run_onset_inhibited_thres)
        stim_run_onset_ratio, stim_run_onset_ratiotype = classify_run_onset_activation_ratio(
            stim_mean, run_onset_activated_thres, run_onset_inhibited_thres)
        
        # modulation index calculation
        MI, MI_early, MI_late = compute_modulation_index(ctrl_mean, stim_mean)
        MI_shuf, MI_early_shuf, MI_late_shuf = compute_modulation_index_shuf(ctrl_matrix, stim_matrix)
        
        # trial by trial variatbility
        baseline_var = compute_trial_by_trial_variability(baseline_matrix)
        ctrl_var = compute_trial_by_trial_variability(ctrl_matrix)
        stim_var = compute_trial_by_trial_variability(stim_matrix)
        
        # spatial information
        baseline_SI = [compute_spatial_information(trains_dist[pyr][trial], occupancy[trial]) for 
                       trial in baseline_idx]
        ctrl_SI = [compute_spatial_information(trains_dist[pyr][trial], occupancy[trial]) for 
                   trial in ctrl_idx]
        stim_SI = [compute_spatial_information(trains_dist[pyr][trial], occupancy[trial]) for 
                   trial in stim_idx]
        
        # temporal information 
        baseline_TI = [compute_temporal_information(trains[pyr][trial][samp_freq*3:],
                                                    bin_size_steps=1) for trial in baseline_idx 
                       if trains[pyr][trial] is not None]
        ctrl_TI = [compute_temporal_information(trains[pyr][trial][samp_freq*3:],
                                                bin_size_steps=1) for trial in ctrl_idx
                   if trains[pyr][trial] is not None]
        stim_TI = [compute_temporal_information(trains[pyr][trial][samp_freq*3:],
                                                bin_size_steps=1) for trial in stim_idx
                   if trains[pyr][trial] is not None]
        
        # good/bad trial mean profiles 
        good_matrix = get_trial_matrix(trains, good_idx, max_samples, pyr)
        good_mean = xp.nanmean(good_matrix, axis=0) if good_idx else xp.array([])  # in case there is no bad trials
        good_sem = sem(good_matrix, axis=0) if good_idx else xp.array([])
        bad_matrix = get_trial_matrix(trains, bad_idx, max_samples, pyr)
        bad_mean = xp.nanmean(bad_matrix, axis=0) if bad_idx else xp.array([])
        bad_sem = sem(bad_matrix, axis=0) if bad_idx else xp.array([])
        
        # good/bad trial mean profiles (MATLAB)
        good_matrix_matlab = get_trial_matrix(trains, good_idx_matlab, max_samples, pyr)
        good_mean_matlab = xp.nanmean(good_matrix_matlab, axis=0) if good_idx_matlab else xp.array([])
        good_sem_matlab = sem(good_matrix_matlab, axis=0) if good_idx_matlab else xp.array([])
        bad_matrix_matlab = get_trial_matrix(trains, bad_idx_matlab, max_samples, pyr)
        bad_mean_matlab = xp.nanmean(bad_matrix_matlab, axis=0) if bad_idx_matlab else xp.array([])
        bad_sem_matlab = sem(bad_matrix_matlab, axis=0) if bad_idx_matlab else xp.array([])

        # transfer stuff from VRAM back to RAM
        if GPU_AVAILABLE:
            baseline_run_onset_ratio = baseline_run_onset_ratio.get()
            stim_run_onset_ratio = stim_run_onset_ratio.get()
            ctrl_run_onset_ratio = ctrl_run_onset_ratio.get()
            baseline_var = baseline_var.get()
            stim_var = stim_var.get()
            ctrl_var = ctrl_var.get()
            baseline_mean = baseline_mean.get()
            baseline_sem = baseline_sem.get()
            stim_mean = stim_mean.get()
            stim_sem = stim_sem.get()
            ctrl_mean = ctrl_mean.get()
            ctrl_sem = ctrl_sem.get()
            good_mean = good_mean.get()
            good_sem = good_sem.get()
            bad_mean = bad_mean.get()
            bad_sem = bad_sem.get()
            good_mean_matlab = good_mean_matlab.get()
            good_sem_matlab = good_sem_matlab.get()
            bad_mean_matlab = bad_mean_matlab.get()
            bad_sem_matlab = bad_sem_matlab.get()
        
        cluname = clu_list[pyr]
        df.loc[cluname] = np.array([prefix,  # rectype
                                    recname,  # recname 
                                    pyr_spike_rate[i],  # spike_rate
                                    pyr in place_cell_idx,  # place_cell
                                    baseline_run_onset_ratio,  # pre_post
                                    stim_run_onset_ratio,  # pre_post_stim
                                    ctrl_run_onset_ratio,  # pre_post_ctrl
                                    baseline_run_onset_ratiotype,  # class
                                    stim_run_onset_ratiotype,  # class_stim
                                    ctrl_run_onset_ratiotype,  # class_ctrl 
                                    baseline_var,  # var
                                    stim_var,  # var_stim
                                    ctrl_var,  # var_ctrl
                                    baseline_SI,  # SI
                                    stim_SI,  # SI_stim
                                    ctrl_SI,  # SI_ctrl
                                    baseline_TI,  # TI
                                    stim_TI,  # TI_stim
                                    ctrl_TI,  # TI_ctrl
                                    baseline_mean,  # prof_mean
                                    baseline_sem,  # prof_sem
                                    stim_mean,  # prof_stim_mean
                                    stim_sem,  # prof_stim_sem
                                    ctrl_mean,  # prof_ctrl_mean
                                    ctrl_sem,  # prof_ctrl_sem
                                    good_mean,  # prof_good_mean
                                    good_sem,  # prof_good_sem
                                    bad_mean,  # prof_bad_mean
                                    bad_sem,  # prof_bad_sem
                                    good_mean_matlab,
                                    good_sem_matlab,
                                    bad_mean_matlab,
                                    bad_sem_matlab,
                                    MI,  # ctrl v stim
                                    MI_early,
                                    MI_late,
                                    MI_shuf,
                                    MI_early_shuf,
                                    MI_late_shuf
                                        ],
                                    dtype='object')
        
    print(f'{recname} done in {time()-t0} s\n')
        
        
#%% save dataframe 
df.to_pickle(fname)
print('\ndataframe saved')