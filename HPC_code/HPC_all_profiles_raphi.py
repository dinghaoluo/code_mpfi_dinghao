# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:34:28 2024
Modified on Tue 10 Dec 2024

genearte profiles for all pyramidal cells in hippocampus recordings, 
    segregated into baseline, ctrl and stim trials 

dependent on HPC_all_extract.py

@author: Dinghao Luo
"""


#%% to-do 
'''
add depth to the dataframe? 
    # depth 
    depth = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Depth.mat'.format(pathname, recname))['depthNeu'][0]
    rel_depth = depth['relDepthNeu'][0][0]
'''


#%% imports 
from pathlib import Path

from time import time
from datetime import timedelta 

import pickle
import pandas as pd 
import numpy as np 
from scipy.stats import sem
from tqdm import tqdm 

# suppress the warning prompts when ctrl and stim have no trials 
import warnings

import support_HPC as support 
from common import mpl_formatting
mpl_formatting()


#%% dataframe initialisation/loading
fpath = Path(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles_raphi.pkl')
if fpath.exists():
    df = pd.read_pickle(fpath)
    print(f'df loaded from {fpath}')
    processed_sess = df['recname'].tolist()
else:
    processed_sess = []
    sess = {
        'recname': [],  # Axxxr-202xxxxx-0x
        'cell_identity': [],  # str, 'pyr' or 'int'
        'depth': [],  # depth relative to layer centre
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
        }
    df = pd.DataFrame(sess)


#%% load paths to recordings 
import rec_list
paths = rec_list.pathHPC_Raphi
mazes = rec_list.pathHPC_Raphi_maze_sess


#%% parameters
global samp_freq, run_onset_bin

# behaviour 
track_length = 200  # in cm
bin_size = 0.1  # in cm
run_onset_bin = 3750  # in bins 

# ephys 
SAMP_FREQ = 1250  # in Hz
MAX_TIME = 10  # collect (for each trial) a maximum of 10 s of spiking-profile
MAX_SAMPLES = SAMP_FREQ * MAX_TIME

# pre_post ratio thresholds 
run_onset_activated_thres = 2/3
run_onset_inhibited_thres = 3/2


#%% parameters and path stems 
beh_stem = Path(r'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCRaphi')
train_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions_raphi')


#%% MAIN
for i, path in enumerate(paths):
    recname = Path(path).name
    
    if recname in processed_sess:
        print(f'{recname} already processed; skipped')
        continue
    else:
        print(recname)
    
    t0 = time()
    
    # load beh dataframe 
    beh_path = beh_stem / f'{recname}.pkl'
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)
        
    speeds = support.load_speeds(beh)
    good_idx, bad_idx = support.get_good_bad_idx(beh)
    
    # import bad beh trial indices from MATLAB pipeline 
    good_idx_matlab, bad_idx_matlab = support.get_good_bad_idx_MATLAB(path,
                                                                      sess=mazes[i])
    
    # calculate occupancy
    distance_bins = np.arange(0, track_length + bin_size, bin_size)
    occupancy = [
        support.calculate_occupancy(s, dt=.02, distance_bins=distance_bins) 
        for s in speeds
        ]
    
    # load spike trains as a list 
    train_path = train_stem / recname / f'{recname}_all_trains.npy'
    clu_list, trains = support.load_train(train_path)
    
    # load spike trains (in distance) as a list
    train_dist_path = Path(path) / f'{recname}_DataStructure_mazeSection1_TrialType1_convSpikesDistAligned_msess{mazes[i]}_Run0.mat'
    trains_dist = support.load_dist_spike_array(train_dist_path)
    
    # get pyr and int ID's and corresponding spike rates
    cell_info_path = Path(path) / f'{recname}_DataStructure_mazeSection1_TrialType1_Info.mat'
    cell_identities, spike_rates = support.get_cell_info(cell_info_path)
    
    # get place cell indices 
    try:
        place_cell_info_path = Path(path) / f'{recname}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run{mazes[i]}_Run0.mat'
        place_cell_idx = support.get_place_cell_idx(place_cell_info_path)
    except FileNotFoundError:  # for some of Raphi's recordings
        place_cell_info_path = Path(path) / f'{recname}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run{mazes[i]}_Run1.mat'
        place_cell_idx = support.get_place_cell_idx(place_cell_info_path)
    
    # get cell depth 
    depth_info_path = Path(path) / f'{recname}_DataStructure_mazeSection1_TrialType1_Depth.mat'
    depths = support.get_relative_depth(depth_info_path)
        
    # behaviour parameters
    beh_MATLAB_path = Path(path) / f'{recname}_DataStructure_mazeSection1_TrialType1_behPar_msess{mazes[i]}.mat'
    (
        baseline_idx,
        stim_idx, 
        ctrl_idx
    ) = support.get_trialtype_idx_MATLAB(beh_MATLAB_path)
    
    baseline_idx = baseline_idx[:-1]
    stim_idx = [t - 1 for t in stim_idx]  # since we skipped trial 1 when extracting trains 
    ctrl_idx = [t - 1 for t in ctrl_idx]

    # iterate over all pyramidal cells 
    for clu in tqdm(range(len(cell_identities)), desc='collecting profiles'):
        # pyr or int (or other if the spike rate is too high or too low)
        # modified 31 Mar 2025
        cell_identity = cell_identities[clu]
        if cell_identity == 'putative_pyr':
            if 0.15<spike_rates[clu]<7:
                cell_identity = 'pyr'
            else:
                cell_identity = 'other'
        
        # depth 
        depth = depths[clu]
        
        # spike-profile matrix
        baseline_matrix = support.get_trial_matrix(
            trains, baseline_idx, MAX_SAMPLES, clu)
        ctrl_matrix = support.get_trial_matrix(
            trains, ctrl_idx, MAX_SAMPLES, clu)
        stim_matrix = support.get_trial_matrix(
            trains, stim_idx, MAX_SAMPLES, clu)
        
        # mean profiles
        baseline_mean = np.nanmean(baseline_matrix, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            ctrl_mean = np.nanmean(ctrl_matrix, axis=0)
            stim_mean = np.nanmean(stim_matrix, axis=0)
        
        # sem profiles 
        baseline_sem = sem(baseline_matrix, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            ctrl_sem = sem(ctrl_matrix, axis=0)
            stim_sem = sem(stim_matrix, axis=0)
        
        # pre-post ratio calculation
        (
            baseline_run_onset_ratio, 
            baseline_run_onset_ratiotype
        ) = support.classify_run_onset_activation_ratio(
            baseline_mean, 
            run_onset_activated_thres, 
            run_onset_inhibited_thres
            )
        if ctrl_idx:
            (
                ctrl_run_onset_ratio, 
                ctrl_run_onset_ratiotype
            ) = support.classify_run_onset_activation_ratio(
                ctrl_mean, 
                run_onset_activated_thres, 
                run_onset_inhibited_thres
                )
            (
                stim_run_onset_ratio, 
                stim_run_onset_ratiotype
            ) = support.classify_run_onset_activation_ratio(
                stim_mean, 
                run_onset_activated_thres,
                run_onset_inhibited_thres
                )
        else:
            ctrl_run_onset_ratio = np.nan
            ctrl_run_onset_ratiotype = np.nan
            stim_run_onset_ratio = np.nan
            stim_run_onset_ratiotype = np.nan
        
        # trial by trial variatbility
        baseline_var = support.compute_trial_by_trial_variability(baseline_matrix)
        if ctrl_idx:
            ctrl_var = support.compute_trial_by_trial_variability(ctrl_matrix)
            stim_var = support.compute_trial_by_trial_variability(stim_matrix)
        else:
            ctrl_var = stim_var = np.nan
        
        # spatial information
        baseline_SI = [support.compute_spatial_information(
            trains_dist[clu][trial], occupancy[trial]) 
            for trial in baseline_idx]
        ctrl_SI = [support.compute_spatial_information(
            trains_dist[clu][trial], occupancy[trial]) 
            for trial in ctrl_idx]
        stim_SI = [support.compute_spatial_information(
            trains_dist[clu][trial], occupancy[trial]) 
            for trial in stim_idx]
        
        # temporal information 
        baseline_TI = [support.compute_temporal_information(
            trains[clu][trial][SAMP_FREQ*3:],
            bin_size_steps=1
            ) for trial in baseline_idx 
            if trains[clu][trial] is not None]
        ctrl_TI = [support.compute_temporal_information(
            trains[clu][trial][SAMP_FREQ*3:],
            bin_size_steps=1) for trial in ctrl_idx
            if trains[clu][trial] is not None]
        stim_TI = [support.compute_temporal_information(
            trains[clu][trial][SAMP_FREQ*3:],
            bin_size_steps=1) for trial in stim_idx
            if trains[clu][trial] is not None]
        
        cluname = clu_list[clu]
        df.loc[cluname] = np.array([recname,  # recname 
                                    cell_identity,  # 'pyr' or 'int'
                                    depth,  # int, relative to layer centre
                                    spike_rates[clu],  # spike_rate
                                    clu in place_cell_idx,  # place_cell
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
                                        ],
                                    dtype='object')
        
    print(f'{recname} done in {str(timedelta(seconds=int(time()-t0)))}\n')
        
        
#%% save dataframe 
df.to_pickle(fpath)
print('\ndataframe saved')