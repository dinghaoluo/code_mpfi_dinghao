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
from tqdm import tqdm 
from time import time
from datetime import timedelta 
import sys 
import pandas as pd

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\HPC_code\utils')
import support_HPC as support 


#%% dataframe initialisation/loading
fname = r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
# if os.path.exists(fname):
if False:
    df = pd.read_pickle(fname)
    print(f'df loaded from {fname}')
    processed_sess = df.index.tolist()
else:
    processed_sess = []
    sess = {
        'rectype': [],  # HPCLC, HPCLCterm
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
        'pre_post_MATLAB': [],
        'pre_post_stim_MATLAB': [],
        'pre_post_ctrl_MATLAB': [],
        'class_MATLAB': [],
        'class_stim_MATLAB': [],
        'class_ctrl_MATLAB': [],
        'var': [],  # trial-by-trial variability in firing
        'var_stim': [],
        'var_ctrl': [],
        'SI': [],  # spatial information
        'SI_stim': [],
        'SI_ctrl': [],
        'TI': [],  # temporal information 
        'TI_stim': [],
        'TI_ctrl': [],
        'var_MATLAB': [],  # trial-by-trial variability in firing
        'var_stim_MATLAB': [],
        'var_ctrl_MATLAB': [],
        'SI_MATLAB': [],  # spatial information
        'SI_stim_MATLAB': [],
        'SI_ctrl_MATLAB': [],
        'TI_MATLAB': [],  # temporal information 
        'TI_stim_MATLAB': [],
        'TI_ctrl_MATLAB': [],
        'prof_mean': [],  # mean firing profile
        'prof_sem': [],
        'prof_stim_mean': [],
        'prof_stim_sem': [],
        'prof_ctrl_mean': [],
        'prof_ctrl_sem': [],
        'prof_mean_MATLAB': [],  # mean firing profile
        'prof_sem_MATLAB': [],
        'prof_stim_mean_MATLAB': [],
        'prof_stim_sem_MATLAB': [],
        'prof_ctrl_mean_MATLAB': [],
        'prof_ctrl_sem_MATLAB': [],
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
SAMP_FREQ = 1250  # in Hz
MAX_TIME = 10  # collect (for each trial) a maximum of 10 s of spiking-profile
MAX_SAMPLES = SAMP_FREQ * MAX_TIME

# pre_post ratio thresholds 
run_onset_activated_thres = 2/3
run_onset_inhibited_thres = 3/2


#%% GPU acceleration
'''
So far it seems that there is nothing to parallelise in this script for signi-
ficant improvement in performance, so let's just stick to CPU computation
'''
print('not using GPU acceleration due to lack of available parallelisation\n')
import numpy as np 
GPU_AVAILABLE = False
from scipy.stats import sem


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
    beh_df = support.load_beh_series(
        r'Z:\Dinghao\code_dinghao\behaviour\all_{}_sessions.pkl'
        .format(prefix), recname
        )
    speeds = support.load_speeds(beh_df)
    good_idx, bad_idx = support.get_good_bad_idx(beh_df)
    
    # import bad beh trial indices from MATLAB pipeline 
    good_idx_matlab, bad_idx_matlab = support.get_good_bad_idx_MATLAB(pathname)
    
    # calculate occupancy
    distance_bins = np.arange(0, track_length + bin_size, bin_size)
    occupancy = [
        support.calculate_occupancy(s, dt=.02, distance_bins=distance_bins) 
        for s in speeds
        ]
    
    # load spike trains as a list 
    clu_list, trains = support.load_train(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{}\{}_all_trains.npy'
        .format(recname, recname)
        )
    
    # load spike trains (in distance) as a list
    trains_dist = support.load_dist_spike_array(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_convSpikesDistAligned_msess1_Run0.mat'
        .format(pathname, recname)
        )
    
    # get pyr and int ID's and corresponding spike rates
    cell_identities, spike_rates = support.get_cell_info(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'
        .format(pathname, recname)
        )
    
    # get place cell indices 
    place_cell_idx = support.get_place_cell_idx(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'
        .format(pathname, recname)
        )
    
    # get cell depth 
    depths = support.get_relative_depth(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_Depth.mat'
        .format(pathname, recname)
        )
    
    # behaviour parameters
    (
        baseline_idx_MATLAB, 
        stim_idx_MATLAB, 
        ctrl_idx_MATLAB
    ) = support.get_trialtype_idx_MATLAB(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.
        format(pathname, recname)
        )
        
    stim_conds = [trial[15] for trial in beh_df['trial_statements']][1:]  # index 15 is the stim condition
    (
        baseline_idx,
        stim_idx,
        ctrl_idx
    ) = support.get_trialtype_idx(
        stim_conds
        )

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
        
        baseline_matrix_MATLAB = support.get_trial_matrix(
            trains, baseline_idx_MATLAB, MAX_SAMPLES, clu)
        ctrl_matrix_MATLAB = support.get_trial_matrix(
            trains, ctrl_idx_MATLAB, MAX_SAMPLES, clu)
        stim_matrix_MATLAB = support.get_trial_matrix(
            trains, stim_idx_MATLAB, MAX_SAMPLES, clu)
        
        # mean profiles
        baseline_mean = np.nanmean(baseline_matrix, axis=0)
        ctrl_mean = np.nanmean(ctrl_matrix, axis=0)
        stim_mean = np.nanmean(stim_matrix, axis=0)
        
        baseline_mean_MATLAB = np.nanmean(baseline_matrix_MATLAB, axis=0)
        ctrl_mean_MATLAB = np.nanmean(ctrl_matrix_MATLAB, axis=0)
        stim_mean_MATLAB = np.nanmean(stim_matrix_MATLAB, axis=0)
        
        # sem profiles 
        baseline_sem = sem(baseline_matrix, axis=0)
        ctrl_sem = sem(ctrl_matrix, axis=0)
        stim_sem = sem(stim_matrix, axis=0)
        
        baseline_sem_MATLAB = sem(baseline_matrix_MATLAB, axis=0)
        ctrl_sem_MATLAB = sem(ctrl_matrix_MATLAB, axis=0)
        stim_sem_MATLAB = sem(stim_matrix_MATLAB, axis=0)
        
        # pre-post ratio calculation
        (
            baseline_run_onset_ratio, 
            baseline_run_onset_ratiotype
        ) = support.classify_run_onset_activation_ratio(
            baseline_mean, 
            run_onset_activated_thres, 
            run_onset_inhibited_thres
            )
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
            
        (
            baseline_run_onset_ratio_MATLAB, 
            baseline_run_onset_ratiotype_MATLAB
        ) = support.classify_run_onset_activation_ratio(
            baseline_mean_MATLAB, 
            run_onset_activated_thres, 
            run_onset_inhibited_thres
            )
        (
            ctrl_run_onset_ratio_MATLAB, 
            ctrl_run_onset_ratiotype_MATLAB
        ) = support.classify_run_onset_activation_ratio(
            ctrl_mean_MATLAB, 
            run_onset_activated_thres, 
            run_onset_inhibited_thres
            )
        (
            stim_run_onset_ratio_MATLAB, 
            stim_run_onset_ratiotype_MATLAB
        ) = support.classify_run_onset_activation_ratio(
            stim_mean_MATLAB, 
            run_onset_activated_thres,
            run_onset_inhibited_thres
            )
        
        # modulation index calculation
        MI, MI_early, MI_late = support.compute_modulation_index(
            ctrl_mean, stim_mean,run_onset_bin=3750
            )
        MI_shuf, MI_early_shuf, MI_late_shuf = support.compute_modulation_index_shuf(
            ctrl_matrix, stim_matrix, run_onset_bin=3750
            )
        
        # trial by trial variatbility
        baseline_var = support.compute_trial_by_trial_variability(baseline_matrix)
        ctrl_var = support.compute_trial_by_trial_variability(ctrl_matrix)
        stim_var = support.compute_trial_by_trial_variability(stim_matrix)
        
        baseline_var_MATLAB = support.compute_trial_by_trial_variability(baseline_matrix_MATLAB)
        ctrl_var_MATLAB = support.compute_trial_by_trial_variability(ctrl_matrix_MATLAB)
        stim_var_MATLAB = support.compute_trial_by_trial_variability(stim_matrix_MATLAB)
        
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
        
        baseline_SI_MATLAB = [support.compute_spatial_information(
            trains_dist[clu][trial], occupancy[trial]) 
            for trial in baseline_idx_MATLAB]
        ctrl_SI_MATLAB = [support.compute_spatial_information(
            trains_dist[clu][trial], occupancy[trial]) 
            for trial in ctrl_idx_MATLAB]
        stim_SI_MATLAB = [support.compute_spatial_information(
            trains_dist[clu][trial], occupancy[trial]) 
            for trial in stim_idx_MATLAB]
        
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
        
        baseline_TI_MATLAB = [support.compute_temporal_information(
            trains[clu][trial][SAMP_FREQ*3:],
            bin_size_steps=1
            ) for trial in baseline_idx_MATLAB
            if trains[clu][trial] is not None]
        ctrl_TI_MATLAB = [support.compute_temporal_information(
            trains[clu][trial][SAMP_FREQ*3:],
            bin_size_steps=1
            ) for trial in ctrl_idx_MATLAB
            if trains[clu][trial] is not None]
        stim_TI_MATLAB = [support.compute_temporal_information(
            trains[clu][trial][SAMP_FREQ*3:],
            bin_size_steps=1
            ) for trial in stim_idx_MATLAB
            if trains[clu][trial] is not None]
        
        # good/bad trial mean profiles 
        good_matrix = support.get_trial_matrix(
            trains, good_idx, MAX_SAMPLES, clu)
        good_mean = np.nanmean(good_matrix, axis=0) if good_idx else np.array([])  # in case there is no bad trials
        good_sem = sem(good_matrix, axis=0) if good_idx else np.array([])
        bad_matrix = support.get_trial_matrix(
            trains, bad_idx, MAX_SAMPLES, clu)
        bad_mean = np.nanmean(bad_matrix, axis=0) if bad_idx else np.array([])
        bad_sem = sem(bad_matrix, axis=0) if bad_idx else np.array([])
        
        # good/bad trial mean profiles (MATLAB)
        good_matrix_matlab = support.get_trial_matrix(
            trains, good_idx_matlab, MAX_SAMPLES, clu)
        good_mean_matlab = np.nanmean(good_matrix_matlab, axis=0) if good_idx_matlab else np.array([])
        good_sem_matlab = sem(good_matrix_matlab, axis=0) if good_idx_matlab else np.array([])
        bad_matrix_matlab = support.get_trial_matrix(
            trains, bad_idx_matlab, MAX_SAMPLES, clu)
        bad_mean_matlab = np.nanmean(bad_matrix_matlab, axis=0) if bad_idx_matlab else np.array([])
        bad_sem_matlab = sem(bad_matrix_matlab, axis=0) if bad_idx_matlab else np.array([])

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
        
        cluname = clu_list[clu]
        df.loc[cluname] = np.array([prefix,  # rectype
                                    recname,  # recname 
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
                                    baseline_run_onset_ratio_MATLAB,
                                    stim_run_onset_ratio_MATLAB,
                                    ctrl_run_onset_ratio_MATLAB, 
                                    baseline_run_onset_ratiotype_MATLAB,
                                    stim_run_onset_ratiotype_MATLAB, 
                                    ctrl_run_onset_ratiotype_MATLAB,
                                    baseline_var,  # var
                                    stim_var,  # var_stim
                                    ctrl_var,  # var_ctrl
                                    baseline_SI,  # SI
                                    stim_SI,  # SI_stim
                                    ctrl_SI,  # SI_ctrl
                                    baseline_TI,  # TI
                                    stim_TI,  # TI_stim
                                    ctrl_TI,  # TI_ctrl
                                    baseline_var_MATLAB,
                                    stim_var_MATLAB,
                                    ctrl_var_MATLAB,
                                    baseline_SI_MATLAB,
                                    stim_SI_MATLAB,
                                    ctrl_SI_MATLAB,
                                    baseline_TI_MATLAB,
                                    stim_TI_MATLAB,
                                    ctrl_TI_MATLAB,
                                    baseline_mean,  # prof_mean
                                    baseline_sem,  # prof_sem
                                    stim_mean,  # prof_stim_mean
                                    stim_sem,  # prof_stim_sem
                                    ctrl_mean,  # prof_ctrl_mean
                                    ctrl_sem,  # prof_ctrl_sem
                                    baseline_mean_MATLAB,  # prof_mean
                                    baseline_sem_MATLAB,  # prof_sem
                                    stim_mean_MATLAB,  # prof_stim_mean
                                    stim_sem_MATLAB,  # prof_stim_sem
                                    ctrl_mean_MATLAB,  # prof_ctrl_mean
                                    ctrl_sem_MATLAB,  # prof_ctrl_sem
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
        
    print(f'{recname} done in {str(timedelta(seconds=int(time()-t0)))}\n')
        
        
#%% save dataframe 
df.to_pickle(fname)
print('\ndataframe saved')