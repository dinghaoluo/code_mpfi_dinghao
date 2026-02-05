# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:41:07 2025

organise properties of individual ROIs into a profile dataframe 
    for axon-GCaMP LC-CA1 recordings

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path 

import numpy as np 
import pandas as pd 
from tqdm import tqdm

import support_LCHPC_axon as support

import peak_detection_functions as pdf

from common_functions import mpl_formatting, smooth_convolve, get_GPU_availability
mpl_formatting()


#%% paths and parameters 
LC_axon_stem  = Path('Z:/Dinghao/code_dinghao/LCHPC_axon_GCaMP')
all_sess_stem = LC_axon_stem / 'all_sessions'


#%% dataframe initialisation/loading
fname = LC_axon_stem / 'LCHPC_axon_GCaMP_all_profiles.pkl'
# if os.path.exists(fname):
if False:
    df = pd.read_pickle(fname)
    print(f'df loaded from {fname}')
    processed_sess = df.index.tolist()
else:
    processed_sess = []
    sess = {
        'recname': [],  # Axxx-202xxxxx-0x
        'roi_type': [],  # str, 'primary' or 'constituent'
        'constituents': [],  # list, filled only if roi_type=='primary'
        'coord': [],  # list of tuples (x, y)
        'size': [],  # pixel count of the ROI
        'run_onset_peak': [],  # booleon
        'run_onset_peak_dFF': [],  # float
        'run_onset_peak_ch2': [],  # as control 
        'mean_profile': [],  # mean fluorescence profile (dFF)
        'sem_profile': [],
        'mean_profile_ch2': [],
        'sem_profile_ch2': [],
        'var': [],  # trialwise variability
        }
    df = pd.DataFrame(sess)



#%% load paths to recordings 
import rec_list
paths = rec_list.pathLCHPCGCaMP


#%% parameters
SMOOTHED = 1
SIGMA = 3  # frames 

SAMP_FREQ = 30  # in Hz
BEF = 3
AFT = 7  # in seconds 
MAX_TIME = 10  # collect (for each trial) a maximum of 10 s of profile
MAX_SAMPLES = SAMP_FREQ * MAX_TIME


#%% GPU acceleration
cp, GPU_AVAILABLE, device_name = get_GPU_availability()
    
    
#%% main 
for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    # load data 
    F_dFF_path        = all_sess_stem / recname / f'{recname}_all_run.npy'
    F_dFF_const_path  = all_sess_stem / recname / 'processed_data' / 'RO_aligned_const_dict.npy'
    F2_dFF_path       = all_sess_stem / recname / f'{recname}_all_run_ch2.npy'
    F2_dFF_const_path = all_sess_stem / recname / 'processed_data' / 'RO_aligned_const_ch2_dict.npy'
    valid_ROIs_path   = all_sess_stem / recname / 'processed_data' / 'valid_rois_dict.npy'
    valid_coords_path = all_sess_stem / recname / 'processed_data' / 'valid_rois_coord_dict.npy'
    const_coords_path = all_sess_stem / recname / 'processed_data' / 'constituent_rois_coord_dict.npy'
    
    F_dFF = np.load(F_dFF_path, allow_pickle=True).item() | np.load(F_dFF_const_path, allow_pickle=True).item()
    F2_dFF = np.load(F2_dFF_path, allow_pickle=True).item() | np.load(F2_dFF_const_path, allow_pickle=True).item()
    
    valid_ROIs_dict = np.load(valid_ROIs_path, allow_pickle=True).item()
    valid_ROIs_coord_dict = np.load(valid_coords_path, allow_pickle=True).item()
    constituent_ROIs_coord_dict = np.load(const_coords_path, allow_pickle=True).item()
    
    # get list of clunames 
    primary_rois = set(
        [int(name.split(' ')[1]) for name in [*valid_ROIs_dict]]
        )
    constituent_rois = set(
        [roi 
         for roi_list in valid_ROIs_dict.values()
         for roi in roi_list]
        )
    all_rois = primary_rois | constituent_rois
    
    for roi in tqdm(all_rois,
                    desc='Collecting ROI profiles'):
        roiname = f'ROI {roi}'
        # print(roiname)
        dFF = F_dFF[roiname]
        dFF2 = F2_dFF[roiname]
        
        if SMOOTHED:
            dFF = smooth_convolve(dFF, sigma=SIGMA, axis=1)
            dFF2 = smooth_convolve(dFF2, sigma=SIGMA, axis=1)
        
        # get mean and sem spiking profiles 
        # returned in lists for parqueting 
        mean_profile, sem_profile, \
        mean_profile_ch2, sem_profile_ch2 = support.compute_mean_sem_dual(
                dFF, dFF2, MAX_SAMPLES
                )

        # identity ('primary' or 'constituent')
        identity = support.get_identity(roi, primary_rois, constituent_rois)
        
        # constituent list 
        if identity=='primary':
            constituents = valid_ROIs_dict[roiname]
        else:
            constituents = None
            
        # coord
        roi_idx = int(roiname.split(' ')[-1])
        if roi_idx in primary_rois:
            coord = valid_ROIs_coord_dict[roiname]
        elif roi_idx in constituent_rois:
            coord = constituent_ROIs_coord_dict[roiname]
    
        # run-onset peak detection 
        peak, mean_prof, shuf_prof = pdf.peak_detection(
            dFF,
            around=4,  # check baseline on a higher threshold
            peak_width=2,  # 2 seconds to include some of the slightly offset peaks 
            min_peak=.5,
            samp_freq=SAMP_FREQ,
            centre_bin=SAMP_FREQ*BEF,
            bootstrap=500,
            no_boundary=True,
            GPU_AVAILABLE=GPU_AVAILABLE,
            VERBOSE=False
            )
        if identity=='primary':
            pdf.plot_peak_v_shuf(
                roiname, mean_prof, shuf_prof, peak,
                peak_width=2,
                savepath=LC_axon_stem / 'peak_detection' / f'{recname} {roiname} {peak}',
                samp_freq=SAMP_FREQ
                )  # plot the detected peaks and save to ...
        
        peak_ch2, _, _ = pdf.peak_detection(
            dFF2,
            around=4,  # check baseline on a higher threshold
            peak_width=2,  # 2 seconds to include some of the slightly offset peaks 
            samp_freq=SAMP_FREQ,
            centre_bin=SAMP_FREQ*BEF,
            bootstrap=500,
            no_boundary=True,
            GPU_AVAILABLE=GPU_AVAILABLE,
            VERBOSE=False
            )

        # trial-wise variability
        variability = support.compute_trialwise_variability(dFF)

        # full ROI name 
        full_roiname = f'{recname} {roiname}'
        
        # put into dataframe 
        df.loc[full_roiname] = np.array(
            [recname,
             identity,
             constituents,
             coord,
             len(coord[0]),  # size of ROI
             peak,
             mean_prof[peak] if peak is not None else np.nan,
             peak_ch2,
             mean_profile,
             sem_profile,
             mean_profile_ch2,
             sem_profile_ch2,
             variability],
            dtype='object'
            )
        
## save dataframe 
df.to_pickle(fname)
print('\nDataframe saved')