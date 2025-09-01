# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:41:07 2025

organise properties of individual ROIs into a profile dataframe 
    for axon-GCaMP LC-CA1 recordings

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import sys 
from tqdm import tqdm
import pandas as pd

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import support_LCHPC_axon as support

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, smooth_convolve
import peak_detection_functions as pdf
mpl_formatting()


#%% dataframe initialisation/loading
fname = r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\LCHPC_axon_GCaMP_all_profiles.pkl'
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
sys.path.append(r'Z:\Dinghao\code_dinghao')
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
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
    if GPU_AVAILABLE:
        print(
            'using GPU-acceleration with '
            f'{str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode("UTF-8"))} '
            'and CuPy'
            )
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
    else:
        print('GPU acceleration unavailable')
except ModuleNotFoundError:
    print('CuPy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
except Exception as e:
    # catch any other unexpected errors and print a general message
    print(f'an error occurred: {e}')
    GPU_AVAILABLE = False
    
    
#%% main 
for path in paths:
    recname = path[-17:]
    print(f'\nprocessing {recname}...')
    
    # load data 
    F_dFF = np.load(
        rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions'
        rf'\{recname}\processed_data\RO_aligned_dict.npy',
        allow_pickle=True
        ).item() | np.load(
            rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions'
            rf'\{recname}\processed_data\RO_aligned_const_dict.npy',
            allow_pickle=True
            ).item()
    F2_dFF = np.load(
        rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions'
        rf'\{recname}\processed_data\RO_aligned_ch2_dict.npy',
        allow_pickle=True
        ).item() | np.load(
            rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions'
            rf'\{recname}\processed_data\RO_aligned_const_ch2_dict.npy',
            allow_pickle=True
            ).item()
    
    valid_ROIs_dict = np.load(
        rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions'
        rf'\{recname}\processed_data\valid_rois_dict.npy',
        allow_pickle=True
        ).item()
    valid_ROIs_coord_dict = np.load(
        rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions'
        rf'\{recname}\processed_data\valid_rois_coord_dict.npy',
        allow_pickle=True
        ).item()
    constituent_ROIs_coord_dict = np.load(
        rf'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\all_sessions'
        rf'\{recname}\processed_data\constituent_rois_coord_dict.npy',
        allow_pickle=True
        ).item()
    
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
                    desc='collecting ROI profiles'):
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
                savepath=r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP\peak_detection'
                         rf'\{recname} {roiname} {peak}',
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
print('\ndataframe saved')