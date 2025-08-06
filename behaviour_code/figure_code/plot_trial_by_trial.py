# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:33:49 2023

plot run bouts 

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import scipy.io as sio
import sys
import mat73
import os 

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, gaussian_kernel_unity
mpl_formatting()


#%% gaussian kernel for speed smoothing 
SAMP_FREQ = 1250  # Hz
gaus_speed = gaussian_kernel_unity(sigma=SAMP_FREQ*0.03)


#%% main
for exp_name in [
        'HPCLC', 
        'HPCLCterm', 
        'LC', 
        'HPCGRABNE', 
        'LCHPCGCaMP',
        'HPCdLightLCOpto',
        'HPCdLightLCOptoInh']:
    
    # make dirs 
    datapath_stem = os.path.join(
        r'Z:\Dinghao\code_dinghao\behaviour\all_experiments', exp_name
        )
    if exp_name == 'HPCLC':
        paths = rec_list.pathHPCLCopt
    elif exp_name == 'HPCLCterm':
        paths = rec_list.pathHPCLCtermopt
    elif exp_name == 'LC':
        paths = rec_list.pathLC
    elif exp_name == 'HPCGRABNE':
        paths = rec_list.pathHPCGRABNE
    elif exp_name == 'LCHPCGCaMP':
        paths = rec_list.pathLCHPCGCaMP
    elif exp_name == 'HPCdLightLCOpto':
        paths = rec_list.pathdLightLCOpto
    elif exp_name == 'HPCdLightLCOptoInh':
        paths = rec_list.pathdLightLCOptoInh
        
    for path in paths:
        recname = path[-17:]
        
        with open(os.path.join(datapath_stem, f'{recname}.pkl'), 'rb') as f:
            try:
                data = pickle.load(f)
            except EOFError:
                print(f'[ERROR] Could not load {recname} — file may be corrupted.')
                continue
        
        output_dir = os.path.join(
            r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles', 
            exp_name, recname
            )
        os.makedirs(output_dir, exist_ok=True)

        alignedRun_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}',  # numbers + r
                                        recname[:14],  # till end of date
                                        recname,
                                        f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat')
        alignedCue_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}',  # numbers + r
                                        recname[:14],  # till end of date
                                        recname,
                                        f'{recname}_DataStructure_mazeSection1_TrialType1_alignCue_msess1.mat')
        alignedRew_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}',  # numbers + r
                                        recname[:14],  # till end of date
                                        recname,
                                        f'{recname}_DataStructure_mazeSection1_TrialType1_alignRew_msess1.mat')
        behave_lfp_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}', 
                                       recname[:14],  # till end of date
                                       recname,
                                       f'{recname}_BehavElectrDataLFP.mat')
        
        if (not os.path.exists(alignedRun_path)
            or not os.path.exists(behave_lfp_path)):
            print(f'\nmissing data for {recname}; skipped')
        else:
            print(f'\n{recname}')
            
        # load data
        beh_lfp = mat73.loadmat(behave_lfp_path)
        tracks = beh_lfp['Track']
        laps = beh_lfp['Laps']
        
        aligned = sio.loadmat(alignedRun_path)['trialsRun'][0][0]
        alignedCue = sio.loadmat(alignedCue_path)['trialsCue'][0][0]  # for cue marking 
        alignedRew = sio.loadmat(alignedRew_path)['trialsRew'][0][0]  # for cue marking 
        
        # read cues 
        cueLfpInd = alignedCue['startLfpInd'].flatten()
        
        # read rewards 
        rewLfpInd = alignedRew['startLfpInd'].flatten()
        
        # read licks 
        lickLfp = laps['lickLfpInd']
        lickLfp_flat = []
        for trial in range(len(lickLfp)):
            if isinstance(lickLfp[trial][0], np.ndarray):  # only when there are licks
                for i in range(len(lickLfp[trial][0])):
                    lickLfp_flat.append(int(lickLfp[trial][0][i]))
            else:
                continue
        lickLfp_flat = np.array(lickLfp_flat)
        speed_MMsec = tracks['speed_MMsecAll']
        for tbin in range(len(speed_MMsec)):
            if speed_MMsec[tbin]<0:
                speed_MMsec[tbin] = (speed_MMsec[tbin-1]+speed_MMsec[tbin+1])/2
        speed_MMsec = np.convolve(speed_MMsec, gaus_speed, mode='same')/10  # /10 for cm
        startLfpInd = aligned['startLfpInd'][0]
        endLfpInd = aligned['endLfpInd'][0]
        
        print('plotting...')
        for t in np.arange(2, len(endLfpInd)-2, 3):
            lfp_indices_t = np.arange(startLfpInd[t]-1250, min(endLfpInd[t+2]+1250, len(speed_MMsec)))
            lap_start = lfp_indices_t[0]
            xaxis = np.arange(0, len(lfp_indices_t)) / SAMP_FREQ
        
            fig, ax = plt.subplots(figsize=(len(lfp_indices_t)/5000, 1.2))
            ax.set(xlabel='time (s)', ylabel='speed (cm/s)',
                   ylim=(0, 1.2 * max(speed_MMsec[lfp_indices_t])),
                   xlim=(0, len(lfp_indices_t) / SAMP_FREQ),
                   title=f'{recname} trials {t-1} to {t+1}')
        
            ax.plot(xaxis, speed_MMsec[lfp_indices_t], color='royalblue', label='speed')

            cueLfpInd_t = cueLfpInd[np.in1d(cueLfpInd, lfp_indices_t)]
            ax.vlines((cueLfpInd_t - lap_start)/SAMP_FREQ, 0, ax.get_ylim()[1], 'darkgrey')
            
            rewLfpInd_t = rewLfpInd[np.in1d(rewLfpInd, lfp_indices_t)]
            ax.vlines((rewLfpInd_t - lap_start)/SAMP_FREQ, ax.get_ylim()[1], ax.get_ylim()[1]*.95, 'forestgreen', linewidth=1.5, zorder=10)
            
            startLfpInd_t = startLfpInd[np.in1d(startLfpInd, lfp_indices_t)]
            ax.vlines((startLfpInd_t - lap_start)/SAMP_FREQ, 0, ax.get_ylim()[1], 'red', linestyle='dashed')
        
            licks_t = lickLfp_flat[np.in1d(lickLfp_flat, lfp_indices_t)]
            ax.vlines((licks_t - lap_start)/SAMP_FREQ, ax.get_ylim()[1]*.96, ax.get_ylim()[1]*.99, 'magenta', linewidth=.8)
            
            for s in ['top', 'right']:
                ax.spines[s].set_visible(False)
        
            save_path = os.path.join(output_dir, f'trials_{t-1}_to_{t+1}')
        
            for ext in ['.pdf', '.png']:
                fig.savefig(save_path+ext, dpi=300, bbox_inches='tight')
            plt.close(fig)