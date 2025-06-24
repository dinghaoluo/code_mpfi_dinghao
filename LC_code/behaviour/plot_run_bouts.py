# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:33:49 2023

plot run bouts 

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.io as sio
import sys
import mat73
import os 
from tqdm import tqdm

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, smooth_convolve, gaussian_kernel_unity
mpl_formatting()


#%% gaussian kernel for speed smoothing 
SAMP_FREQ = 1250  # Hz
gaus_speed = gaussian_kernel_unity(sigma=SAMP_FREQ*0.1)  # same as spike 


#%% load cell profiles
cell_profiles = pd.read_pickle(r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% MAIN
run_bout_dir = r'Z:\Dinghao\code_dinghao\run_bouts'
save_path_base = r'Z:\Dinghao\code_dinghao\run_bouts\fsa_run_bouts_plots_python'

for path in paths:
    recname = path[-17:]
    
    run_bout_path = os.path.join(run_bout_dir, 
                                 f'{recname}_run_bouts_py.csv')
    aligned_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}',  # numbers + r
                                recname[:14],  # till end of date
                                recname,
                                f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat')
    behave_lfp_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}', 
                                   recname[:14],  # till end of date
                                   recname,
                                   f'{recname}_BehavElectrDataLFP.mat')
    clu_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}', 
                            recname[:14],  # till end of date
                            recname,
                            f'{recname}.clu.1')
    res_path = os.path.join(rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}', 
                            recname[:14],  # till end of date
                            recname,
                            f'{recname}.res.1')
    if (not os.path.exists(run_bout_path) 
        or not os.path.exists(aligned_path)
        or not os.path.exists(behave_lfp_path)
        or not os.path.exists(clu_path)
        or not os.path.exists(res_path)):
        print(f'\nmissing data for {recname}; skipped')
    else:
        print(f'\n{recname}')
        
    # load data
    run_bout_table = pd.read_csv(run_bout_path)
    
    beh_lfp = mat73.loadmat(behave_lfp_path)
    tracks = beh_lfp['Track']
    laps = beh_lfp['Laps']
    
    aligned = sio.loadmat(aligned_path)['trialsRun'][0][0]
    
    # spike reading
    clusters = np.loadtxt(clu_path, dtype=int, skiprows=1)  # first line = number of clusters
    spike_times = np.loadtxt(res_path, dtype=int) / (20_000 / 1_250)  # convert to behavioural time scale
    spike_times = spike_times.astype(int)  # ensure integer indices for indexing
    
    unique_clus = [clu for clu in np.unique(clusters) if clu not in [0, 1]]
    clu_to_row = {clu: i for i, clu in enumerate(unique_clus)}  # map cluster ID to row index
    
    max_time = spike_times.max() + 1  # +1 to make sure last time index is included
    spike_map = np.zeros((len(unique_clus), max_time), dtype=int)
    spike_array = np.zeros((len(unique_clus), max_time))
    
    for time, clu in zip(spike_times, clusters):
        if clu in [0, 1]:
            continue  # skip noise
        row = clu_to_row[clu]
        spike_map[row, time] = 1  # set spike bin to 1
    
    for i in range(len(unique_clus)):
        spike_array[i, :] = smooth_convolve(spike_map[i, :], sigma=int(SAMP_FREQ*0.1))
    
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
    
    
    ## plotting 
    save_path_sess = os.path.join(save_path_base, recname)

    # make a separate figure per cluster
    for i, clu in tqdm(enumerate(unique_clus),
                       total=len(unique_clus),
                       desc='plotting'):
        cluname = f'{recname} clu{clu}'
        clu_identity = cell_profiles.loc[cluname]['identity']
        
        for t in np.arange(2, len(endLfpInd)-2, 3):
            lfp_indices_t = np.arange(startLfpInd[t]-1250, min(endLfpInd[t+2]+1250, len(speed_MMsec)))
            lap_start = lfp_indices_t[0]
            
            fig, ax = plt.subplots(figsize=(len(lfp_indices_t)/3000, 2.2))  # smaller and more compact
            ax.set(xlabel='time (s)', ylabel='speed (cm/s)',
                   ylim=(0, 1.2 * max(speed_MMsec[lfp_indices_t])),  # raised slightly
                   xlim=(0, len(lfp_indices_t) / SAMP_FREQ),
                   title=f'{recname} | clu {clu} | trials {t-1} to {t+1}')
            
            xaxis = np.arange(0, len(lfp_indices_t)) / SAMP_FREQ
            ylim = 1.2 * max(speed_MMsec[lfp_indices_t])  # match ylim to ax limit
    
            # plot velocity
            ax.plot(xaxis, speed_MMsec[lfp_indices_t], color='black', label='speed')
            
            # lap starts
            startLfpInd_t = startLfpInd[np.in1d(startLfpInd, lfp_indices_t)]
            ax.vlines((startLfpInd_t - lap_start)/SAMP_FREQ, 0, ylim, 'r', linestyle='dashed', label='lap start')
    
            # run bout starts
            run_bout_t = run_bout_table.iloc[:,1][np.in1d(run_bout_table.iloc[:,1], lfp_indices_t)]
            ax.vlines((run_bout_t - lap_start)/SAMP_FREQ, 0, ylim, 'g', linestyle='dashed', label='run bout')
    
            # secondary axis for spike rate (Hz)
            ax_spk = ax.twinx()
            ax_spk.set_ylabel('firing rate (Hz)', color='orange')
            ax_spk.spines['right'].set_color('orange')
            ax_spk.tick_params(axis='y', colors='orange')
            ax_spk.tick_params(axis='y', labelcolor='orange')
            
            # spike trace for this cluster
            spike_t = spike_array[i, lfp_indices_t]
            spike_rate_hz = spike_t * SAMP_FREQ  # convert from spikes/bin to Hz
            
            # clip extreme peaks for visibility
            spike_rate_hz = np.clip(spike_rate_hz, 0, np.percentile(spike_rate_hz, 99.5))
            
            # plot on secondary axis
            ax_spk.plot(xaxis, spike_rate_hz, color='orange', linewidth=1.2, label=f'spike {clu}')
            
            # optionally match y-lim for visual balance
            ax_spk.set_ylim(0, np.max(spike_rate_hz) * 1.1)
            
            # licks
            licks_t = lickLfp_flat[np.in1d(lickLfp_flat, lfp_indices_t)]
            ax.vlines((licks_t - lap_start)/SAMP_FREQ, ylim, ylim * 0.96, 'magenta', label='licks')
            
            # spines 
            ax.spines['top'].set_visible(False)
            ax_spk.spines['top'].set_visible(False)
            
            # ax.legend(loc='upper right', fontsize=6)
            
            # make sure directory exists
            save_path_sess_clu = os.path.join(save_path_sess, f'{recname} clu{clu} {clu_identity}')
            os.makedirs(save_path_sess_clu, exist_ok=True)
            
            # add ' rb' if run bouts exist in view
            has_rb = len(run_bout_t) > 0
            rb_tag = ' rb' if has_rb else ''
            save_path = os.path.join(save_path_sess_clu, f'trials_{t-1}_to_{t+1}{rb_tag}')
            
            for ext in ['.pdf', '.png']:
                fig.savefig(save_path+ext, dpi=300, bbox_inches='tight')
            plt.close(fig)
