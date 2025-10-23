# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 16:26:59 2025

tonic fluctuation of baseline LC FR?

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
import pandas as pd
import scipy.io as sio  
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt 

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathLC


#%% paths & params
all_sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
LC_beh_stem   = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')
GLM_stem      = Path('Z:/Dinghao/code_dinghao/LC_ephys/GLM')

SAMP_FREQ = 1250  # Hz
nperseg = SAMP_FREQ * 10  # 10 s window
FREQ_RANGE = (0.01, 1)


#%% load cell table
print('loading data...')
cell_prop = pd.read_pickle('Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% main
psd_list = []

for path in paths:
    recname = Path(path).name
    print(recname)
    
    rec_stem = Path(f'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}') / recname[:14] / recname
    aligned_run_path = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
    aligned_run = sio.loadmat(aligned_run_path)['trialsRun'][0][0]
    run_onsets_spike = aligned_run['startLfpInd'][0][1:]
    
    sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions') / recname
    spike_maps = np.load(sess_stem / f'{recname}_smoothed_spike_map.npy', allow_pickle=True)
    
    curr_cell_prop = cell_prop[cell_prop['sessname'] == recname]
    for cluname, row in curr_cell_prop.iterrows():
        if row['identity'] == 'other' or not row['run_onset_peak']:
            continue
        
        clu_idx = int(cluname.split('clu')[-1]) - 2  # to retrieve spike map 
        
        spike_map = spike_maps[clu_idx]
                
        # Welch PSD
        f, Pxx = welch(spike_map, fs=SAMP_FREQ, nperseg=nperseg, detrend='constant')
        
        # restrict to 0.01–1 Hz
        mask = (f >= FREQ_RANGE[0]) & (f <= FREQ_RANGE[1])
        f_slow, Pxx_slow = f[mask], Pxx[mask]
        Pxx_slow /= np.sum(Pxx_slow)

        # log-smooth
        Pxx_smooth = gaussian_filter1d(Pxx_slow, sigma=2)
        
        psd_list.append(Pxx_smooth)
        
        # single-cell plot
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(f_slow, Pxx_slow)
        ax.set(xlabel='Frequency (Hz)',
               ylabel='Power',
               title=cluname)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.show()
        
psd_arr = np.vstack(psd_list)
mean_psd = np.mean(psd_arr, axis=0)
sem_psd = np.std(psd_arr, axis=0) / np.sqrt(psd_arr.shape[0])

plt.figure(figsize=(4, 3))
plt.fill_between(f_slow, mean_psd - sem_psd, mean_psd + sem_psd, color='gray', alpha=0.3)
plt.plot(f_slow, mean_psd, color='k', lw=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density')
plt.title('LC baseline FR power spectrum (0.01–1 Hz)')
plt.tight_layout()
plt.show()