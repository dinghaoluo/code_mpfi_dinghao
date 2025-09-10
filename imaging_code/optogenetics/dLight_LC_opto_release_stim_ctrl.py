# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:12:32 2025

stimulated dopamine release vs endogenous dopamine release

@author: Dinghao Luo
"""

#%% imports
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf  # for txt processing convenience

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\behaviour_code\utils')
import behaviour_functions as bf

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathdLightLCOpto


#%% params
SAMP_FREQ = 30
BEF = 1
AFT = 4
TAXIS = np.arange(-BEF * SAMP_FREQ, AFT * SAMP_FREQ) / SAMP_FREQ
BASELINE_IDX = (TAXIS >= -1.0) & (TAXIS <= -0.15)

WIN_LEN = int((BEF + AFT) * SAMP_FREQ)
PMT_BUFFER_FRAMES = 10

all_sess_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions')
all_miceexp_stem = Path(r'Z:\Dinghao\MiceExp')


#%% iterate sessions
for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    animal = f'ANMD{recname[1:4]}'
    recname_txt = recname.replace('i', '')

    binpath = Path(path) / 'suite2p' / 'plane0' / 'data.bin'
    opspath = Path(path) / 'suite2p' / 'plane0' / 'ops.npy'
    txtpath = all_miceexp_stem / animal / f'{recname_txt}T.txt'
    
    savepath = all_sess_stem / recname
    (savepath / 'processed_data').mkdir(exist_ok=True)
    
    out_fig = savepath / f'{recname}_dLight_profiles_stim_ctrl'
    if (out_fig.with_suffix('.png')).exists():
        print('figure already exists; skipped')
        continue

    # behaviour check 
    beh = bf.process_behavioural_data_imaging(txtpath)
    run_onsets = beh['run_onset_frames']
    stim_methods = [t[15] for t in beh['trial_statements']]
    
    if len(run_onsets) == 0:
        print('no behaviour in session; skipping')
        continue
    elif '2' not in stim_methods:
        print('no valid stim in session; skipped')
        continue
    else:
        print('behaviour session')

    # load data 
    print('loading movie...')
    ops = np.load(opspath, allow_pickle=True).item()
    tot_frames = int(ops['nframes'])
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    shape = (tot_frames, Ly, Lx)
    mov = np.memmap(binpath, mode='r', dtype='int16', shape=shape)
    
    # stim info 
    print('extracting stim info...')
    frame_times = beh['frame_times']
    pulse_times = beh['pulse_times']
    pulse_params = [ls for ls in beh['pulse_descriptions'] if ls]
    
    pulse_width_ON = float(pulse_params[-1][3]) / 1_000_000
    pulse_width    = float(pulse_params[-1][4]) / 1_000_000
    pulse_number   = int(pulse_params[-1][5])
    
    diffs = np.diff(pulse_times)
    split_idx = np.where(diffs >= 1000)[0] + 1
    pulse_trains = np.split(pulse_times, split_idx)
    
    pulse_frames = [
        [ipf.find_nearest(p, frame_times) for p in train]
        for train in pulse_trains
    ]
    
    stim_mask = np.zeros(tot_frames, dtype=bool)
    for train in pulse_frames:
        if len(train) == 0:
            continue
        start = max(0, train[0] - 1)  # small pre-buffer
        # estimate last pulse end in frames
        train_len_s = ( (train[-1] - train[0]) / 1000.0 ) + pulse_width
        end = min(train[-1] + int(pulse_width*SAMP_FREQ) + PMT_BUFFER_FRAMES, tot_frames)
        stim_mask[start:end] = True

    # split trials
    ctrl_trials = [i for i, f in enumerate(stim_methods) if f == '0']
    stim_trials = [i for i, f in enumerate(stim_methods) if f != '0']

    # filter trials
    ctrl_frames = []
    for i in ctrl_trials:
        if i < len(run_onsets):
            f = run_onsets[i]
            if f is not None and not np.isnan(f):
                start = int(f) - int(BEF * SAMP_FREQ)
                end   = int(f) + int(AFT * SAMP_FREQ)
                if start >= 0 and end <= tot_frames:
                    ctrl_frames.append(int(f))

    stim_frames = []
    for i in stim_trials:
        if i < len(run_onsets):
            f = run_onsets[i]
            if f is not None and not np.isnan(f):
                start = int(f) - int(BEF * SAMP_FREQ)
                end   = int(f) + int(AFT * SAMP_FREQ)
                if start >= 0 and end <= tot_frames:
                    stim_frames.append(int(f))

    # extract data 
    print('extracting fluorescence data...')
    F = np.sum(mov, axis=(1, 2)).astype(np.float32)
    dFF = ipf.calculate_dFF(F)
    dFF[stim_mask] = np.nan  # mask out the stim frames 
    
    ctrl_profiles = np.zeros((len(ctrl_frames), WIN_LEN), dtype=np.float32)
    stim_profiles = np.zeros((len(stim_frames), WIN_LEN), dtype=np.float32)
    for j, f in enumerate(ctrl_frames):
        start = f - int(BEF * SAMP_FREQ)
        end   = f + int(AFT * SAMP_FREQ)
        ctrl_profiles[j, :] = dFF[start : end]
    for j, f in enumerate(stim_frames):
        start = f - int(BEF * SAMP_FREQ)
        end   = f + int(AFT * SAMP_FREQ)
        stim_profiles[j, :] = dFF[start : end]

    # single session plot 
    ctrl_mean = np.nanmean(ctrl_profiles, axis=0)
    stim_mean = np.nanmean(stim_profiles, axis=0)
    
    print('plotting...')
    fig, ax = plt.subplots(figsize=(2.4, 2))
    ax.plot(TAXIS, ctrl_mean, 
            c='grey', label='ctrl.', linewidth=1)
    ax.plot(TAXIS, stim_mean, 
            c='royalblue', label='stim.', linewidth=1)
    ax.axvline(0, linestyle='--', linewidth=1, color='red')
    ax.set(xlabel='time from run onset (s)',
           ylabel='ΔF/F',
           title=recname)
    ax.legend(frameon=False)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(f'{out_fig}{ext}', 
                    dpi=300, bbox_inches='tight')