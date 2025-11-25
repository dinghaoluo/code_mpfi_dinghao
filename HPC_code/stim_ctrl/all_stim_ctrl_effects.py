# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:04:59 2025

Estimate the first point of stimulation effects for CA1 cells 

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
import pandas as pd 
import scipy.io as sio 
from scipy.stats import sem, ranksums, ks_2samp
import matplotlib.pyplot as plt 
from tqdm import tqdm

from behaviour_functions import process_txt, detect_run_onsets_teensy
from plotting_functions import plot_ecdfs
from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathHPCLCopt


#%% path stems
all_sess_stem    = Path('Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions')
all_beh_stem     = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments')
mice_exp_stem    = Path('Z:/Dinghao/MiceExp')
single_cell_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/single_cell_stim_effects')
stim_effect_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/stim_effects')


#%% parameters 
SAMP_FREQ = 1250  # Hz

BEF = 1  # in s 
AFT = 4
MAX_LENGTH = (BEF + AFT) * SAMP_FREQ

XAXIS = np.arange(MAX_LENGTH) / SAMP_FREQ - BEF

# for single cell stats 
ALPHA = 0.05
BIN_MS   = 50  # ms
BIN_SIZE = int((BIN_MS/1000) * SAMP_FREQ)
N_BINS = MAX_LENGTH // BIN_SIZE
BIN_CENTRES = (np.arange(N_BINS) * BIN_SIZE + BIN_SIZE/2) / SAMP_FREQ - BEF
SIGNAL_START_BIN = int((BEF * SAMP_FREQ) / BIN_SIZE)


#%% load data
print('loading dataframes...')
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')
df_pyr = cell_profiles[cell_profiles['cell_identity'] == 'pyr']


#%% main (HPCLC)
# sig_dict is cluname: first_sustained_time, first_sustained_sign
# first_sustained_time is the first time bin (in seconds) where sustained
#   activation/inhibition is detected
# first_sustained_sign is the sign of the effect
sig_dict = {}

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    txtpath = mice_exp_stem / f'ANMD{recname[1:4]}r' / recname[:-3] / recname / f'{recname}T.txt'
    beh = process_txt(txtpath)
    
    # get stim and ctrl idx
    stim_conds = [t[15] for t in beh['trial_statements']][1:]
    stim_idx = [trial for trial, cond in enumerate(stim_conds) if cond != '0']
    ctrl_idx = [trial+2 for trial in stim_idx]  
    
    # get online run onsets 
    speed_times = beh['speed_times'][1:]
    run_onset_online = detect_run_onsets_teensy(speed_times)
    run_onset_online = [run for trial, run in enumerate(run_onset_online) 
                        if trial in ctrl_idx]
    
    # get stim times and truncate pulses 
    pulse_times = np.array(beh['pulse_times'])
    diffs = np.diff(pulse_times)
    split_idx = np.where(diffs >= 1000)[0] + 1
    pulse_trains = np.split(pulse_times, split_idx)
    stim_times = [pulse_train[0] for pulse_train in pulse_trains]  # actual stim times 
    
    # get spike maps 
    spike_map_path = all_sess_stem / recname / f'{recname}_smoothed_spike_map.npy'
    spike_maps = np.load(spike_map_path, allow_pickle=True)
    max_time = len(spike_maps[0])
    
    
    ## ---- conversion (teensy-time to spike-time) ---- ##
    # using cue time to align due to least variability 
    rec_stem = Path(f'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}') / recname[:14] / recname
    aligned_cue_path = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignCue_msess1.mat'
    aligned_cue = sio.loadmat(aligned_cue_path)['trialsCue'][0][0]
    
    cue_spike_time  = aligned_cue['startLfpInd'][0]
    cue_teensy_time = [trial[0][0] if trial else np.nan for trial in beh['movie_times']]
    spike_teensy_offset = cue_teensy_time[0] - cue_spike_time[0]
    
    # linear map
    a, b = np.polyfit(cue_teensy_time, cue_spike_time, 1)
    
    def _teensy_to_spike(t_teensy):
        return a*t_teensy + b
    
    # actual conversions
    stim_times_converted       = [int(_teensy_to_spike(t)) for t in stim_times]
    run_onset_online_converted = [int(_teensy_to_spike(t)) for t in run_onset_online]
    ## ---- conversion ends ---- ##
    
    
    # ignore if not enough BEF or AFT
    stim_times_converted       = [stim for stim in stim_times_converted
                                  if BEF*SAMP_FREQ <= stim <= max_time-AFT*SAMP_FREQ]
    run_onset_online_converted = [run for run in run_onset_online_converted
                                  if BEF*SAMP_FREQ <= run <= max_time-AFT*SAMP_FREQ]
    
    # curr session df 
    curr_df_pyr = df_pyr[df_pyr['recname'] == recname]


    ## ---- processing ---- ##
    for idx, session in tqdm(curr_df_pyr.iterrows(), desc='processing', total=len(curr_df_pyr)):
        cluname = idx
        
        clu_idx = int(cluname.split('clu')[1].split(' ')[0]) - 2  # actual index for retrieval 
        spike_map = spike_maps[clu_idx, :]
        
        
        ## ---- alignment ---- ##
        run_aligned = np.zeros((len(run_onset_online_converted), MAX_LENGTH))
        for trial, run in enumerate(run_onset_online_converted):
            run_aligned[trial, :] = spike_map[run - BEF*SAMP_FREQ : run + AFT*SAMP_FREQ]
            
        stim_aligned = np.zeros((len(stim_times_converted), MAX_LENGTH))
        for trial, stim in enumerate(stim_times_converted):
            stim_aligned[trial, :] = spike_map[stim - BEF*SAMP_FREQ : stim + AFT*SAMP_FREQ]
        ## ---- alignment ends ---- ##
        
        
        run_mean  = np.mean(run_aligned, axis=0)
        run_sem   = sem(run_aligned, axis=0)
        stim_mean = np.mean(stim_aligned, axis=0)
        stim_sem  = sem(stim_aligned, axis=0)
        
        
        ## ---- effect test ---- ##
        pvals = np.ones(N_BINS)
        effect_sign = np.zeros(N_BINS, dtype=int)   # -1, 0, +1
        
        for bi in range(N_BINS):
            start = bi * BIN_SIZE
            end   = start + BIN_SIZE
        
            ctrl_bin = np.mean(run_aligned[:,  start:end], axis=1)
            stim_bin = np.mean(stim_aligned[:, start:end], axis=1)
        
            _, p_val = ranksums(ctrl_bin, stim_bin)
            pvals[bi] = p_val
        
            if p_val < ALPHA:
                if np.mean(stim_bin) > np.mean(ctrl_bin):
                    effect_sign[bi] = 1    # stim > ctrl
                else:
                    effect_sign[bi] = -1   # stim < ctrl
        
        first_sustained_time = None
        first_sustained_sign = 0   # +1 or -1
        
        for bi in range(SIGNAL_START_BIN, N_BINS - 4):
            window = effect_sign[bi : bi+5]
        
            # activation (stim > ctrl)
            if np.sum(window == 1) >= 4:
                first_sustained_time = float(BIN_CENTRES[bi])
                first_sustained_sign = 1
                break
        
            # suppression (stim < ctrl)
            if np.sum(window == -1) >= 4:
                first_sustained_time = float(BIN_CENTRES[bi])
                first_sustained_sign = -1
                break
        
        sig_dict[cluname] = [first_sustained_time, first_sustained_sign]
        ## ---- effect test ends ---- ##
        
        
        # plotting 
        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        
        ax.plot(XAXIS, run_mean, c='grey')
        ax.fill_between(XAXIS, run_mean+run_sem,
                               run_mean-run_sem,
                        color='grey', edgecolor='none', alpha=.5)
        ax.plot(XAXIS, stim_mean, c='royalblue')
        ax.fill_between(XAXIS, stim_mean+stim_sem,
                               stim_mean-stim_sem,
                        color='royalblue', edgecolor='none', alpha=.5)
        
        ax.set(xlabel='Time from run onset (s)',
               ylabel='Firing rate (Hz)',
               title=cluname)
        
        ymin, ymax = ax.get_ylim()
        marker_bottom = ymin + 0.90*(ymax - ymin)
        marker_top    = ymin + 0.98*(ymax - ymin)
        for bi in range(N_BINS):
            x = BIN_CENTRES[bi]
            if x < 0:
                continue
            if effect_sign[bi] == 1:
                ax.plot([x, x], [marker_bottom, marker_top], color='red', lw=0.8)
            elif effect_sign[bi] == -1:
                ax.plot([x, x], [marker_bottom, marker_top], color='blue', lw=0.8)
                
        fig.tight_layout()
        
        fig.savefig(
            single_cell_stem / f'{cluname}.png',
            dpi=300,
            bbox_inches='tight'
            )
            
        plt.close()
    ## ---- processing ends ---- ##
    

#%% summary
proportion_act = sum([1 for clu in sig_dict.values() if clu[1] == 1]) / len(sig_dict)
proportion_inh = sum([1 for clu in sig_dict.values() if clu[1] == -1]) / len(sig_dict)

onset_times_act = [clu[0] for clu in sig_dict.values() if clu[1] == 1]
onset_times_inh = [clu[0] for clu in sig_dict.values() if clu[1] == -1]

act_median = np.median(onset_times_act)
inh_median = np.median(onset_times_inh)

# plotting 
bin_edges = np.arange(0.0, max(max(onset_times_act), max(onset_times_inh)), 0.1)

fig, axs = plt.subplots(2,1,figsize=(3,3.4), sharex=True)

axs[0].hist(onset_times_act,
            bins=bin_edges,
            density=True,
            color='indianred',
            edgecolor='k',
            label='activation')

axs[1].hist(onset_times_inh,
            bins=bin_edges,
            density=True,
            color='lightsteelblue',
            edgecolor='k',
            label='inhibition')

axs[0].set(title=f'Activation med={act_median:.3g} s')
axs[1].set(title=f'Inhibition med={inh_median:.3g} s',
           xlabel='Time from run/stim. onset (s)')
for i in range(2):
    axs[i].set(xlim=(0,max(max(onset_times_act), max(onset_times_inh))), 
               yticks=[0,1], ylim=(0,1), ylabel='Density')

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        stim_effect_stem / f'HPCLC_act_inh_full_hist{ext}',
        dpi=300,
        bbox_inches='tight'
        )


# plotting (zoomed in)
bin_edges = np.arange(0.0, max(max(onset_times_act), max(onset_times_inh)), 0.05)

fig, axs = plt.subplots(2,1,figsize=(2.6,3), sharex=True)

axs[0].hist(onset_times_act,
            bins=bin_edges,
            density=True,
            color='indianred',
            edgecolor='k',
            label='activation')

axs[1].hist(onset_times_inh,
            bins=bin_edges,
            density=True,
            color='lightsteelblue',
            edgecolor='k',
            label='inhibition')

axs[0].set(title=f'Activation med={act_median:.3g} s')
axs[1].set(title=f'Inhibition med={inh_median:.3g} s',
           xlabel='Time from run/stim. onset (s)')
for i in range(2):
    axs[i].set(xlim=(0,1), 
               yticks=[0,1], ylim=(0,1), ylabel='Density')

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        stim_effect_stem / f'HPCLC_act_inh_0_1_hist{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    
    
#%% cdfs
ks_stat, ks_p = ks_2samp(onset_times_act, onset_times_inh)

stats_label = (
    f'KS D={ks_stat:.3f}\np={ks_p:.3g}'
)

plot_ecdfs(
    onset_times_act,
    onset_times_inh,
    title='Latency to sustained effect',
    xlabel='Time from run/stim. onset (s)',
    ylabel='Cumulative probability',
    legend_labels=['activation', 'inhibition'],
    colours=['indianred', 'steelblue'],
    save=True,
    savepath=str(stim_effect_stem / 'HPCLC_latency_ecdf'),
    dpi=300,
    figsize=(2.4, 3.0),
    stats_text=stats_label
)