# -*- coding: utf-8 -*-
"""
Created on Thur 13 Feb 14:36:41 2025

analyse and plot stim responses for LC stim recordings

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import fftconvolve

from behaviour_functions import detect_run_onsets_teensy, process_txt
from plotting_functions import plot_violin_with_scatter
from common import gaussian_kernel_unity, mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathLCopt


#%% paths and parameters
mice_exp_stem    = Path('Z:/Dinghao/MiceExp')
all_session_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
stim_raster_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/stim_effects/single_cell_stim_rasters')
stim_raster_stem.mkdir(parents=True, exist_ok=True)

BEF = 3  # s
AFT = 7  # s
SAMP_FREQ = 1250  # hz

MAX_LENGTH = int((BEF + AFT) * SAMP_FREQ)
XAXIS = np.arange(MAX_LENGTH) / SAMP_FREQ - BEF

AMP_WINDOW_LOW_S  = 0  # s
AMP_WINDOW_HIGH_S = 1
AMP_WINDOW_LOW    = int((AMP_WINDOW_LOW_S + BEF) * SAMP_FREQ)
AMP_WINDOW_HIGH   = int((AMP_WINDOW_HIGH_S + BEF) * SAMP_FREQ)

SIGMA_SPIKE = int(SAMP_FREQ * 0.05)  # 50 ms
GAUS_SPIKE = gaussian_kernel_unity(SIGMA_SPIKE, GPU_AVAILABLE=False)


#%% functions
OF_CONSTANT = (2**32 - 1) / 1000  # overflow constant (ms)

def _correct_overflow(times, of_constant=OF_CONSTANT):
    times = np.array(times, dtype=float)

    correction_flag = 0
    correction_time = None
    corrected = [times[0]]

    for i in range(1, len(times)):
        if correction_flag == 1:
            corrected.append(times[i] + of_constant)
        elif times[i] >= times[i - 1] and correction_flag == 0:
            corrected.append(times[i])
        else:
            correction_flag = 1
            correction_time = i
            corrected.append(times[i] + of_constant)

    return corrected, correction_time


def _align_trials(spike_map, events, max_time):
    aligned = []
    for ev in events:
        if ev < BEF * SAMP_FREQ or ev > max_time - AFT * SAMP_FREQ:
            continue
        aligned.append(spike_map[ev - BEF * SAMP_FREQ : ev + AFT * SAMP_FREQ])
    return np.asarray(aligned)


def _load_spike_map(recname, mice_exp_stem, samp_freq=1250):
    """
    reconstruct raw binary spike map from .res and .clu files.

    returns:
    - spike_map: np.ndarray, shape (n_clu, max_time), dtype=int
        binary spike train per cluster (spike clock)
    - clu_ids: list[int]
        cluster IDs corresponding to rows in spike_map
    - max_time: int
        length of spike train in samples
    """

    rec_stem = (
        mice_exp_stem
        / f'ANMD{recname[1:5]}'
        / recname[:14]
        / recname
    )

    clu_path = rec_stem / f'{recname}.clu.1'
    res_path = rec_stem / f'{recname}.res.1'

    if not clu_path.exists() or not res_path.exists():
        raise FileNotFoundError(f'missing res/clu files for {recname}')

    # --- load raw spikes ---
    clusters = np.loadtxt(clu_path, dtype=int, skiprows=1)  # first line = n_clusters
    spike_times = np.loadtxt(res_path, dtype=int)

    # convert from 20 kHz to behavioural clock (1250 Hz)
    spike_times = spike_times / (20_000 / samp_freq)
    spike_times = spike_times.astype(int)

    # valid clusters (exclude noise)
    clu_ids = [clu for clu in np.unique(clusters) if clu not in (0, 1)]
    clu_to_row = {clu: i for i, clu in enumerate(clu_ids)}

    max_time = spike_times.max() + 1
    spike_map = np.zeros((len(clu_ids), max_time), dtype=np.uint8)

    for t, clu in zip(spike_times, clusters):
        if clu in (0, 1):
            continue
        row = clu_to_row[clu]
        spike_map[row, t] = 1

    return spike_map, clu_ids, max_time

def _load_run_rasters(recname, all_session_stem):
    """
    load precomputed run-onset aligned rasters.

    returns:
    - rasters_run: dict[str, np.ndarray]
        cluname -> (n_trials, MAX_LENGTH)
    """
    path = all_session_stem / recname / f'{recname}_all_rasters_run.npy'
    if not path.exists():
        raise FileNotFoundError(f'missing run rasters for {recname}')
    return np.load(path, allow_pickle=True).item()

def _get_clu_idx(clu_key):
    """
    get index into spike_maps from clu name.
    assumes naming like '... cluXX ...' and the '-2' offset used in your other scripts.
    """
    if isinstance(clu_key, str) and 'clu' in clu_key:
        return int(clu_key.split('clu')[1].split(' ')[0]) - 2
    # fallback: if keys are already ints (rare), assume they map directly
    if isinstance(clu_key, (int, np.integer)):
        return int(clu_key)
    raise ValueError(f'cannot parse clu index from key: {clu_key}')


#%% load data
cell_prop = pd.read_pickle(r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')

tagged_keys, putative_keys = [], []
for clu in cell_prop.itertuples():
    if clu.identity == 'tagged':
        tagged_keys.append(clu.Index)
    if clu.identity == 'putative':
        putative_keys.append(clu.Index)


#%% main loop
all_ctrl_amps = []
all_stim_amps = []

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    txtpath = mice_exp_stem / f'ANMD{recname[1:4]}r' / recname[:-3] / recname / f'{recname}T.txt'
    beh = process_txt(txtpath)

    # stim condition per trial
    stim_cds = [trial[15] for trial in beh['trial_statements']][1:]
    stim_idx = [ti for ti, cond in enumerate(stim_cds) if cond != '0']
    ctrl_idx = [ti + 2 for ti in stim_idx]

    # raw spike maps (cell x time in spike clock)
    spike_maps, clu_ids, max_time = _load_spike_map(
        recname,
        mice_exp_stem,
        samp_freq=SAMP_FREQ
    )

    # -------------
    # RE-ALIGNMENT
    # -------------
    speed_times = beh['speed_times'][1:]
    run_onset_online = detect_run_onsets_teensy(speed_times)
    run_onset_online = [run for ti, run in enumerate(run_onset_online) if ti in ctrl_idx]
    run_onset_online, _ = _correct_overflow(run_onset_online)

    pulse_times = np.array(beh['pulse_times'])
    diffs = np.diff(pulse_times)
    split_idx = np.where(diffs >= 1000)[0] + 1
    pulse_trains = np.split(pulse_times, split_idx)
    stim_times = [pt[0] for pt in pulse_trains]  # first pulse of each train
    stim_times, _ = _correct_overflow(stim_times)

    rec_stem = Path(f'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}') / recname[:14] / recname
    cue_path = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignCue_msess1.mat'
    aligned_cue = sio.loadmat(cue_path)['trialsCue'][0][0]

    cue_spike_time  = aligned_cue['startLfpInd'][0]
    cue_teensy_time = [trial[0][0] if trial else np.nan for trial in beh['movie_times']]
    cue_teensy_time, correction_time = _correct_overflow(cue_teensy_time)

    a, b = np.polyfit(cue_teensy_time, cue_spike_time, 1)

    def _teensy_to_spike(t_teensy):
        return int(a * t_teensy + b)

    # IMPORTANT: force-correct stim/run times if cue overflow happened before those events
    if correction_time is not None:
        if len(stim_idx) > 0 and correction_time <= stim_idx[0]:
            stim_times = [t + OF_CONSTANT for t in stim_times]
        if len(ctrl_idx) > 0 and correction_time <= ctrl_idx[0]:
            run_onset_online = [t + OF_CONSTANT for t in run_onset_online]

    stim_onset_spike = [_teensy_to_spike(t) for t in stim_times]
    run_onset_spike  = [_teensy_to_spike(t) for t in run_onset_online]
    # ------------------
    # RE-ALIGNMENT ENDS
    # ------------------


    # session subset if recname exists as a column; otherwise process all keys
    if 'recname' in cell_prop.columns:
        curr_cells = cell_prop[cell_prop['recname'] == recname]
        clu_iter = list(curr_cells.index)
    else:
        clu_iter = list(cell_prop.index)

    for clu in clu_iter:
        if (clu not in tagged_keys) and (clu not in putative_keys):
            continue

        clu_idx = _get_clu_idx(clu)
        if clu_idx < 0 or clu_idx >= spike_maps.shape[0]:
            continue

        # load run-aligned
        rasters_run_disk = _load_run_rasters(recname, all_session_stem)
        cluname = clu
        if cluname not in rasters_run_disk:
            continue
        run_aligned = rasters_run_disk[cluname]

        # load stim-aligned
        spike_map = spike_maps[clu_idx, :]
        stim_aligned = _align_trials(spike_map, stim_onset_spike, max_time)

        if run_aligned.size == 0 or stim_aligned.size == 0:
            continue

        ctrl_mean = np.nanmean(run_aligned, axis=0)
        stim_mean = np.nanmean(stim_aligned, axis=0)
        ctrl_prof = fftconvolve(ctrl_mean, GAUS_SPIKE, mode='same') * SAMP_FREQ
        stim_prof = fftconvolve(stim_mean, GAUS_SPIKE, mode='same') * SAMP_FREQ

        # collect for summary stats
        all_ctrl_amps.append(np.nanmean(ctrl_prof[AMP_WINDOW_LOW:AMP_WINDOW_HIGH]))
        all_stim_amps.append(np.nanmean(stim_prof[AMP_WINDOW_LOW:AMP_WINDOW_HIGH]))


        # PLOTTING
        # use stim range for BOTH twin axes
        ymax = max(np.nanmax(stim_prof), np.nanmax(ctrl_prof)) * 1.05

        fig, axs = plt.subplots(2, 1, figsize=(2.6, 3.2), sharex=True)

        # show single-trial matrices (since we no longer use pre-run-aligned rasters)
        vmin = np.nanmin([run_aligned.min(), stim_aligned.min()])
        vmax = np.nanmax([run_aligned.max(), stim_aligned.max()])

        stim_rows, stim_cols = np.where(stim_aligned > 0)

        axs[0].scatter(
            stim_cols / SAMP_FREQ - BEF,
            stim_rows + 1,
            s=1,
            color='lightsteelblue',
            ec='none',
            rasterized=True
        )
        
        ctrl_rows, ctrl_cols = np.where(run_aligned > 0)

        axs[1].scatter(
            ctrl_cols / SAMP_FREQ - BEF,
            ctrl_rows + 1,
            s=1,
            color='grey',
            ec='none',
            rasterized=True
        )

        axs[0].set(title=f'{clu}\nStim.', ylabel='Trial #')
        axs[1].set(title='Ctrl.', ylabel='Trial #', xlabel='Time from run onset (s)')

        # overlays (same y scale)
        axt0 = axs[0].twinx()
        axt0.plot(XAXIS, stim_prof, color='royalblue', lw=1)
        axt0.set_ylim(0, ymax)
        axt0.set(ylabel='Firing rate (Hz)')
        axt0.spines['top'].set_visible(False)
        
        axt1 = axs[1].twinx()
        axt1.plot(XAXIS, ctrl_prof, color='k', lw=1)
        axt1.set_ylim(0, ymax)
        axt1.set(ylabel='Firing rate (Hz)')
        axt1.spines['top'].set_visible(False)

        for ax in axs:
            ax.set(xlim=(-1, 4), xticks=(0, 2, 4))

        for ax in axs:
            ax.spines['top'].set_visible(False)
        axt0.spines['top'].set_visible(False)
        axt1.spines['top'].set_visible(False)

        fig.tight_layout()

        tag = 'tagged' if clu in tagged_keys else 'putative'
        for ext in ['.png', '.pdf']:
            fig.savefig(
                stim_raster_stem / f'{clu}_{tag}{ext}',
                dpi=300,
                bbox_inches='tight'
            )
        plt.close(fig)


#%% summary stats
plot_violin_with_scatter(
    all_ctrl_amps,
    all_stim_amps,
    'grey',
    'royalblue',
    ylabel='Firing rate (Hz)',
    xticklabels=['Ctrl.', 'Stim.'],
    print_statistics=True,
    save=True,
    savepath='Z:/Dinghao/code_dinghao/LC_ephys/stim_effects/tagged_putative_ctrl_stim_violin'
)
