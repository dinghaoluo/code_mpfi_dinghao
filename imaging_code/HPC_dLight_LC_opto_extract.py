# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:02:49 2025
Modified on Tue 24 June 16:24:15 2025
Modified on Tue 25 Nov 2025

extract opto-LC stimulation + dLight imaging data 
modification notes:
    - 24 June 2025: removed pixel-wise dFF calculation and replaced it with a 
        simplified dFF (stim. / baseline for raw F) that is easy to compute 
        and produces the exact same desideratum (the release map)
    - 25 Nov 2025: changed stim.-alignment method to be detection based on 
        channel 2 (more stable than before)

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
import tifffile

import behaviour_functions as bf 
import imaging_pipeline_functions as ipf
from plotting_functions import plot_violin_with_scatter, add_scale_bar
from common import mpl_formatting, get_GPU_availability
mpl_formatting()

import rec_list
paths = rec_list.pathdLightLCOpto + \
        rec_list.pathdLightLCOptoCtrl + \
        rec_list.pathdLightLCOptoInh + \
        rec_list.pathdLightLCOptoDbhBlock
        
# GPU acceleration
cp, GPU_AVAILABLE = get_GPU_availability()


#%% parameters 
SAMP_FREQ = 30

# post-stim dispersion calculation
BIN_WIDTH = 0.1

# path stems 
mice_exp_stem = Path(r'Z:\Dinghao\MiceExp')
all_sess_stem = Path(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions')


#%% main 
def main(path):
    recname = Path(path).name
    print(f'\n{recname}')
    
    plane_stem = Path(path) / 'suite2p/plane0'
    sessname = recname.replace('i', '')
    
    binpath = plane_stem / 'data.bin'
    bin2path = plane_stem / 'data_chan2.bin'
    opspath = plane_stem / 'ops.npy'
    txtpath = mice_exp_stem / f'ANMD{recname[1:4]}' / f'{sessname}T.txt'
    
    whether_ctrl = '_ctrl' if path in rec_list.pathdLightLCOptoCtrl else ''
    savepath = all_sess_stem / f'{recname}{whether_ctrl}'
    savepath.mkdir(exist_ok=True)
    
    # # check for repeated processing 
    # if ((savepath / f'processed_data/{recname}_pixel_dFF_stim.npy').exists() and
    #     (savepath / f'processed_data/{recname}_pixel_dFF_ch2_stim.npy').exists()):
    #     print(f'processed... skipping {recname}')
    #     return
    
    # load data 
    ops = np.load(opspath, allow_pickle=True).item()
    tot_frames = ops['nframes']
    shape = tot_frames, ops['Ly'], ops['Lx']
    
    print('loading movies and saving references...')
    mov = np.memmap(binpath, mode='r', dtype='int16', shape=shape).astype(np.float32)
    mov2 = np.memmap(bin2path, mode='r', dtype='int16', shape=shape).astype(np.float32)
    
    tot_frames = mov.shape[0]  # once loaded, update tot_frames to be the max frame number, 16 June 2025
    
    
    ref = ipf.plot_reference(mov, recname=recname, outpath=savepath, channel=1)
    ref2 = ipf.plot_reference(mov2, recname=recname, outpath=savepath, channel=2)
    
    # raw traces, which now replace the trace_dFF we used before
    # due to concerns of dFF baselines covering the stim (PMT-off) period
    # 5 Aug 2025
    raw_trace = np.sum(mov, axis=(1,2))
    raw_trace2 = np.sum(mov2, axis=(1,2))
    
    # dFF traces are ONLY used for▲ plotting figure 1 now
    print('computing dFF traces...')
    trace_dFF = ipf.calculate_dFF(raw_trace, sigma=300, t_axis=0,
                                  GPU_AVAILABLE=GPU_AVAILABLE)
    trace2_dFF = ipf.calculate_dFF(raw_trace2, sigma=300, t_axis=0,
                                   GPU_AVAILABLE=GPU_AVAILABLE)
    
    
    # ---------------------------
    # determine stim. timestamps
    # ---------------------------
    # we now use as the primary method derivatives of channel 2 signals to 
    #   determine where step changes happened (both up and down), and only fall
    #   back to the text files for timestamps if this fails 
    
    ## we first read the .txt file to figure out how long the pulse trains are 
    ##  in this recording and to plot the example pulse trace 
    print('retrieving stim. parameters...')
    txt = ipf.process_txt_nobeh(txtpath)
    frame_times = txt['frame_times']
    pulse_times = np.array(txt['pulse_times'])
    pulse_params = txt['pulse_parameters'][-1]  # final set of stim params
    
    pulse_width_ON = float(pulse_params[2]) / 1e6  # s
    pulse_width    = float(pulse_params[3]) / 1e6  # s
    pulse_number   = int(pulse_params[4])
    taper_enabled  = int(pulse_params[7])
    taper_raw      = int(pulse_params[8])
    taper_duration = taper_raw if taper_raw > 1000 else 0
    
    duty_cycle = f'{int(round(100 * pulse_width_ON / pulse_width))}%'
    
    # split by >=1 s gaps (1000 ms) since pulse trains are always separated 
    #   at least by a few seconds
    pulse_diffs = np.diff(pulse_times)
    split_idx = np.where(pulse_diffs >= 1000)[0] + 1
    pulse_trains = np.split(pulse_times, split_idx)
    
    # compute total train duration (ms)
    eg_train = pulse_trains[0]
    t0 = eg_train[0]
    train_rel = eg_train - t0                       # ms
    total_train_duration_ms = round(train_rel[-1] + pulse_width*1000)
    
    # convert all of this to frames
    total_train_duration_s = total_train_duration_ms / 1000
    total_train_duration_frames = int(total_train_duration_s * SAMP_FREQ)
    
    # use 2 × train duration as min interval
    min_interval_frames = 2 * total_train_duration_frames
    if min_interval_frames < 15:
        min_interval_frames = 15  # hard safety floor
        
    ## now we detect stim onsets by step changes in trace2_dFF
    detected_onsets_raw, detected_offsets_raw = ipf.detect_step_pairs(
        trace2_dFF, 
        zthr=100,
        min_interval_frames=min_interval_frames
        )
    detected_onsets  = [f - 1 for f in detected_onsets_raw]  # 1-frame buffer for envelope 
    detected_offsets = [f + 1 for f in detected_offsets_raw]  # same as above
    detected_stim_durations = [off-on for off, on in zip(detected_offsets, detected_onsets)]
    max_stim_duration_s = max(detected_stim_durations) / SAMP_FREQ
    
    detection_printout = (f'''detected based on channel 2:
        {len(detected_onsets)} candidate stim. onset-offset pairs
        max stim. duration: {max(detected_stim_durations)} frames ({max_stim_duration_s} s)''')
    print(detection_printout)
    
    ## now we either use detected_onsets or when that fails fall back to txt
    if len(detected_onsets) == 0:
        print('no detected onset-offset pairs; falling back to .txt pulse_times')
        pulse_frames = [
            [ipf.find_nearest(p, frame_times) for p in train]
            for train in pulse_trains
        ]
        detected_onsets  = [pf[0] for pf in pulse_frames]
        detected_offsets = [pf[-1] for pf in pulse_frames]
    if len(detected_onsets) != len(pulse_trains):
        print('\n\n\n\n******')
        print('WARNING: detection mismatch with text log')
        print(f'    detected {len(detected_onsets)} trains; text log shows {len(pulse_trains)} trains')
        print('******\n\n\n\n')
        
    ## -- PARAMETER DEFINITIONS
    # now defined within the function scope, since we reassign them later in an if statement
    BEF = 2
    AFT = 10
    TAXIS = np.arange(-BEF*SAMP_FREQ, AFT*SAMP_FREQ) / SAMP_FREQ
    BASELINE_IDX = (TAXIS >= -1.0) & (TAXIS < 0)
    STIM_IDX = (TAXIS > max_stim_duration_s) & (TAXIS <= max_stim_duration_s + 1.0)  # note that this is the mask for extracting stim_mean
    ## -- END PARAMETER DEFINITIONS
    
    ## finally we filter out valid start frames
    valid_pulse_start_frames = [
        on for on, off in zip(detected_onsets, detected_offsets)
        if on - BEF*SAMP_FREQ >= 0 and off + AFT*SAMP_FREQ <= tot_frames
    ]
    tot_valid_pulses = len(valid_pulse_start_frames)
    print(f'valid stim. events for alignment: {tot_valid_pulses}')
    
    # post-stim dispersion calculation, 10 Sept 2025
    BIN_START = max_stim_duration_s
    BIN_END   = max_stim_duration_s + 4
    bin_edges = np.arange(BIN_START, BIN_END + BIN_WIDTH, BIN_WIDTH)
    n_bins = len(bin_edges) - 1
    # ---------------------------
    # end
    # ---------------------------
    
    
    # pulse processing 
    print('extracting data...')

    # checks $FM against tot_frame
    if tot_frames<len(frame_times)-3 or tot_frames>len(frame_times):
        Exception('\nWARNING:\ncheck $FM; halting processing for {}\n'.format(recname))
    
    # filter for opto artefact periods 
    pulse_period_frames = np.concatenate([
        np.arange(on, off+1) for on, off in zip(detected_onsets, detected_offsets)
        ])
    
    # filtering
    raw_trace[pulse_period_frames]  = np.nan
    raw_trace2[pulse_period_frames] = np.nan
    
    trace_dFF[pulse_period_frames]  = np.nan
    trace2_dFF[pulse_period_frames] = np.nan
    
    # raw traces aligned
    raw_aligned  = np.zeros((tot_valid_pulses, (BEF+AFT)*SAMP_FREQ), dtype=np.float32)
    raw2_aligned = np.zeros((tot_valid_pulses, (BEF+AFT)*SAMP_FREQ), dtype=np.float32)
    for i, p in enumerate(valid_pulse_start_frames):
        start = p - BEF * SAMP_FREQ
        end   = p + AFT * SAMP_FREQ
        raw_aligned[i, :]  = raw_trace[start:end]
        raw2_aligned[i, :] = raw_trace2[start:end]
    
    # dFF traces aligned 
    trace_dFF_aligned = np.zeros((tot_valid_pulses, (BEF+AFT)*SAMP_FREQ), dtype=np.float32)
    trace2_dFF_aligned = np.zeros((tot_valid_pulses, (BEF+AFT)*SAMP_FREQ), dtype=np.float32)
    for i, p in enumerate(valid_pulse_start_frames):
        start = p - BEF * SAMP_FREQ
        end   = p + AFT * SAMP_FREQ
        trace_dFF_aligned[i, :] = trace_dFF[start:end]
        trace2_dFF_aligned[i, :] = trace2_dFF[start:end]
    trace_dFF_aligned_mean = np.mean(trace_dFF_aligned, axis=0)
    trace2_dFF_aligned_mean = np.mean(trace2_dFF_aligned, axis=0)
    
    # calculate ratios
    # per‐trial raw means
    baseline_raw = np.nanmean(raw_aligned[:,  BASELINE_IDX], axis=1)
    stim_raw = np.nanmean(raw_aligned[:,  STIM_IDX], axis=1)
    baseline2_raw = np.nanmean(raw2_aligned[:, BASELINE_IDX], axis=1)
    stim2_raw = np.nanmean(raw2_aligned[:, STIM_IDX], axis=1)
    
    # per‐trial ΔF/F exactly like pixel dFF (stim − base) / |base|
    dFF = (stim_raw - baseline_raw) / np.abs(baseline_raw)
    dFF2 = (stim2_raw - baseline2_raw) / np.abs(baseline2_raw)
    
    # dFF comp
    baseline_dFF = np.nanmean(trace_dFF_aligned[:,  BASELINE_IDX], axis=1)
    stim_dFF = np.nanmean(trace_dFF_aligned[:,  STIM_IDX], axis=1)
    baseline2_dFF = np.nanmean(trace2_dFF_aligned[:, BASELINE_IDX], axis=1)
    stim2_dFF = np.nanmean(trace2_dFF_aligned[:, STIM_IDX], axis=1)
    
    # for plotting 
    ymin = np.nanmin(trace_dFF_aligned.T)
    ymin2 = np.nanmin(trace2_dFF_aligned.T)
    
    # plotting 
    fig, axs = plt.subplots(3,1,figsize=(3.5,5),
                            sharex=True)
    
    axs[0].plot(TAXIS, trace_dFF_aligned_mean, color='green', linewidth=2)
    axs[0].plot(TAXIS, trace_dFF_aligned.T, color='green', alpha=.05)
    axs[0].text(-2, ymin-.01, 'dLight', ha='left', va='center', color='green', fontsize=10)
    
    axs[1].plot(TAXIS, trace2_dFF_aligned_mean, color='darkred', linewidth=2)
    axs[1].plot(TAXIS, trace2_dFF_aligned.T, color='darkred', alpha=.05)
    axs[1].text(-2, ymin2-.01, 'red ctrl.', ha='left', va='center', color='darkred', fontsize=10)
    
    add_scale_bar(axs[0], x_start=8.5, y_start=ymin-.015, x_len=1, y_len=.02)    
    add_scale_bar(axs[1], x_start=8.5, y_start=ymin2-.015, x_len=1, y_len=.02)
    axs[1].text(8.93, ymin2-.02, '1 s', ha='center', va='top', fontsize=8)
    axs[1].text(8.45, ymin2-.005, '2% ΔF/F', ha='right', va='center', rotation='vertical', fontsize=8)
    
    stim_trace = np.zeros_like(TAXIS)
    for p in range(pulse_number):
        stim_onset = BEF + p * (pulse_width)
        stim_offset = stim_onset + pulse_width_ON
        stim_onset_idx = int(stim_onset * SAMP_FREQ)
        stim_offset_idx = int(stim_offset * SAMP_FREQ)
        stim_trace[stim_onset_idx:stim_offset_idx] = 1
        
    if taper_enabled and taper_duration > 0:
        taper_start_idx = stim_offset_idx
        taper_len = int((taper_duration / 1_000_000) * SAMP_FREQ)  # convert to seconds then to samples (to align to frames)
        taper = np.linspace(1,0, taper_len, endpoint=True)
        stim_trace[taper_start_idx : taper_start_idx+taper_len] = taper[:len(stim_trace) - taper_start_idx]
        
    axs[2].plot(TAXIS, stim_trace, color='k', linewidth=1)
    
    axs[2].set_ylim(0, 2)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ['top', 'right', 'left', 'bottom']:
            ax.spines[s].set_visible(False)
    
    if taper_enabled:
        title_str = (
            f'{recname}\n'
            f'duty cycle = {duty_cycle}\n'
            f'pulse width = {pulse_width} s\n'
            f'pulse(s) per pulse train = {pulse_number}\n'
            f'total pulse trains = {len(detected_onsets)}\n'
            f'taper duration = {taper_duration}'
            )
    else:
        title_str = (
            f'{recname}\n'
            f'duty cycle = {duty_cycle}\n'
            f'pulse width = {pulse_width} s\n'
            f'pulse(s) per pulse train = {pulse_number}\n'
            f'total pulse trains = {len(detected_onsets)}\n'
            )
    
    fig.suptitle(title_str)
    fig.tight_layout()

    for ext in ['.png', '.pdf']:
        fig.savefig(
            savepath / f'{recname}_aligned_stim{ext}',
            dpi=300,
            bbox_inches='tight'
            )
        
    # saving 
    np.save(savepath / f'processed_data/{recname}_wholefield_dFF_stim.npy',
            trace_dFF_aligned_mean)
    np.save(savepath / f'processed_data/{recname}_wholefield_dFF2_stim.npy',
            trace2_dFF_aligned_mean)
        
    # statistics and plotting 
    # clean both arrays before plotting
    valid_mask = (~np.isnan(dFF)) & (~np.isnan(dFF2))
    dFF_valid = dFF[valid_mask]
    dFF2_valid = dFF2[valid_mask]
    plot_violin_with_scatter(
        dFF2_valid, dFF_valid, 
        'darkred', 'green',
        xticklabels=['ref.', 'dLight'],
        ylabel='ΔF/F',
        title=recname,
        save=True,
        savepath=savepath / f'{recname}_dFF_dFF2_violinplot'
        )
    
    baseline_dFF, stim_dFF = map(list, zip(*[
        (b, s) for b, s in zip(baseline_dFF, stim_dFF)
        if not np.isnan(b) and not np.isnan(s)
    ]))
    baseline2_dFF, stim2_dFF = map(list, zip(*[
        (b, s) for b, s in zip(baseline2_dFF, stim2_dFF)
        if not np.isnan(b) and not np.isnan(s)
    ]))
    plot_violin_with_scatter(
        baseline_dFF, stim_dFF, 
        '#8CA082', 'green',
        xticklabels=['baseline', 'stim.'],
        ylabel='ΔF/F',
        title=recname,
        save=True,
        savepath=savepath / f'{recname}_baseline_stim_violinplot'
        )
    plot_violin_with_scatter(
        baseline2_dFF, stim2_dFF, 
        '#8C6464', 'darkred',
        xticklabels=['baseline\nch2', 'stim.\nch2'],
        ylabel='ΔF/F',
        title=recname,
        save=True,
        savepath=savepath / f'{recname}_baseline_stim_ch2_violinplot'
        )
    
    ## pixel-wise extraction 
    # spatial smoothing
    print('performing spatial filtering...')
    mov = ipf.spatial_gaussian_filter(mov, sigma_spatial=1,
                                      GPU_AVAILABLE=GPU_AVAILABLE,
                                      CHUNK=True)
    mov2 = ipf.spatial_gaussian_filter(mov2, sigma_spatial=1,
                                       GPU_AVAILABLE=GPU_AVAILABLE,
                                       CHUNK=True)
    
    # compute dF/F per pixel (stim. / baseline for raw F), 24 June 2025
    pixel_dFF = np.zeros((shape[1], shape[2], len(valid_pulse_start_frames)))
    pixel_dFF2 = np.zeros_like(pixel_dFF)
    
    # we still want F aligned
    pixel_F_aligned = np.zeros((len(valid_pulse_start_frames), 
                                ((BEF+AFT) * SAMP_FREQ),
                                shape[1], 
                                shape[2]))
    pixel_F2_aligned = np.zeros_like(pixel_F_aligned)
    
    for i, p in enumerate(valid_pulse_start_frames):
        temp_F = mov[p - BEF * SAMP_FREQ : p + AFT * SAMP_FREQ, :, :]
        temp_F2 = mov2[p - BEF * SAMP_FREQ : p + AFT * SAMP_FREQ, :, :]
        
        # save F aligned too
        pixel_F_aligned[i, :, :, :] = temp_F
        pixel_F2_aligned[i, :, :, :] = temp_F2
    
        stim_mean = np.mean(temp_F[STIM_IDX, :, :], axis=0)
        baseline_mean = np.mean(temp_F[BASELINE_IDX, :, :], axis=0)
        dFF = (stim_mean - baseline_mean) / np.abs(baseline_mean)
        dFF[np.abs(dFF) > 10] = np.nan  # hard cap
        pixel_dFF[:, :, i] = dFF
    
        stim_mean2 = np.mean(temp_F2[STIM_IDX, :, :], axis=0)
        baseline_mean2 = np.mean(temp_F2[BASELINE_IDX, :, :], axis=0)
        dFF2 = (stim_mean2 - baseline_mean2) / np.abs(baseline_mean2)
        dFF2[np.abs(dFF2) > 10] = np.nan
        pixel_dFF2[:, :, i] = dFF2
        
    np.save(savepath / f'processed_data/{recname}_pixel_dFF_stim.npy',
            pixel_dFF)
    np.save(savepath / f'processed_data/{recname}_pixel_dFF_ch2_stim.npy',
            pixel_dFF2)
    
    # save F aligned 
    np.save(savepath / f'processed_data/{recname}_pixel_F_aligned.npy',
            pixel_F_aligned)
    np.save(savepath / f'processed_data/{recname}_pixel_F2_aligned.npy',
            pixel_F2_aligned)
    
    # generate mean dFF release map as proxy for t-map, 24 June 2025 
    print('computing mean release map...')
    release_map = np.nanmean(pixel_dFF, axis=2)  # shape: (y, x)
    release_map2 = np.nanmean(pixel_dFF2, axis=2)
    
    # save map matrices 
    np.save(savepath / f'processed_data/{recname}_release_map.npy', release_map)
    np.save(savepath / f'processed_data/{recname}_release_map_ch2.npy', release_map2)

    ## plotting - release map ch 1 
    vmin = np.nanpercentile(release_map, 1)
    vmax = np.nanpercentile(release_map, 99)
    
    # edge case check
    if vmin >= 0:
        vmin = -.001
    if vmax <= 0:
        vmax = .001
        
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # panel 1: channel 1 reference
    axs[0].imshow(ref, cmap='gray', interpolation='none')
    axs[0].set_title('channel 1', fontsize=10)
    axs[0].axis('off')
    
    # panel 2: channel 2 reference
    axs[1].imshow(ref2, cmap='gray', interpolation='none')
    axs[1].set_title('channel 2', fontsize=10)
    axs[1].axis('off')
    
    # panel 3: release map heatmap
    im = axs[2].imshow(release_map, cmap='RdBu_r', norm=norm, interpolation='none')
    axs[2].set_title('CH1: stim / baseline (mean)', fontsize=10)
    axs[2].axis('off')
    
    # colourbar for panel 3
    cbar = fig.colorbar(im, ax=axs[2], shrink=0.8, fraction=0.046, pad=0.04)
    cbar.set_label('ΔF/F ratio', fontsize=10)
    cbar.set_ticks([vmin, 0, vmax])
    
    fig.tight_layout()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            savepath / f'{recname}_release_map{ext}',
            dpi=300,
            bbox_inches='tight'
        )
        
    # save as tiff 
    cmap = colormaps['RdBu_r']
    release_map_rgba = cmap(norm(release_map))
    release_map_rgb = (release_map_rgba[..., :3] * 255).astype(np.uint8)
    
    tifffile.imwrite(savepath / f'{recname}_release_map.tiff',
                     release_map_rgb)
    
    
    ## plotting - release map ch 2
    vmin = np.nanpercentile(release_map2, 1)
    vmax = np.nanpercentile(release_map2, 99)
    
    # edge case check
    if vmin >= 0:
        vmin = -.001
    if vmax <= 0:
        vmax = .001
        
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # panel 1: channel 1 reference
    axs[0].imshow(ref, cmap='gray', interpolation='none')
    axs[0].set_title('channel 1', fontsize=10)
    axs[0].axis('off')
    
    # panel 2: channel 2 reference
    axs[1].imshow(ref2, cmap='gray', interpolation='none')
    axs[1].set_title('channel 2', fontsize=10)
    axs[1].axis('off')
    
    # panel 3: release map heatmap
    im = axs[2].imshow(release_map2, cmap='RdBu_r', norm=norm, interpolation='none')
    axs[2].set_title('CH2: stim / baseline (mean)', fontsize=10)
    axs[2].axis('off')
    
    # colourbar for panel 3
    cbar = fig.colorbar(im, ax=axs[2], shrink=0.8, fraction=0.046, pad=0.04)
    cbar.set_label('ΔF/F ratio', fontsize=10)
    cbar.set_ticks([vmin, 0, vmax])
    
    fig.tight_layout()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            savepath / f'{recname}_release_map_ch2{ext}',
            dpi=300,
            bbox_inches='tight'
        )
    
    # save as tiff 
    cmap = colormaps['RdBu_r']
    release_map2_rgba = cmap(norm(release_map2))
    release_map2_rgb = (release_map2_rgba[..., :3] * 255).astype(np.uint8)
    
    tifffile.imwrite(savepath / f'{recname}_release_map_ch2.tiff',
                     release_map2_rgb)
    
    # dispersion rate calculation, 10 Sept 2025 
    print('calculating binned ratios (for dispersion rate analysis)...')
    all_bins_ch1 = []
    all_bins_ch2 = []
    
    for i, p in enumerate(valid_pulse_start_frames):
        temp_F = mov[p - BEF * SAMP_FREQ : p + AFT * SAMP_FREQ, :, :]
        temp_F2 = mov2[p - BEF * SAMP_FREQ : p + AFT * SAMP_FREQ, :, :]
        
        trial_bins_ch1 = np.zeros((shape[1], shape[2], n_bins))
        trial_bins_ch2 = np.zeros_like(trial_bins_ch1)
    
        for b in range(n_bins):
            bin_mask = (TAXIS >= bin_edges[b]) & (TAXIS < bin_edges[b+1])
    
            stim_mean  = np.mean(temp_F[bin_mask, :, :], axis=0)
            baseline_mean = np.mean(temp_F[BASELINE_IDX, :, :], axis=0)
            dFF_bin = (stim_mean - baseline_mean) / np.abs(baseline_mean)
            dFF_bin[np.abs(dFF_bin) > 10] = np.nan
            trial_bins_ch1[:, :, b] = dFF_bin
    
            stim_mean2  = np.mean(temp_F2[bin_mask, :, :], axis=0)
            baseline_mean2 = np.mean(temp_F2[BASELINE_IDX, :, :], axis=0)
            dFF_bin2 = (stim_mean2 - baseline_mean2) / np.abs(baseline_mean2)
            dFF_bin2[np.abs(dFF_bin2) > 10] = np.nan
            trial_bins_ch2[:, :, b] = dFF_bin2
    
        all_bins_ch1.append(trial_bins_ch1)
        all_bins_ch2.append(trial_bins_ch2)
    
    # average across trials → final shape (y, x, n_bins)
    pixel_dFF_bins  = np.nanmean(np.stack(all_bins_ch1, axis=-1), axis=-1)
    pixel_dFF2_bins = np.nanmean(np.stack(all_bins_ch2, axis=-1), axis=-1)
    
    # save arrays
    np.save(savepath / f'processed_data/{recname}_pixel_dFF_bins.npy', pixel_dFF_bins)
    np.save(savepath / f'processed_data/{recname}_pixel_dFF_ch2_bins.npy', pixel_dFF2_bins)
    
    
    ## compute dF/F per pixel IF BEHAVIOUR (run-onset / baseline), 24 June 2025 
    if txt['behaviour']:
        print('behaviour session; compiling run-onset dFF dict...')
        txt = bf.process_behavioural_data_imaging(txtpath)
        run_onsets = txt['run_onset_frames']
        stim_conds = [t[15] for t in txt['trial_statements']]
        stim_idx = [trial for trial, cond in enumerate(stim_conds)
                    if cond!='0']
        stim_idx_et = [trial + 1 for trial in stim_idx if trial + 1 < len(run_onsets)]
        
        # new axes for run 
        BASELINE_IDX_RUN = (TAXIS >= -1.0) & (TAXIS <= -0.15)
        RUN_IDX = (TAXIS >= 0.15) & (TAXIS <= 1.0)
        
        run_onsets = [f for trial, f in enumerate(run_onsets)
                      if not np.isnan(f) 
                      and trial not in stim_idx and trial not in stim_idx_et
                      and f > BEF*SAMP_FREQ
                      and f < tot_frames - AFT*SAMP_FREQ]
        
        pixel_dFF_run = np.zeros((shape[1], shape[2], len(run_onsets)))
        pixel_dFF_run2 = np.zeros_like(pixel_dFF_run)
        for i, f in enumerate(run_onsets):
            temp_F = mov[f-BEF*SAMP_FREQ : f+AFT*SAMP_FREQ, :, :]  # do it for all pixels simultaneously
            temp_F2 = mov2[f-BEF*SAMP_FREQ : f+AFT*SAMP_FREQ, :, :]
            
            run_mean = np.mean(temp_F[RUN_IDX, :, :], axis=0)
            prerun_mean = np.mean(temp_F[BASELINE_IDX_RUN, :, :], axis=0)
            dFF_run = (run_mean - prerun_mean) / np.abs(prerun_mean)
            dFF_run[np.abs(dFF_run) > 10] = np.nan
            pixel_dFF_run[:, :, i] = dFF_run
            
            run_mean2 = np.mean(temp_F2[RUN_IDX, :, :], axis=0)
            prerun_mean2 = np.mean(temp_F2[BASELINE_IDX_RUN, :, :], axis=0)
            dFF_run2 = (run_mean2 - prerun_mean2) / np.abs(prerun_mean2)
            dFF_run2[np.abs(dFF_run2) > 10] = np.nan
            pixel_dFF_run2[:, :, i] = dFF_run2
            
        np.save(savepath / f'processed_data/{recname}_pixel_dFF_run.npy',
                pixel_dFF)
        np.save(savepath / f'processed_data/{recname}_pixel_dFF_ch2_run.npy',
                pixel_dFF2)
        
        # compute mean run-onset aligned release maps
        print('computing mean run-onset release map...')
        release_map_run = np.nanmean(pixel_dFF_run, axis=2)
        release_map_run2 = np.nanmean(pixel_dFF_run2, axis=2)

        # save map matrices
        np.save(savepath / f'processed_data/{recname}_release_map_run.npy', release_map_run)
        np.save(savepath / f'processed_data/{recname}_release_map_run_ch2.npy', release_map_run2)

        ## plotting - run release map ch 2
        vmin = np.nanpercentile(release_map_run, 1)
        vmax = np.nanpercentile(release_map_run, 99)
        
        # edge case check
        if vmin >= 0:
            # no negatives: force vmin = 0, keep vmax
            vmin = -.01
        if vmax <= 0:
            vmax = .01
        
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        
        # plotting
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # panel 1: channel 1 reference
        axs[0].imshow(ref, cmap='gray', interpolation='none')
        axs[0].set_title('channel 1', fontsize=10)
        axs[0].axis('off')

        # panel 2: channel 2 reference
        axs[1].imshow(ref2, cmap='gray', interpolation='none')
        axs[1].set_title('channel 2', fontsize=10)
        axs[1].axis('off')

        # panel 3: run-aligned release map heatmap
        im = axs[2].imshow(release_map_run, cmap='RdBu_r', norm=norm, interpolation='none')
        axs[2].set_title('run-onset / baseline (mean)', fontsize=10)
        axs[2].axis('off')

        cbar = fig.colorbar(im, ax=axs[2], shrink=0.8, fraction=0.046, pad=0.04)
        cbar.set_label('ΔF/F ratio', fontsize=10)
        cbar.set_ticks([vmin, 0, vmax])

        fig.tight_layout()

        for ext in ['.png', '.pdf']:
            fig.savefig(
                savepath / f'{recname}_release_map_run{ext}',
                dpi=300,
                bbox_inches='tight'
            )
        
            
        ## plotting - run release map ch 2
        vmin = np.nanpercentile(release_map_run2, 1)
        vmax = np.nanpercentile(release_map_run2, 99)
        
        # edge case check
        if vmin >= 0:
            vmin = -.001
        if vmax <= 0:
            vmax = .001
            
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
            
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        # panel 1: channel 1 reference
        axs[0].imshow(ref, cmap='gray', interpolation='none')
        axs[0].set_title('channel 1', fontsize=10)
        axs[0].axis('off')
        
        # panel 2: channel 2 reference
        axs[1].imshow(ref2, cmap='gray', interpolation='none')
        axs[1].set_title('channel 2', fontsize=10)
        axs[1].axis('off')
        
        # panel 3: release map heatmap
        im = axs[2].imshow(release_map_run2, cmap='RdBu_r', norm=norm, interpolation='none')
        axs[2].set_title('stim / baseline (mean)', fontsize=10)
        axs[2].axis('off')
        
        # colourbar for panel 3
        cbar = fig.colorbar(im, ax=axs[2], shrink=0.8, fraction=0.046, pad=0.04)
        cbar.set_label('ΔF/F ratio', fontsize=10)
        cbar.set_ticks([vmin, 0, vmax])
        
        fig.tight_layout()
        
        for ext in ['.png', '.pdf']:
            fig.savefig(
                savepath / f'{recname}_release_map_run_ch2{ext}',
                dpi=300,
                bbox_inches='tight'
            )
        
    else:
        print('session with no behaviour; finishing...')
    
    # axon-only imaging session check and processing 
    axon_only_folder = path + '_1100'
    if Path(axon_only_folder).exists():  # if we have a 1100-nm wavelength session 
        print(f'axon-only recording found: {axon_only_folder}')
    
        axon_only_bin_path = Path(axon_only_folder) / 'suite2p/plane0/data_chan2.bin'
        axon_only_ops_path = Path(axon_only_folder) / 'suite2p/plane0/ops.npy'

        if (axon_only_bin_path.exists() and 
            axon_only_ops_path.exists()):
            ops_axon_only = np.load(
                axon_only_ops_path, allow_pickle=True
                ).item()
            tot_frames_axon_only = ops_axon_only['nframes']
            shape_axon_only = (tot_frames_axon_only, 
                               ops_axon_only['Ly'], ops_axon_only['Lx'])
            
            print('loading axon-only movie...')
            mov_axon_only = np.memmap(
                axon_only_bin_path, 
                mode='r', dtype='int16', shape=shape_axon_only
                ).astype(np.float32)
        
            print('computing, plotting and saving axon-only reference..')
            reference_axon_only = np.mean(mov_axon_only, axis=0)
            reference_axon_only = ipf.post_processing_suite2p_gui(
                reference_axon_only
                )
            
            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(reference_axon_only, 
                      aspect='auto', cmap='gist_gray', interpolation='none',
                      extent=[0, 512, 512, 0])
            
            ax.set(xlim=(0,512), ylim=(0,512))

            fig.suptitle('ref 1100 nm')
            fig.tight_layout()
            fig.savefig(savepath / f'{recname}_ref_1100nm.png',
                        dpi=300,
                        bbox_inches='tight')
            
            np.save(savepath / f'processed_data/{recname}_ref_mat_1100nm.npy', 
                    reference_axon_only)
    

#%% execute 
if __name__ == '__main__':
    for path in paths:
        main(path)