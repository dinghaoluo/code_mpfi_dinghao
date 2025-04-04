# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:02:49 2025

extract opto-LC stimulation + dLight imaging data 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 
from scipy.stats import sem
from tqdm import tqdm

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
from plotting_functions import plot_violin_with_scatter, add_scale_bar
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathdLightLCOpto + rec_list.pathdLightLCOptoCtrl


#%% parameters 
SAMP_FREQ = 30

BEF = 2
AFT = 10

TAXIS = np.arange(-BEF*SAMP_FREQ, AFT*SAMP_FREQ) / SAMP_FREQ

BASELINE_IDX = (TAXIS >= -1.0) & (TAXIS <= -0.15)
STIM_IDX = (TAXIS >= 1.15) & (TAXIS < 2.0)


#%% GPU acceleration
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
    if GPU_AVAILABLE:
        print(
            'using GPU-acceleration with '
            f'{str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode("UTF-8"))} '
            'and CuPy')
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
    recname = path.split('\\')[-1]
    print(f'\n{recname}')
    
    binpath = os.path.join(path, 'suite2p/plane0/data.bin')
    bin2path = os.path.join(path, 'suite2p/plane0/data_chan2.bin')
    opspath = os.path.join(path, 'suite2p/plane0/ops.npy')
    txtpath = os.path.join(r'Z:\Dinghao\MiceExp',
                           f'ANMD{recname[1:4]}',
                           f'{recname[:4]}{recname[5:]}T.txt')
    
    # load data 
    ops = np.load(opspath, allow_pickle=True).item()
    tot_frames = ops['nframes']
    shape = tot_frames, ops['Ly'], ops['Lx']
    
    print('loading movies...')
    mov = np.memmap(binpath, mode='r', dtype='int16', shape=shape).astype(np.float32)
    mov2 = np.memmap(bin2path, mode='r', dtype='int16', shape=shape).astype(np.float32)
    
    trace = np.sum(mov, axis=(1,2))
    trace2 = np.sum(mov2, axis=(1,2))
    
    print('computing dFF traces...')
    trace_dFF = ipf.calculate_dFF(trace, sigma=300, t_axis=0,
                                  GPU_AVAILABLE=GPU_AVAILABLE)
    trace2_dFF = ipf.calculate_dFF(trace2, sigma=300, t_axis=0,
                                   GPU_AVAILABLE=GPU_AVAILABLE)
    
    print('processing .txt file...')
    txt = ipf.process_txt_nobeh(txtpath)
    frame_times = txt['frame_times']
    pulse_times = txt['pulse_times']
    pulse_parameters = txt['pulse_parameters']
    
    # pulse parameters 
    print('extracting data...')
    baseline = []; stim = []; stim2 = []
    
    pulse_width_ON = float(pulse_parameters[-1][2])/1000000  # in s
    pulse_width = float(pulse_parameters[-1][3])/1000000  # in s
    pulse_number = int(pulse_parameters[-1][4])
    duty_cycle = f'{int(round(100 * pulse_width_ON/pulse_width, 0))}%'
        
    tot_pulses = int(len(pulse_times) / pulse_number)
    
    pulse_frames = [ipf.find_nearest(p, frame_times) for p in pulse_times]
    pulse_frames = [pulse_frames[p*pulse_number : p*pulse_number+pulse_number] 
                    for p in range(tot_pulses)]
    pulse_start_frames = [p[0] for p in pulse_frames]

    # checks $FM against tot_frame
    if tot_frames<len(frame_times)-3 or tot_frames>len(frame_times):
        Exception('\nWARNING:\ncheck $FM; halting processing for {}\n'.format(recname))
    
    # block out opto artefact periods 
    pulse_period_frames = np.concatenate(
        [np.arange(
            pulse_frames[i][0]-3, 
            pulse_frames[i][0]+round(pulse_width*pulse_number*SAMP_FREQ) + 1  # +1 as a buffer
            )
        for i in range(tot_pulses)]
        )
    trace_dFF[pulse_period_frames] = np.nan
    trace2_dFF[pulse_period_frames] = np.nan
    
    # extract aligned traces
    trace_dFF_aligned = np.zeros((tot_pulses, (BEF+AFT)*SAMP_FREQ))
    trace2_dFF_aligned = np.zeros((tot_pulses, (BEF+AFT)*SAMP_FREQ))
    for i, p in tqdm(enumerate(pulse_start_frames),
                     desc='aligning whole-field dF/F traces to stimulations...'):
        # load traces 
        trace_dFF_aligned[i, :] = trace_dFF[p-(BEF)*SAMP_FREQ : p+(AFT)*SAMP_FREQ]
        trace2_dFF_aligned[i, :] = trace2_dFF[p-(BEF)*SAMP_FREQ : p+(AFT)*SAMP_FREQ]
        
    baseline_mean = np.mean(trace_dFF_aligned[:, BASELINE_IDX], axis=1)
    stim_mean = np.mean(trace_dFF_aligned[:, STIM_IDX], axis=1)
    baseline2_mean = np.mean(trace2_dFF_aligned[:, BASELINE_IDX], axis=1)
    stim2_mean = np.mean(trace2_dFF_aligned[:, STIM_IDX], axis=1)
    
    trace_dFF_aligned_mean = np.mean(trace_dFF_aligned, axis=0)
    trace_dFF_aligned_sem = sem(trace_dFF_aligned, axis=0)
    trace2_dFF_aligned_mean = np.mean(trace2_dFF_aligned, axis=0)
    trace2_dFF_aligned_sem = sem(trace2_dFF_aligned, axis=0)
    
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
    
    axs[2].plot(TAXIS, stim_trace, color='k', linewidth=1)
    
    axs[2].set_ylim(0, 2)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ['top', 'right', 'left', 'bottom']:
            ax.spines[s].set_visible(False)
            
    fig.suptitle(f'{recname}\n'
                 f'duty cycle = {duty_cycle}\n'
                 f'pulse width = {pulse_width} s\n'
                 f'pulse(s) per pulse train = {pulse_number}\n'
                 f'total pulse trains = {tot_pulses}')
    fig.tight_layout()
    
    whether_ctrl = '_ctrl' if path in rec_list.pathdLightLCOptoCtrl else ''
    savepath = os.path.join(
        r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions',
        f'{recname}{whether_ctrl}'
        )
    os.makedirs(savepath, exist_ok=True)
    for ext in ['.png', '.pdf']:
        fig.savefig(
            rf'{savepath}\{recname}_aligned_stim{ext}',
            dpi=300,
            bbox_inches='tight'
            )
        
    # statistics and plotting 
    plot_violin_with_scatter(
        baseline_mean, stim_mean, 
        '#8CA082', 'green',
        xticklabels=['baseline', 'stim.'],
        ylabel='ΔF/F',
        title=recname,
        save=True,
        savepath=rf'{savepath}\{recname}_baseline_stim_violinplot'
        )
    
    plot_violin_with_scatter(
        baseline2_mean, stim2_mean, 
        '#8C6464', 'darkred',
        xticklabels=['baseline\nch2', 'stim.\nch2'],
        ylabel='ΔF/F',
        title=recname,
        save=True,
        savepath=rf'{savepath}\{recname}_baseline_stim_ch2_violinplot'
        )
    
    # pixel-wise extraction only if not a ctrl 
    if path in rec_list.pathdLightLCOpto:
        print('performing spatial filtering...')
        mov_filtered = ipf.spatial_gaussian_filter(mov, sigma_spatial=1,
                                                   GPU_AVAILABLE=GPU_AVAILABLE)
        mov2_filtered = ipf.spatial_gaussian_filter(mov2, sigma_spatial=1,
                                                    GPU_AVAILABLE=GPU_AVAILABLE)
        
        print('computing single-pixel dF/F...')
        # compute dF/F per pixel
        dFF_pix = ipf.calculate_dFF(mov_filtered, sigma=300, t_axis=0, 
                                    GPU_AVAILABLE=GPU_AVAILABLE)
        dFF_pix2 = ipf.calculate_dFF(mov2_filtered, sigma=300, t_axis=0, 
                                     GPU_AVAILABLE=GPU_AVAILABLE)
        
        # mask artefact frames *before* alignment
        print('masking stimulation artefacts in pixel-wise dF/F...')
        dFF_pix[pulse_period_frames, :, :] = np.nan
        dFF_pix2[pulse_period_frames, :, :] = np.nan
        
        # extract aligned dF/F traces per pixel
        aligned_dFF_pix = np.zeros((tot_pulses, (BEF+AFT)*SAMP_FREQ, shape[1], shape[2]))
        aligned_dFF_pix2 = np.zeros((tot_pulses, (BEF+AFT)*SAMP_FREQ, shape[1], shape[2]))
        for i, p in tqdm(enumerate(pulse_start_frames),
                         desc='aligning pixel-wise dF/F traces to stimulations...'):
            aligned_dFF_pix[i] = dFF_pix[p-(BEF)*SAMP_FREQ : p+(AFT)*SAMP_FREQ]
            aligned_dFF_pix2[i] = dFF_pix2[p-(BEF)*SAMP_FREQ : p+(AFT)*SAMP_FREQ]
            
        np.save(rf'{savepath}\{recname}_pixelwise_dFF.npy',
                aligned_dFF_pix)
        np.save(rf'{savepath}\{recname}_pixelwise_dFF_ch2.npy',
                aligned_dFF_pix2)