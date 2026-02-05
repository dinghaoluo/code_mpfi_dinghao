# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:26:31 2024
Modified drastically in Feb 2025:
    - added GPU support and mostly fixed memory-leak problems
Modified on 19 Jan 2026

process and align ROI traces
***this script is specific to axon/dendrite GCaMP data processing, due to the 
   activity merging stage producing special data structures***

overview of the merged data structure:
    - stat.npy contains (per usual) info about the ROIs; AFTer merging and 
        appending the merges, each ROI has 'imerge' and 'inmerge':
        - 'imerge' tells one the constituent ROI(s) of a merged ROI and
        - 'inmerge' tells one of which ROI(s) the current ROI is a part
    - therefore, if one ROI conforms to (1) iscell==True and (2) inmerge<1,
        this ROI is considered valid (sorted)

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np
import pickle 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm
from time import time
from datetime import timedelta
from scipy.stats import sem 

import imaging_pipeline_functions as ipf
import support_LCHPC_axon as support

from common_functions import normalise, smooth_convolve, mpl_formatting, get_GPU_availability
mpl_formatting()

import rec_list
pathHPCLCGCaMP = rec_list.pathLCHPCGCaMP
pathHPCLCGCaMP = pathHPCLCGCaMP[1:]


#%% paths and parameters 
axon_GCaMP_stem  = Path('Z:/Dinghao/code_dinghao/LCHPC_axon_GCaMP')
all_session_stem = axon_GCaMP_stem / 'all_sessions'

beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LCHPCGCaMP')

SAMP_FREQ = 30
BEF = 3
AFT = 7  # in seconds
XAXIS = np.arange((BEF + AFT) * SAMP_FREQ) / SAMP_FREQ - BEF

# correction coefficient for neuropil subtraction 
CORR_COEF = 0.7


#%% GPU acceleration
cp, GPU_AVAILABLE, device_name = get_GPU_availability()
print(f'GPU_AVAILABLE={GPU_AVAILABLE}')


#%% functions
'''
all support functions have been migrated to utils\support_LCHPC_axon.py
'''


#%% processing function
def process_all(path):
    recname = Path(path).name
    print(f'\n{recname}')
    
    ops_path   = Path(path) / 'suite2p/plane0/ops.npy'
    bin_path   = Path(path) / 'suite2p/plane0/data.bin'
    bin2_path  = Path(path) / 'suite2p/plane0/data_chan2.bin'
    stat_path  = Path(path) / 'suite2p/plane0/stat.npy'
    F_path     = Path(path) / 'suite2p/plane0/F.npy'
    F2_path    = Path(path) / 'suite2p/plane0/F_chan2.npy'
    Fneu_path  = Path(path) / 'suite2p/plane0/Fneu.npy'
    F2neu_path = Path(path) / 'suite2p/plane0/Fneu_chan2.npy'
    
    # folder to put processed data and single-session plots 
    proc_path = all_session_stem / recname
    proc_path.mkdir(exist_ok=True)
    
    proc_data_path = proc_path / 'processed_data'
    proc_data_path.mkdir(exist_ok=True)
    
    # load files 
    stat = np.load(stat_path, allow_pickle=True)
    if 'inmerge' not in stat[0]: 
        print('No merging detected; skipped')  # detect merging
        return
    
    ops = np.load(ops_path, allow_pickle=True).item()
    
    tot_frames = ops['nframes']
    
    # behaviour file
    # load beh file 
    beh_path = beh_stem / f'{recname}.pkl'
    with open(beh_path, 'rb') as f:
        beh = pickle.load(f)
    
    run_frames = beh['run_onset_frames']
    rew_frames = beh['reward_frames']
    cue_frames = beh['start_cue_frames']
    
    # filtering
    run_frames = np.asarray(run_frames)
    run_frames = run_frames[run_frames != -1]  # filter out the no-clear-onset trials
    
    rew_frames = np.asarray(rew_frames)
    rew_frames = rew_frames[rew_frames != -1]  # filter out the no-rew trials
    
    cue_frames = np.asarray(cue_frames)
    
    # reference images
    ref_im, ref_ch2_im = ipf.load_or_generate_reference_images(
        proc_path,
        proc_data_path,
        bin_path, bin2_path,
        tot_frames, ops,
        GPU_AVAILABLE,
        VERBOSE=False
        )

    # filter valid ROIs: the end results should only contain end-merge ROIs and 
    # ROIs that have never been merged with other ROIs
    valid_rois_dict = support.filter_valid_rois(stat)
    valid_rois = set(valid_rois_dict)
    
    # 19 Mar 2025: we also want to process the constituent ROIs
    constituent_rois = {roi 
                        for sublist in valid_rois_dict.values() 
                        for roi in sublist}
    all_rois = valid_rois | constituent_rois

    # filtering through channel 2
    support.calculate_and_plot_overlap_indices(
        ref_im, ref_ch2_im, 
        stat, valid_rois, recname, proc_path
        )
    
    # ROI plots
    constituent_rois_coord_dict = support.get_roi_coord_dict(
        ref_im, ref_ch2_im, 
        stat, constituent_rois, recname, proc_path
        )
    valid_rois_coord_dict = support.get_roi_coord_dict(
        ref_im, ref_ch2_im, 
        stat, valid_rois, recname, proc_path
        )
    
    # -----------------
    # dF/F calculation
    # -----------------
    # now we calculate using a rolling percentile, 4 Feb 2026
    t0 = time()
    
    # load traces 
    F     = np.load(F_path, allow_pickle=True)
    F2    = np.load(F2_path, allow_pickle=True)
    Fneu  = np.load(Fneu_path, allow_pickle=True)
    F2neu = np.load(F2neu_path, allow_pickle=True)
    
    # correction using neuropil 
    F_corr  = F - Fneu * CORR_COEF
    F2_corr = F2 - F2neu * CORR_COEF
    
    # calculate dF/F
    F_dFF = ipf.calculate_dFF_percentile(F_corr,
                                         t_axis=1,
                                         window_size=9000,
                                         GPU_AVAILABLE=GPU_AVAILABLE,
                                         device_name=device_name)
    F2_dFF = ipf.calculate_dFF_percentile(F2_corr,
                                          window_size=9000,
                                          t_axis=1,
                                          GPU_AVAILABLE=GPU_AVAILABLE)
    
    # saving to disk 
    np.save(proc_data_path / 'F_dFF.npy', F_dFF)
    np.save(proc_data_path / 'F2_dFF.npy', F2_dFF)
    
    print(f'dF/F computed and saved ({timedelta(seconds=int(time()-t0))})')
    # ----------------------
    # dF/F calculation ends 
    # ----------------------
    
    # plotting: RO-aligned 
    run_path = proc_path / 'RO_aligned_single_roi'
    run_path.mkdir(exist_ok=True)
    
    # valid ROIs RO dict 
    all_run_dict = {}
    all_mean_run_dict = {}
    all_run_ch2_dict = {}
    all_mean_run_ch2_dict = {}
    
    # constituent ROIs RO dict
    all_run_const_dict = {}
    all_mean_run_const_dict = {}
    all_run_const_ch2_dict = {}
    all_mean_run_const_ch2_dict = {}
    
    for roi in tqdm(all_rois, desc='Aligning to run onset'):
        ca = F_dFF[roi]
        ref_ca = F2_dFF[roi]
        
        # align to run 
        filtered_run_frames = []  # ensures monotonity of ordered ascension
        last_frame = float('-inf')  # initialise to negative infinity
        for frame in run_frames:
            if frame > last_frame:
                filtered_run_frames.append(frame)
                last_frame = frame   
        tot_run = len(filtered_run_frames)
        head = 0; tail = len(filtered_run_frames)
        for f in range(tot_run):
            if filtered_run_frames[f]- BEF * SAMP_FREQ < 0:  # in case the 1st run-onset is within BEF s of starting recording
                head+=1
            else:
                break
        for f in range(tot_run-1,-1,-1):
            if filtered_run_frames[f] + AFT * SAMP_FREQ > tot_frames:  # in case the last run-onset is within AFT s of ending recording
                tail-=1
            else:
                break
        tot_trunc = head + (len(filtered_run_frames)-tail)
        run_aligned = np.zeros((tot_run-tot_trunc, (BEF + AFT) * SAMP_FREQ))
        run_aligned_im = np.zeros((tot_run-tot_trunc, (BEF + AFT) * SAMP_FREQ))
        run_aligned_ch2 = np.zeros((tot_run-tot_trunc, (BEF + AFT) * SAMP_FREQ))
        run_aligned_im_ch2 = np.zeros((tot_run-tot_trunc, (BEF + AFT) * SAMP_FREQ))
        for i, r in enumerate(filtered_run_frames[head:tail]):
            run_aligned[i, :] = ca[r-BEF*SAMP_FREQ : r+AFT*SAMP_FREQ]
            run_aligned_im[i, :] = normalise(smooth_convolve(ca[r-BEF*SAMP_FREQ : r+AFT*SAMP_FREQ]))
            run_aligned_ch2[i, :] = ref_ca[r-BEF*SAMP_FREQ : r+AFT*SAMP_FREQ]
            run_aligned_im_ch2[i, :] = normalise(smooth_convolve(ref_ca[r-BEF*SAMP_FREQ : r+AFT*SAMP_FREQ]))
        
        if roi in valid_rois:  # if ROI resulted from a final merge 
            all_run_dict[f'ROI {roi}'] = run_aligned
            all_run_ch2_dict[f'ROI {roi}'] = run_aligned_ch2
            
            run_aligned_mean = np.mean(run_aligned, axis=0)
            run_aligned_sem = sem(run_aligned, axis=0)
            all_mean_run_dict[f'ROI {roi}'] = run_aligned_mean
            
            run_aligned_mean_ch2 = np.mean(run_aligned_ch2, axis=0)
            run_aligned_sem_ch2 = sem(run_aligned_ch2, axis=0)
            all_mean_run_ch2_dict[f'ROI {roi}'] = run_aligned_mean_ch2
                
            fig = plt.figure(figsize=(5, 2.5))
            gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
            ax1 = fig.add_subplot(gs[:, 0]) 
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            fig.subplots_adjust(wspace=.4)
            
            ax1.imshow(ref_im, 
                       cmap='gist_gray')
            ax1.scatter(stat[roi]['xpix'], stat[roi]['ypix'], 
                        color='limegreen', edgecolor='none', s=.1, alpha=1)
            ax1.plot(stat[roi]['xcirc'], stat[roi]['ycirc'], 
                     linewidth=.5, color='white')
            ax1.set(xlim=(0,512), 
                    ylim=(0,512))
            
            ax2.imshow(run_aligned_im, 
                       cmap='Greys', extent=[-BEF, AFT, 0, tot_run], 
                       aspect='auto')
            ax2.set(xticklabels=[],
                    ylabel='trial #')
            
            ax3.plot(XAXIS, run_aligned_mean, 
                     c='darkgreen', linewidth=1)
            ax3.fill_between(XAXIS, run_aligned_mean+run_aligned_sem,
                                    run_aligned_mean-run_aligned_sem,
                             color='darkgreen', alpha=.2, edgecolor='none')
            ax3.set(xlabel='Time from run onset (s)',
                    ylabel='dF/F')
            
            fig.suptitle(f'ROI {roi} run-onset-aligned')
            
            for ext in ('.png', '.pdf'):
                fig.savefig(rf'{run_path}\roi_{roi}{ext}',
                            dpi=300,
                            bbox_inches='tight')
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(2.5, 1.7))
            
            ref_ca_ln, = ax.plot(XAXIS, run_aligned_mean_ch2, 
                                 linewidth=1,
                                 c='indianred', label='tdT', alpha=.5)
            ax.fill_between(XAXIS, run_aligned_mean_ch2+run_aligned_sem_ch2,
                                   run_aligned_mean_ch2-run_aligned_sem_ch2,
                            color='indianred', alpha=.1, edgecolor='none')
            
            ca_ln, = ax.plot(XAXIS, run_aligned_mean, 
                             linewidth=1,
                             c='darkgreen', label='GCaMP',
                             zorder=10)
            ax.fill_between(XAXIS, run_aligned_mean+run_aligned_sem,
                                   run_aligned_mean-run_aligned_sem,
                            color='darkgreen', alpha=.2, edgecolor='none',
                            zorder=10)
            
            ax.legend(fontsize=6, frameon=False)
            
            ax.set(xlabel='Time from run-onset (s)',
                   ylabel='dF/F')           
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            fig.suptitle('ROI {} run-onset-aligned'.format(roi))
            for ext in ('.png', '.pdf'):
                fig.savefig(rf'{run_path}\roi_{roi}_w_ch2{ext}',
                            dpi=300,
                            bbox_inches='tight')
            plt.close(fig)
            
        else:  # if ROI is constituent to a valid ROI 
            all_run_const_dict[f'ROI {roi}'] = run_aligned
            all_run_const_ch2_dict[f'ROI {roi}'] = run_aligned_ch2

    # save valid ROI data
    valid_rois_dict = {f'ROI {roi}': valid_rois_dict[roi] 
                       for roi in valid_rois_dict}  # rename keys to align with other dicts
    np.save(rf'{proc_data_path}\valid_ROIs_dict.npy', valid_rois_dict)
    np.save(rf'{proc_data_path}\valid_ROIs_coord_dict.npy', valid_rois_coord_dict)
    np.save(rf'{proc_data_path}\constituent_ROIs_coord_dict.npy', constituent_rois_coord_dict)

    # save merged ROIs 
    np.save(proc_path / f'{recname}_all_run.npy', all_run_dict)
    np.save(proc_path / f'{recname}_all_run_mean.npy', all_mean_run_dict)
    np.save(proc_path / f'{recname}_all_run_ch2.npy', all_run_ch2_dict)
    np.save(proc_path / f'{recname}_all_run_ch2_mean.npy', all_mean_run_ch2_dict)
    
    # save constituent ROIs 
    np.save(rf'{proc_data_path}\RO_aligned_const_dict.npy', all_run_const_dict)
    np.save(rf'{proc_data_path}\RO_aligned_const_mean_dict.npy', all_mean_run_const_dict)
    np.save(rf'{proc_data_path}\RO_aligned_const_ch2_dict.npy', all_run_const_ch2_dict)
    np.save(rf'{proc_data_path}\RO_aligned_const_mean_ch2_dict.npy', all_mean_run_const_ch2_dict)
    
    # plotting: rew-aligned 
    rew_path = r'{}\rew_aligned_single_roi'.format(proc_path)
    os.makedirs(rew_path, exist_ok=True)

    # valid ROIs RO dict 
    all_rew_dict = {}
    all_mean_rew_dict = {}
    all_rew_ch2_dict = {}
    all_mean_rew_ch2_dict = {}
    
    # constituent ROIs RO dict
    all_rew_const_dict = {}
    all_mean_rew_const_dict = {}
    all_rew_const_ch2_dict = {}
    all_mean_rew_const_ch2_dict = {}

    for roi in tqdm(all_rois, desc='Aligning to rewards'):
        ca = F_dFF[roi]
        ref_ca = F2_dFF[roi]
        
        # align to rew
        filtered_rew_frames = []
        last_frame = float('-inf')
        for frame in rew_frames:
            if frame > last_frame:
                filtered_rew_frames.append(frame)
                last_frame = frame   
        tot_rew = len(filtered_rew_frames)
        head = 0; tail = len(filtered_rew_frames)
        for f in range(tot_rew):
            if filtered_rew_frames[f]-BEF*SAMP_FREQ<0:
                head+=1
            else:
                break
        for f in range(tot_rew-1,-1,-1):
            if filtered_rew_frames[f]+AFT*30>tot_frames:
                tail-=1
            else:
                break
        tot_trunc = head + (len(filtered_rew_frames)-tail)
        rew_aligned = np.zeros((tot_rew-tot_trunc, (BEF+AFT)*30))
        rew_aligned_im = np.zeros((tot_rew-tot_trunc, (BEF+AFT)*30))
        rew_aligned_ch2 = np.zeros((tot_rew-tot_trunc, (BEF+AFT)*30))
        rew_aligned_im_ch2 = np.zeros((tot_rew-tot_trunc, (BEF+AFT)*30))
        for i, r in enumerate(filtered_rew_frames[head:tail]):
            rew_aligned[i, :] = ca[r-BEF*30:r+AFT*30]
            rew_aligned_im[i, :] = normalise(smooth_convolve(ca[r-BEF*30:r+AFT*30]))
            rew_aligned_ch2[i, :] = ref_ca[r-BEF*30:r+AFT*30]
            rew_aligned_im_ch2[i, :] = normalise(smooth_convolve(ref_ca[r-BEF*30:r+AFT*30]))
        
        if roi in valid_rois:  # if ROI resulted from a final merge 
            all_rew_dict[f'ROI {roi}'] = rew_aligned
            all_rew_ch2_dict[f'ROI {roi}'] = rew_aligned_ch2
            
            rew_aligned_mean = np.mean(rew_aligned, axis=0)
            rew_aligned_sem = sem(rew_aligned, axis=0)
            all_mean_rew_dict[f'ROI {roi}'] = rew_aligned_mean
            
            rew_aligned_mean_ch2 = np.mean(rew_aligned_ch2, axis=0)
            rew_aligned_sem_ch2 = sem(rew_aligned_ch2, axis=0)
            all_mean_rew_ch2_dict[f'ROI {roi}'] = rew_aligned_mean_ch2
                
            fig = plt.figure(figsize=(5, 2.5))
            gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
            ax1 = fig.add_subplot(gs[:, 0]) 
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            fig.subplots_adjust(wspace=.4)
            
            ax1.imshow(ref_im, cmap='gist_gray')
            ax1.scatter(stat[roi]['xpix'], stat[roi]['ypix'], color='limegreen', edgecolor='none', s=.1, alpha=1)
            ax1.plot(stat[roi]['xcirc'], stat[roi]['ycirc'], linewidth=.5, color='white')
            ax1.set(xlim=(0,512), 
                    ylim=(0,512))
            
            ax2.imshow(rew_aligned_im, 
                       cmap='Greys', extent=[-BEF, AFT, 0, tot_rew], 
                       aspect='auto')
            ax2.set(xticklabels=[],
                    ylabel='trial #')
            
            ax3.plot(XAXIS, rew_aligned_mean, c='darkgreen', linewidth=1)
            ax3.fill_between(XAXIS, rew_aligned_mean+rew_aligned_sem,
                                    rew_aligned_mean-rew_aligned_sem,
                             color='darkgreen', alpha=.2, edgecolor='none')
            ax3.set(xlabel='time from rew (s)',
                    ylabel='dF/F')
            
            fig.suptitle('ROI {} rew-aligned'.format(roi))
            
            for ext in ('.png', '.pdf'):
                fig.savefig(rf'{rew_path}\roi_{roi}{ext}',
                            dpi=300,
                            bbox_inches='tight')
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(2.5, 1.7))
            axt = ax.twinx()
            axt.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)
            
            ref_ca_ln, = axt.plot(XAXIS, rew_aligned_mean_ch2,
                                  linewidth=1,
                                  c='indianred', label='tdT', alpha=.5)
            axt.fill_between(XAXIS, rew_aligned_mean_ch2+rew_aligned_sem_ch2,
                                    rew_aligned_mean_ch2-rew_aligned_sem_ch2,
                             color='indianred', alpha=.1, edgecolor='none')
            
            ca_ln, = ax.plot(XAXIS, rew_aligned_mean, 
                             linewidth=1,
                             c='darkgreen', label='GCaMP')
            ax.fill_between(XAXIS, rew_aligned_mean+rew_aligned_sem,
                                   rew_aligned_mean-rew_aligned_sem,
                            color='darkgreen', alpha=.2, edgecolor='none')
            
            # manually extract handles for legend (fixes twin axes issue)
            handles = [ca_ln, ref_ca_ln]
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, fontsize=6, frameon=False)
            
            ax.set(xlabel='time from rew (s)',
                   ylabel='dF/F')
            axt.set(ylabel='dF/F')
            
            ax.spines['top'].set_visible(False)
            axt.spines['top'].set_visible(False)
            
            fig.suptitle('ROI {} rew-aligned'.format(roi))
            for ext in ('.png', '.pdf'):
                fig.savefig(rf'{rew_path}\roi_{roi}_w_ch2{ext}',
                            dpi=300,
                            bbox_inches='tight')
            plt.close(fig)
            
        else:  # if ROI is constituent to a valid ROI 
            all_rew_const_dict[f'ROI {roi}'] = rew_aligned
            all_rew_const_ch2_dict[f'ROI {roi}'] = rew_aligned_ch2

    # save merged ROIs 
    np.save(proc_path / f'{recname}_all_rew.npy', all_rew_dict)
    np.save(proc_path / f'{recname}_all_rew_mean.npy', all_mean_rew_dict)
    np.save(proc_path / f'{recname}_all_rew_ch2.npy', all_rew_ch2_dict)
    np.save(proc_path / f'{recname}_all_rew_ch2_mean.npy', all_mean_rew_ch2_dict)
    
    # save constituent ROIs 
    np.save(rf'{proc_data_path}\rew_aligned_const_dict.npy', all_rew_const_dict)
    np.save(rf'{proc_data_path}\rew_aligned_const_mean_dict.npy', all_mean_rew_const_dict)
    np.save(rf'{proc_data_path}\rew_aligned_const_ch2_dict.npy', all_rew_const_ch2_dict)
    np.save(rf'{proc_data_path}\rew_aligned_const_mean_ch2_dict.npy', all_mean_rew_const_ch2_dict)
    
    
    # plotting: cue-aligned 
    cue_path = r'{}\cue_aligned_single_roi'.format(proc_path)
    os.makedirs(cue_path, exist_ok=True)

    # valid ROIs RO dict 
    all_cue_dict = {}
    all_mean_cue_dict = {}
    all_cue_ch2_dict = {}
    all_mean_cue_ch2_dict = {}
    
    # constituent ROIs RO dict
    all_cue_const_dict = {}
    all_mean_cue_const_dict = {}
    all_cue_const_ch2_dict = {}
    all_mean_cue_const_ch2_dict = {}

    for roi in tqdm(all_rois, desc='aligning to cues'):
        ca = F_dFF[roi]
        ref_ca = F2_dFF[roi]
        
        # align to cue
        filtered_cue_frames = []
        last_frame = float('-inf')
        for frame in cue_frames:
            if frame > last_frame:
                filtered_cue_frames.append(frame)
                last_frame = frame   
        tot_cue = len(filtered_cue_frames)
        head = 0; tail = len(filtered_cue_frames)
        for f in range(tot_cue):
            if filtered_cue_frames[f]-BEF*SAMP_FREQ<0:
                head+=1
            else:
                break
        for f in range(tot_cue-1,-1,-1):
            if filtered_cue_frames[f]+AFT*30>tot_frames:
                tail-=1
            else:
                break
        tot_trunc = head + (len(filtered_cue_frames)-tail)
        cue_aligned = np.zeros((tot_cue-tot_trunc, (BEF+AFT)*30))
        cue_aligned_im = np.zeros((tot_cue-tot_trunc, (BEF+AFT)*30))
        cue_aligned_ch2 = np.zeros((tot_cue-tot_trunc, (BEF+AFT)*30))
        cue_aligned_im_ch2 = np.zeros((tot_cue-tot_trunc, (BEF+AFT)*30))
        for i, r in enumerate(filtered_cue_frames[head:tail]):
            cue_aligned[i, :] = ca[r-BEF*30:r+AFT*30]
            cue_aligned_im[i, :] = normalise(smooth_convolve(ca[r-BEF*30:r+AFT*30]))
            cue_aligned_ch2[i, :] = ref_ca[r-BEF*30:r+AFT*30]
            cue_aligned_im_ch2[i, :] = normalise(smooth_convolve(ref_ca[r-BEF*30:r+AFT*30]))
        
        if roi in valid_rois:  # if ROI resulted from a final merge 
            all_cue_dict[f'ROI {roi}'] = cue_aligned
            all_cue_ch2_dict[f'ROI {roi}'] = cue_aligned_ch2
            
            cue_aligned_mean = np.mean(cue_aligned, axis=0)
            cue_aligned_sem = sem(cue_aligned, axis=0)
            all_mean_cue_dict[f'ROI {roi}'] = cue_aligned_mean
            
            cue_aligned_mean_ch2 = np.mean(cue_aligned_ch2, axis=0)
            cue_aligned_sem_ch2 = sem(cue_aligned_ch2, axis=0)
            all_mean_cue_ch2_dict[f'ROI {roi}'] = cue_aligned_mean_ch2
                
            fig = plt.figure(figsize=(5, 2.5))
            gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
            ax1 = fig.add_subplot(gs[:, 0]) 
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            fig.subplots_adjust(wspace=.4)
            
            ax1.imshow(ref_im, cmap='gist_gray')
            ax1.scatter(stat[roi]['xpix'], stat[roi]['ypix'], color='limegreen', edgecolor='none', s=.1, alpha=1)
            ax1.plot(stat[roi]['xcirc'], stat[roi]['ycirc'], linewidth=.5, color='white')
            ax1.set(xlim=(0,512), 
                    ylim=(0,512))
            
            ax2.imshow(cue_aligned_im, 
                       cmap='Greys', extent=[-BEF, AFT, 0, tot_cue], 
                       aspect='auto')
            ax2.set(xticklabels=[],
                    ylabel='trial #')
            
            ax3.plot(XAXIS, cue_aligned_mean, c='darkgreen', linewidth=1)
            ax3.fill_between(XAXIS, cue_aligned_mean+cue_aligned_sem,
                                    cue_aligned_mean-cue_aligned_sem,
                             color='darkgreen', alpha=.2, edgecolor='none')
            ax3.set(xlabel='time from cue (s)',
                    ylabel='dF/F')
            
            fig.suptitle('ROI {} cue-aligned'.format(roi))
            
            for ext in ('.png', '.pdf'):
                fig.savefig(rf'{cue_path}\roi_{roi}{ext}',
                            dpi=300,
                            bbox_inches='tight')
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(2.5, 1.7))
            axt = ax.twinx()
            axt.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)
            
            ref_ca_ln, = axt.plot(XAXIS, cue_aligned_mean_ch2,
                                  linewidth=1,
                                  c='indianred', label='tdT', alpha=.5)
            axt.fill_between(XAXIS, cue_aligned_mean_ch2+cue_aligned_sem_ch2,
                                    cue_aligned_mean_ch2-cue_aligned_sem_ch2,
                             color='indianred', alpha=.1, edgecolor='none')
            
            ca_ln, = ax.plot(XAXIS, cue_aligned_mean, 
                             linewidth=1,
                             c='darkgreen', label='GCaMP')
            ax.fill_between(XAXIS, cue_aligned_mean+cue_aligned_sem,
                                   cue_aligned_mean-cue_aligned_sem,
                            color='darkgreen', alpha=.2, edgecolor='none')
            
            # manually extract handles for legend (fixes twin axes issue)
            handles = [ca_ln, ref_ca_ln]
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, fontsize=6, frameon=False)
            
            ax.set(xlabel='time from cue (s)',
                   ylabel='dF/F')
            axt.set(ylabel='dF/F')
            
            ax.spines['top'].set_visible(False)
            axt.spines['top'].set_visible(False)
            
            fig.suptitle('ROI {} cue-aligned'.format(roi))
            for ext in ('.png', '.pdf'):
                fig.savefig(rf'{cue_path}\roi_{roi}_w_ch2{ext}',
                            dpi=300,
                            bbox_inches='tight')
            plt.close(fig)
            
        else:  # if ROI is constituent to a valid ROI 
            all_cue_const_dict[f'ROI {roi}'] = cue_aligned
            all_cue_const_ch2_dict[f'ROI {roi}'] = cue_aligned_ch2

    # save merged ROIs 
    np.save(proc_path / f'{recname}_all_cue.npy', all_cue_dict)
    np.save(proc_path / f'{recname}_all_cue_mean.npy', all_mean_cue_dict)
    np.save(proc_path / f'{recname}_all_cue_ch2.npy', all_cue_ch2_dict)
    np.save(proc_path / f'{recname}_all_cue_ch2_mean.npy', all_mean_cue_ch2_dict)
    
    # save constituent ROIs 
    np.save(rf'{proc_data_path}\cue_aligned_const_dict.npy', all_cue_const_dict)
    np.save(rf'{proc_data_path}\cue_aligned_const_mean_dict.npy', all_mean_cue_const_dict)
    np.save(rf'{proc_data_path}\cue_aligned_const_ch2_dict.npy', all_cue_const_ch2_dict)
    np.save(rf'{proc_data_path}\cue_aligned_const_mean_ch2_dict.npy', all_mean_cue_const_ch2_dict)
    
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()  # clears CuPy's internal memory pool
        cp.get_default_pinned_memory_pool().free_all_blocks()  # clears pinned memory
        cp._default_memory_pool = cp.cuda.MemoryPool()  # completely resets the memory pool
        cp._default_pinned_memory_pool = cp.cuda.PinnedMemoryPool()  # resets pinned memory
        

#%% main 
def main():
    for rec_path in pathHPCLCGCaMP:
        process_all(rec_path)
        
if __name__ == '__main__':
    main()