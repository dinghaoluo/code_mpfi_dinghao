# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:26:31 2024
Modified drastically in Feb 2025:
    - added GPU support and mostly fixed memory-leak problems

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
from tqdm import tqdm
from time import time
from datetime import timedelta
from scipy.stats import sem 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf
import support_LCHPC_axon as support

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import normalise, smooth_convolve, mpl_formatting
mpl_formatting()

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLCGCaMP = rec_list.pathLCHPCGCaMP


#%% parameters 
SAMP_FREQ = 30
BEF = 3
AFT = 7  # in seconds
XAXIS = np.arange((BEF + AFT) * SAMP_FREQ) / SAMP_FREQ - BEF


#%% load beh dataframe 
print('loading behaviour dataframe...')
df = pd.read_pickle(r'Z:\Dinghao\code_dinghao\behaviour\all_LCHPCGCaMP_sessions.pkl')


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


#%% functions
'''
all support functions have been migrated to utils\support_LCHPC_axon.py
'''


#%% processing function
def process_all(rec_path):
    recname = rec_path[-17:]
    print(f'\n{recname}')
    
    ops_path = rec_path+r'/suite2p/plane0/ops.npy'
    bin_path = rec_path+r'/suite2p/plane0/data.bin'
    bin2_path = rec_path+r'/suite2p/plane0/data_chan2.bin'
    stat_path = rec_path+r'/suite2p/plane0/stat.npy'
    F_path = rec_path+r'/suite2p/plane0/F.npy'
    F2_path = rec_path+r'/suite2p/plane0/F_chan2.npy'
    
    # folder to put processed data and single-session plots 
    proc_path = (r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP'
                 rf'\all_sessions\{recname}')
    os.makedirs(proc_path, exist_ok=True)
    
    proc_data_path = (r'Z:\Dinghao\code_dinghao\LCHPC_axon_GCaMP'
                      rf'\all_sessions\{recname}\processed_data')
    os.makedirs(proc_data_path, exist_ok=True)
    
    # load files 
    stat = np.load(stat_path, allow_pickle=True)
    if 'inmerge' not in stat[0]: 
        sys.exit('halting executation: no merging detected')  # detect merging
    
    ops = np.load(ops_path, allow_pickle=True).item()
    
    tot_frames = ops['nframes']
    
    # behaviour file
    beh = df.loc[recname]
    run_frames = beh['run_onset_frames']
    pump_frames = beh['reward_frames']
    # cue_frames = beh['start_cue_frames']
    # frame_times = beh['frame_times']
    
    # filtering
    run_frames = np.asarray(run_frames)
    run_frames = run_frames[run_frames != -1]  # filter out the no-clear-onset trials
    
    pump_frames = np.asarray(pump_frames)
    pump_frames = pump_frames[pump_frames != -1]  # filter out the no-rew trials
    
    # reference images
    ref_im, ref_ch2_im = ipf.load_or_generate_reference_images(
        proc_path,
        proc_data_path,
        bin_path, bin2_path,
        tot_frames, ops,
        GPU_AVAILABLE
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
        
    # load F trace
    print('computing dFF traces for both channels...'); t0 = time()
    F_dFF = ipf.calculate_dFF(np.load(F_path, allow_pickle=True),
                              t_axis=1,
                              GPU_AVAILABLE=GPU_AVAILABLE)
    F2_dFF = ipf.calculate_dFF(np.load(F2_path, allow_pickle=True),
                               t_axis=1,
                               GPU_AVAILABLE=GPU_AVAILABLE)
    np.save(rf'{proc_data_path}\F_dFF.npy', F_dFF)
    np.save(rf'{proc_data_path}\F2_dFF.npy', F2_dFF)
    print(f'dFF computed and saved ({timedelta(seconds=int(time()-t0))})')
    
    # plotting: RO-aligned 
    run_path = rf'{proc_path}\RO_aligned_single_roi'
    os.makedirs(run_path, exist_ok=True)
    
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
    
    for roi in tqdm(all_rois, desc='aligning to run-onsets'):
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
            ax3.set(xlabel='time from run-onset (s)',
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
            
            ax.set(xlabel='time from run-onset (s)',
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
    np.save(rf'{proc_data_path}\RO_aligned_dict.npy', all_run_dict)
    np.save(rf'{proc_data_path}\RO_aligned_mean_dict.npy', all_mean_run_dict)
    np.save(rf'{proc_data_path}\RO_aligned_ch2_dict.npy', all_run_ch2_dict)
    np.save(rf'{proc_data_path}\RO_aligned_mean_ch2_dict.npy', all_mean_run_ch2_dict)
    
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

    for roi in tqdm(all_rois, desc='aligning to rewards'):
        ca = F_dFF[roi]
        ref_ca = F2_dFF[roi]
        
        # align to rew
        filtered_pump_frames = []
        last_frame = float('-inf')
        for frame in pump_frames:
            if frame > last_frame:
                filtered_pump_frames.append(frame)
                last_frame = frame   
        tot_pump = len(filtered_pump_frames)
        head = 0; tail = len(filtered_pump_frames)
        for f in range(tot_pump):
            if filtered_pump_frames[f]-BEF*SAMP_FREQ<0:
                head+=1
            else:
                break
        for f in range(tot_pump-1,-1,-1):
            if filtered_pump_frames[f]+AFT*30>tot_frames:
                tail-=1
            else:
                break
        tot_trunc = head + (len(filtered_pump_frames)-tail)
        rew_aligned = np.zeros((tot_pump-tot_trunc, (BEF+AFT)*30))
        rew_aligned_im = np.zeros((tot_pump-tot_trunc, (BEF+AFT)*30))
        rew_aligned_ch2 = np.zeros((tot_pump-tot_trunc, (BEF+AFT)*30))
        rew_aligned_im_ch2 = np.zeros((tot_pump-tot_trunc, (BEF+AFT)*30))
        for i, r in enumerate(filtered_pump_frames[head:tail]):
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
                       cmap='Greys', extent=[-BEF, AFT, 0, tot_pump], 
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
    np.save(rf'{proc_data_path}\rew_aligned_dict.npy', all_rew_dict)
    np.save(rf'{proc_data_path}\rew_aligned_mean_dict.npy', all_mean_rew_dict)
    np.save(rf'{proc_data_path}\rew_aligned_ch2_dict.npy', all_rew_ch2_dict)
    np.save(rf'{proc_data_path}\rew_aligned_mean_ch2_dict.npy', all_mean_rew_ch2_dict)
    
    # save constituent ROIs 
    np.save(rf'{proc_data_path}\rew_aligned_const_dict.npy', all_rew_const_dict)
    np.save(rf'{proc_data_path}\rew_aligned_const_mean_dict.npy', all_mean_rew_const_dict)
    np.save(rf'{proc_data_path}\rew_aligned_const_ch2_dict.npy', all_rew_const_ch2_dict)
    np.save(rf'{proc_data_path}\rew_aligned_const_mean_ch2_dict.npy', all_mean_rew_const_ch2_dict)
    
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