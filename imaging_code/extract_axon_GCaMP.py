# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:26:31 2024

process and align ROI traces
***this script is specific to axon/dendrite GCaMP data processing, due to the 
   activity merging stage producing special data structures***

overview of the merged data structure:
    - stat.npy contains (per usual) info about the ROIs; after merging and 
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
from matplotlib.colors import LinearSegmentedColormap
import sys
import os
from tqdm import tqdm
from time import time
from scipy.stats import sem 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import normalise, smooth_convolve, mpl_formatting
mpl_formatting()

# global variables 
bef = 1
aft = 4  # in seconds
xaxis = np.arange((bef+aft)*30)/30-1

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLCGCaMP = rec_list.pathHPCLCGCaMP

df = pd.read_pickle(r'Z:\Dinghao\code_dinghao\behaviour\all_HPCLCGCaMP_sessions.pkl')


#%% GPU acceleration
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
except ModuleNotFoundError:
    print('CuPy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
except Exception as e:
    # catch any other unexpected errors and print a general message
    print(f'An error occurred: {e}')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    # we are assuming that there is only 1 GPU device
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print(f'GPU-acceleration with {str(name)} and cupy')
else:
    print('GPU-acceleartion unavailable')


#%% functions
def calculate_and_plot_overlap_indices(ref_im, ref_ch2_im, stat, valid_rois, recname, border=10):
    """
    calculate overlap indices between ROIs and channel 2, then plot each ROI overlay with reference images.
    
    parameters
    ----------
    ref_im : np.ndarray
        reference image for channel 1.
    ref_ch2_im : np.ndarray
        reference image for channel 2.
    stat : list of dict
        list of ROI dictionaries, each containing 'xpix' and 'ypix' with ROI pixel coordinates.
    valid_rois : list
        list of indices of valid ROIs.
    recname : str
        name of the recording session.
    border : int, optional
        additional padding around ROI for plotting (default is 10).
    
    returns
    -------
    overlap_indices : dict
        dictionary with ROI indices as keys and calculated overlap indices as values.
    """
    overlap_indices = {}
    
    # ensure output directory exists
    output_dir = os.path.join(r'Z:\Dinghao\code_dinghao\axon_GCaMP\single_sessions', recname, 'ROI_ch2_validation')
    os.makedirs(output_dir, exist_ok=True)
    
    # loop over each valid ROI to compute overlap index and generate plots
    for roi in valid_rois:
        xpix = stat[roi]['xpix']
        ypix = stat[roi]['ypix']
        
        # calculate overlap index
        ch2_values = ref_ch2_im[ypix, xpix]
        ch1_values = ref_im[ypix, xpix]
        overlap_index = ch2_values.mean() / ch1_values.mean()
        overlap_indices[roi] = overlap_index
        
        # plot ROI overlay with reference images
        x_min, x_max = xpix.min(), xpix.max()
        y_min, y_max = ypix.min(), ypix.max()
        
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        half_span = max(x_max - x_min, y_max - y_min) // 2 + border
        
        x_min_square = max(0, x_center - half_span)
        x_max_square = min(ref_im.shape[1], x_center + half_span)
        y_min_square = max(0, y_center - half_span)
        y_max_square = min(ref_im.shape[0], y_center + half_span)
        
        channel1_sub = ref_im[y_min_square:y_max_square, x_min_square:x_max_square]
        channel2_sub = ref_ch2_im[y_min_square:y_max_square, x_min_square:x_max_square]
        
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        
        # plot channel 1 with ROI outline
        axes[0].imshow(channel1_sub, cmap='gray', extent=(x_min_square, x_max_square, y_max_square, y_min_square))
        axes[0].scatter(stat[roi]['xpix'], stat[roi]['ypix'], color='limegreen', edgecolor='none', s=1, alpha=1)
        axes[0].set_title(f'ROI {roi}: {round(overlap_index, 3)}')
        
        # plot channel 2 with the same view
        axes[1].imshow(channel2_sub, cmap='gray', extent=(x_min_square, x_max_square, y_max_square, y_min_square))
        axes[1].set_title('channel 2 ref.')
        
        # save the plot
        fig.savefig(os.path.join(output_dir, f'roi_{roi}.png'), dpi=300)
        plt.close(fig)
    
    return overlap_indices

def filter_valid_rois(stat):
    """
    filter ROIs to include only those with the longest or unique 'imerge' lists
    
    fix the issue where serial merges on the same constituent ROIs may cause 
    multiple new ROIs (e.g. ROI 817 may have an imerge-list that is a subset of
    that of ROI 818, in which case we want to eliminate ROI 817),
    13 Nov 2024 Dinghao 
    
    parameters
    ----------
    stat : list
        list of ROI dictionaries, each containing an 'imerge' key with constituent ROIs
    
    returns
    -------
    valid_rois : list
        list of indices of ROIs with the longest or unique 'imerge' lists
    tot_valids : int
        total count of valid ROIs
    """
    valid_rois = []
    
    # create a sorted list of ROI indices based on the length of their 'imerge' lists, from longest to shortest
    sorted_rois = sorted(range(len(stat)), key=lambda roi: len(stat[roi]['imerge']), reverse=True)
    
    # initialise a set to keep track of constituent ROIs that are part of any valid ROI's merge list
    covered_constituents = set()
    
    # loop through the sorted indices
    for roi in sorted_rois:
        imerge_set = set(stat[roi]['imerge'])  # convert the imerge list to a set for easy subset checking
    
        # check if this ROI's constituent ROIs are already covered by any previously added valid ROIs
        if not imerge_set.issubset(covered_constituents):
            # add this ROI index to the valid_rois list
            valid_rois.append(roi)
            
            # update the set to include this ROI's constituents
            covered_constituents.update(imerge_set)
    
    # return valid ROIs and the total count of valid ROIs
    return valid_rois

def plot_roi_overlays(ref_im, ref_ch2_im, stat, valid_rois, recname, proc_path):
    """
    create a 3-panel plot showing ROIs overlayed with channel references, and save it in specified formats.
    
    parameters
    ----------
    ref_im : np.ndarray
        reference image for channel 1
    ref_ch2_im : np.ndarray
        reference image for channel 2
    stat : list of dict
        list of ROI dictionaries, each containing 'xpix' and 'ypix' with ROI pixel coordinates
    valid_rois : list
        list of indices of valid ROIs
    recname : str
        name of the recording session
    proc_path : str
        path for saving processed data
    
    returns
    -------
    valid_roi_dict : dict
        dictionary containing pixel coordinates for each valid ROI
    """
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))
    fig.subplots_adjust(wspace=0.35, top=0.75)
    for ax in axs:
        ax.set(xlim=(0, 512), ylim=(0, 512))
        ax.set_aspect('equal')
    
    # custom color maps for channels 1 and 2
    colors_ch1 = plt.cm.Reds(np.linspace(0, 0.8, 256))
    colors_ch2 = plt.cm.Greens(np.linspace(0, 0.8, 256))
    custom_cmap_ch1 = LinearSegmentedColormap.from_list('mycmap_ch1', colors_ch1)
    custom_cmap_ch2 = LinearSegmentedColormap.from_list('mycmap_ch2', colors_ch2)
    
    # display reference images in channels 1 and 2
    axs[1].imshow(ref_im, cmap=custom_cmap_ch1)
    axs[2].imshow(ref_ch2_im, cmap=custom_cmap_ch2)
    axs[1].set(title='axon-GCaMP')
    axs[2].set(title='Dbh:Ai14')
    
    # overlay ROIs and store their coordinates in valid_roi_dict
    valid_roi_dict = {}
    for roi in valid_rois:
        axs[0].scatter(stat[roi]['xpix'], stat[roi]['ypix'], edgecolor='none', s=0.1, alpha=0.2)
        valid_roi_dict[f'ROI {roi}'] = [stat[roi]['xpix'], stat[roi]['ypix']]
    axs[0].set(title='merged ROIs')
    
    # save ROI dictionary and plot
    np.save(os.path.join(proc_path, 'valid_roi_dict.npy'), valid_roi_dict)
    fig.suptitle(recname)
    for ext in ['.png', '.pdf']:
        fig.savefig(os.path.join(proc_path, f'rois_v_ref{ext}'), dpi=200)
    plt.close(fig)
    
    return valid_roi_dict



#%% main
for rec_path in pathHPCLCGCaMP:
    recname = rec_path[-17:]
    print(f'\n{recname}')
    
    ops_path = rec_path+r'/suite2p/plane0/ops.npy'
    bin_path = rec_path+r'/suite2p/plane0/data.bin'
    bin2_path = rec_path+r'/suite2p/plane0/data_chan2.bin'
    stat_path = rec_path+r'/suite2p/plane0/stat.npy'
    iscell_path = rec_path+r'/suite2p/plane0/iscell.npy'
    F_path = rec_path+r'/suite2p/plane0/F.npy'
    
    # folder to put processed data and single-session plots 
    proc_path = r'Z:\Dinghao\code_dinghao\axon_GCaMP\single_sessions\{}'.format(recname)
    os.makedirs(proc_path, exist_ok=True)
    
    # load files 
    stat = np.load(stat_path, allow_pickle=True)
    if 'inmerge' not in stat[0]: sys.exit('halting executation: no merging detected')  # detect merging
    
    iscell = np.load(iscell_path, allow_pickle=True)
    ops = np.load(ops_path, allow_pickle=True).item()
    
    tot_rois = len(stat)
    tot_frames = ops['nframes']
    
    # behaviour file
    beh = df.loc[recname]
    run_frames = beh['run_onset_frames']
    pump_frames = beh['reward_frames']
    cue_frames = beh['start_cue_frames']
    frame_times = beh['frame_times']
    run_frames = [frame for frame in run_frames if frame != -1]  # filter out the no-clear-onset trials
    pump_frames = [frame for frame in pump_frames if frame != -1]  # filter out the no-rew trials
    
    # reference images
    ref_im, ref_ch2_im = ipf.load_or_generate_reference_images(proc_path,
                                                               bin_path, bin2_path,
                                                               tot_frames, ops,
                                                               GPU_AVAILABLE)

    # filter valid ROIs: the end results should only contain end-merge ROIs and 
    # ROIs that have never been merged with other ROIs
    valid_rois = filter_valid_rois(stat)
    tot_valids = len(valid_rois)

    # filtering through channel 2
    overlap_indices = calculate_and_plot_overlap_indices(ref_im, ref_ch2_im, stat, valid_rois, recname)
    
    # ROI plots
    valid_roi_dict = plot_roi_overlays(ref_im, ref_ch2_im, stat, valid_rois, recname, proc_path)
    
    # load F trace
    F = np.load(F_path, allow_pickle=True)
    
    # plotting: RO-aligned 
    run_path = r'{}\RO_aligned_single_roi'.format(proc_path)
    os.makedirs(run_path, exist_ok=True)
    all_mean_run_dict = {}
    for roi in tqdm(valid_rois, desc='plotting RO-aligned'):
        ca = F[roi]
        start = time()  # timer
        ca = ipf.calculate_dFF(ca, GPU_AVAILABLE=False)  # not using GPU is faster when we are calculating only 1 vector
        
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
            if filtered_run_frames[f]-bef*30<0:
                head+=1
            else:
                break
        for f in range(tot_run-1,-1,-1):
            if filtered_run_frames[f]+aft*30>tot_frames:
                tail-=1
            else:
                break
        tot_trunc = head + (len(filtered_run_frames)-tail)
        run_aligned = np.zeros((tot_run-tot_trunc, (bef+aft)*30))
        run_aligned_im = np.zeros((tot_run-tot_trunc, (bef+aft)*30))
        for i, r in enumerate(filtered_run_frames[head:tail]):
            run_aligned[i, :] = ca[r-bef*30:r+aft*30]
            run_aligned_im[i, :] = normalise(smooth_convolve(ca[r-bef*30:r+aft*30]))
        
        run_aligned_mean = np.mean(run_aligned, axis=0)
        run_aligned_sem = sem(run_aligned, axis=0)
        all_mean_run_dict[f'ROI {roi}'] = run_aligned_mean
            
        fig = plt.figure(figsize=(5, 2.5))
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[:, 0]) 
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        fig.subplots_adjust(wspace=.35)
        
        ax1.imshow(ref_im, cmap='gist_gray')
        ax1.scatter(stat[roi]['xpix'], stat[roi]['ypix'], color='limegreen', edgecolor='none', s=.1, alpha=1)
        ax1.plot(stat[roi]['xcirc'], stat[roi]['ycirc'], linewidth=.5, color='white')
        ax1.set(xlim=(0,512), 
                ylim=(0,512))
        
        image = ax2.imshow(run_aligned_im, cmap='Greys', extent=[-1, 4, 0, tot_run], aspect='auto')
        plt.colorbar(image, shrink=.5, ticks=[0,1], label='norm. dF/F')
        ax2.set(xticklabels=[],
                ylabel='trial #')
        
        ax3.plot(xaxis, run_aligned_mean, c='darkgreen')
        ax3.fill_between(xaxis, run_aligned_mean+run_aligned_sem,
                                run_aligned_mean-run_aligned_sem,
                            color='darkgreen', alpha=.2, edgecolor='none')
        ax3.set(xlabel='time from run-onset (s)',
                ylabel='dF/F')
        
        fig.suptitle('ROI {} run-onset-aligned'.format(roi))
        
        for ext in ('.png', '.pdf'):
            fig.savefig(r'{}\roi_{}{}'.format(run_path, roi, ext),
                        dpi=300)
        plt.close(fig)

    np.save(r'{}\RO_aligned_dict.npy'.format(proc_path), all_mean_run_dict)
    
    
    # plotting: rew-aligned 
    rew_path = r'{}\rew_aligned_single_roi'.format(proc_path)
    os.makedirs(rew_path, exist_ok=True)
    all_mean_rew_dict = {}
    for roi in tqdm(valid_rois, desc='plotting rew-aligned'):
        ca = F[roi]
        start = time()  # timer
        ca = ipf.calculate_dFF(ca, GPU_AVAILABLE=False) 
        
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
            if filtered_pump_frames[f]-bef*30<0:
                head+=1
            else:
                break
        for f in range(tot_pump-1,-1,-1):
            if filtered_pump_frames[f]+aft*30>tot_frames:
                tail-=1
            else:
                break
        tot_trunc = head + (len(filtered_pump_frames)-tail)
        rew_aligned = np.zeros((tot_pump-tot_trunc, (bef+aft)*30))
        rew_aligned_im = np.zeros((tot_pump-tot_trunc, (bef+aft)*30))
        for i, r in enumerate(filtered_pump_frames[head:tail]):
            rew_aligned[i, :] = ca[r-bef*30:r+aft*30]
            rew_aligned_im[i, :] = normalise(smooth_convolve(ca[r-bef*30:r+aft*30]))
        
        rew_aligned_mean = np.mean(rew_aligned, axis=0)
        rew_aligned_sem = sem(rew_aligned, axis=0)
        all_mean_rew_dict[f'ROI {roi}'] = rew_aligned_mean
            
        fig = plt.figure(figsize=(5, 2.5))
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[:, 0]) 
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        fig.subplots_adjust(wspace=.35)
        
        ax1.imshow(ref_im, cmap='gist_gray')
        ax1.scatter(stat[roi]['xpix'], stat[roi]['ypix'], color='limegreen', edgecolor='none', s=.1, alpha=1)
        ax1.plot(stat[roi]['xcirc'], stat[roi]['ycirc'], linewidth=.5, color='white')
        ax1.set(xlim=(0,512), 
                ylim=(0,512))
        
        ax2.imshow(rew_aligned_im, cmap='Greys', extent=[-1, 4, 0, tot_pump], aspect='auto')
        ax2.set(xticks=[],
                ylabel='trial #')
        
        ax3.plot(xaxis, rew_aligned_mean, c='darkgreen')
        ax3.fill_between(xaxis, rew_aligned_mean+rew_aligned_sem,
                                rew_aligned_mean-rew_aligned_sem,
                            color='darkgreen', alpha=.2, edgecolor='none')
        ax3.set(xlabel='time from reward (s)',
                ylabel='dF/F')
        
        fig.suptitle('ROI {} rew-aligned'.format(roi))
        
        for ext in ('.png', '.pdf'):
            fig.savefig(r'{}\roi_{}{}'.format(rew_path, roi, ext),
                        dpi=300)
        plt.close(fig)
            
        
    np.save(r'{}\rew_aligned_dict.npy'.format(proc_path), all_mean_rew_dict)
