# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:21:07 2025

support functions for the LCHPC axon-GCaMP pipeline to reduce cluttering 

@author: Dinghao Luo
"""

#%% imports
import numpy as np 
from scipy.stats import sem
import os 
import sys 
from tqdm import tqdm
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
from skimage.measure import find_contours

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% functions for LCHPC_axon_all_profiles.py
def compute_mean_sem_dual(
        dFF: np.array,
        dFF2: np.array,
        MAX_SAMPLES: int,
        )->tuple:
    dFF = np.array([trace[:MAX_SAMPLES] for trace in dFF])
    dFF2 = np.array([trace[:MAX_SAMPLES] for trace in dFF2])
    
    # returning lists, not arrays, to allow parqueting 
    return (
        list(np.mean(dFF, axis=0)),
        list(sem(dFF, axis=0)),
        list(np.mean(dFF2, axis=0)),
        list(sem(dFF2, axis=0))
        )

def compute_trialwise_variability(dFF: np.array)->float:
    """
    compute trial-by-trial variability for a neuron's spike trains.

    parameters:
    - train: list of numpy arrays, each representing the firing vector of a trial.

    returns:
    - variability_median: variability as 1 - median of pairwise correlations.
    """
    if len(dFF)==0: 
        return np.nan
    
    # threshold each trial by the maximum length of a trial 
    max_length = max([len(v) for v in dFF])
    dFF = [v[:max_length] for v in dFF]
    
    # compute correlation matrix 
    num_trials = len(dFF)
    corr_matrix = np.full((num_trials, num_trials), np.nan)  # initialise
    
    for i in range(num_trials):
        for j in range(i + 1, num_trials):  # only compute upper triangular
            if np.nanstd(dFF[i]) == 0 or np.nanstd(dFF[j]) == 0:
                corr_matrix[i, j] = np.nan
            else:
                corr_matrix[i, j] = np.corrcoef(dFF[i], dFF[j])[0, 1]

    # extract upper triangular correlations
    corr_values = corr_matrix[np.triu_indices(num_trials, k=1)]

    # compute variability metrics
    variability_median = 1 - np.nanmedian(corr_values)  # median-based variability

    return variability_median

def get_identity(
        roi: int,
        primary_rois: set,
        constituent_rois: set
        )->str:
    if roi in primary_rois:
        return 'primary'
    elif roi in constituent_rois:
        return 'constituent'
    else:
        print(f'WARNING: ROI {roi} of unknown identity')
        return None
    

#%% functions for LCHPC_axon_all_extract.py
def calculate_and_plot_overlap_indices(
        ref_im, 
        ref_ch2_im, 
        stat, 
        valid_rois, 
        recname,
        proc_path,
        border=10
        ):
    """
    calculate overlap indices between ROIs and channel 2, and plot aligned ROI overlays with outlines.
    
    parameters:
    - ref_im: np.ndarray
        reference image for channel 1
    - ref_ch2_im: np.ndarray
        reference image for channel 2
    - stat: list of dict
        list of ROI dictionaries, each containing 'xpix' and 'ypix' with ROI pixel coordinates
    - valid_rois: iterable
        list or set of indices of valid ROIs
    - recname: str
        name of the recording session
    - proc_path: str
        path to save ROI overlay plots
    - border: int, optional
        padding around each ROI for cropped sub-image display (default is 10 pixels)
    
    returns:
    - overlap_indices: dict
        dictionary mapping ROI indices to their overlap index (mean ch2 / mean ch1 within ROI)
    """
    overlap_indices = {}
    output_dir = os.path.join(proc_path, 'ROI_ch2_validation')
    os.makedirs(output_dir, exist_ok=True)
    
    for roi in valid_rois:
        xpix = stat[roi]['xpix']
        ypix = stat[roi]['ypix']
        
        # calculate overlap index
        ch2_values = ref_ch2_im[ypix, xpix]
        ch1_values = ref_im[ypix, xpix]
        overlap_index = ch2_values.mean() / ch1_values.mean()
        overlap_indices[roi] = overlap_index
        
        # define crop window
        x_min, x_max = xpix.min(), xpix.max()
        y_min, y_max = ypix.min(), ypix.max()
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        half_span = max(x_max - x_min, y_max - y_min) // 2 + border
        x_min_sq = max(0, x_center - half_span)
        x_max_sq = min(ref_im.shape[1], x_center + half_span)
        y_min_sq = max(0, y_center - half_span)
        y_max_sq = min(ref_im.shape[0], y_center + half_span)

        # extract sub-images
        ch1_sub = ref_im[y_min_sq:y_max_sq, x_min_sq:x_max_sq]
        ch2_sub = ref_ch2_im[y_min_sq:y_max_sq, x_min_sq:x_max_sq]

        # post-process for display
        ch1_proc = ipf.post_processing_suite2p_gui(ch1_sub)
        ch2_proc = ipf.post_processing_suite2p_gui(ch2_sub)

        # create binary mask in cropped coordinates
        roi_mask = np.zeros_like(ch1_sub, dtype=bool)
        for x, y in zip(xpix, ypix):
            x_rel = x - x_min_sq
            y_rel = y - y_min_sq
            if 0 <= x_rel < roi_mask.shape[1] and 0 <= y_rel < roi_mask.shape[0]:
                roi_mask[y_rel, x_rel] = True
        
        # find contours (in y,x coords!)
        contours = find_contours(roi_mask.astype(float), level=0.5)

        # plotting
        fig, axes = plt.subplots(1, 3, figsize=(6, 2.2))
        for ax in axes:
            ax.axis('off')

        axes[0].imshow(ch1_proc, cmap='gray')  # raw channel 1
        axes[0].set_title('ch1 raw')

        axes[1].imshow(ch1_proc, cmap='gray')  # channel 1 with ROI overlay
        # axes[1].scatter(xpix - x_min_sq, ypix - y_min_sq, 
        #                 color='limegreen', s=1, edgecolor='none', alpha=0.5)
        for contour in contours:
            axes[1].plot(contour[:, 1], contour[:, 0], 
                         linewidth=1, color='limegreen')
        axes[1].set_title(f'ch1 + ROI ({round(overlap_index, 3)})')

        axes[2].imshow(ch2_proc, cmap='gray')  # channel 2
        axes[2].set_title('ch2')

        fig.suptitle(f'ROI {roi} overlap')
        fig.tight_layout()

        for ext in ['.png', '.pdf']:
            fig.savefig(os.path.join(output_dir, f'roi_{roi}{ext}'),
                        dpi=300,
                        bbox_inches='tight')
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
    valid_rois_dict : dict
        dictionary where:
            - keys: valid merged ROIs (final merged ROIs)
            - values: list of their constituent ROIs.
    """
    valid_rois_dict = {}

    # create a sorted list of ROI indices based on the length of their 'imerge' lists, from longest to shortest
    sorted_rois = sorted(range(len(stat)), key=lambda roi: len(stat[roi]['imerge']), reverse=True)

    # initialise a set to keep track of constituent ROIs that are part of any valid ROI's merge list
    covered_constituents = set()

    # loop through the sorted indices
    for roi in sorted_rois:
        imerge_set = set(stat[roi]['imerge'])  # convert the imerge list to a set for easy subset checking

        # check if this ROI's constituent ROIs are already covered by any previously added valid ROIs
        if not imerge_set.issubset(covered_constituents):
            # add this ROI index to the valid_rois_dict
            valid_rois_dict[roi] = list(imerge_set)

            # update the set to include this ROI's constituents
            covered_constituents.update(imerge_set)

    return valid_rois_dict

def spatial_median_filter(mov,
                           size=5,
                           GPU_AVAILABLE=False, 
                           dtype='int16', 
                           chunk_size=500):
    """
    apply spatial median filtering to a 3d movie using gpu-safe chunks.

    parameters:
    - mov: 3d array (t, h, w), input movie to be filtered
    - size: int, size of the median filter window (default=5)
    - GPU_AVAILABLE: bool, whether to use cupy for GPU acceleration
    - dtype: data type of the movie (default='int16'; kept for compatibility)
    - chunk_size: int, number of frames to process at a time on gpu

    returns:
    - filtered: 3d array (t, h, w), spatially filtered movie
    """
    T, H, W = mov.shape
    filtered = np.empty((T, H, W), dtype=np.float32)

    if GPU_AVAILABLE:
        import cupy as cp
        import cupyx.scipy.ndimage
        for start in tqdm(range(0, T, chunk_size), 
                          desc='chunk median-filtering on GPU...'):
            end = min(start + chunk_size, T)  # in case the last chunk is smaller
            chunk = cp.asarray(mov[start:end])
            filtered_chunk = cupyx.scipy.ndimage.median_filter(
                chunk, size=(1, size, size)
                )
            filtered[start:end] = filtered_chunk.get()
    else:
        from scipy.ndimage import median_filter as cpu_median_filter
        filtered = cpu_median_filter(mov, size=(1, size, size))

    return filtered

def get_roi_coord_dict(
        ref_im, ref_ch2_im, 
        stat, rois, recname, proc_path,
        plot=True):
    """
    generate a dictionary of ROI pixel coordinates and optionally save a 3-panel reference plot.
    
    parameters:
    - ref_im: np.ndarray
        reference image for channel 1
    - ref_ch2_im: np.ndarray
        reference image for channel 2
    - stat: list of dict
        list of ROI dictionaries, each containing 'xpix' and 'ypix' with ROI pixel coordinates
    - rois: iterable
        list or set of ROI indices to include
    - recname: str
        name of the recording session
    - proc_path: str
        path to save the plot if plotting is enabled
    - plot: bool, optional
        whether to generate and save a 3-panel plot showing merged ROIs, channel 1, and channel 2 (default: True)
    
    returns:
    - roi_coord_dict: dict
        dictionary mapping ROI names (e.g., 'ROI 23') to their [xpix, ypix] coordinates
    """
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(6, 2))
        fig.subplots_adjust(wspace=0.35, top=0.75)
        
        for ax in axs:
            ax.set(xlim=(0, 512), ylim=(0, 512))
            ax.set_aspect('equal')
            ax.set_xticks([])  # remove x ticks
            ax.set_yticks([])  # remove y ticks
        
        # custom color maps for channels 1 and 2
        colors_ch1 = plt.cm.Greens(np.linspace(0, 0.8, 256))
        colors_ch2 = plt.cm.Reds(np.linspace(0, 0.8, 256))
        custom_cmap_ch1 = LinearSegmentedColormap.from_list('mycmap_ch1', colors_ch1)
        custom_cmap_ch2 = LinearSegmentedColormap.from_list('mycmap_ch2', colors_ch2)
        
        # display reference images in channels 1 and 2
        axs[0].set(title='merged ROIs')
        axs[1].imshow(ref_im, cmap=custom_cmap_ch1)
        axs[2].imshow(ref_ch2_im, cmap=custom_cmap_ch2)
        axs[1].set(title='axon-GCaMP')
        axs[2].set(title='Dbh:Ai14')
        
        for roi in rois:
            axs[0].scatter(stat[roi]['xpix'], stat[roi]['ypix'], 
                           edgecolor='none', s=0.1, alpha=0.2)
        
        fig.suptitle(recname)
        for ext in ['.png', '.pdf']:
            fig.savefig(os.path.join(proc_path, f'rois_v_ref{ext}'), dpi=200)
        plt.close(fig)
    
    # store ROI coords in roi_dict
    roi_coord_dict = {}
    for roi in rois:            
        roi_coord_dict[f'ROI {roi}'] = [stat[roi]['xpix'], stat[roi]['ypix']]
      
    return roi_coord_dict