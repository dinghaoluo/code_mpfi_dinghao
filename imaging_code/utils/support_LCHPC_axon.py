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
from tqdm import tqdm
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap


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
    calculate overlap indices between ROIs and channel 2, then save ROI overlays with reference images.

    parameters:
    - ref_im: np.ndarray
        reference image for channel 1
    - ref_ch2_im: np.ndarray
        reference image for channel 2
    - stat: list of dict
        list of ROI dictionaries, each containing 'xpix' and 'ypix' with ROI pixel coordinates
    - valid_rois: list
        list of indices of valid ROIs
    - recname: str
        name of the recording session
    - proc_path: str
        path to save ROI overlay plots
    - border: int, optional
        additional padding around ROI for plotting (default is 10)

    returns:
    - overlap_indices: dict
        dictionary with ROI indices as keys and calculated overlap indices as values
    """
    overlap_indices = {}
    
    # ensure output directory exists
    output_dir = (rf'{proc_path}\ROI_ch2_validation')
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
        fig.savefig(rf'{output_dir}\roi_{roi}.png', 
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
    valid_roi_coord_dict : dict
        dictionary containing pixel coordinates for each valid ROI
    """
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
    axs[1].imshow(ref_im, cmap=custom_cmap_ch1)
    axs[2].imshow(ref_ch2_im, cmap=custom_cmap_ch2)
    axs[1].set(title='axon-GCaMP')
    axs[2].set(title='Dbh:Ai14')
    
    # overlay ROIs and store their coordinates in valid_roi_dict
    valid_roi_coord_dict = {}
    for roi in valid_rois:
        axs[0].scatter(stat[roi]['xpix'], stat[roi]['ypix'], edgecolor='none', s=0.1, alpha=0.2)
        valid_roi_coord_dict[f'ROI {roi}'] = [stat[roi]['xpix'], stat[roi]['ypix']]
    axs[0].set(title='merged ROIs')
    
    # plot 
    fig.suptitle(recname)
    for ext in ['.png', '.pdf']:
        fig.savefig(os.path.join(proc_path, f'rois_v_ref{ext}'), dpi=200)
    plt.close(fig)
    
    return valid_roi_coord_dict