# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:56:07 2025

analyse correlations between different ROIs

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import umap
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import linregress

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLCGCaMP = rec_list.pathHPCLCGCaMP

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import smooth_convolve, mpl_formatting, circ_shuffle
mpl_formatting()


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
def compute_single_trial_correlation_matrix(
        RO_aligned_dict, 
        roi_keys,
        smoothed=True
        ):
    """
    compute the mean single-trial correlation matrix for all rois.

    parameters:
    - RO_aligned_dict: dictionary with trial-by-trial traces (tot_trials x frames per trial)
    - valid_rois: list of roi names

    returns:
    - single_trial_corr_matrix: symmetric matrix of mean single-trial correlations (num_rois x num_rois)
    """
    num_rois = len(roi_keys)
    single_trial_corr_matrix = np.zeros((num_rois, num_rois))

    for i, roi1 in enumerate(roi_keys):
        for j, roi2 in enumerate(roi_keys):
            if i <= j:  # only compute upper triangle to avoid redundant calculations
                corr_value = single_trial_correlation(
                    RO_aligned_dict, 
                    roi1, 
                    roi2,
                    smoothed=smoothed)
                single_trial_corr_matrix[i, j] = corr_value
                single_trial_corr_matrix[j, i] = corr_value  # fill symmetric entry

    return single_trial_corr_matrix

def compute_roi_distance(valid_rois_coord_dict, roi1, roi2, distance_type='centroid'):
    """
    compute the distance between two rois based on the specified method.

    parameters:
    - valid_rois_coord_dict: dictionary with roi x and y pixel coordinates
    - roi1: first roi name
    - roi2: second roi name
    - distance_type: 'centroid' for centroid distance, 'border' for border distance

    returns:
    - distance: computed distance between roi1 and roi2
    """
    x1, y1 = np.array(valid_rois_coord_dict[roi1][0]), np.array(valid_rois_coord_dict[roi1][1])
    x2, y2 = np.array(valid_rois_coord_dict[roi2][0]), np.array(valid_rois_coord_dict[roi2][1])

    if distance_type == 'centroid':
        centroid1 = (np.mean(x1), np.mean(y1))
        centroid2 = (np.mean(x2), np.mean(y2))
        return np.linalg.norm(np.array(centroid1) - np.array(centroid2))

    elif distance_type == 'border':
        # prepare coordinate matrices
        points1 = np.column_stack((x1, y1))  # shape: (num_points_1, 2)
        points2 = np.column_stack((x2, y2))  # shape: (num_points_2, 2)

        if GPU_AVAILABLE:
            # move to GPU
            points1_gpu = cp.array(points1)
            points2_gpu = cp.array(points2)

            # compute pairwise distances on GPU
            distances = cp.linalg.norm(points1_gpu[:, None, :] - points2_gpu[None, :, :], axis=-1)

            return cp.min(distances).item()  # return minimum distance (moved back to CPU)
        else:
            # use CPU version
            return np.min(cdist(points1, points2, metric='euclidean'))

    return None  # invalid distance type

def compute_roi_distance_matrix(
        valid_rois_coord_dict, 
        roi_keys, 
        distance_type='centroid'
        ):
    """
    compute the pairwise distance matrix for all rois based on the specified method.

    parameters:
    - valid_rois_coord_dict: dictionary with roi x and y pixel coordinates
    - valid_rois: list of roi names
    - distance_type: 'centroid' for centroid distance, 'border' for border distance

    returns:
    - distance_matrix: symmetric matrix of roi distances (num_rois x num_rois)
    """
    num_rois = len(roi_keys)
    distance_matrix = np.zeros((num_rois, num_rois))

    for i, roi1 in enumerate(roi_keys):
        for j, roi2 in enumerate(roi_keys):
            if i <= j:  # only compute upper triangle to avoid redundant calculations
                distance = compute_roi_distance(
                    valid_rois_coord_dict, roi1, roi2, distance_type
                    )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # mirror value in lower triangle

    return distance_matrix

def single_trial_correlation(
        RO_aligned_dict, 
        roi1, 
        roi2,
        smoothed=True):
    """
    compute the mean correlation across trials for two rois.

    parameters:
    - RO_aligned_dict: dictionary with trial-by-trial traces (tot_trials x frames per trial)
    - roi1: first roi name
    - roi2: second roi name

    returns:
    - correlation_trial_mean: mean trial-wise correlation between roi1 and roi2
    """
    traces1 = RO_aligned_dict[roi1]  # shape: (tot_trials, frames per trial)
    traces2 = RO_aligned_dict[roi2]  # shape: (tot_trials, frames per trial)

    if smoothed:
        correlations = [
            np.corrcoef(
            smooth_convolve(traces1[i], sigma=2), 
            smooth_convolve(traces2[i], sigma=2)
            )[0, 1] 
            for i in range(traces1.shape[0])
            ]
    else:
        correlations = [
            np.corrcoef(traces1[i], traces2[i])[0, 1] 
            for i in range(traces1.shape[0])
            ]
    
    return np.nanmean(correlations)  # ignore nan values if present


#%% main 
for path in pathHPCLCGCaMP:
    recname = path[-17:]
    print(recname)

    processed_data_path = (
        r'Z:\Dinghao\code_dinghao\axon_GCaMP\all_sessions'
        rf'\{recname}\processed_data'
        )
    
    valid_rois_coord_dict = np.load(
        rf'{processed_data_path}\valid_ROIs_coord_dict.npy',
        allow_pickle=True
        ).item()
    # valid_rois = [*valid_rois_coord_dict]
    
    RO_aligned_mean_dict = np.load(
        rf'{processed_data_path}\RO_aligned_mean_dict.npy',
        allow_pickle=True
        ).item()
    RO_aligned_dict = np.load(
        rf'{processed_data_path}\RO_aligned_dict.npy',
        allow_pickle=True
        ).item()
    
    RO_aligned_mean_dict_sorted = dict(
        sorted(RO_aligned_mean_dict.items(), key=lambda item: np.argmax(item[1]))
        )
    RO_sorted_keys = [*RO_aligned_mean_dict_sorted]
    RO_aligned_dict_sorted = {
        roi: RO_aligned_dict[roi]
        for roi in RO_sorted_keys
        }

    coeff_matrix = np.corrcoef([*RO_aligned_mean_dict_sorted.values()])
    coeff_matrix_single_trial = compute_single_trial_correlation_matrix(
        RO_aligned_dict_sorted, 
        RO_sorted_keys,
        smoothed=True
        )

    dist_matrix = compute_roi_distance_matrix(
        valid_rois_coord_dict,
        RO_sorted_keys,
        distance_type='border'
        )
    
    paired_coeff = coeff_matrix[np.triu_indices_from(coeff_matrix, k=1)]
    paired_coeff_single_trial = coeff_matrix_single_trial[np.triu_indices_from(coeff_matrix, k=1)]
    paired_dist = dist_matrix[np.triu_indices_from(coeff_matrix, k=1)]
    
    # UMAP on activity profiles 
    # prepare data: stack activity profiles into a (n_axons x frames) matrix
    activity_matrix = np.array([RO_aligned_mean_dict[roi] for roi in RO_sorted_keys])
    scaler = StandardScaler()
    activity_matrix_scaled = scaler.fit_transform(activity_matrix)

    # run UMAP
    reducer = umap.UMAP(n_components=2, metric='correlation', random_state=42)
    embedding = reducer.fit_transform(activity_matrix_scaled)  # shape: (n_axons, 2)
    
    # shuffled control 
    shuff_coeff, _, _ = circ_shuffle(
        paired_coeff, 
        num_shuf=500,
        GPU_AVAILABLE=GPU_AVAILABLE
        )
    shuff_coeff_single_trial, _, _ = circ_shuffle(
        paired_coeff_single_trial, 
        num_shuf=500,
        GPU_AVAILABLE=GPU_AVAILABLE
        )
    
    ## plotting 
    # matrices 
    fig, axs = plt.subplots(3,1, figsize=(3.2,9))
    axs[0].imshow(coeff_matrix)
    axs[0].set(title='coeff matrix')
    axs[1].imshow(coeff_matrix_single_trial)
    axs[1].set(title='coeff matrix (single trial)')
    axs[2].imshow(dist_matrix)
    axs[2].set(title='distance matrix')
    fig.suptitle(recname)
    for ext in ['.png', '.pdf']:
        fig.savefig(
            r'Z:\Dinghao\code_dinghao\axon_GCaMP\correlation_analysis'
            rf'\{recname}_matrices{ext}',
            dpi=300,
            bbox_inches='tight'
            )
    plt.close(fig)
    
    (
     slope, 
     intercept, 
     r_value, 
     p_value, 
     std_err
     ) = linregress(
         paired_dist, 
         paired_coeff
         )
    fitted_line = slope * np.array(paired_dist) + intercept
    (
     slope_shuff, 
     intercept_shuff, 
     r_value_shuff, 
     p_value_shuff, 
     std_err_shuff
     ) = linregress(
         paired_dist, 
         shuff_coeff
         )
    fitted_line_shuff = slope_shuff * np.array(paired_dist) + intercept_shuff
    (
     slope_single_trial,
     intercept_single_trial, 
     r_value_single_trial, 
     p_value_single_trial, 
     std_err_single_trial
     ) = linregress(
         paired_dist, 
         paired_coeff_single_trial
         )
    fitted_line_single_trial = slope_single_trial * np.array(paired_dist) + intercept_single_trial
    (
     slope_shuff_single_trial, 
     intercept_shuff_single_trial, 
     r_value_shuff_single_trial, 
     p_value_shuff_single_trial, 
     std_err_shuff_single_trial
     ) = linregress(
         paired_dist, 
         shuff_coeff_single_trial
         )
    fitted_line_shuff_single_trial = slope_shuff_single_trial * np.array(paired_dist) + intercept_shuff_single_trial
    
    # statistics 
    fig, axs = plt.subplots(2,2, figsize=(6,5.6))
    
    axs[0,0].scatter(paired_dist, paired_coeff, s=1, c='k')
    axs[0,0].plot(paired_dist, fitted_line, color='darkred',
                  label=f'$R^2$={r_value**2:.4f}, p={p_value:.4f}')
    axs[0,0].set(title='dist. v. coeff.')
    
    axs[0,1].scatter(paired_dist, shuff_coeff, s=1, c='grey')
    axs[0,1].plot(paired_dist, fitted_line_shuff, color='coral',
                  label=f'$R^2$={r_value_shuff**2:.4f}, p={p_value_shuff:.4f}')
    axs[0,1].set(title='dist. v. coeff. shuff.')

    axs[1,0].scatter(paired_dist, paired_coeff_single_trial, s=1, c='k')
    axs[1,0].plot(paired_dist, fitted_line_single_trial, color='darkred',
                  label=f'$R^2$={r_value_single_trial**2:.4f}, p={p_value_single_trial:.4f}')
    axs[1,0].set(title='dist. v. coeff. (single)')
    
    axs[1,1].scatter(paired_dist, shuff_coeff_single_trial, s=1, c='grey')
    axs[1,1].plot(paired_dist, fitted_line_shuff_single_trial, color='coral',
                  label=f'$R^2$={r_value_shuff_single_trial**2:.4f}, p={p_value_shuff_single_trial:.4f}')
    axs[1,1].set(title='dist. v. coeff. shuff. (single)')
    
    fig.suptitle(recname)
    
    for i in range(2):
        for j in range(2):
            axs[i,j].legend(frameon=False)
            for p in ['top', 'right']:
                axs[i,j].spines[p].set_visible(False)
            
    for ext in ['.png', '.pdf']:
        fig.savefig(
            r'Z:\Dinghao\code_dinghao\axon_GCaMP\correlation_analysis'
            rf'\{recname}_dist_coeff{ext}',
            dpi=300,
            bbox_inches='tight'
            )
    plt.close(fig)
    
    # UMAP 
    fig, ax = plt.subplots(figsize=(3,3))
    ax.scatter(embedding[:,0], embedding[:,1], s=1)
    ax.set(xlabel='dim. 1',
           ylabel='dim. 2',
           title='UMAP embedding of mean prof.')
    fig.suptitle(recname)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            r'Z:\Dinghao\code_dinghao\axon_GCaMP\correlation_analysis'
            rf'\{recname}_UMAP_mean_prof{ext}',
            dpi=300,
            bbox_inches='tight'
            )
    plt.close(fig)