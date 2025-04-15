# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:24:46 2024

utility functions for statistical analyses (imaging data)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
from skimage.morphology import medial_axis, remove_small_objects
import matplotlib.pyplot as plt


#%% axon extraction 
def extract_fibre_centrelines(
        img,
        clip_limit=0.03,
        threshold_percentile=80,
        min_size=100,
        show=True
        ):
    """
    extract the centrelines of fibre-like structures from a 512x512 image.

    parameters:
    - img: 2d array
        the input image (recommended: raw or contrast-enhanced, shape 512x512)
    - clip_limit: float
        contrast limit for adaptive histogram equalisation
    - threshold_percentile: float
        percentile for binarising the image (e.g. 75 keeps brightest 25%)
    - min_size: int
        minimum object size to retain in pixels
    - show: bool
        whether to display the intermediate steps using matplotlib

    returns:
    - medial_skeleton: 2d boolean array
        binary mask of the fibre centrelines
    """
    # normalise to [0, 1]
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()

    # threshold
    thresh = np.percentile(img, threshold_percentile)
    binary = img > thresh

    # remove small objects
    binary = remove_small_objects(binary, min_size=min_size)

    # medial axis skeleton
    medial_skeleton, _ = medial_axis(binary, return_distance=True)

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Original Normalised Image')
        axs[1].imshow(binary, cmap='gray')
        axs[1].set_title('Thresholded Regions')
        axs[2].imshow(medial_skeleton, cmap='gray')
        axs[2].set_title('Fibre Centrelines (Medial Axis)')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return medial_skeleton



#%% colocalisation analysis functions
def shuffle_roi_coordinates(xpix, ypix, map_shape=(512, 512)):
    """
    randomly shuffle a roi's coordinates within a map of given size, preserving its shape

    parameters
    ----------
    xpix : array-like
        array of x-coordinates defining the roi pixels
    ypix : array-like
        array of y-coordinates defining the roi pixels
    map_size : int
        size of the map (n x m) (default = (512, 512))

    returns
    -------
    shuffled_xpix : np.ndarray
        shuffled x-coordinates of the roi
    shuffled_ypix : np.ndarray
        shuffled y-coordinates of the roi
    """
    # calculate the bounding box of the roi
    x_min, x_max = np.min(xpix), np.max(xpix)
    y_min, y_max = np.min(ypix), np.max(ypix)
    
    roi_width = x_max - x_min + 1
    roi_height = y_max - y_min + 1

    # ensure roi fits within map bounds after shuffling
    max_x_start = map_shape[0] - roi_width
    max_y_start = map_shape[1] - roi_height

    if max_x_start < 0 or max_y_start < 0:
        raise ValueError("roi is too large to fit within the map")

    # randomly choose new top-left corner within bounds
    new_x_start = np.random.randint(0, max_x_start + 1)
    new_y_start = np.random.randint(0, max_y_start + 1)

    # shift coordinates by the same offset
    x_offset = new_x_start - x_min
    y_offset = new_y_start - y_min

    shuffled_xpix = xpix + x_offset
    shuffled_ypix = ypix + y_offset

    return shuffled_xpix, shuffled_ypix

def get_mean_ref_intensity_roi(xpix, ypix, ref_map):
    """
    calculate the mean reference intensity around a specified roi, extended outward by a given radius
    
    parameters
    ----------
    xpix : array-like
        array of x-coordinates defining the roi pixels
    ypix : array-like
        array of y-coordinates defining the roi pixels
    ref_map : np.ndarray
        2d array representing the reference intensity map
    
    returns
    -------
    float
        the mean intensity value within the extended roi
    
    notes
    -----
    - the roi is defined as the region covered by the given (xpix, ypix) coordinates
    - the extended region is computed by applying a circular dilation of the specified radius
    - pixels outside the image bounds are ignored during the calculation
    """    
    mask = np.zeros_like(ref_map, dtype=bool)
    mask[ypix, xpix] = True
        
    return np.mean(ref_map[mask])

def get_manders_coefficients(xpix, ypix, ref_map):
    """
    calculate manders' coefficients (M1 and M2) for spatial co-localisation

    parameters
    ----------
    xpix : array-like
        array of x-coordinates defining the roi pixels
    ypix : array-like
        array of y-coordinates defining the roi pixels
    ref_map : np.ndarray
        2d array representing the red channel intensity map

    returns
    -------
    M1 : float
        fraction of green ROI overlapping red signal
    M2 : float
        fraction of red signal overlapping green ROI
    """
    # create a binary mask for the ROI
    mask = np.zeros_like(ref_map, dtype=bool)
    mask[ypix, xpix] = True
    
    # extract intensities
    roi_intensities = ref_map[mask]
    total_red_intensity = np.sum(ref_map)  # sum of all red channel intensities
    total_roi_intensity = np.sum(roi_intensities)  # sum of red intensities within the ROI
    
    # ensure mask size matches input pixel count
    assert np.sum(mask) == len(xpix), 'Mismatch: ROI mask size does not match input pixel count'

    # compute Manders' coefficients
    M1 = total_roi_intensity / np.sum(mask)  # fraction of green ROI associated with red
    M2 = total_roi_intensity / total_red_intensity  # fraction of red associated with green ROI

    return M1, M2