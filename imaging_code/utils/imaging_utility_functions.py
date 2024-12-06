# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:24:46 2024

utility functions for statistical analyses (imaging data)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
from skimage.morphology import dilation, disk


#%% functions 
def circ_shuffle(arr, alpha=.01, num_shuf=5000):
    """
    Parameters
    ----------
    arr : array
        dFF array.
    alpha : float, optional
        significance threshold. The default is 0.01.
    num_shuf : int, optional
        how many times to shuffle. The default is 5000.

    Returns
    -------
    list of mean, alpha and 1-alpha shuf.
    """
    
    vector = False  # default to 2D array 
    
    sig_perc = (1-alpha)*100  # significance for percentile
    try:
        tot_trial, tot_time = arr.shape
    except ValueError:  # if input is 1D
        tot_time = arr.shape[0]
        vector = True
    
    shuf_mean_array = np.zeros([num_shuf, tot_time])
    
    for i in range(num_shuf):
        if vector:
            rand_shift = np.random.randint(1, tot_time)
            shuf_mean_array[i,:]+=np.roll(arr, -rand_shift)
        else:
            for t in range(tot_trial):
                rand_shift = np.random.randint(1, tot_time)
                shuf_mean_array[i,:]+=np.roll(arr[t,:], -rand_shift)
    if not vector:
        shuf_mean_array/=num_shuf

    return [np.mean(shuf_mean_array, axis=0), 
            np.percentile(shuf_mean_array, sig_perc, axis=0, method='midpoint'),
            np.percentile(shuf_mean_array, 100-sig_perc, axis=0, method='midpoint')]


def gaussian_kernel_unity(sigma):
    kernel_size = int(6 * sigma + 1)
    x = np.arange(kernel_size) - (kernel_size // 2)
    kernel = np.exp(-(x**2 / (2 * sigma**2)))
    kernel /= kernel.sum()  # normalisation to ensure the unity sum
    return kernel 


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