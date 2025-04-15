# -*- coding: utf-8 -*-
"""
Created on Fri 11 April 11:11:42 2025

fibre segmentation attempt 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import cv2
from skimage.measure import regionprops


#%% functions 
def mser_detection(
        img, 
        delta=5, 
        min_area=30, 
        max_area=500, 
        max_variation=1.0
        ):
    """
    perform MSER detection on a 2D grayscale image.

    parameters:
    - img: 2d array
    - delta, min_area, max_area, max_variation: MSER parameters

    returns:
    - regs: 3d boolean array of shape (n_regions, H, W)
    """
    img = img.astype(np.float32)
    
    # normalisation
    img -= img.min()
    img /= img.max()
    
    # OpenCV MSER only works on uint8 (originally designed for grayscale natural images implemented with integers )
    img_u8 = (img * 255).astype(np.uint8)

    mser = cv2.MSER_create(delta, min_area, max_area)
    mser.setMaxVariation(max_variation)
    regions, _ = mser.detectRegions(img_u8)

    h, w = img_u8.shape
    regs = np.zeros((len(regions), h, w), dtype=bool)
    for i, region in enumerate(regions):
        rr, cc = region[:, 1], region[:, 0]
        regs[i, rr, cc] = True
    return regs

def detect_fibres_from_ref(
        ref_image,
        mser_params=None,
        median_filter_size=(3, 3),
        threshold_percentile=40,
        clip_percentiles=(0, 99),
        show=False
        ):
    """
    detect fibre-like ROIs from a 2D reference image using MSER.

    parameters:
    - ref_image: 2d array (e.g. 1100nm mean image)
    - mser_params: dict of MSER parameters
    - median_filter_size: tuple for spatial filtering
    - threshold_percentile: float for background masking
    - clip_percentiles: (low, high) for contrast clipping
    - return_skeleton: bool, whether to return medial axis mask
    - show: bool, whether to show overlay plot

    returns:
    - roi_mask: binary mask of all detected ROIs
    - roi_stats: list of {'xpix', 'ypix'}
    - centreline (optional): binary skeleton mask
    """
    # spatially filter the image
    ref_filtered = median_filter(ref_image, size=median_filter_size)

    # threshold for background masking
    thresh_val = np.percentile(ref_filtered, threshold_percentile)
    binary_mask = np.zeros_like(ref_filtered, dtype=bool)
    binary_mask[ref_filtered > thresh_val] = True

    # apply mask + clip contrast to suppress extreme values 
    ref_masked = ref_filtered.copy()
    ref_masked[~binary_mask] = 0
    ref_masked = np.clip(
        ref_masked,
        np.percentile(ref_masked, clip_percentiles[0]),
        np.percentile(ref_masked, clip_percentiles[1])
    )

    # MSER detection
    if mser_params is None:
        mser_params = {
            "min_area": 30,
            "max_area": 500,
            "delta": 5,
            "max_variation": 1.0
        }
    rois = mser_detection(ref_masked, **mser_params)
    roi_mask = np.any(rois, axis=0)

    # build ROI dictionary 
    roi_dict = []
    for i in range(rois.shape[0]):
        ypix, xpix = np.where(rois[i])
        roi_dict.append({'xpix': xpix, 'ypix': ypix})

    if show:
        fig, axs = plt.subplots(1, 2, dpi=300)
        axs[0].imshow(ref_image, cmap='gray',
                      vmin=np.percentile(ref_image, 0.5),
                      vmax=np.percentile(ref_image, 99.9))
        axs[0].set_title('mean image')
        axs[1].imshow(ref_image, cmap='gray',
                      vmin=np.percentile(ref_image, 0.5),
                      vmax=np.percentile(ref_image, 99.9))
        axs[1].imshow(np.where(roi_mask, roi_mask, np.nan), cmap='Set1', alpha=0.5)
        axs[1].set_title(f'n_roi = {len(roi_dict)}')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return roi_mask, roi_dict
    
    
#%% MSER
ref_image = np.load(r'Z:\Dinghao\code_dinghao\HPC_dLight_LC_opto\all_sessions\A116i-20250410-02\processed_data\A116i-20250410-02_ref_mat_1100nm.npy')
roi_mask, roi_dict = detect_fibres_from_ref(ref_image, show=True)


#%% processing 
roi_labelled = np.zeros_like(ref_image, dtype=np.int32)
for i, roi in enumerate(roi_dict):
    roi_labelled[roi['ypix'], roi['xpix']] = i + 1 

props = regionprops(roi_labelled)

filtered_roi_dict = {}
for region in props:
    area = region.area
    eccentricity = region.eccentricity
    solidity = region.solidity
    aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-6)
    perimeter = region.perimeter  # Jingyu 20250414
    thinness = 4 * np.pi * area / (perimeter ** 2)  # Jingyu 20250414
    if (
        area > 10 and
        aspect_ratio > 2 and
        solidity < 0.7 and
        eccentricity > 0.85 and
        thinness < .5
    ):
        ypix, xpix = region.coords[:, 0], region.coords[:, 1]
        filtered_roi_dict[region.label-1] = {'xpix': xpix, 'ypix': ypix}

roi_mask_filtered = np.zeros_like(roi_mask, dtype=bool)
for roi_coords in [*filtered_roi_dict.values()]:
    roi_mask_filtered[roi_coords['ypix'], roi_coords['xpix']] = True
    
fig, ax = plt.subplots(dpi=300)
ax.imshow(ref_image, cmap='gray',
          vmin=np.percentile(ref_image, 0.5),
          vmax=np.percentile(ref_image, 99.9))
ax.imshow(np.where(roi_mask_filtered, roi_mask_filtered, np.nan), 
          cmap='Set1', alpha=0.5)
ax.set_title(f'n_roi = {len(filtered_roi_dict)}')
ax.axis('off')
plt.tight_layout()
plt.show()