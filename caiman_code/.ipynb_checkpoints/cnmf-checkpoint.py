# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: caiman
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import numpy as np
import sys

import caiman as cm

if ('Z:/Dinghao/code_mpfi_dinghao/caiman_code/caiman' in sys.path) == False:
    sys.path.append('Z:/Dinghao/code_mpfi_dinghao/caiman_code/caiman')
import utils as utl


# %% [markdown]
# # Preprare suite2p data (motion-corrected) for Caiman

# %%
# save data.bin as caiman memory mapped file
p_ops = Path(r"Z:/Nico/AC918-20231017_02/ops.npy")
try:
    p_memmap = next(p_ops.parent.glob('memmap_*'))
    print(f'Found memmap file. Using: {p_memmap}')
except StopIteration:
    p_memmap = utl.save_data_as_mmap(p_ops, last_frame=5000, crop=True)

# load memory mapped file
Yr, dims, num_frames = cm.load_memmap(str(p_memmap))
images = np.reshape(Yr.T, [num_frames] + list(dims), order='F')

# load reference image
img_mean = utl.load_ref_img(p_ops)

# %% [markdown]
# # Choose parameters for CNMF

# %%
# general dataset-dependent parameters
fr = 30                     # imaging rate in frames per second
decay_time = 0.4            # length of a typical transient in seconds
dxy = (2., 2.)              # spatial resolution in x and y in (um per pixel)

# CNMF parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system (set p=2 if there is visible rise time in data)
gnb = 2                     # number of global background components (set to 1 or 2)
merge_thr = 0.85            # merging threshold, max correlation allowed
bas_nonneg = True           # enforce nonnegativity constraint on calcium traces (technically on baseline)
rf = 30 # default: 15                     # half-size of the patches in pixels (patch width is rf*2 + 1)
stride_cnmf = 15 # default: 10             # amount of overlap between the patches in pixels (overlap is stride_cnmf+1) 
K = 4                       # number of components per patch
gSig = np.array([4, 4])     # expected half-width of neurons in pixels (Gaussian kernel standard deviation)
gSiz = 2*gSig + 1           # Gaussian kernel width and hight
method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data see demo_dendritic.ipynb)
ssub = 1                    # spatial subsampling during initialization 
tsub = 1                    # temporal subsampling during intialization

# parameters for component evaluation
min_SNR = 2.0               # signal to noise ratio for accepting a component
rval_thr = 0.85             # space correlation threshold for accepting a component
cnn_thr = 0.99              # threshold for CNN based classifier
cnn_lowest = 0.1            # neurons with cnn probability lower than this value are rejected

parameter_dict = {
    'fr': fr,
    'dxy': dxy,
    'decay_time': decay_time,
    'p': p,
    'nb': gnb,
    'rf': rf,
    'K': K, 
    'gSig': gSig,
    'gSiz': gSiz,
    'stride': stride_cnmf,
    'method_init': method_init,
    'rolling_sum': True,
    'only_init': True,
    'ssub': ssub,
    'tsub': tsub,
    'merge_thr': merge_thr, 
    'bas_nonneg': bas_nonneg,
    'min_SNR': min_SNR,
    'rval_thr': rval_thr,
    'use_cnn': False,
    'min_cnn_thr': cnn_thr,
    'cnn_lowest': cnn_lowest
                  }

# investigate CNMF patches
cnmf_patch_width = rf*2 + 1
cnmf_patch_overlap = stride_cnmf + 1
cnmf_patch_stride = cnmf_patch_width - cnmf_patch_overlap

#patch_ax = cm.utils.visualization.view_quilt(
#    img_mean, 
#    cnmf_patch_stride, 
#    cnmf_patch_overlap, 
#    vmin=np.percentile(np.ravel(img_mean),50), 
#    vmax=np.percentile(np.ravel(img_mean),99.5),
#    figsize=(10,10))
#patch_ax.set_title(f'CNMF Patches Width {cnmf_patch_width}, Overlap {cnmf_patch_overlap}')

# %% [markdown]
# # Define output folder

# %%
# define output folder
p_out = p_ops.parent / 'K_4_p_2_decay_time_6'
p_out.mkdir(exist_ok=True)

modified_parameters = {
    'K': 4,
    'p': 2,
    'decay_time': 6,
    }
new_parameter_dict = parameter_dict.copy()
new_parameter_dict.update(modified_parameters)

# %% [markdown]
# # Run CNMF

# %%
utl.run_cnmf(images, parameter_dict, p_out)


# %% [markdown]
# # load saved data

# %%
# load previous CNMF fit
cnmf_refit = utl.load_cnmf(p_out)

# %% [markdown]
# # Save videos and masks (long)
# This will create the following files in `p_out`
# - `neural_activity.tif`: all components found by CNMF
# - `background.tif`: background component(s)
# - `resudial.tif`: original data minus the components
# - `roi.zip`: controus of components to be loaded in ImageJ
#
# This may take up to an hour and is heavy on the RAM.
#  
# <font color='red'>ATTENTION</font> This will overwrite the files if they already exist.
#
# Writing ROI files for ImageJ requires [roifile](https://github.com/cgohlke/roifile/),
# which can be installed with `pip install roifile` inside the `caiman` conda environment.

# %%
# write tifs
utl.write_results_tifs(cnmf_refit.estimates, Yr, dims, p_out)

# write roi file
utl.save_rois_imagej(cnmf_refit.estimates, dims, perc=50, p_roi=p_out / 'roi.zip')

# create mock suite2p files
utl.create_suite2p_files(cnmf_refit.estimates, Yr, p_ops, p_out / 'mock_suite2p')

# %% [markdown]
# # investigate components with caiman tools

# %%
cnmf_refit.estimates.nb_view_components(img=img_mean, denoised_color='red')
cnmf_refit.estimates.view_components(img=img_mean, denoised_color='red')


# %% [markdown]
# # Batch mode: loop over parameters
# This is a template to explore multiple parameter sets for CNMF. It recreates the steps above and saves each result in a separate folder. Make sure to modify `p_out` accordingly.

# # %%
# def cnmf_wrapper(folder, parameter_dict):

#     # define output folder
#     p_out = p_ops.parent / folder
#     p_out.mkdir(exist_ok=True, parents=True)

#     # run CNMF with new parameters
#     utl.run_cnmf(images, parameter_dict, p_out)

#     # load again from disk
#     cnmf_refit = utl.load_cnmf(p_out)

#     # write tifs
#     utl.write_results_tifs(cnmf_refit.estimates, Yr, dims, p_out)

#     # write roi file
#     utl.save_rois_imagej(cnmf_refit.estimates, dims, perc=50, p_roi=p_out / 'roi.zip')

#     # create mock suite2p files
#     utl.create_suite2p_files(cnmf_refit.estimates, Yr, p_ops, p_out / 'mock_suite2p')


# # %%
# # full parameter sweep
# for k in [4, 5, 6]:
#     for g in [3, 4, 5]:
#         # new parameters
#         gSig = np.array([g, g]) 
#         gSiz = 2*gSig + 1    
#         modified_parameters = {
#             'K': k,
#             'gSig': gSig,
#             'gSiz': gSiz,
#             }
#         new_parameter_dict = parameter_dict.copy()
#         new_parameter_dict.update(modified_parameters)
#         cnmf_wrapper(f'K_{k}_gSig_{g}',  parameter_dict)

# # %%
# # selective parameter combinations
# parent_folder = 'parameter_search'

# # default
# cnmf_wrapper(f'{parent_folder}/default', parameter_dict)

# # 2nd order because visible rise time
# new_parameter_dict = parameter_dict.copy()
# new_parameter_dict['p'] = 2
# cnmf_wrapper(f'{parent_folder}/p_2', new_parameter_dict)

# # decay times
# for d in [1, 2, 3]:
#     new_parameter_dict = parameter_dict.copy()
#     new_parameter_dict['decay_time'] = d
#     cnmf_wrapper(f'{parent_folder}/decay_time_{d}', new_parameter_dict)
# # K
# for k in [5, 7, 9]:
#     new_parameter_dict = parameter_dict.copy()
#     new_parameter_dict['K'] = k
#     cnmf_wrapper(f'{parent_folder}/K_{k}', new_parameter_dict)

# # baseline nonnegativity
# new_parameter_dict = parameter_dict.copy()
# new_parameter_dict['bas_nonneg'] = False
# cnmf_wrapper(f'{parent_folder}/bas_nonneg_False', new_parameter_dict)

