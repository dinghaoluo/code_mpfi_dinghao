# -*- coding: utf-8 -*-
"""
Created originally by Yingxue Wang
Modified on 2 Sept 2025

using linear regression to demix motion artefacts from imaging recordings 
    (dual colour)
modified from Yingxue's original pipeline

I have packed all the scripts into functions 

@author: Dinghao Luo
"""

#%% imports 
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(r'Z:\Yingxue\code\PythonMotionCorrection')

import Motion_Per_ROI

import Generate_masks

import Regression_Red_From_Green_BehAvg_Template as rrfgbt

import Regression_Red_From_Green as rrfg

import Single_trial_dLight


#%% functions 
def generate_masks(mov, movr, path_result, edge_pixel_num=15, grid_size=16):
    """
    generate masks and save them to disk, also plot mean images.

    parameters:
    - mov: np.ndarray, green channel movie [frames, height, width]
    - movr: np.ndarray, red channel movie [frames, height, width]
    - path_result: Path, directory to save masks and figures
    - edge_pixel_num: int, number of pixels trimmed from edges
    - grid_size: int, size of grid for ROI pixel count heatmap

    returns:
    - dict: dictionary containing generated masks
    """
    path_result.mkdir(parents=True, exist_ok=True)

    # trim movies
    shape = mov.shape
    xlim = shape[1]
    ylim = shape[2]
    mov_tmp_r = movr[:, edge_pixel_num:ylim-edge_pixel_num, :]
    mov_tmp_r = mov_tmp_r[:, :, edge_pixel_num:xlim-edge_pixel_num]
    mov_tmp_g = mov[:, edge_pixel_num:ylim-edge_pixel_num, :]
    mov_tmp_g = mov_tmp_g[:, :, edge_pixel_num:xlim-edge_pixel_num]

    # mean images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle('Mean image Ch1 vs Ch2', fontsize=16)

    im1 = ax1.imshow(np.mean(mov_tmp_g, axis=0), cmap='gray')
    ax1.set_title('Mean_Ch1')
    fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04, label='F')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    im2 = ax2.imshow(np.mean(mov_tmp_r, axis=0), cmap='gray')
    ax2.set_title('Mean_Ch2')
    fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04, label='F')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    fig.savefig(Path(path_result, 'mean_images.png'),
                dpi=300,
                bbox_inches='tight')
    plt.close(fig)

    # create masks
    mean_img_red = movr.mean(axis=0)
    global_axon_mask = Generate_masks.generate_and_save_axon_mask(
        mean_img=mean_img_red,
        output_dir=path_result,
        output_filename_base='global_axon_mask',
        tophat_radius=5,
        intensity_quantile=0.97,
        min_size=10
    )

    mean_img_green = mov.mean(axis=0)
    global_dlight_mask = Generate_masks.generate_and_save_dlight_mask(
        mean_img=mean_img_green,
        output_dir=path_result,
        output_filename_base='global_dlight_mask',
        gaussian_sigma=1.5,
        peak_min_distance=5,
        adaptive_block_size=5
    )

    global_axon_dlight_mask = Generate_masks.AND_mask(
        mask1=global_axon_mask,
        mask2=global_dlight_mask,
        mask1_name='axon',
        mask2_name='dlight',
        output_dir=path_result,
        output_filename_base='global_axon_and_dlight_mask'
    )

    global_axon_or_dlight_mask = Generate_masks.OR_mask(
        mask1=global_axon_mask,
        mask2=global_dlight_mask,
        mask1_name='axon',
        mask2_name='dlight',
        output_dir=path_result,
        output_filename_base='global_axon_or_dlight_mask'
    )

    global_axon_mask_dilated = Generate_masks.dilate_mask(global_axon_mask, iterations=4)

    dilated_global_axon_and_dlight_mask = Generate_masks.AND_mask(
        mask1=global_axon_mask_dilated,
        mask2=global_dlight_mask,
        mask1_name='dilated_axon',
        mask2_name='dlight',
        output_dir=path_result,
        output_filename_base='dilated_global_axon_and_dlight_mask'
    )

    dilated_global_axon_or_dlight_mask = Generate_masks.OR_mask(
        mask1=global_axon_mask_dilated,
        mask2=global_dlight_mask,
        mask1_name='dilated_axon',
        mask2_name='dlight',
        output_dir=path_result,
        output_filename_base='dilated_global_axon_or_dlight_mask'
    )

    # pixel count heatmap
    Motion_Per_ROI.plot_pixel_count_heatmap(
        global_axon_mask=global_axon_mask,
        grid_size=grid_size,
        mean_img=global_axon_mask
    )

    return {
        'global_axon_mask': global_axon_mask,
        'global_dlight_mask': global_dlight_mask,
        'global_axon_dlight_mask': global_axon_dlight_mask,
        'global_axon_or_dlight_mask': global_axon_or_dlight_mask,
        'dilated_global_axon_and_dlight_mask': dilated_global_axon_and_dlight_mask,
        'dilated_global_axon_or_dlight_mask': dilated_global_axon_or_dlight_mask
    }


def process_behavior(beh_file, output_dir, num_imaging_frames=10_000, imaging_rate=30.0, do_plots=True):
    """
    process behavioural data and align with imaging frames, then save to disk.

    parameters:
    - beh_file: str or Path, path to the behavioural .pkl file
    - output_dir: str or Path, directory to save processed behavioural data
    - num_imaging_frames: int, number of imaging frames in the recording
    - imaging_rate: float, imaging frame rate (hz)
    - do_plots: bool, whether to generate diagnostic plots

    returns:
    - None
    """
    import Align_Beh_Imaging

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Align_Beh_Imaging.process_and_save_behavioral_data(
        input_pkl_path=beh_file,
        output_dir=output_dir,
        num_imaging_frames=num_imaging_frames,
        imaging_rate=imaging_rate,
        do_plots=do_plots
    )


def run_event_locked_analysis(
    mov,
    movr,
    path_result,
    path_masks,
    beh_npz,
    grid_size=16,
    pre=90,
    post=120,
    imaging_rate=30.0
):
    """
    run event-locked whole-fov analyses for dual-colour recordings
        (red = axon; green = dlight)

    parameters:
    - mov: np.ndarray or memmap, green channel movie (dlight)
    - movr: np.ndarray or memmap, red channel movie (axon)
    - path_result: Path, output directory for results
    - path_masks: Path, directory containing precomputed masks
    - beh_npz: Path, path to processed behavioural .npz
    - grid_size: int, border size for edge trimming
    - pre: int, number of frames before event
    - post: int, number of frames after event
    - imaging_rate: float, imaging frame rate in hz

    returns:
    - None
    """
    import FRedPerTrial

    path_result = Path(path_result)
    path_masks = Path(path_masks)
    path_result.mkdir(parents=True, exist_ok=True)
    path_masks.mkdir(parents=True, exist_ok=True)

    # load behavioural events
    with np.load(beh_npz) as ld:
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None
        reward_frames = ld['reward_frames'] if 'reward_frames' in ld else None

    if run_onset_frames is None or reward_frames is None:
        raise RuntimeError('missing required event frames (run_onset_frames or reward_frames) in behaviour npz')

    # load masks
    global_axon_mask = np.load(path_masks / 'global_axon_mask.npy')

    # run analyses (whole-fov, red and green)
    FRedPerTrial.run_whole_fov_analysis(
        movr,
        global_axon_mask,
        run_onset_frames,
        path_result,
        chan_name='red_axon',
        event_name='run_onset',
        mask_name='axon_mask',
        edge_pixels=grid_size,
        pre_frames=pre,
        post_frames=post,
        imaging_rate=imaging_rate
    )

    FRedPerTrial.run_whole_fov_analysis(
        mov,
        global_axon_mask,
        run_onset_frames,
        path_result,
        chan_name='dlight',
        event_name='run_onset',
        mask_name='axon_mask',
        edge_pixels=grid_size,
        pre_frames=pre,
        post_frames=post,
        imaging_rate=imaging_rate
    )


def run_regression_with_behavioral_templates(
    mov,
    movr,
    path_result,
    path_masks,
    beh_npz,
    grid_size=16,
    pre=90,
    post=120,
    imaging_rate=30.0,
    baseline_period_s=(-3.0, -1.0),
    blocks_of_interest=None,
    neighborhood_size=15,
    correction_params=None,
):
    """
    Run regression using behaviour-averaged red kernels to demix motion artefacts
    from dual-colour recordings (red = axon, green = dLight), using the
    NON-dilated axon∧dLight mask. Performs a single-block test with validation
    plots, then a full-grid regression over selected blocks.

    Parameters
    ----------
    mov : np.ndarray or np.memmap
        Green channel movie (dLight).
    movr : np.ndarray or np.memmap
        Red channel movie (axon).
    path_result : str or Path
        Output directory for results.
    path_masks : str or Path
        Directory containing precomputed masks (.npy files).
    beh_npz : str or Path
        Path to processed behavioural .npz file (contains event frames).
    grid_size : int, default 16
        Grid size for block division.
    pre, post : int, default 90, 120
        Frames before/after the event for alignment windows.
    imaging_rate : float, default 30.0
        Imaging frame rate (Hz).
    baseline_period_s : tuple[float, float], default (-3.0, -1.0)
        Baseline window (seconds) relative to event for ΔF/F baseline.
    blocks_of_interest : dict[str, tuple[int, int]] | None
        Mapping of block labels to coordinates for grid pipeline; if None,
        a default set is used.
    neighborhood_size : int, default 15
        Local background neighbourhood size for regression.
    correction_params : dict | None
        Optional override for pipeline correction parameters.

    Returns
    -------
    None
    """
    path_result = Path(path_result)
    path_masks = Path(path_masks)
    path_result.mkdir(parents=True, exist_ok=True)
    path_masks.mkdir(parents=True, exist_ok=True)

    # --- behaviour frames ---
    with np.load(beh_npz) as ld:
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None
        reward_frames = ld['reward_frames'] if 'reward_frames' in ld else None
    if run_onset_frames is None or reward_frames is None:
        raise RuntimeError('missing required event frames (run_onset_frames or reward_frames)')

    # --- masks (NON-dilated) ---
    global_axon_dlight_mask = np.load(path_masks / 'global_axon_and_dlight_mask.npy')
    global_axon_or_dlight_mask = np.load(path_masks / 'global_axon_or_dlight_mask.npy')

    # --- full-grid regression ---
    if blocks_of_interest is None:
        blocks_of_interest = {
            'Block_A': (8, 8),
            'Block_B': (23, 8),
            'Block_C': (8, 23),
            'Block_D': (23, 23),
        }
    if correction_params is None:
        correction_params = {
            'grid_size': grid_size,
            'pre_frames': pre,
            'post_frames': post,
            'imaging_rate': imaging_rate,
            'baseline_period_s': baseline_period_s,
            'kernel_smoothing_sigma': 1.5,
            'n_jobs': -1,
        }

    mean_img_red = np.mean(movr, axis=0)

    rrfgbt.run_regression_pipeline(
        mov=mov,
        movr=movr,
        global_mask_green=global_axon_dlight_mask,
        global_mask_red=global_axon_dlight_mask,
        global_neuropil_rev=global_axon_or_dlight_mask,
        run_onset_frames=run_onset_frames,
        reward_frames=reward_frames,
        mean_img=mean_img_red,
        mask_str='G_axon_dlight_R_axon_dlight',
        output_dir=path_result,
        blocks_of_interest=blocks_of_interest,
        correction_params=correction_params,
        neighborhood_size=neighborhood_size
    )


def run_regression_with_behavioral_templates_dilated(
    mov,
    movr,
    path_result,
    path_masks,
    beh_npz,
    grid_size=16,
    pre=90,
    post=120,
    imaging_rate=30.0,
    baseline_period_s=(-3.0, -1.0),
    blocks_of_interest=None,
    neighborhood_size=15,
    correction_params=None,
):
    """
    Run regression using behaviour-averaged red kernels to demix motion artefacts
    from dual-colour recordings (red = axon, green = dLight), using the
    DILATED axon∧dLight mask. Performs a single-block validation AND a full-grid
    regression over selected blocks.

    Parameters
    ----------
    mov : np.ndarray or np.memmap
        Green channel movie (dLight).
    movr : np.ndarray or np.memmap
        Red channel movie (axon).
    path_result : str or Path
        Output directory for results.
    path_masks : str or Path
        Directory containing precomputed masks (.npy files).
    beh_npz : str or Path
        Path to processed behavioural .npz file (contains event frames).
    grid_size : int, default 16
        Grid size for block division.
    pre, post : int, default 90, 120
        Frames before/after the event for alignment windows.
    imaging_rate : float, default 30.0
        Imaging frame rate (Hz).
    baseline_period_s : tuple[float, float], default (-3.0, -1.0)
        Baseline window (seconds) relative to event for ΔF/F baseline.
    blocks_of_interest : dict[str, tuple[int, int]] | None
        Mapping of block labels to coordinates for grid pipeline; if None,
        a default set is used.
    neighborhood_size : int, default 15
        Local background neighbourhood size for regression.
    correction_params : dict | None
        Optional override for pipeline correction parameters.

    Returns
    -------
    None
    """
    path_result = Path(path_result)
    path_masks = Path(path_masks)
    path_result.mkdir(parents=True, exist_ok=True)
    path_masks.mkdir(parents=True, exist_ok=True)

    # --- behaviour frames ---
    with np.load(beh_npz) as ld:
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None
        reward_frames = ld['reward_frames'] if 'reward_frames' in ld else None
    if run_onset_frames is None or reward_frames is None:
        raise RuntimeError('missing required event frames (run_onset_frames or reward_frames)')

    # --- masks (DILATED) ---
    dilated_global_axon_dlight_mask = np.load(path_masks / 'dilated_global_axon_and_dlight_mask.npy')
    global_axon_or_dlight_mask = np.load(path_masks / 'global_axon_or_dlight_mask.npy')

    # --- full-grid regression ---
    if blocks_of_interest is None:
        blocks_of_interest = {
            'Block_A': (8, 8),
            'Block_B': (23, 8),
            'Block_C': (8, 23),
            'Block_D': (23, 23),
        }
    if correction_params is None:
        correction_params = {
            'grid_size': grid_size,
            'pre_frames': pre,
            'post_frames': post,
            'imaging_rate': imaging_rate,
            'baseline_period_s': baseline_period_s,
            'kernel_smoothing_sigma': 1.5,
            'n_jobs': -1,
        }

    mean_img_red = np.mean(movr, axis=0)

    rrfgbt.run_regression_pipeline(
        mov=mov,
        movr=movr,
        global_mask_green=dilated_global_axon_dlight_mask,
        global_mask_red=dilated_global_axon_dlight_mask,
        global_neuropil_rev=global_axon_or_dlight_mask,
        run_onset_frames=run_onset_frames,
        reward_frames=reward_frames,
        mean_img=mean_img_red,
        mask_str='G_dilated_axon_dlight_R_dilated_axon_dlight',
        output_dir=path_result,
        blocks_of_interest=blocks_of_interest,
        correction_params=correction_params,
        neighborhood_size=neighborhood_size,
    )


def run_regression_whole_trace(
    mov,
    movr,
    path_result,
    path_masks,
    beh_npz,
    grid_size=16,
    pre=90,
    post=120,
    imaging_rate=30.0,
    blocks_of_interest=None,
    neighborhood_size=15,
    correction_params=None
):
    """
    run regression using the whole red trace (axon) to demix motion artefacts
    from dual-colour recordings (red = axon, green = dlight).

    parameters:
    - mov: np.ndarray or memmap, green channel movie (dlight)
    - movr: np.ndarray or memmap, red channel movie (axon)
    - path_result: Path, output directory for results
    - path_masks: Path, directory containing precomputed masks
    - beh_npz: Path, path to processed behavioural .npz file
    - grid_size: int, grid size for block division
    - pre: int, frames before event
    - post: int, frames after event
    - imaging_rate: float, imaging frame rate (hz)
    - blocks_of_interest: dict, optional mapping of block labels to coordinates
    - neighborhood_size: int, size of neighbourhood for regression
    - correction_params: dict, overrides default regression parameters

    returns:
    - None
    """
    path_result = Path(path_result)
    path_masks = Path(path_masks)
    path_result.mkdir(parents=True, exist_ok=True)

    # behaviour frames
    with np.load(beh_npz) as ld:
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None
        reward_frames = ld['reward_frames'] if 'reward_frames' in ld else None
    if run_onset_frames is None or reward_frames is None:
        raise RuntimeError('missing required event frames')

    # masks
    global_axon_dlight_mask = np.load(path_masks / 'global_axon_and_dlight_mask.npy')
    global_axon_or_dlight_mask = np.load(path_masks / 'global_axon_or_dlight_mask.npy')

    # full-grid regression
    if blocks_of_interest is None:
        blocks_of_interest = {
            'Block_A': (8, 8),
            'Block_B': (23, 8),
            'Block_C': (8, 23),
            'Block_D': (23, 23),
        }

    if correction_params is None:
        correction_params = {
            'grid_size': grid_size,
            'imaging_rate': imaging_rate,
            'smoothing_sigma': 0,
            'n_jobs': -1
        }

    rrfg.run_regression_pipeline(
        mov=mov,
        movr=movr,
        global_mask_green=global_axon_dlight_mask,
        global_mask_red=global_axon_dlight_mask,
        global_neuropil_rev=global_axon_or_dlight_mask,
        run_onset_frames=run_onset_frames,
        mean_img=np.mean(movr, axis=0),
        mask_str='G_axon_dlight_R_axon_dlight',
        output_dir=path_result,
        blocks_of_interest=blocks_of_interest,
        correction_params=correction_params,
        neighborhood_size=neighborhood_size,
        pre_frames=pre,
        post_frames=post
    )


def run_regression_whole_trace_dilated(
    mov,
    movr,
    path_result,
    path_masks,
    beh_npz,
    grid_size=16,
    pre=90,
    post=120,
    imaging_rate=30.0,
    blocks_of_interest=None,
    neighborhood_size=15,
    correction_params=None
):
    """
    same as `run_regression_whole_trace`, but uses the DILATED axon∧dLight mask
    instead of the standard one.

    parameters: see `run_regression_whole_trace`

    returns:
    - None
    """
    path_result = Path(path_result)
    path_masks = Path(path_masks)
    path_result.mkdir(parents=True, exist_ok=True)

    # behaviour frames
    with np.load(beh_npz) as ld:
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None
        reward_frames = ld['reward_frames'] if 'reward_frames' in ld else None
    if run_onset_frames is None or reward_frames is None:
        raise RuntimeError('missing required event frames')

    # masks (dilated)
    dilated_global_axon_dlight_mask = np.load(path_masks / 'dilated_global_axon_and_dlight_mask.npy')
    global_axon_or_dlight_mask = np.load(path_masks / 'global_axon_or_dlight_mask.npy')

    # full-grid regression
    if blocks_of_interest is None:
        blocks_of_interest = {
            'Block_A': (8, 8),
            'Block_B': (23, 8),
            'Block_C': (8, 23),
            'Block_D': (23, 23),
        }

    if correction_params is None:
        correction_params = {
            'grid_size': grid_size,
            'imaging_rate': imaging_rate,
            'smoothing_sigma': 0,
            'n_jobs': -1
        }

    rrfg.run_regression_pipeline(
        mov=mov,
        movr=movr,
        global_mask_green=dilated_global_axon_dlight_mask,
        global_mask_red=dilated_global_axon_dlight_mask,
        global_neuropil_rev=global_axon_or_dlight_mask,
        run_onset_frames=run_onset_frames,
        mean_img=np.mean(movr, axis=0),
        mask_str='G_dilated_axon_dlight_R_dilated_axon_dlight',
        output_dir=path_result,
        blocks_of_interest=blocks_of_interest,
        correction_params=correction_params,
        neighborhood_size=neighborhood_size,
        pre_frames=pre,
        post_frames=post
    )
    
    
def run_regression_single_trial(
    mov,
    movr,
    path_result,
    path_masks,
    beh_npz,
    grid_size=16,
    pre=90,
    post=120,
    imaging_rate=30.0,
    neighborhood_size=15,
    correction_params=None
):
    """
    run single-trial regression (red from green) using axon∧dLight mask,
    with validation plots and full-grid regression.

    parameters:
    - mov: np.ndarray or memmap, green channel movie (dlight)
    - movr: np.ndarray or memmap, red channel movie (axon)
    - path_result: Path, output directory for results
    - path_masks: Path, directory containing precomputed masks
    - beh_npz: Path, path to processed behavioural .npz file
    - grid_size, pre, post, imaging_rate, neighborhood_size: analysis params
    - correction_params: dict, overrides default regression parameters
    """
    import Regression_Red_From_Green_Single_Trial as rrfgst

    path_result = Path(path_result)
    path_masks = Path(path_masks)

    # --- behaviour frames ---
    with np.load(beh_npz) as ld:
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None
        reward_frames = ld['reward_frames'] if 'reward_frames' in ld else None
    if run_onset_frames is None:
        raise RuntimeError('missing run_onset_frames in behaviour npz')

    # --- masks (standard axon∧dLight) ---
    global_axon_dlight_mask = np.load(path_masks / 'global_axon_and_dlight_mask.npy')
    global_axon_or_dlight_mask = np.load(path_masks / 'global_axon_or_dlight_mask.npy')

    # --- grid-level regression ---
    if correction_params is None:
        correction_params = {'grid_size': grid_size, 'pre_frames': pre, 'post_frames': post, 'n_jobs': -1}
    mean_img_red = np.mean(movr, axis=0)

    rrfgst.run_regression_pipeline(
        mov=mov, movr=movr,
        global_mask_green=global_axon_dlight_mask,
        global_mask_red=global_axon_dlight_mask,
        global_neuropil_rev=global_axon_or_dlight_mask,
        run_onset_frames=run_onset_frames,
        mean_img=mean_img_red,
        mask_str='G_axon_dlight_R_axon_dlight',
        output_dir=path_result,
        blocks_of_interest={'Block_A': (8, 8), 'Block_B': (23, 8), 'Block_C': (8, 23), 'Block_D': (23, 23)},
        correction_params=correction_params,
        neighborhood_size=neighborhood_size,
        imaging_rate=imaging_rate
    )


def run_regression_single_trial_dilated(
    mov,
    movr,
    path_result,
    path_masks,
    beh_npz,
    grid_size=16,
    pre=90,
    post=120,
    imaging_rate=30.0,
    neighborhood_size=15,
    correction_params=None
):
    """
    run single-trial regression (red from green) using DILATED axon∧dLight mask,
    with validation plots and full-grid regression.

    parameters: same as run_regression_single_trial
    """
    import Regression_Red_From_Green_Single_Trial as rrfgst

    path_result = Path(path_result)
    path_masks = Path(path_masks)

    # --- behaviour frames ---
    with np.load(beh_npz) as ld:
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None
        reward_frames = ld['reward_frames'] if 'reward_frames' in ld else None
    if run_onset_frames is None:
        raise RuntimeError('missing run_onset_frames in behaviour npz')

    # --- masks (DILATED) ---
    dilated_global_axon_dlight_mask = np.load(path_masks / 'dilated_global_axon_and_dlight_mask.npy')
    global_axon_or_dlight_mask = np.load(path_masks / 'global_axon_or_dlight_mask.npy')

    # --- grid-level regression ---
    if correction_params is None:
        correction_params = {'grid_size': grid_size, 'pre_frames': pre, 'post_frames': post, 'n_jobs': -1}
    mean_img_red = np.mean(movr, axis=0)

    rrfgst.run_regression_pipeline(
        mov=mov, movr=movr,
        global_mask_green=dilated_global_axon_dlight_mask,
        global_mask_red=dilated_global_axon_dlight_mask,
        global_neuropil_rev=global_axon_or_dlight_mask,
        run_onset_frames=run_onset_frames,
        mean_img=mean_img_red,
        mask_str='G_dilated_axon_dlight_R_dilated_axon_dlight',
        output_dir=path_result,
        blocks_of_interest={'Block_A': (8, 8), 'Block_B': (23, 8), 'Block_C': (8, 23), 'Block_D': (23, 23)},
        correction_params=correction_params,
        neighborhood_size=neighborhood_size,
        imaging_rate=imaging_rate
    )


def run_single_trial_dlight_analysis_template(
    path_result,
    beh_npz,
    imaging_rate=30.0,
    n_shuffles=1000,
    p_value_threshold=0.05
):
    """
    run single-trial dLight analysis using corrected traces from
    TEMPLATE regression (axon∧dLight masks, dilated variant included).

    parameters:
    - path_result: Path, output directory containing masks and regression outputs
    - beh_npz: Path, processed behaviour npz
    - imaging_rate: float, imaging frame rate (hz)
    - n_shuffles: int, shuffles for permutation test
    - p_value_threshold: float, significance cutoff

    returns:
    - None
    """
    path_result = Path(path_result)
    beh_npz = Path(beh_npz)

    # load behaviour event frames
    with np.load(beh_npz) as ld:
        speed_trace = ld['speed'] if 'speed' in ld else None
        lick_trace = ld['licks'] if 'licks' in ld else None
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None

    # load mask for plotting/weighting
    global_axon_mask = np.load(path_result / 'masks' / 'global_axon_mask.npy')

    for mask_str in ['G_axon_dlight_R_axon_dlight',
                     'G_dilated_axon_dlight_R_dilated_axon_dlight']:
        result_dir = path_result / 'regression_with_template' / mask_str
        corrected_file = result_dir / 'all_corrected_traces_regress_with_template.npz'

        with np.load(corrected_file) as ld:
            corrected_dlight = ld['corrected_dlight']
            red_trace = ld['red_trace']
            pre = ld['pre_frames'].item()
            post = ld['post_frames'].item()

        # population-level significance
        Single_trial_dLight.run_and_save_significance_analysis(
            corrected_traces_grid=corrected_dlight,
            red_traces_grid=red_trace,
            speed_trace=speed_trace,
            lick_trace=lick_trace,
            event_frames=run_onset_frames,
            mean_img=global_axon_mask,
            output_dir=result_dir,
            event_name='Run_onset',
            pre_frames=pre,
            post_frames=post,
            imaging_rate=imaging_rate,
            p_value_threshold=p_value_threshold,
            n_shuffles=n_shuffles
        )


def run_single_trial_dlight_analysis_single_trial(
    path_result,
    beh_npz,
    imaging_rate=30.0,
    n_shuffles=1000,
    p_value_threshold=0.05
):
    """
    run single-trial dLight analysis using corrected traces from
    SINGLE-TRIAL regression (axon∧dLight masks, dilated variant included).
    """
    path_result = Path(path_result)
    beh_npz = Path(beh_npz)

    with np.load(beh_npz) as ld:
        speed_trace = ld['speed'] if 'speed' in ld else None
        lick_trace = ld['licks'] if 'licks' in ld else None
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None

    global_axon_mask = np.load(path_result / 'masks' / 'global_axon_mask.npy')

    for mask_str in ['G_axon_dlight_R_axon_dlight',
                     'G_dilated_axon_dlight_R_dilated_axon_dlight']:
        result_dir = path_result / 'regression_single_trial' / mask_str
        corrected_file = result_dir / 'all_corrected_traces_regress_with_template.npz'

        with np.load(corrected_file) as ld:
            corrected_dlight = ld['corrected_dlight']
            red_trace = ld['red_trace']
            pre = ld['pre_frames'].item()
            post = ld['post_frames'].item()

        Single_trial_dLight.run_and_save_significance_analysis(
            corrected_traces_grid=corrected_dlight,
            red_traces_grid=red_trace,
            speed_trace=speed_trace,
            lick_trace=lick_trace,
            event_frames=run_onset_frames,
            mean_img=global_axon_mask,
            output_dir=result_dir,
            event_name='Run_onset',
            pre_frames=pre,
            post_frames=post,
            imaging_rate=imaging_rate,
            p_value_threshold=p_value_threshold,
            n_shuffles=n_shuffles
        )


def run_single_trial_dlight_analysis_whole_trace(
    path_result,
    beh_npz,
    imaging_rate=30.0,
    n_shuffles=1000,
    p_value_threshold=0.05
):
    """
    run single-trial dLight analysis using corrected traces from
    WHOLE-TRACE regression (axon∧dLight masks, dilated variant included).
    """
    path_result = Path(path_result)
    beh_npz = Path(beh_npz)

    with np.load(beh_npz) as ld:
        speed_trace = ld['speed'] if 'speed' in ld else None
        lick_trace = ld['licks'] if 'licks' in ld else None
        run_onset_frames = ld['run_onset_frames'] if 'run_onset_frames' in ld else None

    global_axon_mask = np.load(path_result / 'masks' / 'global_axon_mask.npy')

    for mask_str in ['G_axon_dlight_R_axon_dlight',
                     'G_dilated_axon_dlight_R_dilated_axon_dlight']:
        result_dir = path_result / 'regression_whole_red_trace' / mask_str
        corrected_file = result_dir / 'all_corrected_traces_regress_with_template.npz'

        with np.load(corrected_file) as ld:
            corrected_dlight = ld['corrected_dlight']
            red_trace = ld['red_trace']
            pre = ld['pre_frames'].item()
            post = ld['post_frames'].item()
            
        Single_trial_dLight.run_and_save_significance_analysis(
            corrected_traces_grid=corrected_dlight,
            red_traces_grid=red_trace,
            speed_trace=speed_trace,
            lick_trace=lick_trace,
            event_frames=run_onset_frames,
            mean_img=global_axon_mask,
            output_dir=result_dir,
            event_name='Run_onset',
            pre_frames=pre,
            post_frames=post,
            imaging_rate=imaging_rate,
            p_value_threshold=p_value_threshold,
            n_shuffles=n_shuffles
        )
        
        
def run_post_amp_analysis_whole_trace(
    path_result,
    beh_npz,
    mask_strs=('G_axon_dlight_R_axon_dlight', 'G_dilated_axon_dlight_R_dilated_axon_dlight'),
    post_time_window=(0.2, 1.0),
    p_value_threshold=0.05,
    imaging_rate=30.0,
    n_shuffles=1000
):
    """
    run single-trial post-stimulus amplitude significance analysis on dlight signals
    after regression with whole red trace.

    parameters:
    - path_result: Path to result directory
    - beh_npz: Path to processed behaviour npz
    - mask_strs: tuple of mask subfolder names to process
    - block_to_analyze: grid coordinates for trial-level inspection
    - post_time_window: tuple (s) defining post-event window
    - p_value_threshold: float, permutation test threshold
    - imaging_rate: float, Hz
    - n_shuffles: int, shuffles for permutation test
    """
    import Single_trial_dLight_post_amp_signif as stpost

    path_result = Path(path_result)

    # behaviour
    with np.load(beh_npz) as ld:
        speed_trace       = ld.get('speed')
        lick_trace        = ld.get('licks')
        run_onset_frames  = ld.get('run_onset_frames')
        reward_frames     = ld.get('reward_frames')
    if run_onset_frames is None:
        raise RuntimeError('missing run_onset_frames in behaviour npz')

    global_axon_mask = np.load(path_result / 'masks' / 'global_axon_mask.npy')

    def load_corrected_traces(result_dir):
        corrected_file_to_load = result_dir / 'all_corrected_traces_regress_with_template.npz'
        with np.load(corrected_file_to_load) as ld:
            return (
                ld['original_dlight'],
                ld['corrected_dlight'],
                ld['red_trace'],
                ld['global_mask_green'],
                ld['pre_frames'].item(),
                ld['post_frames'].item(),
            )

    for mask_str in mask_strs:
        result_dir = path_result / 'regression_whole_red_trace' / mask_str
        _, corrected_dlight, red_trace, _, pre, post = load_corrected_traces(result_dir)

        # population-level significance
        stpost.run_and_save_significance_analysis(
            corrected_traces_grid=corrected_dlight,
            red_traces_grid=red_trace,
            speed_trace=speed_trace,
            lick_trace=lick_trace,
            event_frames=run_onset_frames,
            mean_img=global_axon_mask,
            output_dir=result_dir,
            event_name='Run_onset',
            pre_frames=pre,
            post_frames=post,
            imaging_rate=imaging_rate,
            p_value_threshold=p_value_threshold,
            n_shuffles=n_shuffles,
            post_time_window=post_time_window
        )


def run_post_amp_analysis_single_trial(
    path_result,
    beh_npz,
    mask_strs=('G_axon_dlight_R_axon_dlight', 'G_dilated_axon_dlight_R_dilated_axon_dlight'),
    post_time_window=(0.2, 1.0),
    p_value_threshold=0.05,
    imaging_rate=30.0,
    n_shuffles=1000
):
    """
    run single-trial post-stimulus amplitude significance analysis on dlight signals
    after regression with single-trial red traces.

    same parameters as whole-trace variant.
    """
    import Single_trial_dLight_post_amp_signif as stpost

    path_result = Path(path_result)

    # behaviour
    with np.load(beh_npz) as ld:
        speed_trace       = ld.get('speed')
        lick_trace        = ld.get('licks')
        run_onset_frames  = ld.get('run_onset_frames')
    if run_onset_frames is None:
        raise RuntimeError('missing run_onset_frames in behaviour npz')

    global_axon_mask = np.load(path_result / 'masks' / 'global_axon_mask.npy')

    def load_corrected_traces(result_dir):
        with np.load(result_dir / 'all_corrected_traces_regress_with_template.npz') as ld:
            return (
                ld['original_dlight'],
                ld['corrected_dlight'],
                ld['red_trace'],
                ld['global_mask_green'],
                ld['pre_frames'].item(),
                ld['post_frames'].item(),
            )

    for mask_str in mask_strs:
        result_dir = path_result / 'regression_single_trial' / mask_str
        _, corrected_dlight, red_trace, _, pre, post = load_corrected_traces(result_dir)

        stpost.run_and_save_significance_analysis(
            corrected_traces_grid=corrected_dlight,
            red_traces_grid=red_trace,
            speed_trace=speed_trace,
            lick_trace=lick_trace,
            event_frames=run_onset_frames,
            mean_img=global_axon_mask,
            output_dir=result_dir,
            event_name='Run_onset',
            pre_frames=pre,
            post_frames=post,
            imaging_rate=imaging_rate,
            p_value_threshold=p_value_threshold,
            n_shuffles=n_shuffles,
            post_time_window=post_time_window
        )


def run_post_amp_analysis_template(
    path_result,
    beh_npz,
    mask_strs=('G_axon_dlight_R_axon_dlight', 'G_dilated_axon_dlight_R_dilated_axon_dlight'),
    post_time_window=(0.2, 1.0),
    p_value_threshold=0.05,
    imaging_rate=30.0,
    n_shuffles=1000
):
    """
    run single-trial post-stimulus amplitude significance analysis on dlight signals
    after regression with behaviour-averaged template kernels.
    """
    import Single_trial_dLight_post_amp_signif as stpost

    path_result = Path(path_result)

    with np.load(beh_npz) as ld:
        speed_trace      = ld.get('speed')
        lick_trace       = ld.get('licks')
        run_onset_frames = ld.get('run_onset_frames')
    if run_onset_frames is None:
        raise RuntimeError('missing run_onset_frames in behaviour npz')

    global_axon_mask = np.load(path_result / 'masks' / 'global_axon_mask.npy')

    def load_corrected_traces(result_dir):
        with np.load(result_dir / 'all_corrected_traces_regress_with_template.npz') as ld:
            return (
                ld['original_dlight'],
                ld['corrected_dlight'],
                ld['red_trace'],
                ld['global_mask_green'],
                ld['pre_frames'].item(),
                ld['post_frames'].item(),
            )

    for mask_str in mask_strs:
        result_dir = path_result / 'regression_with_template' / mask_str
        _, corrected_dlight, red_trace, _, pre, post = load_corrected_traces(result_dir)

        stpost.run_and_save_significance_analysis(
            corrected_traces_grid=corrected_dlight,
            red_traces_grid=red_trace,
            speed_trace=speed_trace,
            lick_trace=lick_trace,
            event_frames=run_onset_frames,
            mean_img=global_axon_mask,
            output_dir=result_dir,
            event_name='Run_onset',
            pre_frames=pre,
            post_frames=post,
            imaging_rate=imaging_rate,
            p_value_threshold=p_value_threshold,
            n_shuffles=n_shuffles,
            post_time_window=post_time_window
        )
