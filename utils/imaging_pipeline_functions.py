# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:50:53 2024

functions for the Python imaging pipeline
modified: added GPU acceleration using cupy, 1 Nov 2024 Dinghao 

@author: Dinghao Luo
@contributor: Jingyu Cao
"""


#%% imports 
import numpy as np
import os 
import scipy.ndimage
from time import time 
from datetime import timedelta
from tqdm import tqdm
import gc
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt 


#%% constants 
# overflow constant 
of_constant = (2**32-1)/1000


#%% utilities
def sum_mat(matrix):
    """
    this is just a sum sum function
    """
    return sum(map(sum, matrix))

def convolve_gaussian(
        arr, 
        sigma,
        t_axis=0,
        GPU_AVAILABLE=False
        ):
    """
    convolve an array with a gaussian kernel along a specified axis.

    parameters:
    - arr: numpy array
        input array to convolve (1D or multi-dimensional).
    - sigma: float
        standard deviation of the gaussian kernel.
    - t_axis: int, default=0
        axis along which convolution is performed.
    - GPU_AVAILABLE: bool, default=False
        whether to perform convolution on GPU using cupy.

    returns:
    - convolved array: numpy array
        gaussian-convolved array with same shape as input.
    """
    if GPU_AVAILABLE:
        import cupyx.scipy.ndimage as cpximg
        # arr_gpu = cp.array(arr)
        # we assume that the array in on GPU already
        return cpximg.gaussian_filter1d(arr, sigma, axis=t_axis, mode='reflect')
    else:
        return scipy.ndimage.gaussian_filter1d(arr, sigma, axis=t_axis, mode='reflect')

def rolling_min(
        arr, 
        win, 
        t_axis=0,
        GPU_AVAILABLE=False
        ):
    """
    calculate rolling minimum along a specified axis of an array.

    parameters:
    - arr: numpy array
        input array (1D or multi-dimensional).
    - win: int
        size of the rolling window.
    - t_axis: int, default=0
        axis along which rolling minimum is computed.
    - GPU_AVAILABLE: bool, default=False
        whether to perform computation on GPU using cupy.

    returns:
    - minimum-filtered array: numpy array
        rolling minimum array with same shape as input.
    """
    if GPU_AVAILABLE:
        import cupyx.scipy.ndimage as cpximg
        # arr_gpu = cp.array(arr)
        return cpximg.minimum_filter1d(arr, size=win, axis=t_axis, mode='reflect')
    else:
        return scipy.ndimage.minimum_filter1d(arr, size=win, axis=t_axis, mode='reflect')

def rolling_max(
        arr, 
        win, 
        t_axis=0,
        GPU_AVAILABLE=False
        ):
    """
    calculate rolling maximum along a specified axis of an array.

    parameters:
    - arr: numpy array
        input array (1D or multi-dimensional).
    - win: int
        size of the rolling window.
    - t_axis: int, default=0
        axis along which rolling maximum is computed.
    - GPU_AVAILABLE: bool, default=False
        whether to perform computation on GPU using cupy.

    returns:
    - maximum-filtered array: numpy array
        rolling maximum array with same shape as input.
    """
    if GPU_AVAILABLE:
        import cupyx.scipy.ndimage as cpximg
        # arr_gpu = cp.array(arr)
        return cpximg.maximum_filter1d(arr, size=win, axis=t_axis, mode='reflect')
    else:
        return scipy.ndimage.maximum_filter1d(arr, size=win, axis=t_axis, mode='reflect')

def calculate_dFF(
        F_array,
        sigma=300,
        t_axis=0,
        GPU_AVAILABLE=False,
        CHUNK=False,
        chunk_size=2000
        ):
    """
    calculate dF/F for fluorescence traces using gaussian smoothing and rolling min-max baseline.
    
    parameters:
    - F_array: np.ndarray
        fluorescence trace array, can be 1D or ND (time must be along t_axis).
    - sigma: int, default=300
        standard deviation for the gaussian smoothing filter.
    - t_axis: int, default=0
        axis corresponding to time.
    - GPU_AVAILABLE: bool, default=False
        whether to use CuPy to accelerate filtering operations.
    - CHUNK: bool, default=False
        whether to apply the dF/F calculation in memory-safe chunks.
    - chunk_size: int, default=10000
        number of timepoints per chunk if CHUNK is True.
    
    returns:
    - dFF: np.ndarray
        array of same shape as F_array, containing the computed dF/F values.
    """
    window = sigma * 6
    T = F_array.shape[t_axis]

    if not CHUNK:
        # full-array processing
        if GPU_AVAILABLE:
            import cupy as cp
            F_array = cp.array(F_array, dtype=cp.float32)
        else:
            F_array = F_array.astype(np.float32, copy=False)

        baseline = convolve_gaussian(F_array, sigma, t_axis, GPU_AVAILABLE)
        baseline = rolling_min(baseline, window, t_axis, GPU_AVAILABLE)
        baseline = rolling_max(baseline, window, t_axis, GPU_AVAILABLE)
        dFF = (F_array - baseline) / baseline

        return dFF.get() if GPU_AVAILABLE else dFF

    # chunked processing
    pad = window // 2
    slices = []

    if GPU_AVAILABLE:
        import cupy as cp

        for start in tqdm(range(0, T, chunk_size),
                          desc='chunked dFF calculation on GPU...'):
            chunk_start = max(0, start - pad)
            chunk_end = min(T, start + chunk_size + pad)

            # extract and move to GPU
            slicer = [slice(None)] * F_array.ndim
            slicer[t_axis] = slice(chunk_start, chunk_end)
            chunk_gpu = cp.array(F_array[tuple(slicer)], dtype=cp.float32)

            baseline = convolve_gaussian(chunk_gpu, sigma, t_axis, GPU_AVAILABLE=True)
            baseline = rolling_min(baseline, window, t_axis, GPU_AVAILABLE=True)
            baseline = rolling_max(baseline, window, t_axis, GPU_AVAILABLE=True)

            dFF_chunk = (chunk_gpu - baseline) / baseline
            dFF_chunk = dFF_chunk.get()

            # trim overlap/padding
            slicer_out = [slice(None)] * dFF_chunk.ndim
            slicer_out[t_axis] = slice(pad, -pad) if (start > 0 and chunk_end < T) else \
                                 slice(pad, None) if start > 0 else \
                                 slice(None, -pad) if chunk_end < T else \
                                 slice(None)
            slices.append(dFF_chunk[tuple(slicer_out)])

        return np.concatenate(slices, axis=t_axis)

    else:
        for start in tqdm(range(0, T, chunk_size),
                          desc='chunked dFF calculation on CPU...'):
            chunk_start = max(0, start - pad)
            chunk_end = min(T, start + chunk_size + pad)

            slicer = [slice(None)] * F_array.ndim
            slicer[t_axis] = slice(chunk_start, chunk_end)
            chunk = F_array[tuple(slicer)].astype(np.float32, copy=False)

            baseline = convolve_gaussian(chunk, sigma, t_axis, GPU_AVAILABLE=False)
            baseline = rolling_min(baseline, window, t_axis, GPU_AVAILABLE=False)
            baseline = rolling_max(baseline, window, t_axis, GPU_AVAILABLE=False)

            dFF_chunk = (chunk - baseline) / baseline

            slicer_out = [slice(None)] * dFF_chunk.ndim
            slicer_out[t_axis] = slice(pad, -pad) if (start > 0 and chunk_end < T) else \
                                 slice(pad, None) if start > 0 else \
                                 slice(None, -pad) if chunk_end < T else \
                                 slice(None)
            slices.append(dFF_chunk[tuple(slicer_out)])

        return np.concatenate(slices, axis=t_axis)
    
def calculate_dFF_abs(
        F_array, 
        window=1800, 
        sigma=300, 
        GPU_AVAILABLE=False
        ): # Jingyu, 3/16/25 for testing std, cv and dFF
    """
    calculate absolute dF/F using a smoothed rolling baseline.

    parameters:
    - F_array: np.ndarray
        fluorescence traces (ROIs x time).
    - window: int, default=1800
        rolling window size for baseline computation.
    - sigma: int, default=300
        gaussian smoothing parameter before rolling min/max.
    - GPU_AVAILABLE: bool, default=False
        whether to use CuPy to accelerate processing.

    returns:
    - dFF_abs: np.ndarray
        array of same shape as F_array, containing absolute dF/F values.
    """

def filter_outlier(
        F_array, 
        std_threshold=10
        ):
    """
    remove high-amplitude artefact outliers from each ROI trace using local interpolation.
    
    parameters:
    - F_array: np.ndarray
        array of ROI traces (ROIs x timepoints).
    - std_threshold: float, default=10
        threshold (in SD) for marking outliers.
    
    returns:
    - cleaned_array: np.ndarray
        same shape as F_array with outliers smoothed using neighbours.
    """
    means = F_array.mean(axis=1)  # mean of each ROI trace
    stds = F_array.std(axis=1)  # std of each ROI trace 
    tot_roi = F_array.shape[0]
    
    for r in range(tot_roi):
        outlier_ind = np.where(F_array[r]>=means[r]+stds[r]*std_threshold)[0]
        
        for i in outlier_ind:  # this is assuming that outliers are artefacts that do not show continuity
            if i+1<F_array.shape[1]:
                F_array[r,i] = (F_array[r,i-1]+F_array[r,i+1])/2
            else:
                F_array[r,i] = F_array[r,i-1]
            
    return F_array

def spatial_gaussian_filter(
        movie,
        sigma_spatial=1,
        GPU_AVAILABLE=False,
        CHUNK=False,
        chunk_size=2000
        ):
    """
    apply spatial gaussian filtering (only over y and x) to a 3D movie in memory-efficient chunks.

    parameters:
    - movie: np.ndarray
        3D array (frames x height x width).
    - sigma_spatial: float, default=1
        spatial std for gaussian filter over height and width.
    - GPU_AVAILABLE: bool, default=False
        use CuPy GPU filtering if available.
    - CHUNK: bool, default=False
        whether to filter in smaller chunks along the frame axis.
    - chunk_size: int, default=500
        number of frames per chunk.

    returns:
    - filtered_movie: np.ndarray
        array of same shape as movie, spatially filtered.
    """
    T, H, W = movie.shape

    if not CHUNK:
        # full-array processing
        if GPU_AVAILABLE:
            import cupy as cp
            from cupyx.scipy.ndimage import gaussian_filter
            movie_gpu = cp.array(movie, dtype=cp.float32)
            filtered = gaussian_filter(movie_gpu, sigma=(0, sigma_spatial, sigma_spatial))
            return cp.asnumpy(filtered)
        else:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(movie, sigma=(0, sigma_spatial, sigma_spatial))

    # chunked processing
    filtered = np.empty_like(movie)

    if GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter

        for start in tqdm(range(0, T, chunk_size),
                          desc='chunked spatial-filtering frames on GPU...'):
            end = min(start + chunk_size, T)
            chunk_gpu = cp.asarray(movie[start:end])
            filtered_chunk = gaussian_filter(chunk_gpu, sigma=(0, sigma_spatial, sigma_spatial))
            filtered[start:end] = cp.asnumpy(filtered_chunk)

    else:
        from scipy.ndimage import gaussian_filter

        for start in tqdm(range(0, T, chunk_size),
                          desc='chunked spatial-filtering frames on CPU...'):
            end = min(start + chunk_size, T)
            filtered[start:end] = gaussian_filter(movie[start:end], sigma=(0, sigma_spatial, sigma_spatial))

    return filtered


#%% grid functions 
def check_stride_border(stride, border, dim=512):
    """
    check whether the image dimensions align with the grid configuration.

    parameters:
    - stride: int
        number of pixels between each grid point.
    - border: int
        number of pixels ignored on each border.
    - dim: int, default=512
        image dimension (assumes square image).

    returns:
    - is_valid: bool
        True if stride fits evenly, else False (also prints a warning).
    """
    if not np.mod(512-border*2, stride)==0:
        print('\n***\nWARNING:\nborder does not fit stride.\n***\n')
    return np.mod(512-border*2, stride)==0


def make_grid(stride=8, dim=512, border=0):
    """
    create a list of grid points spanning an image dimension.

    parameters:
    - stride: int, default=8
        spacing between grid points.
    - dim: int, default=512
        size of the image dimension.
    - border: int, default=0
        number of pixels to exclude from edges.

    returns:
    - grid: list[int]
        list of pixel positions for grid lines.
    """
    return list(np.arange(0+border, dim-border, stride))

def run_grid(
        frame, 
        grids, 
        tot_grid, 
        stride=8, 
        GPU_AVAILABLE=False
        ):
    """
    extract grid patches from a 2D image.

    parameters:
    - frame: np.ndarray
        2D input image (height x width).
    - grids: list[int]
        grid positions (usually generated by make_grid).
    - tot_grid: int
        total number of grid patches to extract.
    - stride: int, default=8
        side length of each patch.
    - GPU_AVAILABLE: bool, default=False
        whether to use CuPy to extract grid patches.

    returns:
    - gridded: np.ndarray
        3D array of shape (tot_grid x stride x stride).
    """
    # plot the mean image (Z-projection)
    grid_count = 0

    if GPU_AVAILABLE:
        frame_gpu = cp.array(frame)
        gridded_gpu = cp.zeros((tot_grid, stride, stride))  # initialise array in VRAM
        for hgp in grids:
            for vgp in grids:
                gridded_gpu[grid_count,:,:] = frame_gpu[hgp:hgp+stride, vgp:vgp+stride]
                grid_count+=1
        gridded = gridded_gpu.get()  # move back to RAM
    else:
        gridded = np.zeros((tot_grid, stride, stride))  # initialise array in RAM
        for hgp in grids:
            for vgp in grids:
                gridded[grid_count,:,:] = frame[hgp:hgp+stride, vgp:vgp+stride]
                grid_count+=1
            
    return gridded

def plot_reference(
        mov, 
        outpath=r'', 
        recname='',
        grids=-1, 
        stride=-1, 
        dim=512, 
        channel=1,
        GPU_AVAILABLE=False
        ): 
    """
    plot a reference image (mean Z-projection) with optional grid annotations
    
    parameters
    ----------
    mov : np.ndarray or cupy.ndarray
        imaging data as a 3D array (frames x height x width)
    grids : list[int] or int, optional
        grid line positions for annotation; set to -1 to disable grid processing
    stride : int, optional
        stride length between grid lines; required if grids is not -1
    dim : int, optional
        dimension of the imaging data for plotting (default: 512)
    channel : int, optional
        channel number to annotate in the title (default: 1)
    outpath : str, optional
        output path to save the reference image and array (default: '')
    GPU_AVAILABLE : bool, optional
        flag to enable GPU processing using cupy (default: False)
    
    returns
    -------
    ref_im : np.ndarray
        processed reference image (mean Z-projection)
    
    notes
    -----
    - If `grids` is not -1, the function will annotate the reference image with
      vertical and horizontal grid lines based on the provided positions and stride.
    - If `GPU_AVAILABLE` is True, the function uses cupy to process data on the GPU,
      otherwise it defaults to numpy for CPU-based processing.
    - The function saves the reference image and the numpy array (`ref_im`) to the
      specified `outpath`.
    """
    proc_path = rf'{outpath}\processed_data'
    os.makedirs(proc_path, exist_ok=True)
    
    if grids!=-1:  # if one is using grid-processing; else don't do anything
        boundary_low = grids[0]
        boundary_high = grids[-1]+stride

    # plot the mean image (Z-projection)
    if GPU_AVAILABLE:
        mov_gpu = cp.array(mov)  # move to VRAM
        ref_im_gpu = cp.mean(mov_gpu, axis=0)
        ref_im = ref_im_gpu.get()  # move back to RAM
    else:
        ref_im = np.mean(mov, axis=0)

    ref_im = post_processing_suite2p_gui(ref_im)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(ref_im, aspect='auto', cmap='gist_gray', interpolation='none',
              extent=[0, dim, dim, 0])
    if grids!=-1:
        for i in range(len(grids)):  # vertical lines 
            ax.plot([grids[i], grids[i]], [boundary_low, boundary_high], color='grey', linewidth=1, alpha=.5)
            ax.plot([boundary_low, boundary_high], [grids[i], grids[i]], color='grey', linewidth=1, alpha=.5)
        ax.plot([grids[-1]+stride, grids[-1]+stride], [boundary_low, boundary_high], color='grey', linewidth=1, alpha=.5)  # last vertical line 
        ax.plot([boundary_low, boundary_high], [grids[-1]+stride, grids[-1]+stride], color='grey', linewidth=1, alpha=.5)  # last horizontal line
    ax.set(xlim=(0,dim), ylim=(0,dim))

    fig.suptitle(f'ref ch{channel}')
    fig.tight_layout()
    if grids!=-1:  # grid-analysis 
        fig.savefig(rf'{outpath}\{recname}_ref_ch{channel}_{stride}.png',
                    dpi=300,
                    bbox_inches='tight')
    else:  # regular 
        fig.savefig(rf'{outpath}\{recname}_ref_ch{channel}.png',
                    dpi=300,
                    bbox_inches='tight')
    np.save(rf'{proc_path}\{recname}_ref_mat_ch{channel}.npy', ref_im)
    plt.close(fig)

    # explicitly clear VRAM
    if GPU_AVAILABLE:
        del mov_gpu, ref_im_gpu
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return ref_im


def load_or_generate_reference_images(
        proc_path, proc_data_path, 
        bin_path, bin2_path, tot_frames, ops, 
        GPU_AVAILABLE
        ):
    """
    load or generate reference images for channels 1 and 2
    
    parameters
    ----------
    proc_path : str
        path to the saved reference images 
    bin_path : str
        path to the binary file for channel 1 data
    bin2_path : str
        path to the binary file for channel 2 data
    tot_frames : int
        total number of frames in the imaging data
    ops : dict
        suite2p operations dictionary containing image dimensions
    GPU_AVAILABLE : bool
        flag for GPU availability
    
    returns
    -------
    ref_im : np.ndarray
        reference image for channel 1
    ref_ch2_im : np.ndarray
        reference image for channel 2
    """
    ref_path = rf'{proc_data_path}\ref_mat_ch1.npy'
    ref_ch2_path = rf'{proc_data_path}\ref_mat_ch2.npy'
    if any(not os.path.exists(path) 
           for path in [proc_path, ref_path, ref_ch2_path]):
        os.makedirs(proc_path, exist_ok=True)
        os.makedirs(proc_data_path, exist_ok=True)
        shape = (tot_frames, ops['Ly'], ops['Lx'])
        print('generating reference images...')
        
        recname = proc_path.split('\\')[-1]
        
        # generate reference image for channel 1
        start = time()
        try:
            mov = np.memmap(bin_path, mode='r', dtype='int16', shape=shape)
            ref_im = plot_reference(
                mov, 
                recname=recname,
                channel=1, 
                outpath=proc_path, 
                GPU_AVAILABLE=GPU_AVAILABLE
                )
        except Exception as e:
            raise IOError(f'failed to memory-map .bin file: {e}')
        finally:
            if mov is not None:
                mov._mmap.close()
                del mov
                gc.collect()
        print(f'ref done ({str(timedelta(seconds=int(time() - start)))})')
        
        # generate reference image for channel 2
        start = time()
        try:
            mov2 = np.memmap(bin2_path, mode='r', dtype='int16', shape=shape)
            ref_ch2_im = plot_reference(
                mov2, 
                recname=recname,
                channel=2,
                outpath=proc_path, 
                GPU_AVAILABLE=GPU_AVAILABLE
                )
        except Exception as e:
            raise IOError(f'failed to memory-map .bin file: {e}')
        finally:
            if mov2 is not None:
                mov2._mmap.close()
                del mov2
                gc.collect()
        print(f'ref_ch2 done ({str(timedelta(seconds=int(time() - start)))})')
        
    else:
        print(f'ref images already generated\nloading ref_im from {ref_path}...')
        try:
            ref_im = np.load(ref_path, allow_pickle=True)
            ref_ch2_im = np.load(ref_ch2_path, allow_pickle=True)
        except Exception as e:
            raise IOError(f'paths exist but failure occurred when loading ref images: {e}')
    
    return ref_im, ref_ch2_im
        
    
def post_processing_suite2p_gui(img_orig):
    """
    apply percentile-based contrast normalisation and rescale to 8-bit for GUI display.
    
    parameters:
    - img_orig: np.ndarray
        input 2D image.
    
    returns:
    - img_proc: np.ndarray
        normalised image (uint8, range 0–255).
    """
    # normalize to 1st and 99th percentile
    perc_low, perc_high = np.percentile(img_orig, [1, 99])
    img_proc = (img_orig - perc_low) / (perc_high - perc_low)
    img_proc = np.maximum(0, np.minimum(1, img_proc))

    # convert to uint8
    img_proc *= 255
    img_proc = img_proc.astype(np.uint8)

    return img_proc
    
def find_nearest(value, arr):
    """
    find the index of the value in arr that is closest to the input value.

    parameters:
    - value: the target value to compare against
    - arr: a list of numeric values

    returns:
    - nearest_value_index: the index of the value in arr closest to the input value
    """
    nearest_value = min(arr, key=lambda x: abs(x-value))
    nearest_value_index = arr.index(nearest_value)
    return nearest_value_index
    
    
#%% behaviour file processing
def process_txt(txtfile):
    """
    parse behavioural .txt log file into structured trial-wise data.
       
    parameters:
    - txtfile: str
        path to the behavioural log file.
       
    returns:
    - logfile: dict
        dictionary containing fields like 'speed_times', 'lick_times', etc.
    """
    curr_logfile = {} 
    file = open(txtfile, 'r')
    
    line = ['']
    while line[0] != '$TR':
        line = get_next_line(file)
        
    lick_times = []
    pump_times = []
    movie_times = []
    speed_times = []
    motor_times = []
    pulse_times = []
    frame_times = []
    
    mt_trial = []
    wt_trial = []
    lt_trial = []
    pt_trial = []
    mv_trial = []
    pc_trial = []
    pulse_command_list = []
    current_pulse_command = []
    
    trial_statements = []
    
    while line[0].find('$') == 0:
        if line[0] == '$TR' and len(speed_times)>0: # need to update motor_times here - but ignore motors before first trial started. 
            motor_times.append(mt_trial)
            mt_trial = []
            trial_statements.append(line)
        if line[0] == '$MV':
            mv_trial.append([float(line[1]), float(line[2])])
        if line[0] == '$WE':
            wt_trial.append([float(line[1]), float(line[2])*.04*50, float(line[3])])  # 2nd value in each line is the number of clicks per 20 ms, and each click corresponds to .04 cm, Dinghao, 20240625
            # wt_trial.append([float(line[1]), float(line[2])*.04*50, float(line[3])])  # old way of doing this, reading in the 2nd value directly, but it is not the true speed
        if line[0] == '$LE' and line[3] == '1':
            lt_trial.append([float(line[1]), float(line[2])]) 
        if line[0] == '$PE' and line[3] == '1':
            pt_trial.append(float(line[1]))
        if line[0] == '$MT':
            mt_trial.append([float(line[1]), float(line[2])])
        if line[0] == '$PC':
            pc_trial.append(float(line[1]))
        if line[0] == '$PP':
            current_pulse_command = line       
        if line[0] == '$NT':
            lick_times.append(lt_trial)
            movie_times.append(mv_trial)
            pump_times.append(pt_trial)
            speed_times.append(wt_trial)
            pulse_times.append(pc_trial)
            pulse_command_list.append(current_pulse_command)
            lt_trial = []
            mv_trial = []
            pt_trial = []
            wt_trial = []
            pc_trial = []
        if line[0] == '$FM' and line[2] == '0':
            frame_times.append(float(line[1]))
        line = get_next_line(file)
        
    curr_logfile['speed_times'] = speed_times
    curr_logfile['movie_times'] = movie_times
    curr_logfile['lick_times'] = lick_times
    curr_logfile['pump_times'] = pump_times
    curr_logfile['motor_times'] = motor_times
    curr_logfile['pulse_times'] = pulse_times
    curr_logfile['frame_times'] = frame_times
    curr_logfile['trial_statements'] = trial_statements
    curr_logfile['pulse_descriptions'] = pulse_command_list
    
    return curr_logfile

def process_txt_nobeh(txtfile):
    """
    simplified parser for behavioural .txt log file when no behavioural data is expected.
    
    parameters:
    - txtfile: str
        path to the log file.
    
    returns:
    - logfile: dict
    """
    curr_logfile = {} 
    file = open(txtfile, 'r')
    
    line = get_next_line(file)
        
    pulse_times = []
    pulse_parameters = []
    frame_times = []
    wheel_dummy = 0
    
    while line[0].find('$') == 0:
        if line[0] == '$WE':
            wheel_dummy += 1
        if line[0] == '$PC':
            pulse_times.append(float(line[1]))
        if line[0] == '$PP':
            pulse_parameters.append([s for s in line[1:]])
        if line[0] == '$FM' and line[2] == '0':
            frame_times.append(float(line[1]))
        line = get_next_line(file)
        
    curr_logfile['pulse_times'] = pulse_times
    curr_logfile['pulse_parameters'] = pulse_parameters
    curr_logfile['frame_times'] = frame_times
    
    # flag for 'well, actually this one HAS behaviour', 24 June 2025
    curr_logfile['behaviour'] = False
    if wheel_dummy > 5:
        curr_logfile['behaviour'] = True
    
    return curr_logfile

def get_next_line(file):
    """
    read and return the next non-empty, comma-split line from a file.
    
    parameters:
    - file: file object
        open text file with behavioural data.
    
    returns:
    - line: list[str]
        split line contents (ignores empty lines).
    """
    line = file.readline().rstrip('\n').split(',')
    if len(line) == 1: # read an empty line
        line = file.readline().rstrip('\n').split(',')
    return line


def correct_overflow(data, label):
    """
    correct time-overflow in behavioural signals (e.g. due to 32-bit counter wraparound).

    parameters:
    - data: list
        behavioural time series data (trial-structured).
    - label: str
        name of the data type: one of ['speed', 'lick', 'pump', 'movie', 'frame'].

    returns:
    - corrected_data: list
        overflow-corrected version of input data.
    """
    tot_trial = len(data)
    new_data = []
    
    if label=='speed':
        curr_time = data[0][0][0]
        for t in range(tot_trial):
            if data[t][-1][0]-curr_time>=0:  # if the last speed cell is within overflow, then simply append
                new_data.append(data[t])
                curr_time = data[t][-1][0]
            else:  # once overflow is detected, do not update curr_time
                new_trial = []
                curr_trial = data[t]
                curr_length = len(curr_trial)
                for s in range(curr_length):
                    if curr_trial[s][0]-curr_time>0:
                        new_trial.append(curr_trial[s])
                    else:
                        new_trial.append([curr_trial[s][0]+of_constant, curr_trial[s][1], curr_trial[s][2]])
                new_data.append(new_trial)              
    if label == 'lick':  # added by Jingyu 6/22/2024
        first_trial_with_licks = next(x for x in data if len(x)!=0)  # in case the first trial has no licks, Dinghao, 20240626
        curr_time = first_trial_with_licks[0][0]
        for t in range(tot_trial):
            if len(data[t])==0:  # if there is no lick, append an empty list
                new_data.append([])
            elif data[t][-1][0]-curr_time>=0:  # if the last lick cell is within overflow, then simply append
                new_data.append(data[t])
                curr_time = data[t][-1][0]
            else:  # once overflow is detected, do not update curr_time
                new_trial = []
                curr_trial = data[t]
                curr_length = len(curr_trial)
                for s in range(curr_length):
                    if curr_trial[s][0]-curr_time>0:
                        new_trial.append(curr_trial[s])
                    else:
                        new_trial.append([curr_trial[s][0]+of_constant, curr_trial[s][1]])
                new_data.append(new_trial)
    if label=='pump':
        first_trial_with_pump = next(x for x in data if len(x)!=0)  # in case the first trial has no pump, Dinghao, 20240704
        curr_time = first_trial_with_pump[0]
        for t in range(tot_trial):
            if len(data[t])!=0:
                if data[t][0]-curr_time>=0:
                    new_data.append(data[t][0])
                    curr_time = data[t][0]
                else:  # once overflow is detected, do not update curr_time
                    new_data.append(data[t][0]+of_constant)
    if label=='movie':
        curr_time = data[0][0][0]
        for t in range(tot_trial):
            if data[t][-1][0]-curr_time>=0:
                new_data.append(data[t])
                curr_time = data[t][-1][0]
            else:  # once overflow is detected, do not update curr_time
                new_trial = []
                curr_trial = data[t]
                curr_length = len(curr_trial)
                for s in range(curr_length):
                    if curr_trial[s][0]-curr_time>=0: #Jingyu 8/18/24
                        new_trial.append(curr_trial[s])
                    else:
                        new_trial.append([curr_trial[s][0]+of_constant, curr_trial[s][1]])
                new_data.append(new_trial)
    if label=='frame':
        curr_time = data[0]
        for f in data:
            if f-curr_time>=0:
                new_data.append(f)
                curr_time = f
            else:  # once overflow is detected, do not update curr_time
                new_data.append(f+of_constant)
    
    return new_data

def get_onset(uni_speeds, uni_times, threshold=0.3):  # 0.3 seconds
    """
    find the index corresponding to the start of a continuous running episode.

    parameters:
    - uni_speeds: list[float]
        speed values sampled over time.
    - uni_times: list[float]
        time values in milliseconds.
    - threshold: float, default=0.3
        duration (s) of continuous fast running required.

    returns:
    - onset_index: float
        timestamp of run onset; returns -1 if none found.
    """
    count = 0
    for i in range(len(uni_speeds)):
        count = fast_in_a_row(uni_speeds[i], count, 10)
        if count>threshold*1000:
            index = uni_times[i]-threshold*1000
            break
    if count<threshold*1000:
        index = -1  # cannot find clear run-onsets
    return index

def fast_in_a_row(speed_value, count, threshold):
    """
    count consecutive samples above a speed threshold.
    
    parameters:
    - speed_value: float
        current speed value.
    - count: int
        current count of consecutive fast samples.
    - threshold: float
        minimum speed for "fast".
    
    returns:
    - updated_count: int
        updated counter value.
    """
    if speed_value > threshold:
        count-=-1
    else:
        count=0
    return count