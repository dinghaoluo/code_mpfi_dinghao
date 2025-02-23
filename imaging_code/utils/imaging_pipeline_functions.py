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
from time import time 
from datetime import timedelta
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
    Parameters
    ----------
    matrix : numpy array
        A 2D array to be summed over.

    Returns
    -------
        Sum of all values in the 2D array.
    """
    return sum(map(sum, matrix))

def gaussian_kernel_unity(sigma):
    """
    Calculates a 1D Gaussian convolution kernel that sums to unity
    """
    kernel_size = int(6 * sigma + 1)
    x = np.arange(kernel_size) - (kernel_size // 2)
    kernel = np.exp(-(x**2 / (2 * sigma**2)))
    kernel /= kernel.sum()  # normalisation to ensure the unity sum
    return kernel

def convolve_gaussian(arr, sigma, GPU_AVAILABLE):
    kernel = gaussian_kernel_unity(sigma) 
    pad_width = len(kernel) // 2  # pad symmetrically at the edges to eliminate edge effects 

    if GPU_AVAILABLE:
        kernel = cp.array(kernel)  # move to VRAM
        if len(arr.shape)>1:  # more than 1 ROIs
            arr_gpu_padded = cp.pad(cp.array(arr), ((0, 0), (pad_width, pad_width)), mode='reflect')
            # apply convolution across axis 1 without explicit looping
            return cp.apply_along_axis(lambda x: cp.convolve(x, kernel, mode='same'), 
                                       axis=1, 
                                       arr=arr_gpu_padded)[:, pad_width:-pad_width]  # do not need to .get() since GPU_AVAILABLE doesn't change 
        else:  # 1 ROI
            arr_gpu_padded = cp.pad(cp.array(arr), (pad_width, pad_width), mode='reflect')
            return cp.convolve(arr_gpu_padded, kernel, mode='same')[pad_width:-pad_width]
    else:
        if len(arr.shape)>1:  # more than 1 ROIs
            arr_padded = np.pad(arr, ((0, 0), (pad_width, pad_width)), mode='reflect')
            return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 
                                       axis=1, 
                                       arr=arr_padded)[:, pad_width:-pad_width]
        else:  # 1 ROI
            arr_padded = np.pad(arr, (pad_width, pad_width), mode='reflect')
            return np.convolve(arr_padded, kernel, mode='same')[pad_width:-pad_width]

def rolling_min(arr, win, GPU_AVAILABLE):
    if len(arr.shape)>1:  # 2D
        length = arr.shape[1]
    else:  # 1D
        length = arr.shape[0]
    
    if GPU_AVAILABLE:
        half_win = int(cp.ceil(win/2))
        arr_gpu = cp.array(arr)  # move to VRAM
        if len(arr.shape)>1:
            array_padding = cp.hstack((arr_gpu[:,:half_win], arr_gpu, arr_gpu[:,-half_win:length]))
            output = cp.array([cp.min(array_padding[:,i:i+win], axis=1) for i in range(half_win,length+half_win)]).T
        else:
            array_padding = cp.hstack((arr_gpu[:half_win], arr_gpu, arr_gpu[-half_win:length]))
            output = cp.array([cp.min(array_padding[i:i+win]) for i in range(half_win,length+half_win)]).T
        return output
    else:
        half_win = int(np.ceil(win/2))
        if len(arr.shape)>1:
            array_padding = np.hstack((arr[:,:half_win], arr, arr[:,-half_win:length]))
            output = np.array([np.min(array_padding[:,i:i+win], axis=1) for i in range(half_win,length+half_win)]).T
        else:
            array_padding = np.hstack((arr[:half_win], arr, arr[-half_win:length]))
            output = np.array([np.min(array_padding[i:i+win]) for i in range(half_win,length+half_win)]).T
        return output

def rolling_max(arr, win, GPU_AVAILABLE):
    if len(arr.shape)>1:  # 2D
        length = arr.shape[1]
    else:  # 1D
        length = arr.shape[0]
    
    if GPU_AVAILABLE:
        half_win = int(cp.ceil(win/2))
        arr_gpu = cp.array(arr)  # move to VRAM
        if len(arr.shape)>1:
            array_padding = cp.hstack((arr_gpu[:,:half_win], arr_gpu, arr_gpu[:,-half_win:length]))
            output = cp.array([cp.max(array_padding[:,i:i+win], axis=1) for i in range(half_win,length+half_win)]).T
        else:
            array_padding = cp.hstack((arr_gpu[:half_win], arr_gpu, arr_gpu[-half_win:length]))
            output = cp.array([cp.max(array_padding[i:i+win]) for i in range(half_win,length+half_win)]).T
        return output
    else:
        half_win = int(np.ceil(win/2))
        if len(arr.shape)>1:
            array_padding = np.hstack((arr[:,:half_win], arr, arr[:,-half_win:length]))
            output = np.array([np.max(array_padding[:,i:i+win], axis=1) for i in range(half_win,length+half_win)]).T
        else:
            array_padding = np.hstack((arr[:half_win], arr, arr[-half_win:length]))
            output = np.array([np.max(array_padding[i:i+win]) for i in range(half_win,length+half_win)]).T
        return output


def calculate_dFF(F_array, window=1800, sigma=300, GPU_AVAILABLE=False):
    """
    Parameters
    ----------
    F_array : numpy array
        array with fluorescence traces for each ROI.
    window : int, default=1800
        Window for calculating baselines.
    sigma : int, default=300
        Sigma for Gaussian filter.

    Returns
    -------
        2D array containing the dFF for each ROI.

    """
    # print('GPU_AVAILABLE: {}'.format(GPU_AVAILABLE))
    baseline = convolve_gaussian(F_array, sigma, GPU_AVAILABLE)
    baseline = rolling_min(baseline, window, GPU_AVAILABLE)
    baseline = rolling_max(baseline, window, GPU_AVAILABLE)
    
    if GPU_AVAILABLE: F_array = cp.array(F_array)  # if GPU_AVAILABLE, baseline will remain on GPU
    
    dFF = (F_array-baseline)/baseline
    
    if GPU_AVAILABLE:
        return dFF.get()
    else:
        return dFF

def filter_outlier(F_array, std_threshold=10):
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


#%% grid functions 
def check_stride_border(stride, border, dim=512):
    """
    Parameters
    ----------
    stride : int
        How many pixels per grid (stride x stride).
    border : int
        How many pixels to ignore at the border of the movie.
    dim : int, default=512
        Dimensions of the movie.

    Returns
    -------
    NONE
    """
    if not np.mod(512-border*2, stride)==0:
        print('\n***\nWARNING:\nborder does not fit stride.\n***\n')
    return np.mod(512-border*2, stride)==0


def make_grid(stride=8, dim=512, border=0):
    """
    Parameters
    ----------
    stride : int, default=8
        How many pixels per grid.
    dim : int, default=512
        x/y dimension; either should do since we are imaging squared images.

    Returns
    -------
    a list of grid points.
    """
    return list(np.arange(0+border, dim-border, stride))

def run_grid(frame, grids, tot_grid, stride=8, GPU_AVAILABLE=False):
    """
    Parameters
    ----------
    frame : array
        current frame as an array (default dim.=512x512).
    grid_list : list 
        a list of grid points.
    tot_grid : int
        total number of grids.
    stride : int, default=8
        how many pixels per grid.

    Returns
    -------
    gridded : array
        3-dimensional array at tot_grid x stride x stride.
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

def plot_reference(mov, grids=-1, stride=-1, dim=512, channel=1, outpath=r'', GPU_AVAILABLE=False): 
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
        fig.savefig(r'{}\ref_ch{}_{}.png'.format(outpath, channel, stride),
                    dpi=300,
                    bbox_inches='tight')
        np.save(r'{}\ref_mat_ch{}.npy'.format(outpath, channel), ref_im)
    else:  # regular 
        fig.savefig(r'{}\ref_ch{}.png'.format(outpath, channel),
                    dpi=300,
                    bbox_inches='tight')
        np.save(r'{}\ref_mat_ch{}.npy'.format(outpath, channel), ref_im)
    plt.close(fig)

    # explicitly clear VRAM
    if GPU_AVAILABLE:
        del mov_gpu, ref_im_gpu
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return ref_im


def load_or_generate_reference_images(proc_path, bin_path, bin2_path, tot_frames, ops, GPU_AVAILABLE):
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
    ref_path = os.path.join(proc_path, 'ref_mat_ch1.npy')
    ref_ch2_path = os.path.join(proc_path, 'ref_mat_ch2.npy')
    if not os.path.exists(proc_path) or not os.path.exists(ref_path) or not os.path.exists(ref_ch2_path):
        os.makedirs(proc_path, exist_ok=True)  # create proc_path if it does not already exist 
        shape = (tot_frames, ops['Ly'], ops['Lx'])
        print('generating reference images...')
        
        # Generate reference image for channel 1
        start = time()
        try:
            mov = np.memmap(bin_path, mode='r', dtype='int16', shape=shape)
            ref_im = plot_reference(mov, channel=1, outpath=proc_path, GPU_AVAILABLE=GPU_AVAILABLE)
        except Exception as e:
            raise IOError(f'failed to memory-map .bin file: {e}')
        finally:
            if mov is not None:
                mov._mmap.close()
        print('ref done ({})'.format(str(timedelta(seconds=int(time() - start)))))
        
        # Generate reference image for channel 2
        start = time()
        try:
            mov2 = np.memmap(bin2_path, mode='r', dtype='int16', shape=shape)
            ref_ch2_im = plot_reference(mov2, channel=2, outpath=proc_path, GPU_AVAILABLE=GPU_AVAILABLE)
        except Exception as e:
            raise IOError(f'failed to memory-map .bin file: {e}')
        finally:
            if mov2 is not None:
                mov2._mmap.close()
        print('ref_ch2 done ({})'.format(str(timedelta(seconds=int(time() - start)))))
        
    else:
        print(f'ref images already generated\nloading ref_im from {ref_path}...')
        try:
            ref_im = np.load(ref_path, allow_pickle=True)
            ref_ch2_im = np.load(ref_ch2_path, allow_pickle=True)
        except Exception as e:
            raise IOError(f'paths exist but failure occurred when loading ref images: {e}')
    
    return ref_im, ref_ch2_im
        
    
def post_processing_suite2p_gui(img_orig):
    '''
    no idea what this does but ok
    apparently it does something to the image
    '''
    # normalize to 1st and 99th percentile
    perc_low, perc_high = np.percentile(img_orig, [1, 99])
    img_proc = (img_orig - perc_low) / (perc_high - perc_low)
    img_proc = np.maximum(0, np.minimum(1, img_proc))

    # convert to uint8
    img_proc *= 255
    img_proc = img_proc.astype(np.uint8)

    return img_proc
    
def find_nearest(value, arr):
    # return value and index of nearest value in arr to input value
    nearest_value = min(arr, key=lambda x: abs(x-value))
    nearest_value_index = arr.index(nearest_value)
    return nearest_value_index
    
    
#%% behaviour file processing
def process_txt(txtfile):
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
    

def get_next_line(file):
    line = file.readline().rstrip('\n').split(',')
    if len(line) == 1: # read an empty line
        line = file.readline().rstrip('\n').split(',')
    return line


def correct_overflow(data, label):
    """
    Parameters
    ----------
    data : list
        speed_times, pump_times, frame_times, movie_times etc.
    label : str
        the label of the data array (eg. 'speed').

    Returns
    -------
    new_data : list
        data corrected for overflow.
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
    if speed_value > threshold:
        count-=-1
    else:
        count=0
    return count