# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:54:04 2022

a collection of commonly used functions

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import os


#%% os functions 
def scan_directory_tree(path, indent='', is_first_level=True):
    output = ''

    if not os.path.isdir(path):
        print('path does not point to a valid directory')
        return None 

    ignore_folders = {'__pycache__', '.git', '.vscode', '.ipynb_checkpoints', 'defunct_code'}
    items = sorted(os.listdir(path))

    for i, item in enumerate(items):
        if item in ignore_folders:
            continue

        full_path = os.path.join(path, item)
        prefix = '|- '

        if os.path.isdir(full_path):
            output += f'{indent}{prefix}**{item}**\n'
            output += scan_directory_tree(full_path, indent + '|    ', is_first_level=False)
        else:
            output += f'{indent}{prefix}*{item}*\n'

    # wrap the output in code block because it turned out that GitHub collapses
    # even non-breaking spaces (\u00A0) into a single space... the only way to circumvent 
    # that is to wrap everything in a code block
    if is_first_level:
        output = '```\n' + output + '```'

    return output


#%% plot formatting (to produce PDFs that allow Illustrator editing)
def mpl_formatting():
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


#%% data normalisation 
def normalise(data, axis=1):
    if data.size==0:
        print('array size is not valid')
        return None 
    
    if len(data.shape)>1:
        return np.apply_along_axis(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)), axis=axis, arr=data)
    else:
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
def normalise_to_all(data, alldata):  # data needs to be a 1-d vector/list
    norm_data = (data - min(alldata))/(max(alldata) - min(alldata))
    return norm_data


#%% data processing
def circ_shuffle(arr, alpha=.01, num_shuf=5000, GPU_AVAILABLE=False):
    '''
    performs circular shuffling on an input array.
    
    parameters:
    ----------
    arr : array
        dff array to be shuffled.
    alpha : float, optional
        significance threshold; defaults to 0.01.
    num_shuf : int, optional
        number of shuffle iterations; defaults to 5000.
    GPU_AVAILABLE : bool, optional
        if True, uses GPU acceleration with cupy; defaults to False.
    
    returns:
    -------
    tuple
        contains the following:
        - mean of shuffled arrays (array).
        - alpha percentile of shuffled arrays (array).
        - 1-alpha percentile of shuffled arrays (array).
    '''
    sig_perc = (1-alpha)*100  # significance for percentile
    vector = False  # default to 2D array; vector is only True when arr is 1D
    try:
        tot_row, tot_col = arr.shape
    except ValueError:  # if input is 1D
        tot_col = arr.shape[0]
        vector = True
    
    if GPU_AVAILABLE:
        try:
            import cupy as cp 
            xp = cp  # naive me never realised this was possible but hey, don't mind reducing the number of lines, 9 Dec 2024
        except ImportError:
            raise ImportError('CuPy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions.')
    else:
        xp = np
    
    shuf_mean_array = xp.zeros([num_shuf, tot_col])
    for i in range(num_shuf):
        if vector:  # if there is only 1 dimension
            rand_shift = xp.random.randint(1, tot_col)
            shuf_mean_array[i,:]+=xp.roll(arr, -rand_shift)
        else:
            for row in range(tot_row):
                rand_shift = xp.random.randint(1, tot_col)
                shuf_mean_array[i,:]+=xp.roll(arr[row,:], -rand_shift)
            shuf_mean_array/=num_shuf
        
    mean_shuf = xp.mean(shuf_mean_array, axis=0)
    sig_shuf = xp.percentile(shuf_mean_array, sig_perc, axis=0, method='midpoint')
    neg_sig_shuf = xp.percentile(shuf_mean_array, 100-sig_perc, axis=0, method='midpoint')
    
    if GPU_AVAILABLE:
        mean_shuf = mean_shuf.get()
        sig_shuf = sig_shuf.get()
        neg_sig_shuf = neg_sig_shuf.get()

    return mean_shuf, sig_shuf, neg_sig_shuf  # implicit tuple packing--another TIL moment, 9 Dec 2024

def smooth_convolve(data, sigma=3, axis=1):
    '''
    applies gaussian smoothing to a 1D or 2D array using convolution.

    parameters:
    ----------
    data : array
        input array to be smoothed.
    sigma : float, optional
        standard deviation of the gaussian kernel; defaults to 3 (good for imaging 
        processing).
    axis : int, optional
        axis along which to apply the smoothing; defaults to 1.

    returns:
    -------
    array
        smoothed array with the same shape as the input.

    raises:
    ------
    ValueError
        if the input array is empty or if sigma is not positive.
    '''
    if data.size == 0:
        raise ValueError('input array is empty.')

    # create a Gaussian kernel
    kernel = gaussian_kernel_unity(sigma)
    pad_width = len(kernel) // 2  # padding size

    # handle padding and convolution
    if data.ndim == 1:  # 1D array
        data_padded = np.pad(data, pad_width, mode='reflect')
        smoothed = np.convolve(data_padded, kernel, mode='same')[pad_width:-pad_width]
    elif data.ndim > 1:  # multidimensional array
        pad_config = [(0, 0)] * data.ndim  # no padding for non-convolved axes
        pad_config[axis] = (pad_width, pad_width)
        data_padded = np.pad(data, pad_config, mode='reflect')
        smoothed = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'),
            axis=axis,
            arr=data_padded
        )
        slice_config = [slice(None)] * data.ndim
        slice_config[axis] = slice(pad_width, -pad_width)
        smoothed = smoothed[tuple(slice_config)]
    else:
        raise ValueError('input array must have at least one dimension.')

    return smoothed


def gaussian_kernel_unity(sigma):
    '''
    generates a normalised gaussian kernel.
    
    parameters:
    ----------
    sigma : float
        standard deviation of the gaussian distribution.
    
    returns:
    -------
    array
        gaussian kernel with unity sum, centred around zero.
    '''
    kernel_size = int(6 * sigma + 1)
    x = np.arange(kernel_size) - (kernel_size // 2)
    kernel = np.exp(-(x**2 / (2 * sigma**2)))
    kernel /= kernel.sum()  # normalisation to ensure the unity sum
    return kernel 

def replace_outlier(arr, method='std', k=5):
    '''
    replaces outliers with linearly interpolated values.
    
    parameters:
    ----------
    arr : array-like
        input array to process. must be 1D or 2D with one singleton dimension.
    method : str, optional
        method for outlier detection: 'std' (default) or 'mad'.
    k : float, optional
        threshold for outlier detection; defaults to 5 which should account for 
        biological variability in most cases.
    
    returns:
    -------
    array
        array with outliers replaced by interpolated values.
    
    raises:
    ------
    ValueError
        if input array has more than 2 dimensions or invalid shape.
    '''
    if not isinstance(arr, np.ndarray):
        raise TypeError('input is not a NumPy array')
    
    shape = arr.shape 
    if len(shape) == 1:
        pass
    elif len(shape) == 2:
        arr = arr.ravel()
    else:
        raise ValueError('input array has more than 2 dimensions')
        
    if method == 'std':
        mean = np.mean(arr)
        std = np.std(arr)
        outliers = np.abs(arr-mean) > k*std  # mask
    elif method == 'mad':
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        outliers = np.abs(arr - median) > k * mad
    else:
        raise ValueError('invalid method; choose "std" (standard deviation) or "mad" (median absolute deviation)')
    
    # interpolato to replace outliers 
    if not np.any(outliers):  # if there are no outliers then don't bother
        return arr
    else:
        from scipy.interpolate import interp1d
        indices = np.arange(len(arr))
        valid_indices = indices[~outliers]
        valid_values = arr[~outliers]
        interp_func = interp1d(valid_indices, valid_values, kind='linear', fill_value='extrapolate')
        arr[outliers] = interp_func(indices[outliers])
        return arr


#%% calculate sem using cupy 
def sem_gpu(arr, axis=0, ddof=1):
    import cupy as cp 
    n = arr.shape[axis]
    arr_gpu = cp.array(arr)  # move to VRAM
    s = cp.std(arr_gpu, axis=axis, ddof=ddof) / cp.sqrt(n)
    return s.get()  # move back to VRAM