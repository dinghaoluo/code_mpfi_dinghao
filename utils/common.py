# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:54:04 2022

a collection of commonly used functions

@author: Dinghao Luo
"""

import numpy as np 
import os


# os functions 
def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def scan_directory_tree(path, indent='', is_first_level=True):
    output = ''
    
    if not os.path.isdir(path):
        print('path does not point to a valid directory')
        return None 
    
    ignore_folders = {'__pycache__', '.git', '.vscode', '.ipynb_checkpoints'}
    items = sorted(os.listdir(path))
    
    for i, item in enumerate(items):
        if item in ignore_folders:
           continue  # kkip ignored folders
           
        full_path = os.path.join(path, item)
        is_last = (i == len(items) - 1)
        prefix = '└── ' if is_last else '├── '

        if os.path.isdir(full_path):
            output += f'{indent}{prefix}**{item}**\n'
            output += scan_directory_tree(full_path, indent + ('    ' if is_last else '│   '), is_first_level=False)
        else:
            output += f'{indent}{prefix}{item}\n'
    
    return output


# formatting 
def mpl_formatting():
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


# normalise data 
def normalise(data):  # data needs to be a 1-d vector/list
    norm_data = (data - min(data))/(max(data) - min(data))
    return norm_data

def normalise_to_all(data, alldata):  # data needs to be a 1-d vector/list
    norm_data = (data - min(alldata))/(max(alldata) - min(alldata))
    return norm_data


# smoth data 
def gaussian_unity(sigma=3):  # generate a Gaussian filter that sums to unity
    gx = np.arange(-sigma*3, sigma*3, 1)
    gaussian_filter = [1 / (sigma*np.sqrt(2*np.pi)) * 
                       np.exp(-x**2/(2*sigma**2)) for x in gx]
    gaussian_filter /= np.sum(gaussian_filter)  # ensures unity
    return gaussian_filter

def smooth_convolve(data, sigma=3):  # sigma in frames
    return np.convolve(data, gaussian_unity(sigma), mode='same')


# calculate sem using cupy 
def sem_gpu(arr, axis=0, ddof=1):
    import cupy as cp 
    n = arr.shape[axis]
    arr_gpu = cp.array(arr)  # move to VRAM
    s = cp.std(arr_gpu, axis=axis, ddof=ddof) / cp.sqrt(n)
    return s.get()  # move back to VRAM