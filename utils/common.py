# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:54:04 2022

a collection of commonly used functions

@author: Dinghao Luo
"""

import numpy as np 
import os


# os shortcuts
def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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