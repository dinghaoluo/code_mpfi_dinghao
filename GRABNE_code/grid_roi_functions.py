# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:50:53 2024

functions for grid-ROI processing

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 


#%% main 
def sum_mat(matrix):
    return sum(map(sum, matrix))

def make_grid(stride=8, dim=512):
    """
    Parameters
    ----------
    stride : int, default=8
        how many pixels per grid.
    dim : int, default=512
        x/y dimension; either should do since we are imaging squared images.

    Returns
    -------
    a list of grid points.
    """
    return list(np.arange(0, dim, stride))

def run_grid(frame, grids, tot_grid, stride=8):
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
    gridded = np.zeros((tot_grid, stride, stride))
    
    grid_count = 0
    for hgp in grids:
        for vgp in grids:
            gridded[grid_count,:,:] = frame[hgp:hgp+stride, vgp:vgp+stride]
            grid_count+=1
            
    return gridded