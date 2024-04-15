# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:04:39 2023

plot waveforms, CCG's and rasters of all LC cells

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt


#%% load files 
waveforms = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_waveforms.npy',
                    allow_pickle=True).item()

rasters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()
    
CCGs = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_ccg.npy', 
               allow_pickle=True).item()


#%% function
def get_waveform(cluname):
    return waveforms[cluname][0,:], waveforms[cluname][1,:]

def get_raster(cluname):
    return

def get_CCG(cluname):
    return


#%% plotting loop
fig, axs = plt.subplot_mosaic('AB;CC', figsize=(8,8))
