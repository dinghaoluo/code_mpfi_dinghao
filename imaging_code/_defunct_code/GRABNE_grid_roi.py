# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:29:58 2024

divide frames into grid-ROIs

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import sys
import matplotlib.pyplot as plt 
import pandas as pd 

# import pre-processing functions 
if ('Z:\Dinghao\code_mpfi_dinghao\GRABNE_code' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_mpfi_dinghao\GRABNE_code')
import grid_roi_functions as grf

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% parameters 
stride = 16  # how many pixels x/y for each grid
grids = grf.make_grid(stride)
tot_grid = len(grids)**2


#%% load data 
ops = np.load('Z:/Dinghao/2p_recording/YW5016_d11_920_2/suite2p/plane0/ops.npy', allow_pickle=True).item()
tot_frames = ops['nframes']
shape = tot_frames, ops['Ly'], ops['Lx']
mov = np.memmap('Z:/Dinghao/2p_recording/YW5016_d11_920_2/suite2p/plane0/data.bin', 
                mode='r', dtype='int16', shape=shape)
mov2 = np.memmap('Z:/Dinghao/2p_recording/YW5016_d11_920_2/suite2p/plane0/data.bin', 
                 mode='r', dtype='int16', shape=shape)


#%% compute reference 
ref_im = np.mean(mov, axis=0)
ref_im2 = np.mean(mov2, axis=0)


#%% plot references
fig, ax = plt.subplots(figsize=(4,4))

ax.imshow(ref_im, aspect='auto', cmap='gist_gray', interpolation='none',
          extent=[0, 32, 32, 0])
# ax.imshow(ref_im2, cmap='Reds', aspect='auto', alpha=.8)

for g in grids:
    ax.plot([0,32], [g/stride,g/stride], color='white', linewidth=.2, alpha=.2)
    ax.plot([g/stride,g/stride], [0,32], color='white', linewidth=.2, alpha=.2)

ax.set(xlim=(0,32), ylim=(0,32))


#%% test parameter 
test_frames = 1000


#%% get gridded 
# initialise gridded movie
gridded = np.zeros((test_frames, tot_grid, stride, stride)) 
for f in range(test_frames):
    gridded[f] = grf.run_grid(mov[f,:,:], grids, tot_grid, stride)


#%% get traces   
# initialise trace array
grid_trace = np.zeros((tot_grid, test_frames))
for g in range(tot_grid):
    for f in range(test_frames):
        grid_trace[g,f] = grf.sum_mat(gridded[f,g,:,:])
        

#%% plotting 
fig, ax = plt.subplots(figsize=(5,6))

xaxis = np.linspace(0, 240/30, 240)

for g in range(64):
    ax.plot(xaxis, normalise(grid_trace[g,:240])+g, alpha=.8, linewidth=1)

for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)

ax.set(xlabel='time (s)',
       yticks=[],
       ylim=(-1, 65))