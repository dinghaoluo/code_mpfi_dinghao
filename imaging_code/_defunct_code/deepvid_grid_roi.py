# -*- coding: utf-8 -*-
"""
Created on Thur Apr 25 15:03:44 2024

divide frames into grid-ROIs for DeepVID testing 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import sys
import matplotlib.pyplot as plt 
import scipy.io as sio 
import matplotlib as plc

# import pre-processing functions 
if ('Z:\Dinghao\code_mpfi_dinghao\imaging_code' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_mpfi_dinghao\imaging_code')
import grid_roi_functions as grf

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% parameters 
stride = 16  # how many pixels x/y for each grid
grids = grf.make_grid(stride)
tot_grid = len(grids)**2


#%% load data 
ops = np.load('Z:/Colin/NeuroimagingPipeline/data/interim/deepcadrt/results/test_202404161333/E_05_Iter_10780/suite2p/plane0/ops.npy', allow_pickle=True).item()
tot_frames = ops['nframes']
shape = tot_frames, ops['Ly'], ops['Lx']
mov = np.memmap('Z:/Colin/NeuroimagingPipeline/data/interim/deepcadrt/results/test_202404161333/E_05_Iter_10780/suite2p/plane0/data.bin', 
                mode='r', dtype='int16', shape=shape)


#%% compute reference 
ref_im = np.mean(mov, axis=0)


#%% plot references
fig, ax = plt.subplots(figsize=(4,4))

ax.imshow(ref_im, aspect='auto', cmap='gist_gray', interpolation='none',
          extent=[0, 32, 32, 0])

for g in grids:
    ax.plot([0,32], [g/stride,g/stride], color='white', linewidth=.2, alpha=.2)
    ax.plot([g/stride,g/stride], [0,32], color='white', linewidth=.2, alpha=.2)

ax.set(xlim=(0,32), ylim=(0,32))


#%% behavioural parameters 
behEvents = sio.loadmat('Z:/Jingyu/2P_Recording/AC926/AC926-20240305/02/denoise=1_rolling=max_pix=0.04_peak=0.03_iterations=1_norm=max_neuropillam=True/Channel1/A926-20240305-02BTDT.mat')['behEventsTdt'][0]
pumps = behEvents['pump'][0][:,0]
pumps = pumps[range(0,len(pumps),2)]


#%% get gridded 
# initialise gridded movie
gridded = np.zeros((tot_frames, tot_grid, stride, stride)) 
for f in range(tot_frames):
    gridded[f] = grf.run_grid(mov[f,:,:], grids, tot_grid, stride)


#%% get traces
# initialise trace array
grid_trace = np.zeros((tot_grid, tot_frames))
for g in range(tot_grid):
    for f in range(tot_frames):
        grid_trace[g,f] = grf.sum_mat(gridded[f,g,:,:])  # the trace of each grid is the summed amplitude in the grid over all the frames
        

#%% align to pumps 
tot_pumps = len(pumps)
bef = 1
aft = 4
pump_aligned = np.zeros((tot_grid, (tot_pumps-1)*(bef+aft)*30))
for i, p in enumerate(pumps[:-1]):
    pump_aligned[:, i*(bef+aft)*30:(i+1)*(bef+aft)*30] = grid_trace[:, p-bef*30:p+aft*30]


#%% plotting
tot_files = tot_grid/8  # this generates 128 files
plots_per_file = 8
col_plots = 4
row_plots = 2
plot_pos = np.arange(1, plots_per_file+1)
grid_count = 0

plc.rcParams['figure.figsize'] = (6*2, row_plots*3)

for f in range(tot_files):
    fig = plt.figure(1)
    
    for p in range(plots_per_file):
        curr_grid_trace = pump_aligned[grid_count, :]
        curr_grid_map = np.zeros((tot_pumps-1, (bef+aft)*30))
        for i in range(tot_pumps-1):
            curr_grid_map[i, :] = normalise(curr_grid_trace[i*(bef+aft)*30:(i+1)*(bef+aft)*30])
        
        ax = fig.add_subplot(row_plots, col_plots, plot_pos[p])
        ax.set(title='grid {}'.format(grid_count),
               xlabel='time (s)', ylabel='trial #')
        ax.imshow(curr_grid_map, aspect='auto', extent=[-1,4,1,tot_pumps], cmap='Greys')
        
        grid_count+=1
        
    fig.suptitle('pump_aligned')
    fig.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\dLight\DeepVid_grid_test\grid{}_{}.png'.format(grid_count-8, grid_count))
        