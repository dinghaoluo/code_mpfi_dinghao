# -*- coding: utf-8 -*-
"""
Created on Mon 13 May 17:13:42 2024

This code is composed of several distinct steps, each eventually able to run 
on their own:
    1. Define gridmesh
    2. Extract traces based on gridmeshed movie 
    3. Align traces to separate trial variables

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import sys
import matplotlib.pyplot as plt 
import scipy.io as sio 
import matplotlib as plc
import os

# import pre-processing functions 
if ('Z:\Dinghao\code_mpfi_dinghao\imaging_code' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_mpfi_dinghao\imaging_code')
import grid_roi_functions as grf

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% path
pathname = 'Z:\Dinghao\2p_recording\A087i\20240506\01\suite2p\plane0'
# opsfile = pathname+'\\'+'ops.npy'
opsfile = r'Z:/Jingyu/2P_Recording/AC934/AC934-20240510/02/RegOnly/suite2p/plane0/ops.npy'
# binfile = pathname+'\\'+'data.bin'
binfile = r'Z:\Jingyu\2P_Recording\AC934\AC934-20240510\02\RegOnly\suite2p\plane0\data.bin'


#%% define gridmesh
# parameters 
stride = 16  # how many pixels x/y for each grid
grids = grf.make_grid(stride)
tot_grid = len(grids)**2

# load data 
ops = np.load(opsfile, allow_pickle=True).item()
tot_frames = ops['nframes']
shape = tot_frames, ops['Ly'], ops['Lx']
mov = np.memmap(binfile, mode='r', dtype='int16', shape=shape)


#%% plot references (optional)
plot_ref = False  # switch 
if plot_ref:
    grf.plot_reference(mov, grids, stride)


#%% extract traces 
grid_trace = np.zeros((tot_grid, tot_frames))
# for f in range(tot_frames):
for f in range(100):
    curr_frame = grf.run_grid(mov[f,:,:], grids, tot_grid, stride)
    for g in range(tot_grid):
        grid_trace[g, f] = grf.sum_mat(curr_frame[g])


#%% save grid traces (optional but recommended)
save_grids = 1
if save_grids:
    outdirroot = pathname+'\\'+'grid_extract'
    if not os.path.exists(outdirroot):
        os.makedirs(outdirroot)
    np.save(outdirroot+'\\'+'grid_traces.npy')


#%% behavioural parameters 
txt = grf.process_txt('Z:/Jingyu/mice-expdata/AC934/A934-20240509-02T.txt')
speed_times = txt['speed_times']
frame_times = txt['frame_times']


#%% align to...
# available: 'run', 'reward', 'cue'
align_to = 'run'
# align_to = 'reward'
# align_to = 'cue'


#%% find run-onsets
raw_speed_times = txt['speed_times']
run_onsets = []

for trial in range(len(raw_speed_times)):
    speed_times = grf.correct_overflow(raw_speed_times[trial])[0]
    times = [s[0] for s in speed_times]
    speeds = [s[1] for s in speed_times]
    uni_time = np.linspace(times[0], times[-1], int((times[-1] - times[0])))
    uni_speed = np.interp(uni_time, times, speeds)  # interpolation for speed
    
    run_onsets.append(grf.get_onset(uni_speed, uni_time))


#%% 
pumps = behEvents['pump'][0][:,0]
pumps = pumps[range(0,len(pumps),2)]


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
        