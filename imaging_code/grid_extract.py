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
import matplotlib as plc
import os

# import pre-processing functions 
if ('Z:\Dinghao\code_mpfi_dinghao\imaging_code' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_mpfi_dinghao\imaging_code')
import grid_roi_functions as grf

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise, smooth_convolve


#%% customise parameters
# align_to's available: 'run', 'reward', 'cue'
# align_to = 'run'
align_to = 'reward'
# align_to = 'cue'

# how many seconds before and after each align_to landmark 
bef = 1; aft = 4  # in seconds 

# how many pixels x/y for each grid
stride = 32

# do we want to save the grid_traces if extracted 
save_grids = 1


#%% path
# pathname = r'Z:\Dinghao\2p_recording\A087i\20240506\01\suite2p\plane0\'
rec_path = r'Z:/Jingyu/2P_Recording/AC928/AC928-20240327/02/'
suite2p_path = r'Z:/Jingyu/2P_Recording/AC928/AC928-20240327/02/RegOnly/suite2p/plane0/'
# opsfile = pathname+'ops.npy'
opsfile = suite2p_path+'ops.npy'
# binfile = pathname+'data.bin'
binfile = suite2p_path+'data.bin'


#%% define gridmesh
# parameters 
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


#%% do we need to extract traces? 
extract_trace = True  # default to True
extract_root = rec_path+'grid_extract'
extract_file_path = extract_root+'\\'+'grid_traces.npy'
if os.path.isfile(extract_file_path):
    extract_trace = False


#%% extract traces 
if extract_trace:
    grid_traces = np.zeros((tot_grid, tot_frames))
    for f in range(tot_frames):
        curr_frame = grf.run_grid(mov[f,:,:], grids, tot_grid, stride)
        for g in range(tot_grid):
            grid_traces[g, f] = grf.sum_mat(curr_frame[g])


#%% save grid traces (optional but recommended)
if extract_trace and save_grids:
    if not os.path.exists(extract_root):
        os.makedirs(extract_root)
    np.save(extract_file_path, grid_traces)


#%% read grid traces (if exists)
if extract_trace==False:
    grid_traces = np.load(extract_file_path, allow_pickle=True)


#%% timestamps 
txt = grf.process_txt('Z:/Jingyu/mice-expdata/AC928/A928-20240327-02T.txt')

# pumps
pump_times = txt['pump_times']

# speed
speed_times = txt['speed_times']

# cues 
movie_times = txt['movie_times']

# frames
frame_times = txt['frame_times']
first_frame = frame_times[0]; last_frame = frame_times[-1]

tot_trial = len(speed_times)


#%% **fill in dropped $FM signals
# since the 2P system does not always successfully transmit frame signals to
# the behavioural recording system every time it acquires a frame, one needs to
# manually interpolate the frame signals in between 2 frame signals that are 
# further apart than 50 ms
for i in range(len(frame_times)-1):
    if frame_times[i+1]-frame_times[i]>50:
        interp_fm = (frame_times[i+1]+frame_times[i])/2
        frame_times.insert(i+1, interp_fm)


#%% run-onset aligned
if align_to=='run': 
    
    # find run-onsets
    run_onsets = []
    
    for trial in range(tot_trial):
        corrected_speed_times = grf.correct_overflow(speed_times[trial])[0]
        times = [s[0] for s in corrected_speed_times]
        speeds = [s[1] for s in corrected_speed_times]
        uni_time = np.linspace(times[0], times[-1], int((times[-1] - times[0])))
        uni_speed = np.interp(uni_time, times, speeds)  # interpolation for speed
        
        run_onsets.append(grf.get_onset(uni_speed, uni_time))
        
    # filter run-onsets ([first_frame, last_frame])
    run_onsets = [t for t in run_onsets if t>first_frame and t<last_frame]
    
    # get aligned frame numbers
    run_frames = []
    for trial in range(len(run_onsets)):
        if run_onsets!=-1:  # if there is a clear run-onset in this trial
            run_frames.append(grf.find_nearest(run_onsets[trial], frame_times))  # find the nearest frame
        else:
            run_frames.append(-1)
            
    # align traces to run-onsets
    tot_run = len(run_onsets)
    run_aligned = np.zeros((tot_grid, (tot_run-1)*(bef+aft)*30))
    for i, r in enumerate(run_frames[:-1]):
        run_aligned[:, i*(bef+aft)*30:(i+1)*(bef+aft)*30] = grid_traces[:, r-bef*30:r+aft*30]
    

#%% pump aligned
if align_to=='reward':
    
    # filter pumps ([first_frame, last_frame])
    pumps = [t[0] for t in pump_times if t[0]>first_frame and t[0]<last_frame]
    
    # get aligned frame numbers
    pump_frames = []
    for trial in range(len(pumps)):
        pump_frames.append(grf.find_nearest(pumps[trial], frame_times))

    # align traces to pumps
    tot_pump = len(pumps)
    pump_aligned = np.zeros((tot_grid, (tot_pump-1)*(bef+aft)*30))
    for i, p in enumerate(pump_frames[:-1]):
        pump_aligned[:, i*(bef+aft)*30:(i+1)*(bef+aft)*30] = grid_traces[:, p-bef*30:p+aft*30]
        
        
#%% cue-onset aligned
if align_to=='cue':
    
    # filter pumps ([first_frame, last_frame])
    cues = [t[0][0] for t in movie_times if t[0][0]>first_frame and t[0][0]<last_frame]
    
    # get aligned frame numbers
    cue_frames = []
    for trial in range(len(pumps)):
        cue_frames.append(grf.find_nearest(cues[trial], frame_times))

    # align traces to pumps
    tot_cue = len(cues)
    cue_aligned = np.zeros((tot_grid, (tot_cue-1)*(bef+aft)*30))
    for i, c in enumerate(cue_frames[:-1]):
        cue_aligned[:, i*(bef+aft)*30:(i+1)*(bef+aft)*30] = grid_traces[:, c-bef*30:c+aft*30]


#%% plotting 
dimension = len(grids)
tot_plot = dimension**2

plc.rcParams['figure.figsize'] = (dimension*2, dimension*2)

fig = plt.figure(1)
for p in range(tot_plot):
    if align_to=='run':
        curr_grid_trace = run_aligned[p, :]
    if align_to=='reward':
        curr_grid_trace = pump_aligned[p, :]
    if align_to=='cue':
        curr_grid_trace = cue_aligned[p, :]
    curr_grid_map = np.zeros((tot_run-1, (bef+aft)*30))
    for i in range(tot_run-1):
        # curr_grid_map[i, :] = normalise(smooth_convolve(curr_grid_trace[i*(bef+aft)*30:(i+1)*(bef+aft)*30]))
        curr_grid_map[i, :] = normalise(curr_grid_trace[i*(bef+aft)*30:(i+1)*(bef+aft)*30])
    
    ax = fig.add_subplot(dimension, dimension, p+1)
    ax.set(xlabel='time (s)', ylabel='trial #')
    ax.imshow(curr_grid_map, aspect='auto', extent=[-1,4,1,tot_run], cmap='Greys')

fig.tight_laytout()

if align_to=='run':
    fig.suptitle('run_aligned')
    fig.savefig('{}/grid_traces_run_aligned.png'.format(extract_root),
                dpi=300,
                bbox_inches='tight')
if align_to=='reward':
    fig.suptitle('reward aligned')
    fig.savefig('{}/grid_traces_reward_aligned.png'.format(extract_root),
                dpi=300,
                bbox_inches='tight')
if align_to=='cue':
    fig.suptitle('cue aligned')
    fig.savefig('{}/grid_traces_cue_aligned.png'.format(extract_root),
                dpi=300,
                bbox_inches='tight')



#%% plotting
# tot_files = tot_grid/8  # this generates 64 files
# plots_per_file = 8
# col_plots = 4
# row_plots = 2
# plot_pos = np.arange(1, plots_per_file+1)
# grid_count = 0

# plc.rcParams['figure.figsize'] = (6*2, row_plots*3)

# for f in range(2):
#     fig = plt.figure(1)
    
#     for p in range(plots_per_file):
#         curr_grid_trace = run_aligned[grid_count, :]
#         curr_grid_map = np.zeros((tot_run-1, (bef+aft)*30))
#         for i in range(tot_run-1):
#             curr_grid_map[i, :] = normalise(curr_grid_trace[i*(bef+aft)*30:(i+1)*(bef+aft)*30])
        
#         ax = fig.add_subplot(row_plots, col_plots, plot_pos[p])
#         ax.set(title='grid {}'.format(grid_count),
#                xlabel='time (s)', ylabel='trial #')
#         ax.imshow(curr_grid_map, aspect='auto', extent=[-1,4,1,tot_run], cmap='Greys')
        
#         grid_count+=1
        
#     fig.suptitle('run_aligned')
#     fig.tight_layout()
#     plt.show()
    
    # fig.savefig('Z:\Dinghao\code_dinghao\dLight\DeepVid_grid_test\grid{}_{}.png'.format(grid_count-8, grid_count))
        