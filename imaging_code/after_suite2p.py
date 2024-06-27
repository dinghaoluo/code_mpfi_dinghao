# -*- coding: utf-8 -*-
"""
Created on Mon 13 May 17:13:42 2024

This code is for post-pocessing of suite2p detected ROIs, will be able to:

    1. Align F trace to behaviour recording txt file
    2. Generate behaviour variables file for each session
    2. 
    3. 

@author: Dinghao Luo & Jingyu Cao
"""


#%% imports 
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.stats import sem
import sys
import matplotlib.pyplot as plt 
import matplotlib as plc
import os
from time import time
from datetime import timedelta

# import pre-processing functions 
if ('Z:\Dinghao\code_mpfi_dinghao\imaging_code' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_mpfi_dinghao\imaging_code')
import grid_roi_functions as grf

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise, smooth_convolve  # !be extra cautious when using smooth_convolve!

if ("Z:\Jingyu\Code\Python" in sys.path) == False:
    sys.path.append("Z:\Jingyu\Code\Python")
from utils_Jingyu import get_dff
    
    
#%% customise parameters
# align to...
align_run = 1
align_rew = 1

# params for plotting
plot_traces=0
plot_heatmaps=0
# how many seconds before and after each align_to landmark 
bef = 1; aft = 4  # in seconds 
xaxis = np.arange((bef+aft)*30)/30-bef  # xaxis for plotting

#calculate dFF?
dFF = 1


#%% load data
session = 'AC928-20240327-02'
# rec_path = r'Z:\Dinghao\2p_recording\A091i\A091i-20240611\A091i-20240611-01'
rec_path = r"Z:\Jingyu\2P_Recording\{}\{}\{}".format(session[0:5],session[0:-3],session[-2:])
beh_path = r"Z:\Jingyu\mice-expdata\{}\A{}T.txt".format(session[0:5],session[2:])
out_path = r"Z:\Jingyu\2P_Recording\{}\data_analysis".format(session[0:5])
# rec_path = r"Z:\Jingyu\2P_Recording\AC926\AC926-20240306\02"
# out_path = r"Z:\Jingyu\2P_Recording\AC926\data_analysis\suite2p_ROIs"

if not os.path.exists(out_path):
    os.makedirs(out_path)
# if 'Dinghao' in rec_path:
#     reg_path = rec_path+r'\registered\suite2p\plane0'
if 'Jingyu' in rec_path:
    suite2p_path = rec_path+r'\denoise=1_rolling=max_pix=0.04_peak=0.03_iterations=1_norm=max_neuropillam=True\suite2p\plane0'

opsfile = suite2p_path+r'\ops.npy'
F_file = suite2p_path+r'\F.npy'
F_ch2_file = suite2p_path+r'\F_chan2.npy'
F_neu_file = suite2p_path+r'\Fneu.npy'
F_neu_ch2_file = suite2p_path+r'\Fneu_ch2.npy'
statsfile = suite2p_path+r'\stat.npy'

if not os.path.exists(statsfile):
    print('no suite2p results found')

# beh file
print('\nreading behaviour file...')
# txt = grf.process_txt('Z:/Dinghao/MiceExp/ANMD091/A091-20240611-01T.txt')
txt = grf.process_txt(beh_path)

#load ROIs and trace
F_all = np.load(F_file, allow_pickle=True)
F_ch2_all = np.load(F_ch2_file, allow_pickle=True)
F_neu_all = np.load(F_neu_file, allow_pickle=True)

tot_roi = F_all.shape[0]
tot_frames = F_all.shape[1]

if dFF:
    print('calculating dFF...')
    
    F_all = get_dff(F_all)
    F_ch2_all = get_dff(F_ch2_all)
    F_neu_all = get_dff(F_neu_all)
    
#%% timestamps
print('\ndetermining behavioural timestamps...') 
# pumps
pump_times = txt['pump_times']

# speed
speed_times = txt['speed_times']

# cues 
movie_times = txt['movie_times']

# frames
frame_times = txt['frame_times']

#licks
lick_times = txt['lick_times']

tot_trial = len(speed_times)

#%% correct overflow
print('\ncorrecting overflow...')
pump_times = grf.correct_overflow(pump_times, 'pump')
speed_times = grf.correct_overflow(speed_times, 'speed')
movie_times = grf.correct_overflow(movie_times, 'movie')
frame_times = grf.correct_overflow(frame_times, 'frame')
lick_times = grf.correct_overflow(lick_times, 'lick')
first_frame = frame_times[0]; last_frame = frame_times[-1]

#%% **fill in dropped $FM signals
# since the 2P system does not always successfully transmit frame signals to
# the behavioural recording system every time it acquires a frame, one needs to
# manually interpolate the frame signals in between 2 frame signals that are 
# further apart than 50 ms
print('\nfilling in dropped $FM statements...')
for i in range(len(frame_times)-1):
    if frame_times[i+1]-frame_times[i]>50:
        interp_fm = (frame_times[i+1]+frame_times[i])/2
        frame_times.insert(i+1, interp_fm)

# check the difference between imaging frames and FM signal:
if len(frame_times)-3<=tot_frames<=len(frame_times):
    print('frame number check passed')
elif tot_frames>len(frame_times):
    print('Actuall frames > FM')
elif len(frame_times)>tot_frames+3:
    print('frame difference > 3')
    
#%% calculate dimension for plotting
n_col = int(tot_roi**0.5)
n_row = int(np.ceil(tot_roi/n_col))

# run-onset aligned
if align_run==1: 
    print('\nplotting traces aligned to RUN...')
    
    # find run-onsets
    run_onsets = []
    
    for trial in range(tot_trial):
        times = [s[0] for s in speed_times[trial]]
        speeds = [s[1] for s in speed_times[trial]]
        uni_time = np.linspace(times[0], times[-1], int((times[-1] - times[0])))
        uni_speed = np.interp(uni_time, times, speeds)  # interpolation for speed
        
        run_onsets.append(grf.get_onset(uni_speed, uni_time))
        
    # filter run-onsets ([first_frame, last_frame])
    run_onsets = [t for t in run_onsets if t>first_frame and t<last_frame]
    
    # get aligned frame numbers
    run_frames = []
    for trial in range(len(run_onsets)):
        if run_onsets!=-1:  # if there is a clear run-onset in this trial
            rf = grf.find_nearest(run_onsets[trial], frame_times)
            if rf!=0:
                run_frames.append(rf)  # find the nearest frame
        else:
            run_frames.append(-1)
            
    # align traces to run-onsets
    tot_run = len(run_onsets)
    run_aligned = np.zeros((tot_roi, tot_run-1, (bef+aft)*30))  # roi x trial x frame bin
    run_aligned_ch2 = np.zeros((tot_roi, tot_run-1, (bef+aft)*30))
    run_aligned_neu = np.zeros((tot_roi, tot_run-1, (bef+aft)*30))
    for roi in range(tot_roi):
        for i, r in enumerate(run_frames[:-1]):
            run_aligned[roi, i, :] = F_all[roi][r-bef*30:r+aft*30]
            run_aligned_ch2[roi, i, :] = F_ch2_all[roi][r-bef*30:r+aft*30]
            run_aligned_neu[roi, i, :] = F_neu_all[roi][r-bef*30:r+aft*30]
    
    if plot_heatmaps:
        print('plotting heatmaps...')
        # heatmap chan1
        fig = plt.figure(1)
        for p in range(tot_roi):
            curr_roi_trace = run_aligned[p, :, :]
            curr_roi_map = np.zeros((tot_run-1, (bef+aft)*30))
            for i in range(tot_run-1):
                curr_roi_map[i, :] = normalise(curr_roi_trace[i, :])
                
            ax = fig.add_subplot(n_row, n_col, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='roi {}'.format(p))
            ax.imshow(curr_roi_map, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
        # fig.suptitle('run_aligned')
        fig.tight_layout()
        if dFF:
            fig.savefig('{}/suite2pROIs_run_dFF_aligned.png'.format(out_path),
                        dpi=300,
                        bbox_inches='tight')
        fig.savefig('{}/suite2pROIs_run_aligned.png'.format(out_path),
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        # heatmap chan2
        fig = plt.figure(1)
        for p in range(tot_roi):
            curr_roi_trace_ch2 = run_aligned_ch2[p, :, :]
            curr_roi_map_ch2 = np.zeros((tot_run-1, (bef+aft)*30))
            for i in range(tot_run-1):
                curr_roi_map_ch2[i, :] = normalise(curr_roi_trace_ch2[i, :])
                
            ax = fig.add_subplot(n_col, n_row, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='roi {}'.format(p))
            ax.imshow(curr_roi_map_ch2, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
        # fig.suptitle('run_aligned')
        fig.tight_layout()
        if dFF:
            fig.savefig('{}/suite2pROIs_run_dFF_aligned_ch2.png'.format(out_path),
                        dpi=300,
                        bbox_inches='tight')
        fig.savefig('{}/suite2pROIs_run_aligned_ch2.png'.format(out_path),
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        # heatmap neuropil
        fig = plt.figure(1)
        for p in range(tot_roi):
            curr_roi_trace_neu = run_aligned_neu[p, :, :]
            curr_roi_map_neu = np.zeros((tot_run-1, (bef+aft)*30))
            for i in range(tot_run-1):
                curr_roi_map_neu[i, :] = normalise(curr_roi_trace_neu[i, :])           
            ax = fig.add_subplot(n_col, n_row, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='roi {}'.format(p))
            ax.imshow(curr_roi_map_neu, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
        # fig.suptitle('run_aligned')
        fig.tight_layout()
        if dFF:
            fig.savefig('{}/suite2pROIs_run_dFF_aligned_neu.png'.format(out_path),
                        dpi=300,
                        bbox_inches='tight')
        fig.savefig('{}/suite2pROIs_run_aligned_neu.png'.format(out_path),
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    if plot_traces:
        print('plotting combined averaged traces...')
        # average combined 
        fig = plt.figure(1, figsize=(n_row*4, n_col*2))
        for p in range(56,57):
            curr_roi_trace = run_aligned[p, :, :]
            curr_roi_trace_ch2 = run_aligned_ch2[p, :, :]
            curr_roi_trace_neu = run_aligned_neu[p, :, :]
            mean_trace = np.mean(curr_roi_trace, axis=0)
            mean_trace_ch2 = np.mean(curr_roi_trace_ch2, axis=0)
            mean_trace_neu = np.mean(curr_roi_trace_neu, axis=0)
            sem_trace = sem(curr_roi_trace, axis=0)
            sem_trace_ch2 = sem(curr_roi_trace_ch2, axis=0)
            sem_trace_neu = sem(curr_roi_trace_neu, axis=0)
            
            ax = fig.add_subplot(n_row, n_col, p+1)
            ax.set(xlim=(-bef,aft),xlabel='time (s)', ylabel='F', title='roi {}'.format(p))
            if dFF:
                ax.set(xlabel='time (s)', ylabel='dFF', title='roi {}'.format(p))
            ax.plot(xaxis, mean_trace, color='lightseagreen', linewidth=.8)
            ax.fill_between(xaxis, mean_trace+sem_trace,
                                   mean_trace-sem_trace,
                            color='lightseagreen', edgecolor='none', alpha=.2)
            # ax2 = ax.twinx()
            # ax2.plot(xaxis, mean_trace_ch2, color='rosybrown', linewidth=.8)
            # ax2.fill_between(xaxis, mean_trace_ch2+sem_trace_ch2,
            #                         mean_trace_ch2-sem_trace_ch2,
            #                  color='rosybrown', edgecolor='none', alpha=.2)
            ax3 = ax.twinx()
            ax3.plot(xaxis, mean_trace_neu, color='burlywood', linewidth=.8)
            ax3.fill_between(xaxis, mean_trace_neu+sem_trace_neu,
                                    mean_trace_neu-sem_trace_neu,
                             color='burlywood', edgecolor='none', alpha=.2)
            ax.axvspan(0, 0, color='grey', alpha=.5, linestyle='dashed', linewidth=1)
        # fig.suptitle('run_aligned_ch2')
        fig.tight_layout()
        if dFF:
            fig.savefig('{}/suite2pROIs_avgdFF_run_aligned.png'.format(out_path),
                        dpi=300,
                        bbox_inches='tight')
        fig.savefig('{}/suite2pROIs_avg_run_aligned.png'.format(out_path),
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)

# pump aligned
if align_rew==1:
    print('\nplotting traces aligned to REW...')
    
    # filter pumps ([first_frame, last_frame])
    pumps = [t for t in pump_times if t>first_frame and t<last_frame]
    
    # get aligned frame numbers
    pump_frames = []
    for trial in range(len(pumps)):
        pump_frames.append(grf.find_nearest(pumps[trial], frame_times))

    # align traces to pumps
    tot_pump = len(pumps)
    pump_aligned = np.zeros((tot_roi, (tot_pump-1), (bef+aft)*30))
    pump_aligned_ch2 = np.zeros((tot_roi, (tot_pump-1), (bef+aft)*30))
    pump_aligned_neu = np.zeros((tot_roi, (tot_pump-1), (bef+aft)*30))
    for i, p in enumerate(pump_frames[:-1]):
        pump_aligned[:, i, :] = F_all[:, p-bef*30:p+aft*30]
        pump_aligned_ch2[:, i, :] = F_ch2_all[:, p-bef*30:p+aft*30]
        pump_aligned_neu[:, i, :] = F_neu_all[:, p-bef*30:p+aft*30]
    
    if plot_heatmaps:
        print('plotting heatmaps...')
        
        # heatmap ch1
        fig = plt.figure(1)
        for p in range(tot_roi):
            curr_roi_trace = pump_aligned[p, :, :]
            curr_roi_map = np.zeros((tot_pump-1, (bef+aft)*30))
            for i in range(tot_pump-1):
                curr_roi_map[i, :] = normalise(curr_roi_trace[i, :])
            
            ax = fig.add_subplot(n_row, n_col, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
            ax.imshow(curr_roi_map, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
    
        # fig.suptitle('reward_aligned')
        fig.tight_layout()
        if dFF:
            fig.savefig('{}/suite2pROIs_rew_dFF_aligned.png'.format(out_path),
                        dpi=300,
                        bbox_inches='tight')
        fig.savefig('{}/suite2pROIs_rew_aligned.png'.format(out_path),
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
            
        # heatmap ch2
        fig = plt.figure(1)
        for p in range(tot_roi):
            curr_roi_trace_ch2 = pump_aligned_ch2[p, :, :]
            curr_roi_map_ch2 = np.zeros((tot_pump-1, (bef+aft)*30))
            for i in range(tot_pump-1):
                curr_roi_map_ch2[i, :] = normalise(curr_roi_trace_ch2[i, :])
            
            ax = fig.add_subplot(n_row, n_col, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
            ax.imshow(curr_roi_map_ch2, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
    
        # fig.suptitle('reward_aligned_ch2')
        fig.tight_layout()
        if dFF:
            fig.savefig('{}/suite2pROIs_rew_dFF_aligned_ch2.png'.format(out_path),
                        dpi=300,
                        bbox_inches='tight')
        fig.savefig('{}/suite2pROIs_rew_aligned_ch2.png'.format(out_path),
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
        plt.show()
        plt.close(fig)
        
        # heatmap neu
        fig = plt.figure(1)
        for p in range(tot_roi):
            curr_roi_trace_neu = pump_aligned_neu[p, :, :]
            curr_roi_map_neu = np.zeros((tot_pump-1, (bef+aft)*30))
            for i in range(tot_pump-1):
                curr_roi_map_neu[i, :] = normalise(curr_roi_trace_neu[i, :])
            
            ax = fig.add_subplot(n_row, n_col, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
            ax.imshow(curr_roi_map_neu, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
    
        # fig.suptitle('reward_aligned_neu')
        fig.tight_layout()
        if dFF:
            fig.savefig('{}/suite2pROIs_rew_dFF_aligned_neu.png'.format(out_path),
                        dpi=300,
                        bbox_inches='tight')
        fig.savefig('{}/suite2pROIs_rew_aligned_neu.png'.format(out_path),
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
        plt.show()
        plt.close(fig)

    if plot_traces:
        print('plotting combined averaged traces...')
        # average combined 
        fig = plt.figure(1, figsize=(n_row*4, n_col*2))
        for p in range(tot_roi):
            curr_roi_trace = pump_aligned[p, :, :]
            curr_roi_trace_ch2 = pump_aligned_ch2[p, :, :]
            curr_roi_trace_neu = pump_aligned_neu[p, :, :]
            mean_trace = np.mean(curr_roi_trace, axis=0)
            mean_trace_ch2 = np.mean(curr_roi_trace_ch2, axis=0)
            mean_trace_neu = np.mean(curr_roi_trace_neu, axis=0)
            sem_trace = sem(curr_roi_trace, axis=0)
            sem_trace_ch2 = sem(curr_roi_trace_ch2, axis=0)
            sem_trace_neu = sem(curr_roi_trace_neu, axis=0)
            
            ax = fig.add_subplot(n_row, n_col, p+1)
            ax.set(xlabel='time (s)', ylabel='F', title='roi {}'.format(p))
            if dFF:
                ax.set(xlabel='time (s)', ylabel='dFF', title='roi {}'.format(p))
            ax.plot(xaxis, mean_trace, color='lightseagreen', linewidth=.8)
            ax.fill_between(xaxis, mean_trace+sem_trace,
                                   mean_trace-sem_trace,
                            color='lightseagreen', edgecolor='none', alpha=.2)
            ax2 = ax.twinx()
            ax2.plot(xaxis, mean_trace_ch2, color='rosybrown', linewidth=.8)
            ax2.fill_between(xaxis, mean_trace_ch2+sem_trace_ch2,
                                    mean_trace_ch2-sem_trace_ch2,
                             color='rosybrown', edgecolor='none', alpha=.2)
            ax3 = ax.twinx()
            ax3.plot(xaxis, mean_trace_neu, color='burlywood', linewidth=.8)
            ax3.fill_between(xaxis, mean_trace_neu+sem_trace_neu,
                                    mean_trace_neu-sem_trace_neu,
                             color='burlywood', edgecolor='none', alpha=.2)
            ax.axvspan(0, 0, color='grey', alpha=.5, linestyle='dashed', linewidth=1)
        # fig.suptitle('run_aligned_ch2')
        fig.tight_layout()
        if dFF:
            fig.savefig('{}/suite2pROIs_avgdFF_rew_aligned.png'.format(out_path),
                        dpi=300,
                        bbox_inches='tight')
        fig.savefig('{}/suite2pROIs_avg_rew_aligned.png'.format(out_path),
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)

#%% get lick frames
lick_frames = []
for t in lick_times:
    if t[0][0]>first_frame and t[-1][0]<last_frame:
        # tmp_licks = [l[0] for l in t]
        # licks.append(tmp_licks)
        licks = [grf.find_nearest(lick[0], frame_times) for lick in t]
        lick_frames.append(licks)

#%% interpolation for speed to match with total nunmber of frames
speeds = []
speeds_times = []
for t in speed_times: # get all speeds for the whole session
    if t[0][0]>first_frame and t[-1][0]<last_frame: # filter speeds ([first_frame, last_frame])
        speeds.extend([p[1] for p in t])
        speeds_times.extend([p[0] for p in t])
uni_time = np.linspace(first_frame,last_frame, tot_frames)        
uni_speeds = np.interp(uni_time,speeds_times,speeds)  # interpolation for speed, match with total nunmber of frames

#%% DEMO PLOT: plot consective trials with speed and licks and dLight
f_start=5000
f_end=6000
roi =10


F_trace = F_all[roi,f_start:f_end]
rew = [f for f in pump_frames if f_start<f<f_end]
run = [f for f in run_frames if f_start<f<f_end]
lick = [f for f in np.hstack(lick_frames) if f_start<f<f_end]
speed = uni_speeds[f_start:f_end]

plt.rcParams['axes.labelsize']= 11
plt.rcParams['xtick.labelsize']= 10
plt.rcParams['ytick.labelsize']= 10

fig, ax = plt.subplots(figsize=(12, 2),dpi=300)
xaxis = np.arange(f_start, f_end)
ax.plot(xaxis,F_trace, color = 'green', linewidth=0.8, label='dLight3.6')
ax.set(ylabel='dLight_dFF')
ax1 = ax.twinx()
ax1.plot(xaxis,speed, color = 'steelblue', linewidth=0.8, alpha=0.8, label='speed (cm/s)')
ax1.set(ylim=(0,100))
ax1.vlines(run,0,800, color='orange', linewidth=0.8, alpha=.8, label='run onset')
ax1.spines['top'].set_visible(False)
ax1.tick_params(right=False, labelright=False)
ax1.vlines(lick, 85,95, color='darkslateblue', linewidth=0.2, alpha=0.8, label='licks')
ax1.vlines(rew, 0,800, color='deepskyblue', linewidth=0.8, alpha=1, label='reward')
ax1.set(ylabel='speed (cm/s)')
ax1.yaxis.label.set_color('steelblue')
ax1.tick_params(top=False,
               bottom=False,
               left=False,
               right=True,
               labelleft=False,
               labelright=True,
               labelbottom=True)
ax.spines['top'].set_visible(False) 
ax.set(xlabel='frames', xlim=(f_start,f_end))
fig.legend(prop={'size':4}, loc=(0.065, 0.77), frameon = False)
fig.tight_layout()


#%%

plt.rcParams['axes.labelsize']= 6
plt.rcParams['xtick.labelsize']= 5
plt.rcParams['ytick.labelsize']= 5 

fig, ax = plt.subplots(figsize=(16, 2),dpi=300)
xaxis = np.arange(len(F_trace))/30
ax.plot(xaxis,F_trace, color = 'green', linewidth=0.5, label='dLight3.6')
ax.set(ylabel='dLight_dFF')
ax1 = ax.twinx()
ax1.plot(xaxis,speeds, color = 'lightsteelblue', linewidth=0.5, alpha=0.8, label='speed (mm/s)')
ax1.set(ylim=(0,1000))
ax1.vlines(runs/500,0,800, color='orange', linewidth=0.3, alpha=1, label='run onset')
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(right=False, labelright=False)
ax1.vlines(lick_frames/30, 850,900, color='darkslateblue', linewidth=0.1, alpha=0.5, label='licks')
ax1.vlines(rews/30, 850, 900, color='deepskyblue', linewidth=0.5, alpha=1, label='reward')
ax.spines[['top','right']].set_visible(False) 
ax.set(xlabel='Time (s)', xlim=(0), ylim=(-0.6, 0.6))
fig.legend(prop={'size':4}, loc=1, frameon = False)
fig.tight_layout()
plt.show()        

#%% DEMO PLOT: plot heatmap for one roi:
# plotting parameters
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
   
roi = 13
run = 0
rew = 1
smooth = 1
sigma=1
heatmap = 1
trace = 1
c_dLight = 'lightseagreen'
c_neu = 'rosybrown'
if run:
    plot_array = run_aligned[roi,:,:]
    plot_array_neu = run_aligned_neu[roi,:,:]

if rew:
    plot_array = pump_aligned[roi,:,:]
    plot_array_neu = pump_aligned_neu[roi,:,:]
    
for i in range(plot_array.shape[0]):
    plot_array[i,:] = normalise(plot_array[i,:])
    plot_array_neu[i,:] = normalise(plot_array_neu[i,:])

if smooth:
    plot_array = gaussian_filter(plot_array, sigma, axes=1)
    plot_array_neu = gaussian_filter(plot_array_neu, sigma, axes=1)


xaxis = np.arange((bef+aft)*30)/30-bef  # xaxis for plotting

if heatmap:
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(plot_array, cmap='Greys', aspect='auto',vmin=np.percentile(plot_array,1), vmax=np.percentile(plot_array,99))
    ax.axvspan(bef*30, bef*30, color='black', linewidth=1, linestyle='dashed', alpha=.5)
    # ax.set(xlabel='time (s)',ylabel='n_ROI',xticks=np.linspace(0, (0.5+aft)*30, 8), xticklabels=np.arange(-0.5,aft+0.5,0.5))
    ax.set(xlabel='time (s)',ylabel='n_trial', xticks=np.linspace(0, (bef+aft)*30, 2*(bef+aft)+1), xticklabels=np.arange(-bef,aft+0.5,0.5))
    fig.tight_layout()
    if run:
        fig.savefig(r"Z:\Jingyu\plots_20240624\demo_heatmap_AC928-20240327-02-roi{}_ro_ramp.pdf".format(roi), dpi=300, transparent=True, bbox_inches="tight", format="pdf")
    if rew:
        fig.savefig(r"Z:\Jingyu\plots_20240624\demo_heatmap_AC928-20240327-02-roi{}_reward_peak.pdf".format(roi), dpi=300, transparent=True, bbox_inches="tight", format="pdf")
    plt.show()
    # plt.close()

if trace:
    fig, ax = plt.subplots(figsize=(3,1.5))
    trace_mean = np.mean(plot_array, axis=0)
    trace_sem = sem(plot_array, axis=0)
    trace_neu_mean = np.mean(plot_array_neu, axis=0)
    trace_neu_sem = sem(plot_array_neu, axis=0)
    
    ax.plot(xaxis, trace_mean, lw=.8,color=c_dLight, label='dLight3.6')
    ax.plot(xaxis, trace_neu_mean, lw=.8, color=c_neu,label='Neuropil')
    ax.fill_between(xaxis, trace_mean+trace_sem, trace_mean-trace_sem,
                     color=c_dLight, edgecolor='none', alpha=.2)
    ax.fill_between(xaxis, trace_neu_mean+trace_neu_sem, trace_neu_mean-trace_neu_sem,
                     color=c_neu, edgecolor='none', alpha=.2)
    ax.axvspan(0, 0, color='black', linewidth=.5, linestyle='--', alpha=.5)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='time (s)',ylabel='dFF', xlim=(-bef,aft))
    ax.legend(prop={'size':4}, loc=(0.8, 0.77), frameon = False)
    # ax.set(xlabel='time (s)',ylabel='dFF', xticks=np.linspace(0, (bef+aft)*30, 2*(bef+aft)+1), xticklabels=np.arange(-bef,aft+0.5,0.5))
    fig.tight_layout()
    if run:
        fig.savefig(r"Z:\Jingyu\plots_20240624\demo_avgTrace_AC928-20240327-02-roi{}_ro_ramp.pdf".format(roi), dpi=300, transparent=False, bbox_inches="tight", format="pdf")
    if rew:
        fig.savefig(r"Z:\Jingyu\plots_20240624\demo_avgTrace_AC928-20240327-02-roi{}_reward_peak.pdf".format(roi), dpi=300, transparent=False, bbox_inches="tight", format="pdf")
    plt.show()
#%% cue-onset aligned
# if align_cue==1:
    
#     # filter pumps ([first_frame, last_frame])
#     cues = [t[0][0] for t in movie_times if t[0][0]>first_frame and t[0][0]<last_frame]
    
#     # get aligned frame numbers
#     cue_frames = []
#     for trial in range(len(pumps)):
#         cue_frames.append(grf.find_nearest(cues[trial], frame_times))

#     # align traces to pumps
#     tot = len(cues)
#     cue_aligned = np.zeros((tot_grid, (tot-1)*(bef+aft)*30))
#     for i, c in enumerate(cue_frames[:-1]):
#         cue_aligned[:, i*(bef+aft)*30:(i+1)*(bef+aft)*30] = grid_traces[:, c-bef*30:c+aft*30]