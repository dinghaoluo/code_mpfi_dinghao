# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:02:34 2024

A script to host the entire suite2p and grid ROI pipeline

@author: Dinghao Luo
"""


#%% imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as plc
from scipy.stats import sem 
from time import time
from datetime import timedelta

# import pre-processing functions 
if (r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise, smooth_convolve  # !be extra cautious when using smooth_convolve!


#%% grid pipeline function 
def run_grid_pipeline(rec_path, recname, reg_path, txt_path, 
                      stride, border, 
                      plot_ref, 
                      smooth, save_grids, 
                      bef, aft,
                      align_run, align_rew, align_cue):    
    opsfile = reg_path+r'\ops.npy'
    binfile = reg_path+r'\data.bin'
    bin2file = reg_path+r'\data_chan2.bin'
    
    extract_path = rec_path+r'_grid_extract'
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    # beh file
    print('\nreading behaviour file...')
    txt = ipf.process_txt(txt_path)


    ## define gridmesh
    # parameters 
    grids = ipf.make_grid(stride=stride, dim=512, border=border)
    tot_grid = len(grids)**2
    
    # load data 
    ops = np.load(opsfile, allow_pickle=True).item()
    tot_frames = ops['nframes']
    shape = tot_frames, ops['Ly'], ops['Lx']
    mov = np.memmap(binfile, mode='r', dtype='int16', shape=shape)
    mov2 = np.memmap(bin2file, mode='r', dtype='int16', shape=shape)


    ## plotting parameters
    dimension = len(grids)
    tot_plot = dimension**2
    xaxis = np.arange((bef+aft)*30)/30-bef  # xaxis for plotting 
    plc.rcParams['figure.figsize'] = (dimension*2, dimension*2)


    ## plot references (optional)
    if plot_ref:
        if not os.path.exists(extract_path+r'\ref_ch1_{}.png'.format(stride)):
            print('\ngenerating reference images...')
            
            t0 = time()  # timer
            ipf.plot_reference(mov, grids, stride, 512, 1, extract_path)
            print('ref done ({})'.format(str(timedelta(seconds=int(time()-t0)))))
            ipf.plot_reference(mov2, grids, stride, 512, 2, extract_path)
            print('ref_ch2 done ({})'.format(str(timedelta(seconds=int(time()-t0)))))


    ## do we need to extract traces? (channel 1)
    extract_trace = True  # default to True
    extract_file_path = extract_path+r'\grid_traces_{}.npy'.format(stride)
    if os.path.exists(extract_file_path):
        extract_trace = False


    ## extract traces (this is going to take the most time)
    if extract_trace:
        print('\nch1 trace extraction starts')
        t0 = time()  # timer
        grid_traces = np.zeros((tot_grid, tot_frames))
        
        for f in range(tot_frames):
            # progress report 
            for p in [.25, .5, .75]:
                if f==int(tot_frames*p):
                    print('{} ({}%) frames done ({})'.format(f, int(p*100), str(timedelta(seconds=int(time()-t0)))))
            
            curr_frame = ipf.run_grid(mov[f,:,:], grids, tot_grid, stride)
            for g in range(tot_grid):
                grid_traces[g, f] = ipf.sum_mat(curr_frame[g])
        print('ch1 trace extraction complete ({})'.format(str(timedelta(seconds=int(time()-t0)))))
    

    ## save grid traces (optional but recommended)
    if extract_trace and save_grids:
        np.save(extract_file_path, grid_traces)
        print('ch1 traces saved to {}\n'.format(extract_file_path))

    
    ## do we need to extract traces? (channel 2)
    extract_trace_ch2 = True  # default to True
    extract_file_path_ch2 = extract_path+r'\grid_traces_{}_ch2.npy'.format(stride)
    if os.path.exists(extract_file_path_ch2):
        extract_trace_ch2 = False


    ## extract traces (channel 2)
    if extract_trace_ch2:
        print('ch2 trace extraction starts')
        t0 = time()  # timee
        grid_traces_ch2 = np.zeros((tot_grid, tot_frames))
        
        for f in range(tot_frames):
            # progress report 
            for p in [.25, .5, .75]:
                if f==int(tot_frames*p):
                    print('{} ({}%) frames done ({})'.format(f, int(p*100), str(timedelta(seconds=int(time()-t0)))))
            
            curr_frame = ipf.run_grid(mov2[f,:,:], grids, tot_grid, stride)
            for g in range(tot_grid):
                grid_traces_ch2[g, f] = ipf.sum_mat(curr_frame[g])
        print('ch2 trace extraction complete ({})'.format(str(timedelta(seconds=int(time()-t0)))))
    

    ## save grid traces (channel 2)
    if extract_trace_ch2 and save_grids:
        np.save(extract_file_path_ch2, grid_traces_ch2)
        print('ch2 traces saved to {}\n'.format(extract_file_path_ch2))


    ## read grid traces (if exists)
    if extract_trace==False:
        grid_traces = np.load(extract_file_path, allow_pickle=True)
        grid_traces_ch2 = np.load(extract_file_path_ch2, allow_pickle=True)
        print('\ntraces read from {}'.format(extract_path))


    ## timestamps
    print('\ndetermining behavioural timestamps...') 
    # pumps
    pump_times = txt['pump_times']
    # speed
    speed_times = txt['speed_times']
    # cues 
    movie_times = txt['movie_times'] 
    # frames
    frame_times = txt['frame_times']
    
    tot_trial = len(speed_times)
    # tot_frame = len(frame_times)

    ## correct overflow
    print('\ncorrecting overflow...')
    pump_times = ipf.correct_overflow(pump_times, 'pump')
    speed_times = ipf.correct_overflow(speed_times, 'speed')
    movie_times = ipf.correct_overflow(movie_times, 'movie')
    frame_times = ipf.correct_overflow(frame_times, 'frame')
    first_frame = frame_times[0]; last_frame = frame_times[-1]


    ## **fill in dropped $FM signals
    # since the 2P system does not always successfully transmit frame signals to
    # the behavioural recording system every time it acquires a frame, one needs to
    # manually interpolate the frame signals in between 2 frame signals that are 
    # further apart than 50 ms
    print('\nfilling in dropped $FM statements...')
    for i in range(len(frame_times)-1):
        if frame_times[i+1]-frame_times[i]>50:
            interp_fm = (frame_times[i+1]+frame_times[i])/2
            frame_times.insert(i+1, interp_fm)
            
    # checks $FM against tot_frame
    if tot_frames<len(frame_times)-3 or tot_frames>len(frame_times):
        print('\nWARNING:\ncheck $FM; halting processing for {}\n'.format(recname))
        align_run=0; align_rew=0; align_cue=0  # if tot_frame is more than sync signals or fewer than syncs-3, then halt


    ## run-onset aligned
    if align_run==1: 
        print('\nplotting traces aligned to RUN...')
        
        # find run-onsets
        run_onsets = []
        
        for trial in range(tot_trial):
            times = [s[0] for s in speed_times[trial]]
            speeds = [s[1] for s in speed_times[trial]]
            uni_time = np.linspace(times[0], times[-1], int((times[-1] - times[0])))
            uni_speed = np.interp(uni_time, times, speeds)  # interpolation for speed
            
            run_onsets.append(ipf.get_onset(uni_speed, uni_time))
            
        # filter run-onsets ([first_frame, last_frame])
        run_onsets = [t for t in run_onsets if t>first_frame and t<last_frame]
        
        # get aligned frame numbers
        run_frames = []
        for trial in range(len(run_onsets)):
            if run_onsets!=-1:  # if there is a clear run-onset in this trial
                rf = ipf.find_nearest(run_onsets[trial], frame_times)
                if rf!=0:
                    run_frames.append(rf)  # find the nearest frame
            else:
                run_frames.append(-1)
                
        # align traces to run-onsets
        tot_run = len(run_onsets)
        run_aligned = np.zeros((tot_grid, tot_run-1, (bef+aft)*30))  # grid x trial x frame bin
        run_aligned_ch2 = np.zeros((tot_grid, tot_run-1, (bef+aft)*30))
        for i, r in enumerate(run_frames[:-1]):
            run_aligned[:, i, :] = grid_traces[:, r-bef*30:r+aft*30]
            run_aligned_ch2[:, i, :] = grid_traces_ch2[:, r-bef*30:r+aft*30]
        
        print('plotting heatmaps...')
        # heatmap chan1
        fig = plt.figure(1, figsize=(dimension*2.5, dimension*2))
        for p in range(tot_plot):
            curr_grid_trace = run_aligned[p, :, :]
            curr_grid_map = np.zeros((tot_run-1, (bef+aft)*30))
            for i in range(tot_run-1):
                curr_grid_map[i, :] = normalise(curr_grid_trace[i, :])
                
            ax = fig.add_subplot(dimension, dimension, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
            ax.imshow(curr_grid_map, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
        fig.suptitle('run_aligned')
        fig.tight_layout()
        fig.savefig('{}/grid_traces_{}_run_aligned.png'.format(extract_path, stride),
                    dpi=120,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        # heatmap chan2
        fig = plt.figure(1, figsize=(dimension*2.5, dimension*2))
        for p in range(tot_plot):
            curr_grid_trace_ch2 = run_aligned_ch2[p, :, :]
            curr_grid_map_ch2 = np.zeros((tot_run-1, (bef+aft)*30))
            for i in range(tot_run-1):
                curr_grid_map_ch2[i, :] = normalise(curr_grid_trace_ch2[i, :])
            
            ax = fig.add_subplot(dimension, dimension, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
            ax.imshow(curr_grid_map_ch2, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
        fig.suptitle('run_aligned_ch2')
        fig.tight_layout()
        fig.savefig('{}/grid_traces_{}_run_aligned_ch2.png'.format(extract_path, stride),
                    dpi=120,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print('plotting combined averaged traces...')
        # average combined 
        fig = plt.figure(1, figsize=(dimension*4, dimension*3))
        for p in range(tot_plot):
            curr_grid_trace = run_aligned[p, :, :]
            curr_grid_trace_ch2 = run_aligned_ch2[p, :, :]
            mean_trace = np.mean(curr_grid_trace, axis=0)
            mean_trace_ch2 = np.mean(curr_grid_trace_ch2, axis=0)
            sem_trace = sem(curr_grid_trace, axis=0)
            sem_trace_ch2 = sem(curr_grid_trace_ch2, axis=0)
            
            ax = fig.add_subplot(dimension, dimension, p+1)
            ax.set(xlabel='time (s)', ylabel='F', title='grid {}'.format(p))
            ax.plot(xaxis, mean_trace, color='limegreen', linewidth=1, zorder=10)
            ax.fill_between(xaxis, mean_trace+sem_trace,
                                   mean_trace-sem_trace,
                            color='limegreen', edgecolor='none', alpha=.2, zorder=10)
            ax2 = ax.twinx()
            ax2.plot(xaxis, mean_trace_ch2, color='red', linewidth=1, zorder=1, alpha=.55)
            ax2.fill_between(xaxis, mean_trace_ch2+sem_trace_ch2,
                                    mean_trace_ch2-sem_trace_ch2,
                             color='red', edgecolor='none', alpha=.1, zorder=1)
            ax.axvspan(0, 0, color='grey', alpha=.5, linestyle='dashed', linewidth=1)
        fig.suptitle('run_aligned_ch2')
        fig.tight_layout()
        fig.savefig('{}/grid_traces_{}_avg_run_aligned.png'.format(extract_path, stride),
                    dpi=120,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    
    ## pump aligned
    if align_rew==1:
        print('\nplotting traces aligned to REW...')
        
        # filter pumps ([first_frame, last_frame])
        pumps = []
        for i, p in enumerate(pump_times[:-1]):
            if p>first_frame and p<last_frame:
                if p!=0 and p-bef*30>0:
                    pumps.append(p)
                else:
                    pumps.append([])
        
        # get aligned frame numbers
        pump_frames = []
        for trial in range(len(pumps)):
            pump_frames.append(ipf.find_nearest(pumps[trial], frame_times))
    
        # align traces to pumps
        tot_pump = len(pumps)
        pump_aligned = np.zeros((tot_grid, (tot_pump-2), (bef+aft)*30))
        pump_aligned_ch2 = np.zeros((tot_grid, (tot_pump-2), (bef+aft)*30))
        for i, p in enumerate(pump_frames[1:-1]):
            pump_aligned[:, i, :] = grid_traces[:, p-bef*30:p+aft*30]
            pump_aligned_ch2[:, i, :] = grid_traces_ch2[:, p-bef*30:p+aft*30]
            
        print('plotting heatmaps...')
        # heatmap ch1
        fig = plt.figure(1, figsize=(dimension*2.5, dimension*2))
        for p in range(tot_plot):
            curr_grid_trace = pump_aligned[p, :, :]
            curr_grid_map = np.zeros((tot_pump-2, (bef+aft)*30))
            for i in range(tot_pump-2):
                curr_grid_map[i, :] = normalise(curr_grid_trace[i, :])
            
            ax = fig.add_subplot(dimension, dimension, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
            ax.imshow(curr_grid_map, aspect='auto', extent=[-bef,aft,1,tot_pump], cmap='Greys')
    
        fig.suptitle('reward_aligned')
        fig.tight_layout()
        fig.savefig('{}/grid_traces_{}_rew_aligned.png'.format(extract_path, stride),
                    dpi=120,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
            
        # heatmap ch2
        fig = plt.figure(1, figsize=(dimension*2.5, dimension*2))
        for p in range(tot_plot):
            curr_grid_trace_ch2 = pump_aligned_ch2[p, :, :]
            curr_grid_map_ch2 = np.zeros((tot_pump-2, (bef+aft)*30))
            for i in range(tot_pump-2):
                curr_grid_map_ch2[i, :] = normalise(curr_grid_trace[i, :])
            
            ax = fig.add_subplot(dimension, dimension, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
            ax.imshow(curr_grid_map_ch2, aspect='auto', extent=[-bef,aft,1,tot_pump], cmap='Greys')
    
        fig.suptitle('reward_aligned_ch2')
        fig.tight_layout()
        fig.savefig('{}/grid_traces_{}_rew_aligned_ch2.png'.format(extract_path, stride),
                    dpi=120,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print('plotting combined averaged traces...')
        # average combined 
        fig = plt.figure(1, figsize=(dimension*4, dimension*3))
        for p in range(tot_plot):
            curr_grid_trace = pump_aligned[p, :, :]
            curr_grid_trace_ch2 = pump_aligned_ch2[p, :, :]
            mean_trace = np.mean(curr_grid_trace, axis=0)
            mean_trace_ch2 = np.mean(curr_grid_trace_ch2, axis=0)
            sem_trace = sem(curr_grid_trace, axis=0)
            sem_trace_ch2 = sem(curr_grid_trace_ch2, axis=0)
            
            ax = fig.add_subplot(dimension, dimension, p+1)
            ax.set(xlabel='time (s)', ylabel='F', title='grid {}'.format(p))
            ax.plot(xaxis, mean_trace, color='limegreen', linewidth=1, zorder=10)
            ax.fill_between(xaxis, mean_trace+sem_trace,
                                   mean_trace-sem_trace,
                            color='limegreen', edgecolor='none', alpha=.2, zorder=10)
            ax2 = ax.twinx()
            ax2.plot(xaxis, mean_trace_ch2, color='red', linewidth=1, zorder=1, alpha=.55)
            ax2.fill_between(xaxis, mean_trace_ch2+sem_trace_ch2,
                                    mean_trace_ch2-sem_trace_ch2,
                             color='red', edgecolor='none', alpha=.1, zorder=1)
            ax.axvspan(0, 0, color='grey', alpha=.5, linestyle='dashed', linewidth=1)
        fig.suptitle('rew_aligned_ch2')
        fig.tight_layout()
        fig.savefig('{}/grid_traces_{}_avg_rew_aligned.png'.format(extract_path, stride),
                    dpi=120,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
            
            
    ## cue-onset aligned
    if align_cue==1:
        print('\nplotting traces aligned to CUE...')
        
        # filter cues ([first_frame, last_frame])
        cues = [t for t in movie_times if t>first_frame and t<last_frame]
        
        # get aligned frame numbers
        cue_frames = []
        for trial in range(len(cues)):
            cue_frames.append(ipf.find_nearest(cues[trial], frame_times))
    
        # align traces to cues
        tot_cue = len(cues)
        cue_aligned = np.zeros((tot_grid, (tot_cue-1), (bef+aft)*30))
        cue_aligned_ch2 = np.zeros((tot_grid, (tot_cue-1), (bef+aft)*30))
        for i, c in enumerate(cue_frames[:-1]):
            cue_aligned[:, i, :] = grid_traces[:, c-bef*30:c+aft*30]
            cue_aligned_ch2[:, i, :] = grid_traces_ch2[:, c-bef*30:c+aft*30]
            
        print('plotting heatmaps...')
        # heatmap ch1
        fig = plt.figure(1, figsize=(dimension*2.5, dimension*1.5))
        for p in range(tot_plot):
            curr_grid_trace = cue_aligned[p, :, :]
            curr_grid_map = np.zeros((tot_cue-1, (bef+aft)*30))
            for i in range(tot_cue-1):
                curr_grid_map[i, :] = normalise(curr_grid_trace[i, :])
            
            ax = fig.add_subplot(dimension, dimension, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
            ax.imshow(curr_grid_map, aspect='auto', extent=[-bef,aft,1,tot_cue], cmap='Greys')
    
        fig.suptitle('cue_aligned')
        fig.tight_layout()
        fig.savefig('{}/grid_traces_{}_cue_aligned.png'.format(extract_path, stride),
                    dpi=120,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
            
        # heatmap ch2
        fig = plt.figure(1, figsize=(dimension*2.5, dimension*1.5))
        for p in range(tot_plot):
            curr_grid_trace_ch2 = cue_aligned_ch2[p, :, :]
            curr_grid_map_ch2 = np.zeros((tot_cue-1, (bef+aft)*30))
            for i in range(tot_cue-1):
                curr_grid_map_ch2[i, :] = normalise(curr_grid_trace[i, :])
            
            ax = fig.add_subplot(dimension, dimension, p+1)
            ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
            ax.imshow(curr_grid_map_ch2, aspect='auto', extent=[-bef,aft,1,tot_cue], cmap='Greys')
    
        fig.suptitle('cue_aligned_ch2')
        fig.tight_layout()
        fig.savefig('{}/grid_traces_{}_cue_aligned_ch2.png'.format(extract_path, stride),
                    dpi=120,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print('plotting combined averaged traces...')
        # average combined 
        fig = plt.figure(1, figsize=(dimension*2.5, dimension*1.5))
        for p in range(tot_plot):
            curr_grid_trace = cue_aligned[p, :, :]
            curr_grid_trace_ch2 = cue_aligned_ch2[p, :, :]
            mean_trace = np.mean(curr_grid_trace, axis=0)
            mean_trace_ch2 = np.mean(curr_grid_trace_ch2, axis=0)
            sem_trace = sem(curr_grid_trace, axis=0)
            sem_trace_ch2 = sem(curr_grid_trace_ch2, axis=0)
            
            ax = fig.add_subplot(dimension, dimension, p+1)
            ax.set(xlabel='time (s)', ylabel='F', title='grid {}'.format(p))
            ax.plot(xaxis, mean_trace, color='limegreen', linewidth=1)
            ax.fill_between(xaxis, mean_trace+sem_trace,
                                   mean_trace-sem_trace,
                            color='limegreen', edgecolor='none', alpha=.2)
            ax2 = ax.twinx()
            ax2.plot(xaxis, mean_trace_ch2, color='red', linewidth=1)
            ax2.fill_between(xaxis, mean_trace_ch2+sem_trace_ch2,
                                    mean_trace_ch2-sem_trace_ch2,
                             color='red', edgecolor='none', alpha=.2)
            ax.axvspan(0, 0, color='grey', alpha=.5, linestyle='dashed', linewidth=1)
        fig.suptitle('rew_aligned_ch2')
        fig.tight_layout()
        fig.savefig('{}/grid_traces_{}_avg_cue_aligned.png'.format(extract_path, stride),
                    dpi=120,
                    bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        
    ## end statement 
    print('processing for {} is finished\n\n\n'.format(recname))
    

#%% suite2p pipeline function
def run_suite2p_pipeline(rec_path, recname, reg_path, txt_path,
                         plot_ref, plot_heatmap, plot_trace,
                         smooth, dFF,
                         bef, aft,
                         align_run, align_rew, align_cue):
    
    ops_path = reg_path+r'\ops.npy'
    F_path = reg_path+r'\F.npy'
    F2_path = reg_path+r'\F_chan2.npy'
    Fneu_path = reg_path+r'\Fneu.npy'
    Fneu2_path = reg_path+r'\Fneu_ch2.npy'
    stat_path = reg_path+r'\stat.npy'

    if not os.path.exists(stat_path):
        print('Suite2p ROI extraction file not found')

    # make directory
    extract_path = rec_path+r'_roi_extract'
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    # beh file
    print('\nreading behaviour file...')
    txt = ipf.process_txt(txt_path)

    #load ROIs and trace
    F_all = np.load(F_path, allow_pickle=True)
    F_all2 = np.load(F2_path, allow_pickle=True)
    Fneu_all = np.load(Fneu_path, allow_pickle=True)

    tot_roi = F_all.shape[0]
    tot_frames = F_all.shape[1]

    if dFF:
        print('calculating dFF...')
        F_all = ipf.calculate_dFF(F_all)
        F_all2 = ipf.calculate_dFF(F_all2)
        Fneu_all = ipf.calculate_dFF(Fneu_all)
        
        
    ## timestamps
    print('\ndetermining behavioural timestamps...') 
    # pumps
    pump_times = txt['pump_times']

    # speed
    speed_times = txt['speed_times']

    # cues 
    movie_times = txt['movie_times']

    # frames
    frame_times = txt['frame_times']

    # licks
    lick_times = txt['lick_times']

    tot_trial = len(speed_times)
    

    ## correct overflow
    print('\ncorrecting overflow...')
    pump_times = ipf.correct_overflow(pump_times, 'pump')
    speed_times = ipf.correct_overflow(speed_times, 'speed')
    movie_times = ipf.correct_overflow(movie_times, 'movie')
    frame_times = ipf.correct_overflow(frame_times, 'frame')
    # lick_times = ipf.correct_overflow(lick_times, 'lick')
    first_frame = frame_times[0]; last_frame = frame_times[-1]


    ## **fill in dropped $FM signals
    # since the 2P system does not always successfully transmit frame signals to
    # the behavioural recording system every time it acquires a frame, one needs to
    # manually interpolate the frame signals in between 2 frame signals that are 
    # further apart than 50 ms
    print('\nfilling in dropped $FM statements...')
    for i in range(len(frame_times)-1):
        if frame_times[i+1]-frame_times[i]>50:
            interp_fm = (frame_times[i+1]+frame_times[i])/2
            frame_times.insert(i+1, interp_fm)
            
    # checks $FM against tot_frame
    if tot_frames<len(frame_times)-3 or tot_frames>len(frame_times):
        print('\nWARNING:\ncheck $FM; halting processing for {}\n'.format(recname))
        align_run=0; align_rew=0; align_cue=0  # if tot_frame is more than sync signals or fewer than syncs-3, then halt
        
        
    ## calculate dimension for plotting
    n_col = int(tot_roi**0.5)
    n_row = int(np.ceil(tot_roi/n_col))
    
    xaxis = (np.arange((bef+aft)*30)-bef*30)/30

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
            
            run_onsets.append(ipf.get_onset(uni_speed, uni_time))
            
        # filter run-onsets ([first_frame, last_frame])
        run_onsets = [t for t in run_onsets if t>first_frame and t<last_frame]
        
        # get aligned frame numbers
        run_frames = []
        for trial in range(len(run_onsets)):
            if run_onsets!=-1:  # if there is a clear run-onset in this trial
                rf = ipf.find_nearest(run_onsets[trial], frame_times)
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
                run_aligned_ch2[roi, i, :] = F_all2[roi][r-bef*30:r+aft*30]
                run_aligned_neu[roi, i, :] = Fneu_all[roi][r-bef*30:r+aft*30]
        
        if plot_heatmap:
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
                fig.savefig('{}/suite2pROI_run_dFF_aligned.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            else:
                fig.savefig('{}/suite2pROI_run_aligned.png'.format(extract_path),
                            dpi=120,
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
                fig.savefig('{}/suite2pROI_run_dFF_aligned_ch2.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            else:
                fig.savefig('{}/suite2pROI_run_aligned_ch2.png'.format(extract_path),
                            dpi=120,
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
                fig.savefig('{}/suite2pROI_run_dFF_aligned_neu.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            else:
                fig.savefig('{}/suite2pROI_run_aligned_neu.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            plt.show()
            plt.close(fig)
        
        if plot_trace:
            print('plotting combined averaged traces...')
            # average combined 
            fig = plt.figure(1, figsize=(n_row*4, n_col*3))
            for p in range(tot_roi):
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
                ax.plot(xaxis, mean_trace, color='darkgreen', linewidth=.8)
                ax.fill_between(xaxis, mean_trace+sem_trace,
                                       mean_trace-sem_trace,
                                color='darkgreen', edgecolor='none', alpha=.2)
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
                fig.savefig('{}/suite2pROI_avgdFF_run_aligned.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            else:
                fig.savefig('{}/suite2pROI_avg_run_aligned.png'.format(extract_path),
                            dpi=120,
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
            pump_frames.append(ipf.find_nearest(pumps[trial], frame_times))

        # align traces to pumps
        tot_pump = len(pumps)
        pump_aligned = np.zeros((tot_roi, (tot_pump-1), (bef+aft)*30))
        pump_aligned_ch2 = np.zeros((tot_roi, (tot_pump-1), (bef+aft)*30))
        pump_aligned_neu = np.zeros((tot_roi, (tot_pump-1), (bef+aft)*30))
        for i, p in enumerate(pump_frames[:-1]):
            pump_aligned[:, i, :] = F_all[:, p-bef*30:p+aft*30]
            pump_aligned_ch2[:, i, :] = F_all2[:, p-bef*30:p+aft*30]
            pump_aligned_neu[:, i, :] = Fneu_all[:, p-bef*30:p+aft*30]
        
        if plot_heatmap:
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
                fig.savefig('{}/suite2pROI_rew_dFF_aligned.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            else:
                fig.savefig('{}/suite2pROI_rew_aligned.png'.format(extract_path),
                            dpi=120,
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
                fig.savefig('{}/suite2pROI_rew_dFF_aligned_ch2.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            else:
                fig.savefig('{}/suite2pROI_rew_aligned_ch2.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
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
                fig.savefig('{}/suite2pROI_rew_dFF_aligned_neu.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            else:
                fig.savefig('{}/suite2pROI_rew_aligned_neu.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            plt.show()
            plt.close(fig)
            
        if plot_trace:
            print('plotting combined averaged traces...')
            # average combined 
            fig = plt.figure(1, figsize=(n_row*4, n_col*3))
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
                ax.plot(xaxis, mean_trace, color='darkgreen', linewidth=.8)
                ax.fill_between(xaxis, mean_trace+sem_trace,
                                       mean_trace-sem_trace,
                                color='darkgreen', edgecolor='none', alpha=.2)
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
                fig.savefig('{}/suite2pROI_avgdFF_rew_aligned.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            else:
                fig.savefig('{}/suite2pROI_avg_rew_aligned.png'.format(extract_path),
                            dpi=120,
                            bbox_inches='tight')
            plt.show()
            plt.close(fig)


    # #%% get lick frames
    # lick_frames = []
    # for t in lick_times:
    #     if t[0][0]>first_frame and t[-1][0]<last_frame:
    #         # tmp_licks = [l[0] for l in t]
    #         # licks.append(tmp_licks)
    #         licks = [grf.find_nearest(lick[0], frame_times) for lick in t]
    #         lick_frames.append(licks)

    # #%% interpolation for speed to match with total nunmber of frames
    # speeds = []
    # speeds_times = []
    # for t in speed_times: # get all speeds for the whole session
    #     if t[0][0]>first_frame and t[-1][0]<last_frame: # filter speeds ([first_frame, last_frame])
    #         speeds.extend([p[1] for p in t])
    #         speeds_times.extend([p[0] for p in t])
    # uni_time = np.linspace(first_frame,last_frame, tot_frames)        
    # uni_speeds = np.interp(uni_time,speeds_times,speeds)  # interpolation for speed, match with total nunmber of frames

    # #%% DEMO PLOT: plot consective trials with speed and licks and dLight
    # f_start=5000
    # f_end=6000
    # roi =10


    # F_trace = F_all[roi,f_start:f_end]
    # rew = [f for f in pump_frames if f_start<f<f_end]
    # run = [f for f in run_frames if f_start<f<f_end]
    # lick = [f for f in np.hstack(lick_frames) if f_start<f<f_end]
    # speed = uni_speeds[f_start:f_end]

    # plt.rcParams['axes.labelsize']= 11
    # plt.rcParams['xtick.labelsize']= 10
    # plt.rcParams['ytick.labelsize']= 10

    # fig, ax = plt.subplots(figsize=(12, 2),dpi=120)
    # xaxis = np.arange(f_start, f_end)
    # ax.plot(xaxis,F_trace, color = 'green', linewidth=0.8, label='dLight3.6')
    # ax.set(ylabel='dLight_dFF')
    # ax1 = ax.twinx()
    # ax1.plot(xaxis,speed, color = 'steelblue', linewidth=0.8, alpha=0.8, label='speed (cm/s)')
    # ax1.set(ylim=(0,100))
    # ax1.vlines(run,0,800, color='orange', linewidth=0.8, alpha=.8, label='run onset')
    # ax1.spines['top'].set_visible(False)
    # ax1.tick_params(right=False, labelright=False)
    # ax1.vlines(lick, 85,95, color='darkslateblue', linewidth=0.2, alpha=0.8, label='licks')
    # ax1.vlines(rew, 0,800, color='deepskyblue', linewidth=0.8, alpha=1, label='reward')
    # ax1.set(ylabel='speed (cm/s)')
    # ax1.yaxis.label.set_color('steelblue')
    # ax1.tick_params(top=False,
    #                bottom=False,
    #                left=False,
    #                right=True,
    #                labelleft=False,
    #                labelright=True,
    #                labelbottom=True)
    # ax.spines['top'].set_visible(False) 
    # ax.set(xlabel='frames', xlim=(f_start,f_end))
    # fig.legend(prop={'size':4}, loc=(0.065, 0.77), frameon = False)
    # fig.tight_layout()


    # #%%

    # plt.rcParams['axes.labelsize']= 6
    # plt.rcParams['xtick.labelsize']= 5
    # plt.rcParams['ytick.labelsize']= 5 

    # fig, ax = plt.subplots(figsize=(16, 2),dpi=120)
    # xaxis = np.arange(len(F_trace))/30
    # ax.plot(xaxis,F_trace, color = 'green', linewidth=0.5, label='dLight3.6')
    # ax.set(ylabel='dLight_dFF')
    # ax1 = ax.twinx()
    # ax1.plot(xaxis,speeds, color = 'lightsteelblue', linewidth=0.5, alpha=0.8, label='speed (mm/s)')
    # ax1.set(ylim=(0,1000))
    # ax1.vlines(runs/500,0,800, color='orange', linewidth=0.3, alpha=1, label='run onset')
    # ax1.spines[['top', 'right']].set_visible(False)
    # ax1.tick_params(right=False, labelright=False)
    # ax1.vlines(lick_frames/30, 850,900, color='darkslateblue', linewidth=0.1, alpha=0.5, label='licks')
    # ax1.vlines(rews/30, 850, 900, color='deepskyblue', linewidth=0.5, alpha=1, label='reward')
    # ax.spines[['top','right']].set_visible(False) 
    # ax.set(xlabel='Time (s)', xlim=(0), ylim=(-0.6, 0.6))
    # fig.legend(prop={'size':4}, loc=1, frameon = False)
    # fig.tight_layout()
    # plt.show()        

    # #%% DEMO PLOT: plot heatmap for one roi:
    # # plotting parameters
    # import matplotlib
    # plt.rcParams['font.family'] = 'Arial'
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42
       
    # roi = 13
    # run = 0
    # rew = 1
    # smooth = 1
    # sigma=1
    # heatmap = 1
    # trace = 1
    # c_dLight = 'lightseagreen'
    # c_neu = 'rosybrown'
    # if run:
    #     plot_array = run_aligned[roi,:,:]
    #     plot_array_neu = run_aligned_neu[roi,:,:]

    # if rew:
    #     plot_array = pump_aligned[roi,:,:]
    #     plot_array_neu = pump_aligned_neu[roi,:,:]
        
    # for i in range(plot_array.shape[0]):
    #     plot_array[i,:] = normalise(plot_array[i,:])
    #     plot_array_neu[i,:] = normalise(plot_array_neu[i,:])

    # if smooth:
    #     plot_array = gaussian_filter(plot_array, sigma, axes=1)
    #     plot_array_neu = gaussian_filter(plot_array_neu, sigma, axes=1)


    # xaxis = np.arange((bef+aft)*30)/30-bef  # xaxis for plotting

    # if heatmap:
    #     fig, ax = plt.subplots(figsize=(3,3))
    #     ax.imshow(plot_array, cmap='Greys', aspect='auto',vmin=np.percentile(plot_array,1), vmax=np.percentile(plot_array,99))
    #     ax.axvspan(bef*30, bef*30, color='black', linewidth=1, linestyle='dashed', alpha=.5)
    #     # ax.set(xlabel='time (s)',ylabel='n_ROI',xticks=np.linspace(0, (0.5+aft)*30, 8), xticklabels=np.arange(-0.5,aft+0.5,0.5))
    #     ax.set(xlabel='time (s)',ylabel='n_trial', xticks=np.linspace(0, (bef+aft)*30, 2*(bef+aft)+1), xticklabels=np.arange(-bef,aft+0.5,0.5))
    #     fig.tight_layout()
    #     if run:
    #         fig.savefig(r"Z:\Jingyu\plots_20240624\demo_heatmap_AC928-20240327-02-roi{}_ro_ramp.pdf".format(roi), dpi=120, transparent=True, bbox_inches="tight", format="pdf")
    #     if rew:
    #         fig.savefig(r"Z:\Jingyu\plots_20240624\demo_heatmap_AC928-20240327-02-roi{}_reward_peak.pdf".format(roi), dpi=120, transparent=True, bbox_inches="tight", format="pdf")
    #     plt.show()
    #     # plt.close()

    # if trace:
    #     fig, ax = plt.subplots(figsize=(3,1.5))
    #     trace_mean = np.mean(plot_array, axis=0)
    #     trace_sem = sem(plot_array, axis=0)
    #     trace_neu_mean = np.mean(plot_array_neu, axis=0)
    #     trace_neu_sem = sem(plot_array_neu, axis=0)
        
    #     ax.plot(xaxis, trace_mean, lw=.8,color=c_dLight, label='dLight3.6')
    #     ax.plot(xaxis, trace_neu_mean, lw=.8, color=c_neu,label='Neuropil')
    #     ax.fill_between(xaxis, trace_mean+trace_sem, trace_mean-trace_sem,
    #                      color=c_dLight, edgecolor='none', alpha=.2)
    #     ax.fill_between(xaxis, trace_neu_mean+trace_neu_sem, trace_neu_mean-trace_neu_sem,
    #                      color=c_neu, edgecolor='none', alpha=.2)
    #     ax.axvspan(0, 0, color='black', linewidth=.5, linestyle='--', alpha=.5)
    #     ax.spines[['right', 'top']].set_visible(False)
    #     ax.set(xlabel='time (s)',ylabel='dFF', xlim=(-bef,aft))
    #     ax.legend(prop={'size':4}, loc=(0.8, 0.77), frameon = False)
    #     # ax.set(xlabel='time (s)',ylabel='dFF', xticks=np.linspace(0, (bef+aft)*30, 2*(bef+aft)+1), xticklabels=np.arange(-bef,aft+0.5,0.5))
    #     fig.tight_layout()
    #     if run:
    #         fig.savefig(r"Z:\Jingyu\plots_20240624\demo_avgTrace_AC928-20240327-02-roi{}_ro_ramp.pdf".format(roi), dpi=120, transparent=False, bbox_inches="tight", format="pdf")
    #     if rew:
    #         fig.savefig(r"Z:\Jingyu\plots_20240624\demo_avgTrace_AC928-20240327-02-roi{}_reward_peak.pdf".format(roi), dpi=120, transparent=False, bbox_inches="tight", format="pdf")
    #     plt.show()u