# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:26:31 2024

process and align ROI traces
***this script is specific to axon/dendrite GCaMP data processing, due to the 
   activity merging stage producing special data structures***

overview of the merged data structure:
    - stat.npy contains (per usual) info about the ROIs; after merging and 
        appending the merges, each ROI has 'imerge' and 'inmerge':
        - 'imerge' tells one the constituent ROI(s) of a merged ROI and
        - 'inmerge' tells one of which ROI(s) the current ROI is a part
    - therefore, if one ROI conforms to (1) iscell==True and (2) inmerge<1,
        this ROI is considered valid (sorted)

@author: Dinghao Luo
"""



#%% imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import sys
import os
from time import time
from datetime import timedelta
from scipy.stats import sem 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import normalise, smooth_convolve, generate_dir, mpl_formatting

# global variables 
bef = 1
aft = 4  # in seconds
xaxis = np.arange((bef+aft)*30)/30-1

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPCLCGCaMP = rec_list.pathHPCLCGCaMP

df = pd.read_pickle(r'Z:\Dinghao\code_dinghao\behaviour\all_HPCLCGCaMP_sessions.pkl')


#%% GPU acceleration
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
except ModuleNotFoundError:
    print('cupy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
except Exception as e:
    # catch any other unexpected errors and print a general message
    print(f'An error occurred: {e}')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    # we are assuming that there is only 1 GPU device
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print(f'GPU-acceleration with {str(name)} and cupy')
else:
    print('GPU-acceleartion unavailable')


#%% main
for rec_path in pathHPCLCGCaMP:
    recname = rec_path[-17:]
    print(recname)
    
    ops_path = rec_path+r'/suite2p/plane0/ops.npy'
    bin_path = rec_path+r'/suite2p/plane0/data.bin'
    bin2_path = rec_path+r'/suite2p/plane0/data_chan2.bin'
    stat_path = rec_path+r'/suite2p/plane0/stat.npy'
    iscell_path = rec_path+r'/suite2p/plane0/iscell.npy'
    F_path = rec_path+r'/suite2p/plane0/F.npy'
    
    # folder to put processed data and single-session plots 
    proc_path = r'Z:\Dinghao\code_dinghao\axon_GCaMP\single_sessions\{}'.format(recname)
    generate_dir(proc_path)
    ref_path = proc_path+r'/ref_ch1.png'
    
    # load files 
    stat = np.load(stat_path, allow_pickle=True)
    if 'inmerge' not in stat[0]: sys.exit('halting executation: no merging detected')  # detect merging
    
    iscell = np.load(iscell_path, allow_pickle=True)
    ops = np.load(ops_path, allow_pickle=True).item()
    
    tot_rois = len(stat)
    tot_frames = ops['nframes']
    shape = tot_frames, ops['Ly'], ops['Lx']
    mov = np.memmap(bin_path, mode='r', dtype='int16', shape=shape)
    mov2 = np.memmap(bin2_path, mode='r', dtype='int16', shape=shape)
    
    # behaviour file
    beh = df.loc[recname]
    run_frames = beh['run_onset_frames']
    pump_frames = beh['pump_frames']
    cue_frames = beh['cue_frames']
    frame_times = beh['frame_times']
    for i in range(len(run_frames)-1, -1, -1):  # filter out the no-clear-onset trials
        if run_frames[i]==-1:
            del run_frames[i]
    for i in range(len(pump_frames)-1, -1, -1):  # filter out the no-rew trials
        if pump_frames[i]==-1:
            del pump_frames[i]
    
    # checks $FM against tot_frame
    if tot_frames<len(frame_times)-3 or tot_frames>len(frame_times):
        print('\nWARNING:\ncheck $FM; halting processing for {}\n'.format(recname))
        align_run=0; align_rew=0; align_cue=0  # if tot_frame is more than sync signals or fewer than syncs-3, then halt
        
    
    # reference images
    if not os.path.exists(ref_path):
        print('plotting reference images...')
        start = time()  # timer
        ref_im = ipf.plot_reference(mov, channel=1, outpath=proc_path, GPU_AVAILABLE=GPU_AVAILABLE)
        print('ref done ({})'.format(str(timedelta(seconds=int(time()-start)))))
        ref_ch2_im = ipf.plot_reference(mov2, channel=2, outpath=proc_path, GPU_AVAILABLE=GPU_AVAILABLE)
        print('ref_ch2 done ({})'.format(str(timedelta(seconds=int(time()-start)))))
    else:
        print(f'ref images already plotted; loading ref_im from {ref_path}...')
        ref_im = np.load(ref_path[:-4]+'.npy', allow_pickle=True)


    # find valid ROIs
    valid_rois = []
    for roi in range(tot_rois):
        # if this ROI was sorted as a 'cell' & is not merged into another ROI
        if iscell[roi][0]==1 and stat[roi]['inmerge']<1:
            valid_rois.append(roi)
    tot_valids = len(valid_rois)

    
    # ROI plots 
    fig, axs = plt.subplots(1,3, figsize=(6,2))
    fig.subplots_adjust(wspace=.35, top=.75)
    for i in range(3):
        axs[i].set(xlim=(0,512), ylim=(0,512))
        axs[i].set_aspect('equal')
    
    colors_ch1 = plt.cm.Reds(np.linspace(0, 0.8, 256))
    colors_ch2 = plt.cm.Greens(np.linspace(0, 0.8, 256))
    custom_cmap_ch1 = LinearSegmentedColormap.from_list('mycmap', colors_ch1)
    custom_cmap_ch2 = LinearSegmentedColormap.from_list('mycmap', colors_ch2)
    axs[1].imshow(ref_im, cmap=custom_cmap_ch1)
    axs[2].imshow(ref_ch2_im, cmap=custom_cmap_ch2)
    axs[1].set(title='axon-GCaMP')
    axs[2].set(title='Dbh:Ai14')
    
    for roi in valid_rois:
        axs[0].scatter(stat[roi]['xpix'], stat[roi]['ypix'], edgecolor='none', s=.1, alpha=.2)
    axs[0].set(title='merged ROIs')
    
    fig.suptitle(recname)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(r'{}\rois_v_ref{}'.format(proc_path, ext),
                    dpi=200)
    plt.close(fig)
    
    
    # plotting
    F = np.load(F_path, allow_pickle=True)
    
    run_path = r'{}\run_aligned'.format(proc_path)
    generate_dir(run_path)
    
    all_mean_run = np.zeros((tot_valids, (bef+aft)*30)); counter = 0
    for roi in valid_rois:
        print(f'ROI {roi}')
        ca = F[roi]
        start = time()  # timer
        ca = ipf.calculate_dFF(ca, GPU_AVAILABLE=False)  # not using GPU is faster when we are calculating only 1 vector
        
        # align to run 
        filtered_run_frames = []  # ensures monotonity of ordered ascension
        last_frame = float('-inf')  # initialise to negative infinity
        for frame in run_frames:
            if frame > last_frame:
                filtered_run_frames.append(frame)
                last_frame = frame   
        tot_run = len(filtered_run_frames)
        head = 0; tail = len(filtered_run_frames)
        for f in range(tot_run):
            if filtered_run_frames[f]-bef*30<0:
                head+=1
            else:
                break
        for f in range(tot_run-1,-1,-1):
            if filtered_run_frames[f]+aft*30>tot_frames:
                tail-=1
            else:
                break
        tot_trunc = head + (len(filtered_run_frames)-tail)
        run_aligned = np.zeros((tot_run-tot_trunc, (bef+aft)*30))  # grid x trial x frame bin
        run_aligned_im = np.zeros((tot_run-tot_trunc, (bef+aft)*30))  # grid x trial x frame bin
        for i, r in enumerate(filtered_run_frames[head:tail]):
            run_aligned[i, :] = ca[r-bef*30:r+aft*30]
            run_aligned_im[i, :] = normalise(smooth_convolve(ca[r-bef*30:r+aft*30]))
        
        run_aligned_mean = np.mean(run_aligned, axis=0)
        run_aligned_sem = sem(run_aligned, axis=0)
        all_mean_run[counter, :] = run_aligned_mean
            
        fig = plt.figure(figsize=(5, 2.5))
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[:, 0]) 
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        fig.subplots_adjust(wspace=.35)
        
        ax1.imshow(ref_im, cmap='gist_gray')
        ax1.scatter(stat[roi]['xpix'], stat[roi]['ypix'], color='limegreen', edgecolor='none', s=.1, alpha=1)
        ax1.plot(stat[roi]['xcirc'], stat[roi]['ycirc'], linewidth=.5, color='white')
        ax1.set(xlim=(0,512), 
                ylim=(0,512))
        
        ax2.imshow(run_aligned_im, cmap='Greys', extent=[-1, 4, 0, tot_run], aspect='auto')
        ax2.set(xticks=[],
                ylabel='trial #')
        
        ax3.plot(xaxis, run_aligned_mean, c='darkgreen')
        ax3.fill_between(xaxis, run_aligned_mean+run_aligned_sem,
                                run_aligned_mean-run_aligned_sem,
                            color='darkgreen', alpha=.2, edgecolor='none')
        ax3.set(xlabel='time from run-onset (s)',
                ylabel='dF/F')
        
        fig.suptitle('ROI {} run-onset-aligned'.format(roi))
        
        fig.savefig(r'{}\roi_{}.png'.format(run_path, roi),
                    dpi=300)
        plt.close(fig)
        
        counter+=1
            
    
    # plot heatmap of all 
    max_pt = {}  # argmax for average for all cells
    for roi in range(tot_valids):
        max_pt[roi] = np.argmax(all_mean_run[roi])
    def sort_helper(x):
        return max_pt[x]
    keys = sorted(np.arange(tot_valids), key=sort_helper)
    im_matrix = np.zeros((tot_valids, (bef+aft)*30))
    for i, roi in enumerate(keys):
        im_matrix[i, :] = all_mean_run[roi]
    
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(im_matrix, aspect='auto', extent=[-1,4,0,tot_rois])
    ax.set(xlabel='time from run-onset (s)',
           ylabel='ROI #')
    fig.savefig(r'{}\run_aligned_sorted.png'.format(proc_path, recname))
    plt.close(fig)
# %%
