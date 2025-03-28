# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:24:39 2024

analyse the colocalisation between a signal channel and a ref channel
current config.: dLight v Dbh:Ai14

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import sys 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf
import imaging_utility_functions as iuf

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


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
# replace the following lines 
sys.path.append(r'Z:\Jingyu\Code\Python')
import anm_list_running
pathDbhdLight = anm_list_running.AC923 + anm_list_running.AC925

for recname in pathDbhdLight:
    print(f'\ngenerating/loading reference images for {recname}')
    ops_path = r'Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly\suite2p\plane0\ops.npy'.format(recname[:5], recname[:-3], recname[-2:])
    bin_path = r'Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly\suite2p\plane0\data.bin'.format(recname[:5], recname[:-3], recname[-2:])
    bin2_path = r'Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly\suite2p\plane0\data_chan2.bin'.format(recname[:5], recname[:-3], recname[-2:])
    proc_path = r'Z:\Dinghao\code_dinghao\dLight\{}'.format(recname)
    
    # special stat file seperate from RegOnly
    stat_path = r'Z:\Jingyu\2P_Recording\{}\{}\{}\denoise=1_rolling=max_pix=0.04_peak=0.03_iterations=1_norm=max_neuropillam=True\suite2p\plane0\stat.npy'.format(recname[:5], recname[:-3], recname[-2:])
    
    ops = np.load(ops_path, allow_pickle=True).item()
    tot_frames = ops['nframes']

    stat = np.load(stat_path, allow_pickle=True)

    # if one day one needs to use the image matrices, just enable assignment here
    ref, ref_ch2 = ipf.load_or_generate_reference_images(proc_path,
                                                         bin_path, bin2_path,
                                                         tot_frames, ops,
                                                         GPU_AVAILABLE)
    
    # overlap iamge 
    fig, ax = plt.subplots(figsize=(3,3))
    ax.set(xlim=(0, 512), 
           ylim=(0, 512),
           title=recname)
    ax.set_aspect('equal')
    
    # display reference images in channels 1 and 2
    ax.imshow(ref, cmap='Greens', alpha=.8)
    ax.imshow(ref_ch2, cmap='Reds', alpha=.6)
    
    fig.savefig(os.path.join(proc_path, 'overlap_dLight_Dbh.png'),
                dpi=500,
                bbox_inches='tight')
    
    # ROI overlay
    for roi in stat:
        ax.scatter(roi['xpix'], roi['ypix'], color='darkgreen', edgecolor='none', s=0.1, alpha=0.5)
        
    ax.set(title=recname)
    fig.savefig(os.path.join(proc_path, 'overlap_dLight_Dbh_w_ROIs.png'),
                dpi=500,
                bbox_inches='tight')
    
    
    # colocalisation analysis
    mean_ref_intensities = []
    mean_manders = []
    for roi in tqdm(stat, desc='calculating ROI-ref colocalisation'):
        mean_ref_intensities.append(iuf.get_mean_ref_intensity_roi(roi['xpix'], 
                                                                   roi['ypix'], 
                                                                   ref_ch2))
        mean_manders.append(iuf.get_manders_coefficients(roi['xpix'],
                                                         roi['ypix'], 
                                                         ref_ch2))
        
    fig, ax = plt.subplots(figsize=(3,3))
    ax.set(xlim=(0, 512),
           ylim=(0, 512),
           title=f'{recname} shuf. ROIs')
    ax.set_aspect('equal')
    ax.imshow(ref, cmap='Greens', alpha=.8)
    ax.imshow(ref_ch2, cmap='Reds', alpha=.6)
    
    mean_ref_intensities_shuf = []
    mean_manders_shuf = []
    for roi in tqdm(stat, desc='calculating shuffled ROI-ref colocalisation'):
        shuf_x, shuf_y = iuf.shuffle_roi_coordinates(roi['xpix'], roi['ypix'])
        ax.scatter(shuf_x, shuf_y, color='darkgreen', edgecolor='none', s=0.1, alpha=0.5)
        mean_ref_intensities_shuf.append(iuf.get_mean_ref_intensity_roi(shuf_x, 
                                                                        shuf_y, 
                                                                        ref_ch2))
        mean_manders_shuf.append(iuf.get_manders_coefficients(shuf_x, 
                                                              shuf_y, 
                                                              ref_ch2))
    
    fig.savefig(os.path.join(proc_path, 'overlap_dLight_Dbh_w_shuf_ROIs.png'),
                dpi=500,
                bbox_inches='tight')
    
    pf.plot_violin_with_scatter(mean_ref_intensities, mean_ref_intensities_shuf,'darkgreen', 'grey', paired=False,
                                xticklabels=['dLight', 'shuf.\ndLight'],
                                ylabel='ref. intensity',
                                save=True, 
                                savepath=os.path.join(proc_path, 'mean_intensity_v_shuf.png'))
    # pf.plot_violin_with_scatter([manders[0] for manders in mean_manders], 
    #                             [manders[0] for manders in mean_manders_shuf], 
    #                             'darkgreen', 'grey', paired=False,
    #                             xticklabels=['dLight', 'shuf.\ndLight'],
    #                             ylabel='M1',
    #                             save=True, 
    #                             savepath=os.path.join(proc_path, 'M1_v_shuf.png'))
    # pf.plot_violin_with_scatter([manders[1] for manders in mean_manders], 
    #                             [manders[1] for manders in mean_manders_shuf], 
    #                             'darkred', 'grey', paired=False,
    #                             xticklabels=['dLight', 'shuf.\ndLight'],
    #                             ylabel='M2',
    #                             save=True,
    #                             savepath=os.path.join(proc_path, 'M2_v_shuf.png'))