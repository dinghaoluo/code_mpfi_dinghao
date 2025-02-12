# -*- coding: utf-8 -*-
"""
Created on Mon 10 Feb 11:41:21 2025

Perform Fourier transform on NE data

@author: Dinghao Luo
"""


#%% imports 
import sys 
import numpy as np
import matplotlib.pyplot as plt 
from numpy.fft import fft

sys.path.append(r'Z:\Dinghao\code_dinghao_mpfi\utils')
from common import mpl_formatting 
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE

    
#%% Gaussian filter 
sigma = 150  # 5 seconds 
gx = np.arange(-sigma*6, sigma*6, 1)
gaussian_filter = [1 / (sigma*np.sqrt(2*np.pi)) * 
                   np.exp(-x**2/(2*sigma**2)) for x in gx]


#%% parameters 
# time 
samp_freq = 30
dt = 1/samp_freq  # in seconds 


#%% load data 
for path in pathGRABNE:
    recname = path[-17:]
    print(recname)
    
    traces_whole_field = np.load(
        r'{}_grid_extract/grid_traces_dFF_496.npy'
        .format(path),
        allow_pickle=True)
    traces_whole_field = np.squeeze(traces_whole_field)
    
    traces_grids = np.load(
        r'{}_grid_extract/grid_traces_dFF_31.npy'
        .format(path),
        allow_pickle=True)
    tot_grids = traces_grids.shape[0]
    
    # session-specific parameters 
    T = (traces_whole_field.shape[0]-sigma*2)*dt  # total rec time in seconds
    df = 1/T  # freq resolution 
    fNQ = 1/dt/2  # Nyquist 
    faxis = np.arange(0, fNQ, df)
    
    
    ## whole-field
    F_conv_whole_field = np.convolve(traces_whole_field, gaussian_filter, mode='same')
    
    fft_whole_field = fft(traces_whole_field - traces_whole_field.mean())  # fft of the centralised trace
    spectrum = 2 * dt ** 2 / T * (fft_whole_field * fft_whole_field.conj())  # get power 
    
    fig, axs = plt.subplots(1,2, figsize=(6,2.8))
    
    for i in range(2):
        # plot the same figure but truncate x-axis for axs[1]
        axs[i].plot(faxis, spectrum[:len(faxis)], color='darkgreen', lw=1)
        axs[i].set(xlabel='frequency (Hz)', ylabel='power (a.u.)',
                   ylim=(0, max(spectrum[:len(faxis)])),
                   title='whole-field Fourier transform')
    
    axs[0].set(xlim=(0,0.3), xticks=(0,.2,.4,.6,.8,1),
               title='behaviour-relevance')
    axs[1].set(xlim=(0,0.1), title='tonic oscillations')
    
    fig.suptitle(f'{recname} whole-field NE')
    
    fig.tight_layout()

    fig.savefig(
        r'Z:\Dinghao\code_dinghao\GRABNE\single_session_tonic_fft_whole_field\{}.png'
        .format(recname),
        dpi=300,
        bbox_inches='tight')
    
    plt.close(fig)
    

    ## single grids
    all_specs = []  # container for spectra
    for i in range(tot_grids):
    
        # convolution
        F_conv = np.convolve(traces_grids[i,:], gaussian_filter, mode='same')
        
        # # dFF 
        # dFF = []
        # for j, f in enumerate(traces_grids[i,:][sigma: -sigma]):
        #     dFF.append(f/np.mean(F_conv[j:j+sigma*2]))
        # dFF = np.asarray(dFF)
        
        curr_fft = fft(traces_grids[i,:] - traces_grids[i,:].mean())
        spectrum = 2 * dt ** 2 / T * (curr_fft * curr_fft.conj())
        all_specs.append(spectrum)
        
    # plotting 
    tot_plot = tot_grids
    n_col = 16
    n_row = int(np.ceil(tot_plot/n_col))
    
    fig = plt.figure(1, figsize=(n_col*1.6, n_row*1.3))
    for p in range(tot_plot):
        curr_spec = all_specs[p]
        
        ax = fig.add_subplot(n_row, n_col, p+1)
        ax.set(xlabel='frequency (Hz)', xlim=(0,0.1), 
               ylim=(0, max(curr_spec[:len(faxis)])),
               title='grid {}'.format(p+1))
        ax.plot(faxis, curr_spec.real[:len(faxis)], color='darkgreen', lw=1)
    
    fig.tight_layout()
    plt.show()
    
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\GRABNE\single_session_tonic_fft_grids\{}.png'
        .format(recname),
        dpi=300, 
        bbox_inches='tight')
    
    plt.close(fig)