# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:42:50 2024

Perform Fourier transform on dF/F

@author: Dinghao Luo
"""


#%% imports 
import sys 
import numpy as np
import matplotlib.pyplot as plt 
from numpy.fft import fft

# plotting parameters
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% recording list
if r'Z:\Dinghao\code_dinghao' not in sys.path:
    sys.path.append('Z:\Dinghao\code_dinghao')
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
    
    roi_traces = np.load(r'{}/processed/suite2p/plane0/F.npy'.format(path),
                         allow_pickle=True)
        
    tot_roi = roi_traces.shape[0]
    

    # session-specific parameters 
    T = (roi_traces.shape[1]-sigma*2)*dt  # total rec time in seconds
    df = 1/T  # freq resolution 
    fNQ = 1/dt/2  # Nyquist 
    faxis = np.arange(0, fNQ, df)
    

    ## main loop
    all_specs = []  # container for spectra
    for i in range(tot_roi):
    
        # convolution
        F_conv = np.convolve(roi_traces[i,:], gaussian_filter, mode='same')
        
        # dFF 
        dFF = []
        for j, f in enumerate(roi_traces[i,:][sigma: -sigma]):
            dFF.append(f/np.mean(F_conv[j:j+sigma*2]))
        dFF = np.asarray(dFF)
        
        curr_fft = fft(roi_traces[i,:] - roi_traces[i,:].mean())
        spectrum = 2 * dt ** 2 / T * (curr_fft * curr_fft.conj())
        all_specs.append(spectrum[:int(dFF.shape[0]/2)])
        
        
    ## plotting 
    tot_plot = tot_roi
    n_col = 5
    n_row = int(np.ceil(tot_plot/n_col))
    
    fig = plt.figure(1, figsize=(n_col*1.6, n_row*1.2))
    for p in range(tot_plot):
        curr_spec = all_specs[p]
        
        ax = fig.add_subplot(n_row, n_col, p+1)
        ax.set(xlabel='frequency (Hz)', xticks=[0,1,2],
               title='roi {}'.format(p+1))
        ax.plot(faxis[:3500], curr_spec.real[:3500], color='darkgreen', lw=1)
    
    fig.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_tonic_fft_rois\{}.png'.format(recname),
                dpi=120, bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\GRABNE\single_session_tonic_fft_rois\{}.pdf'.format(recname),
                bbox_inches='tight')
    
    plt.close(fig)
    