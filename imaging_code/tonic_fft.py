# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:42:50 2024

Perform Fourier transform on dF/F

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt 
from numpy.fft import fft


#%% load data 
Fall = sio.loadmat(r'Z:/Jingyu/2P_Recording/AC926/AC926-20240306/02/denoise=1_rolling=max_pix=0.04_peak=0.03_iterations=1_norm=max_neuropillam=True/suite2p/plane0/Fall.mat')

F = Fall['F']

tot_roi = F.shape[0]


#%% Gaussian filter 
sigma = 150  # 5 seconds 
gx = np.arange(-sigma*6, sigma*6, 1)
gaussian_filter = [1 / (sigma*np.sqrt(2*np.pi)) * 
                   np.exp(-x**2/(2*sigma**2)) for x in gx]


#%% parameters 
# time 
samp_freq = 30
dt = 1/samp_freq  # in seconds 
T = (F.shape[1]-sigma*2)*dt  # in seconds

# frequency
df = 1 / T
fNQ = 1 / dt / 2
faxis = np.arange(0, fNQ, df)


#%% main 
all_specs = []

for i in range(tot_roi):

    # convolution
    F_conv = np.convolve(F[i,:], gaussian_filter, mode='same')
    
    # dFF 
    dFF = []
    for j, f in enumerate(F[i,:][sigma: -sigma]):
        dFF.append(f/np.mean(F_conv[j:j+sigma*2]))
    dFF = np.asarray(dFF)
    
    curr_fft = fft(F[i,:] - F[i,:].mean())
    spectrum = 2 * dt ** 2 / T * (curr_fft * curr_fft.conj())
    all_specs.append(spectrum[:int(dFF.shape[0]/2)])
    
    
#%% plotting 
tot_plot = tot_roi
row = 50
col = 5

fig = plt.figure(1)
for p in range(tot_plot):
    curr_spec = all_specs[p]
    
    ax = fig.add_subplot(row, col, p+1)
    ax.set(xlabel='frequency (Hz)')
    ax.plot(faxis[:1500], curr_spec.real[:1500])

plt.show()