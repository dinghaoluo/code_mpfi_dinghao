# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:26:56 2025

playground script 

@author: Dinghao Luo
"""

#%% imports 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.signal import fftconvolve

from common import mpl_formatting
mpl_formatting()


#%% plotting 
# HPC curve 
taxis_HPC = np.linspace(-.5, 6, 550)
sigmoid_onset = 1 / (1 + np.exp(-5 * (taxis_HPC - 0.3)))  # onset at ~0.3 s
exponential_tail = np.exp(-taxis_HPC / 2.5)  # slow decay

HPC = sigmoid_onset * exponential_tail
HPC /= np.max(HPC)

# HPC_mod: reduced amplitude version
HPC_mod = HPC * 0.25

# LC curve
taxis_LC = np.linspace(-1, 1, 200)
gauss_std = 0.25 # standard deviation in seconds (sharp peak)
gauss_peak = np.exp(-0.5 * (taxis_LC / gauss_std)**2)
gauss_peak /= np.max(gauss_peak)  # normalise to peak at 1

# dopamine trace: starts at time -1, similar rise to LC Gaussian, slower decay
taxis_decay = np.linspace(0, 6, 600)
tau_decay_mod = 4.4
decay = np.exp(-taxis_decay / tau_decay_mod)
dopamine = fftconvolve(gauss_peak, decay)[:600]
dopamine /= np.max(dopamine)
taxis_mod = np.linspace(0, 6, 600)
taxis_mod = taxis_mod - .9

fig, ax = plt.subplots(figsize=(2.2, 1.6))
# ax.plot(taxis_HPC, HPC_mod, c='lightcoral', alpha=.6)
ax.plot(taxis_HPC, HPC, c='firebrick', alpha=.8)
# ax.plot(taxis_HPC, 1-HPC, c='purple', alpha=.8)
# ax.plot(taxis_LC, gauss_peak, c='royalblue')
ax.plot(taxis_mod, dopamine, c='darkgreen', ls='--')

# scaled LC peaks
ax.plot(taxis_LC, gauss_peak * 1, c='royalblue', lw=2, alpha=1)
# ax.plot(taxis_LC, gauss_peak * .7, c='royalblue', alpha=0.5)

# LC stim 
# ax.plot(taxis_LC, gauss_peak * .75, c='grey', alpha=.6)
# ax.plot(taxis_LC, gauss_peak * 1, c='royalblue', alpha=1)

# remove left, right and top spines
for s in ['left', 'right', 'top', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(-1, 5), xticks=[],
       ylim=(0, 1.1), yticks=[])

plt.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\paper\figures_other\trace_LC_stim_HPC_diversity_response.pdf',
            dpi=300,
            bbox_inches='tight')


#%% simplified/generic 
# HPC curve 
t = np.linspace(-1, 5, 500)
hpc_generic = 1 / (1 + np.exp(-8 * (t - 0)))  # sigmoid at t = 0

# LC curves 
# 1
gauss_std = 0.25
gauss_peak = np.exp(-0.5 * (t / gauss_std)**2)
gauss_peak /= np.max(gauss_peak)
lc_tonic = np.copy(gauss_peak)
peak_idx = np.argmax(lc_tonic)
lc_tonic[peak_idx:] = lc_tonic[peak_idx]  # flatten from peak onward

# 2
taxis_LC = np.linspace(-1, 1, 200)
gauss_std = 0.25 # standard deviation in seconds (sharp peak)
gauss_peak = np.exp(-0.5 * (taxis_LC / gauss_std)**2)
gauss_peak /= np.max(gauss_peak)  # normalise to peak at 1

# plot
fig, ax = plt.subplots(figsize=(2.2, 1.6))
# ax.plot(t, hpc_generic, c='firebrick', lw=3, ls='dashed')
# ax.plot(t, 1-hpc_generic, c='purple', lw=2, ls='dashed')
# ax.plot(t, lc_tonic, c='royalblue', lw=2, ls='dashed')  # LC tonic trace
ax.plot(taxis_LC, gauss_peak, c='royalblue', lw=2) 

# remove spines and ticks
for s in ['left', 'right', 'top', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(-1, 5), xticks=[], ylim=(0, 1.1), yticks=[])

plt.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\paper\figures_other\trace_LC_phasic_solid.png',
            dpi=300,
            bbox_inches='tight')