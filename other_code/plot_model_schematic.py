# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:26:56 2025

playground script 

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import fftconvolve

from common import mpl_formatting
mpl_formatting()


#%% paths 
save_stem = Path('Z:/Dinghao/code_dinghao/model_schematics')


#%% LC curve
taxis_LC = np.linspace(-1, 1, 200)
gauss_std = 0.25 # standard deviation in seconds (sharp peak)
gauss_peak = np.exp(-0.5 * (taxis_LC / gauss_std)**2)
gauss_peak /= np.max(gauss_peak)  # normalise to peak at 1

fig, ax = plt.subplots(figsize=(2.2, 1.6))
ax.plot(taxis_LC, gauss_peak, c='royalblue')

for s in ['left', 'right', 'top', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(-1, 5), xticks=[],
       ylim=(0, 1.1), yticks=[])

plt.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        save_stem / f'trace_LC{ext}',
        dpi=300,
        bbox_inches='tight'
        )


#%% HPC curve 
taxis_HPC = np.linspace(-1, 6, 550)
sigmoid_onset = 1 / (1 + np.exp(-6.5 * (taxis_HPC - 0.3)))  # onset at ~0.3 s
exponential_tail = np.exp(-taxis_HPC / 2.5)  # slow decay

HPC = sigmoid_onset * exponential_tail
HPC /= np.max(HPC)

# HPC_mod: reduced amplitude version
HPC_mod = HPC * 0.25

fig, ax = plt.subplots(figsize=(2.2, 1.6))
ax.plot(taxis_HPC, HPC, c='firebrick', lw=1.5, alpha=1)

for s in ['left', 'right', 'top', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(-1, 5), xticks=[],
       ylim=(0, 1.1), yticks=[])

plt.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        save_stem / f'trace_HPC{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    

#%% DA trace 
# dopamine trace: starts at time -1, similar rise to LC Gaussian, slower decay
taxis_decay = np.linspace(0, 6, 600)
tau_decay_mod = 4.4
decay = np.exp(-taxis_decay / tau_decay_mod)
dopamine = fftconvolve(gauss_peak, decay)[:600]
dopamine /= np.max(dopamine)
taxis_mod = np.linspace(0, 6, 600)
taxis_mod = taxis_mod - .9

fig, ax = plt.subplots(figsize=(2.2, 1.6))
ax.plot(taxis_mod, dopamine, c='darkgreen')

for s in ['left', 'right', 'top', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(-1, 5), xticks=[],
       ylim=(0, 1.1), yticks=[])

plt.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        save_stem / f'trace_DA{ext}',
        dpi=300,
        bbox_inches='tight'
        )


#%% combined 
fig, ax = plt.subplots(figsize=(2.2, 1.6))
ax.plot(taxis_LC, gauss_peak, c='royalblue')
ax.plot(taxis_HPC, HPC, c='firebrick', lw=1.5, alpha=1)
ax.plot(taxis_mod, dopamine, c='darkgreen')

for s in ['left', 'right', 'top', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(-1, 5), xticks=[],
       ylim=(0, 1.1), yticks=[])

plt.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        save_stem / f'trace_combined{ext}',
        dpi=300,
        bbox_inches='tight'
        )
