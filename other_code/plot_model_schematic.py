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


#%% plotting 
# HPC curve 
taxis_HPC = np.linspace(0, 5, 500)
tau_rise = 0.35
tau_decay = 5
HPC = np.exp(-taxis_HPC / tau_decay) - np.exp(-taxis_HPC / tau_rise)
HPC /= np.max(HPC)

HPC_mod = np.exp(-taxis_HPC / 3) - np.exp(-taxis_HPC / tau_rise)
HPC_mod /= np.max(HPC_mod)/.25

# LC curve
taxis_LC = np.linspace(-1, 1, 200)
gauss_std = 0.25 # standard deviation in seconds (sharp peak)
gauss_peak = np.exp(-0.5 * (taxis_LC / gauss_std)**2)
gauss_peak /= np.max(gauss_peak)  # normalise to peak at 1

# dopamine trace: starts at time -1, similar rise to LC Gaussian, slower decay
taxis_decay = np.linspace(0, 6, 600)
tau_decay_mod = 3.4
decay = np.exp(-taxis_decay / tau_decay_mod)
dopamine = fftconvolve(gauss_peak, decay)[:600]
dopamine /= np.max(dopamine)
taxis_mod = np.linspace(0, 6, 600)
taxis_mod = taxis_mod - 1

fig, ax = plt.subplots(figsize=(2.2, 1.6))
ax.plot(taxis_HPC, HPC_mod, c='lightcoral', alpha=.6)
ax.plot(taxis_HPC, HPC, c='firebrick', alpha=1)
# ax.plot(taxis_LC, gauss_peak, c='royalblue')
# ax.plot(taxis_mod, dopamine, c='darkgreen', ls='--')

# scaled LC peaks
# ax.plot(taxis_LC, gauss_peak * .9, c='royalblue', alpha=.8)
# ax.plot(taxis_LC, gauss_peak * 1, c='royalblue', alpha=1)
# ax.plot(taxis_LC, gauss_peak * .7, c='royalblue', alpha=0.5)
# ax.plot(taxis_LC, gauss_peak * .5, c='royalblue', alpha=0.3)

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

fig.savefig(r'Z:\Dinghao\paper\figures_other\trace_HPC_reduced.png',
            dpi=300,
            bbox_inches='tight')