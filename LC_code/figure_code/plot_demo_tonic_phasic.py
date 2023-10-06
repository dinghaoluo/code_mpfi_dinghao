# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:28:36 2023

create example tonic vs phasic spike histograms

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 


#%% create spike trains
tonic_low = np.random.uniform(.1, .5, 300)
tonic_high = np.random.uniform(.25, .65, 125)
tonic_low_aft = np.random.uniform(.1, .4, 200)

tonic_hist = np.concatenate((tonic_low, tonic_high, tonic_low_aft))


phasic_low = np.random.uniform(0.1, 0.5, 100)
phasic_high = np.random.uniform(0.5, 1, 10)
phasic_low_aft = np.random.uniform(0.1, 0.5, 55)
phasic_high_aft = np.random.uniform(0.6, 1, 8)
phasic_low_aft_aft = np.random.uniform(0.1, 0.5, 100)

phasic_hist = np.concatenate((phasic_low, phasic_high, phasic_low_aft, phasic_high_aft, phasic_low_aft_aft))


#%% plotting
fig, ax = plt.subplots(figsize=(5,2))
ax.set(xticks=[], yticks=[])

minute_axis = np.arange(len(tonic_hist))/60
ax.plot(minute_axis, tonic_hist, 'k', linewidth=1)

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\LC_figures\eg_tonic_hist.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)


fig, ax = plt.subplots(figsize=(5,2))

ax.plot(phasic_hist, 'k', linewidth=1)
ax.set(xticks=[], yticks=[])

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\LC_figures\eg_phasic_hist.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)