# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:53:29 2025

IBL dataset access test 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 

from one.api import ONE 
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import brainbox.behavior.wheel as wh


#%% main 
sessions = one.search()


#%% search insertions 
one.search_terms('remote', 'insertions')
pids = one.search_insertions(atlas_acronym=['CA3'], query_type='remote')

session, probe = one.pid2eid(pids[5])

one.list_datasets(session)

spikes_clusters = one.load_dataset(session, 'alf/probe00/pykilosort/spikes.clusters.npy')
spikes_times = one.load_dataset(session, 'alf/probe00/pykilosort/spikes.times.npy')

wheel = one.load_object(session, 'wheel', collection='alf')
licks_times = one.load_dataset(session, 'alf/licks.times.npy')


#%% try 
all_clusters = sorted(np.unique(spikes_clusters))

clu_dict = {all_clusters[i] : 
            spikes_times[[idx for idx, clu in enumerate(spikes_clusters) if clu==i]] 
            for i in all_clusters}
    
    
#%% plotting 
plt.scatter(clu_dict[5], [0]*len(clu_dict[5]), s=5)
plt.set(xlim=(0,1000))


#%% plotting 
pos, t = wh.interpolate_position(wheel.timestamps, wheel.position)
sec = 1000  # Number of seconds to plot
plt.figure()

# Plot the interpolated data points
mask = t < (t[0] + sec)
plt.plot(t[mask], pos[mask], '.', markeredgecolor='lightgrey', markersize=1)

# Plot the original data
mask = wheel.timestamps < (wheel.timestamps[0] + sec)
plt.plot(wheel.timestamps[mask], wheel.position[mask], 'r+', markersize=6)

# Labels etc.
plt.xlabel('time / sec')
plt.ylabel('position / rad')
plt.box(on=None)
plt.show()


#%% plotting 
plt.scatter(licks_times, [0]*len(licks_times), s=1)
