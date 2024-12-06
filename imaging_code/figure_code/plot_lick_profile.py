# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:46:46 2024

plot the lick profile of GRABNE sessions

@author: Dinghao Luo
"""


#%% imports 
import pandas as pd 
import matplotlib.pyplot as plt 


#%% load behav data 
behaviour = pd.read_pickle('Z:/Dinghao/code_dinghao/behaviour/all_GRABNE_sessions.pkl')
lick_series = behaviour['lick_distances']

licks = []
for key in lick_series.keys():
    tot_trials = len(lick_series[key])
    for trial in range(tot_trials):
        licks.extend(lick_series[key][trial])


#%% main 
fig, ax = plt.subplots(figsize=(2.4,2))

ax.hist(licks, bins=100, range=(0, 220), density=True, color='orchid')

ax.set(xlim=(30,220), xlabel='distance (cm)',
       ylim=(0, 0.05), yticks=[0, 0.04], ylabel='histogram of licks')
for s in ['top','right']: ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show(fig)
fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\lick_profile_GRABNE.png',
            dpi=300,
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\lick_profile_GRABNE.pdf',
    
            bbox_inches='tight')

#%% same to time 
lick_series = behaviour['lick_times']
starts = behaviour['run_onsets']

licks = []
for key in lick_series.keys():
    tot_trials = len(lick_series[key])
    for trial in range(tot_trials):
        start = starts[key][trial]
        licks.extend([(t[0]-start)/10 for t in lick_series[key][trial]])


#%% main 
fig, ax = plt.subplots(figsize=(2.4,2))

ax.hist(licks, bins=100, range=(0,1000), density=True, color='orchid')

ax.set(xlabel='time (s)',
       ylabel='histogram of licks')
for s in ['top','right']: ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show(fig)
fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\lick_profile_GRABNE_time.png',
            dpi=300,
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\lick_profile_GRABNE_time.pdf',
            bbox_inches='tight')