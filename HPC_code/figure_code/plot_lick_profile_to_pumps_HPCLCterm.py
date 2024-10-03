# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:46:46 2024
Modified on Fri  Sept 20 15:15:12 2024 to plot lick to pump distribution 

plot the lick profile of HPC sessions

@author: Dinghao Luo
"""


#%% imports 
import pandas as pd 
import matplotlib.pyplot as plt 

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load behav data 
behaviour = pd.read_pickle('Z:/Dinghao/code_dinghao/behaviour/all_HPCLCterm_sessions.pkl')
lick_series = behaviour['lick_times']
pump_series = behaviour['pumps']

# updated to eliminate nested for loops, 3 Oct 2024 Dinghao 
lick_to_pumps = [
    (lick[0]-pump_series[key][trial])/1250
    for key in lick_series.keys()
    for trial in range(len(lick_series[key]))
    if lick_series[key][trial]  # this is to make sure that the current trial has licks
    for lick in lick_series[key][trial]]


#%% main 
fig, ax = plt.subplots(figsize=(2,1.7))

# create histogram 
ax.hist(lick_to_pumps, bins=60, range=(-5, 1), density=True, color='orchid')

ax.set(xlim=(-5, 1), xlabel='time to reward (s)', 
       yticks=[0, 0.5], ylabel='histogram of licks')
for s in ['top','right']: ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()
for ext in ['png', 'pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\lick_to_pumps_HPCLCterm.{}'.format(ext),
                dpi=300,
                bbox_inches='tight')