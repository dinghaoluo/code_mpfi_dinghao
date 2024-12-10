# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:46:46 2024
Modified on Fri  Sept 20 15:15:12 2024 to plot lick to pump distribution 
Modified on Tue 10 Dec 2024 to accommodate all recording lists

plot the lick-to-pump profiles (both time and distance)

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


#%% load data 
exp_name = 'LC'  # HPCLC, HPCLCterm, LC, HPCGRABNE, HPCLCGCaMP
df = pd.read_pickle(r'Z:/Dinghao/code_dinghao/behaviour/all_{}_sessions.pkl'.format(
    exp_name))


#%% load behav data 
lick_times = df['lick_times']
reward_times = df['reward_times']

lick_distances = df['lick_distances']
reward_distances = df['reward_distances']

# updated to eliminate nested for loops, 3 Oct 2024 Dinghao 
lick_to_pumps_times = [
    (lick-reward_times[key][trial][0])/1250
    for key in lick_times.keys()
    for trial in range(len(lick_times[key]))
    if lick_times[key][trial] and reward_times[key][trial]  # this is to make sure that the current trial has licks and rewards
    for lick in lick_times[key][trial]]

lick_to_pumps_distances = [
    lick-reward_distances[key][trial][0]
    for key in lick_distances.keys()
    for trial in range(len(lick_distances[key]))
    if reward_distances[key][trial]  # this is to make sure that the current trial has licks and rewards
    for lick in lick_distances[key][trial]]


#%% main 
fig, ax = plt.subplots(figsize=(1.9,1.5))

# create histogram 
ax.hist(lick_to_pumps_times, bins=30, range=(-5, 1), density=True, color='orchid')

ax.set(xlim=(-5, 1), xlabel='time to reward (s)', 
       ylabel='hist. licks')
for s in ['top','right']: ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()
for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\lick2pump_profile_time\lick_to_pumps_{}{}'.format(exp_name, ext),
                dpi=300,
                bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(1.9,1.5))

# create histogram
ax.hist(lick_to_pumps_distances, bins=30, range=(-100, 20), density=True, color='orchid')
ax.vlines(0, 0, 0.035, 'darkgreen')

ax.set(xlim=(-100, 20), xlabel='dist. to reward (cm)', 
       ylabel='hist. licks')
for s in ['top','right']: ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()
for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\lick2pump_profile_distance\lick_to_pumps_{}{}'.format(exp_name, ext),
                dpi=300,
                bbox_inches='tight')
plt.close(fig)