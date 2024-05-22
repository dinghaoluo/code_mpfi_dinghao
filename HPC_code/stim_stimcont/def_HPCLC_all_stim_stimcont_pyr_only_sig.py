# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:00:55 2024

Quantify significantly responding cells for HPCLC activation

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from ast import literal_eval


#%% load data 
df = pd.read_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.csv')


#%% initialisation
tot_excited = []
tot_inhibited = []
tot_clu = []


#%% iterate over all 
recname = df['recname'][0]  # initialise recname as 1st recording 

excited = 0
inhibited = 0
clu = 0

for cluname, row in df.iterrows():
    
    # if new recording session, first save counters to tot_ lists, and then reset all counters
    if row['recname'] != recname:
        # save data to lists
        tot_excited.append(excited)
        tot_inhibited.append(inhibited)
        tot_clu.append(clu)
        
        # reset counters 
        recname = row['recname']
        excited = 0
        inhibited = 0
        clu = 0
    
    n_bins = len(literal_eval(row['sig_bins']))
    
    if n_bins > 300:
        if row['excited']:
            excited+=1
        else:
            inhibited+=1
            
    clu+=1
    

#%% proportions
prop_excited = np.array([i/j for i, j in zip(tot_excited, tot_clu)])
prop_inhibited = np.array([i/j for i, j in zip(tot_inhibited, tot_clu)])
    

#%% plotting 
fig, ax = plt.subplots(figsize=(1.5,3))

mean_excited = np.mean(prop_excited)
mean_inhibited = np.mean(prop_inhibited)

# make points of 0 slightly above 0 for visualisation (does not affect the bars or statistics)
prop_excited[np.where(prop_excited==0.0)[0]] = 0.01
prop_inhibited[np.where(prop_inhibited==0.0)[0]] = 0.01

# jitters for visualisation 
jitter_exc = np.random.uniform(-.1, .1, len(prop_excited))
jitter_inh = np.random.uniform(-.1, .1, len(prop_inhibited))

ax.bar(1, mean_excited, 0.5, color='white', edgecolor='orange')
ax.bar(2, mean_inhibited, 0.5, color='white', edgecolor='forestgreen')
ax.scatter([1]*len(prop_excited)+jitter_exc, prop_excited, s=8, c='none', ec='grey')
ax.scatter([2]*len(prop_inhibited)+jitter_inh, prop_inhibited, s=8, c='none', ec='grey')

ax.set(ylim=(0,.6), xlim=(.5, 2.5),
       xticks=[1,2], xticklabels=['excited','inhibited'])

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
plt.show()

# fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_term_stim_all_responsive_divided_bar.png',
#             dpi=500,
#             bbox_inches='tight')