# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 9:37:55 2024

Quantify significantly responding cells for both HPCLC and HPCLCterm activation

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import sem
import pandas as pd
from ast import literal_eval


#%% load data
df = pd.read_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.csv') 
df_term = pd.read_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_diff_profiles_pyr_only.csv')


#%% initialisation of global variables
tot_excited = []
tot_inhibited = []
tot_clu = []

tot_excited_term = []
tot_inhibited_term = []
tot_clu_term = []


#%% iterate over all (HPCLC)
recname = df['recname'][0]  # initialise recname as 1st recording 

excited = 0
inhibited = 0
clu = 0

regulated = []
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
            regulated.append('up')
        else:
            inhibited+=1
            regulated.append('down')
    else:
        regulated.append('none')
            
    clu+=1
    
df['regulated'] = regulated
df.to_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.csv')
    

#%% iterate over all (HPCLCterm)
recname = df_term['recname'][0]  # initialise recname as 1st recording 

excited = 0
inhibited = 0
clu = 0

regulated = []
for cluname, row in df_term.iterrows():
    
    # if new recording session, first save counters to tot_ lists, and then reset all counters
    if row['recname'] != recname:
        # save data to lists
        tot_excited_term.append(excited)
        tot_inhibited_term.append(inhibited)
        tot_clu_term.append(clu)
        
        # reset counters 
        recname = row['recname']
        excited = 0
        inhibited = 0
        clu = 0
    
    n_bins = len(literal_eval(row['sig_bins']))
    
    if n_bins > 300:
        if row['excited']:
            excited+=1
            regulated.append('up')
        else:
            inhibited+=1
            regulated.append('down')
    else:
        regulated.append('none')
            
    clu+=1
    
df_term['regulated'] = regulated
df_term.to_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_diff_profiles_pyr_only.csv')
    

#%% proportions
prop_excited = np.array([i/j for i, j in zip(tot_excited, tot_clu)])
prop_inhibited = np.array([i/j for i, j in zip(tot_inhibited, tot_clu)])

prop_excited_term = np.array([i/j for i, j in zip(tot_excited_term, tot_clu_term)])
prop_inhibited_term = np.array([i/j for i, j in zip(tot_inhibited_term, tot_clu_term)])


#%% plotting 
fig, ax = plt.subplots(figsize=(4,3))

# means 
mean_excited = np.mean(prop_excited)
mean_inhibited = np.mean(prop_inhibited)
mean_excited_term = np.mean(prop_excited_term)
mean_inhibited_term = np.mean(prop_inhibited_term)

# sems
sem_excited = sem(prop_excited)
sem_inhibited = sem(prop_inhibited)
sem_excited_term = sem(prop_excited_term)
sem_inhibited_term = sem(prop_inhibited_term)

# make points of 0 slightly above 0 for visualisation (does not affect the bars or statistics)
prop_excited[np.where(prop_excited==0.0)[0]] = 0.01
prop_inhibited[np.where(prop_inhibited==0.0)[0]] = 0.01
prop_excited_term[np.where(prop_excited_term==0.0)[0]] = 0.01
prop_inhibited_term[np.where(prop_inhibited_term==0.0)[0]] = 0.01

# jitters for visualisation 
jitter_exc = np.random.uniform(-.1, .1, len(prop_excited))
jitter_inh = np.random.uniform(-.1, .1, len(prop_inhibited))
jitter_exc_term = np.random.uniform(-.1, .1, len(prop_excited_term))
jitter_inh_term = np.random.uniform(-.1, .1, len(prop_inhibited_term))

ax.bar(1, mean_excited, 0.5, linewidth=2, yerr=sem_excited, capsize=3, color='white', edgecolor='darkorange')
ax.bar(2, mean_inhibited, 0.5, linewidth=2, yerr=sem_inhibited, capsize=3, color='white', edgecolor='darkgreen')
ax.bar(3, mean_excited_term, 0.5, linewidth=2, yerr=sem_excited_term, capsize=3, color='white', edgecolor='darkorange')
ax.bar(4, mean_inhibited_term, 0.5, linewidth=2, yerr=sem_inhibited_term, capsize=3, color='white', edgecolor='darkgreen')
ax.scatter([1]*len(prop_excited)+jitter_exc, prop_excited, s=8, c='none', ec='grey')
ax.scatter([2]*len(prop_inhibited)+jitter_inh, prop_inhibited, s=8, c='none', ec='grey')
ax.scatter([3]*len(prop_excited_term)+jitter_exc_term, prop_excited_term, s=8, c='none', ec='grey')
ax.scatter([4]*len(prop_inhibited_term)+jitter_inh_term, prop_inhibited_term, s=8, c='none', ec='grey')

# styling 
ax.set(ylim=(0,.55), xlim=(.5, 4.5),
       xticks=[1,2,3,4], xticklabels=['upmod.','downmod.','upmod. (term.)','downmod. (term.)'])
fig.suptitle('modulation by LC activation (pyr)')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
plt.xticks(rotation=45)
    
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_all_responsive_pyr_only_divided_bar.png',
                dpi=300,
                bbox_inches='tight')