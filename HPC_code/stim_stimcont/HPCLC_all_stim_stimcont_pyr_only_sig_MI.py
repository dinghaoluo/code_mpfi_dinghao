# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:00:55 2024

Quantify significantly responding cells for HPCLC activation

modification:
    - 17 May 2024, modified to include modulation index

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

tot_f2excited = []
tot_f3excited = []
tot_f4excited = []
tot_f5excited = []
tot_f2inhibited = []
tot_f3inhibited = []
tot_f4inhibited = []
tot_f5inhibited = []

tot_seed = []

tot_clu = []


#%% iterate over all 
recname = df['recname'][0]  # initialise recname as 1st recording 

excited = 0
inhibited = 0

# n-fold counts 
f2excited = 0
f3excited = 0
f4excited = 0
f5excited = 0
f2inhibited = 0
f3inhibited = 0
f4inhibited = 0
f5inhibited = 0

seeds = 0

# recording names 
recs = []

# modified to include MI's of both categories 
excited_MI = []
inhibited_MI = []

clu = 0

for cluname, row in df.iterrows():
    
    # if new recording session, first save counters to tot_lists, and then reset all counters
    if row['recname'] != recname:
        # save data to lists
        tot_excited.append(excited)
        tot_inhibited.append(inhibited)
        
        tot_f2excited.append(f2excited)
        tot_f3excited.append(f3excited)
        tot_f4excited.append(f4excited)
        tot_f5excited.append(f5excited)
        tot_f2inhibited.append(f2inhibited)
        tot_f3inhibited.append(f3inhibited)
        tot_f4inhibited.append(f4inhibited)
        tot_f5inhibited.append(f5inhibited)
        
        tot_clu.append(clu)
        
        recs.append(recname)
        
        # reset counters 
        recname = row['recname']
        excited = 0
        inhibited = 0
        
        f2excited = 0
        f3excited = 0
        f4excited = 0
        f5excited = 0
        f2inhibited = 0
        f3inhibited = 0
        f4inhibited = 0
        f5inhibited = 0
        
        seeds = 0
        
        clu = 0
    
    n_bins = len(literal_eval(row['sig_bins']))
    
    if n_bins > 300:
        mi = row['MI']
        if row['excited']:
            excited+=1
            excited_MI.append(mi)
            if mi>2: f2excited+=1
            if mi>3: f3excited+=1
            if mi>4: f4excited+=1
            if mi>5: f5excited+=1
        else:
            inhibited+=1
            inhibited_MI.append(mi)
            if mi<.5: f2inhibited+=1
            if mi<.33: f3inhibited+=1
            if mi<.25: f4inhibited+=1
            if mi<.2: f5inhibited+=1
        if literal_eval(row['sig_bins'])[100]<1.0:  # if gets activated before 1.0 s
            seeds+=1
            
    clu+=1
    

#%% proportions
prop_excited = np.array([i/j for i, j in zip(tot_excited, tot_clu)])
prop_inhibited = np.array([i/j for i, j in zip(tot_inhibited, tot_clu)])

prop_f2excited = np.array([i/j for i, j in zip(tot_f2excited, tot_clu)])
prop_f3excited = np.array([i/j for i, j in zip(tot_f3excited, tot_clu)])
prop_f4excited = np.array([i/j for i, j in zip(tot_f4excited, tot_clu)])
prop_f5excited = np.array([i/j for i, j in zip(tot_f5excited, tot_clu)])
prop_f2inhibited = np.array([i/j for i, j in zip(tot_f2inhibited, tot_clu)])
prop_f3inhibited = np.array([i/j for i, j in zip(tot_f3inhibited, tot_clu)])
prop_f4inhibited = np.array([i/j for i, j in zip(tot_f4inhibited, tot_clu)])
prop_f5inhibited = np.array([i/j for i, j in zip(tot_f5inhibited, tot_clu)])
    

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


#%% MI analysis 
fig, ax = plt.subplots(figsize=(1.5,5))

vp = ax.violinplot([inhibited_MI, excited_MI],
                   positions=[1, 1], showextrema=False, showmeans=True)
vp['bodies'][0].set_color('forestgreen')
vp['bodies'][0].set_edgecolor('darkgreen')
vp['bodies'][1].set_color('orange')
vp['bodies'][1].set_edgecolor('darkorange')
for i in [0, 1]:
    vp['bodies'][i].set_alpha(.2)

jit1 = np.random.uniform(-.04, .04, len(excited_MI))
jit2 = np.random.uniform(-.04, .04, len(inhibited_MI))
ax.scatter([1]*len(excited_MI)+jit1, excited_MI, s=3, c='orange', lw=.2, ec='darkorange')
ax.scatter([1]*len(inhibited_MI)+jit2, inhibited_MI, s=3, c='forestgreen', lw=.2, ec='darkgreen')

ax.plot([.5, 1.5], [1, 1], color='grey', linestyle='dashed', linewidth=1)

for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(.65, 1.35), xticks=[],
       title='modulation index\n(significant only)')


#%% n-fold quantification 
f2exc = 0; f2inh = 0
f3exc = 0; f3inh = 0
f4exc = 0; f4inh = 0
f5exc = 0; f5inh = 0

for i in excited_MI: 
    if i>2: f2exc+=1
    if i>3: f3exc+=1
    if i>4: f4exc+=1
    if i>5: f5exc+=1
for i in inhibited_MI: 
    if i<.5: f2inh+=1
    if i<.33: f3inh+=1
    if i<.25: f4inh+=1
    if i<.2: f5inh+=1
    
fig, ax = plt.subplots(figsize=(2.5, 1.5))

darkorange_list = [(1, .549, 0, .25), (1, .549, 0, .5), (1, .549, 0, .75), (1, .549, 0, 1)]
forestgreen_list = [(.133, .545, .133, .25), (.133, .545, .133, .5), (.133, .545, .133, .75), (.133, .545, .133, 1)]

ax.barh([2,3,4,5], [f2exc, f3exc, f4exc, f5exc], color=darkorange_list)
ax.barh([2,3,4,5], [-f2inh, -f3inh, -f4inh, -f5inh], color=forestgreen_list)

ax.text(f2exc+.5, 2-.1, str(f2exc))
ax.text(f3exc+.5, 3-.1, str(f3exc))
ax.text(f4exc+.5, 4-.1, str(f4exc))
ax.text(f5exc+.5, 5-.1, str(f5exc))
ax.text(-f2inh-7.5, 2-.1, str(f2inh))
ax.text(-f3inh-4.5, 3-.1, str(f3inh))
ax.text(-f4inh-4.5, 4-.1, str(f4inh))
ax.text(-f5inh-4.5, 5-.1, str(f5inh))

ax.set(xlim=(-40, 40),
       yticks=[2,3,4,5], yticklabels=['2-fold', '3-fold', '4-fold', '5-fold'])

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
    
#%% n-fold divided bar
fig, ax = plt.subplots(figsize=(5,2))

mean_f2excited = np.mean(prop_f2excited)
mean_f3excited = np.mean(prop_f3excited)
mean_f4excited = np.mean(prop_f4excited)
mean_f5excited = np.mean(prop_f5excited)
mean_f2inhibited = np.mean(prop_f2inhibited)
mean_f3inhibited = np.mean(prop_f3inhibited)
mean_f4inhibited = np.mean(prop_f4inhibited)
mean_f5inhibited = np.mean(prop_f5inhibited)

# jitters for visualisation 
jitter_f2exc = np.random.uniform(-.2, .2, len(prop_f2excited))
jitter_f3exc = np.random.uniform(-.2, .2, len(prop_f3excited))
jitter_f4exc = np.random.uniform(-.2, .2, len(prop_f4excited))
jitter_f5exc = np.random.uniform(-.2, .2, len(prop_f5excited))
jitter_f2inh = np.random.uniform(-.2, .2, len(prop_f2inhibited))
jitter_f3inh = np.random.uniform(-.2, .2, len(prop_f3inhibited))
jitter_f4inh = np.random.uniform(-.2, .2, len(prop_f4inhibited))
jitter_f5inh = np.random.uniform(-.2, .2, len(prop_f5inhibited))

ax.bar(1, mean_f2excited, 1, color=darkorange_list[0], edgecolor='darkorange')
ax.bar(2, mean_f2inhibited, 1, color=forestgreen_list[0], edgecolor='forestgreen')
ax.bar(4, mean_f3excited, 1, color=darkorange_list[1], edgecolor='darkorange')
ax.bar(5, mean_f3inhibited, 1, color=forestgreen_list[1], edgecolor='forestgreen')
ax.bar(7, mean_f4excited, 1, color=darkorange_list[2], edgecolor='darkorange')
ax.bar(8, mean_f4inhibited, 1, color=forestgreen_list[2], edgecolor='forestgreen')
ax.bar(10, mean_f5excited, 1, color=darkorange_list[3], edgecolor='darkorange')
ax.bar(11, mean_f5inhibited, 1, color=forestgreen_list[3], edgecolor='forestgreen')
ax.scatter([1]*len(prop_f2excited)+jitter_f2exc, prop_f2excited, s=8, c='none', ec='grey')
ax.scatter([2]*len(prop_f2inhibited)+jitter_f2inh, prop_f2inhibited, s=8, c='none', ec='grey')
ax.scatter([4]*len(prop_f3excited)+jitter_f3exc, prop_f3excited, s=8, c='none', ec='grey')
ax.scatter([5]*len(prop_f3inhibited)+jitter_f3inh, prop_f3inhibited, s=8, c='none', ec='grey')
ax.scatter([7]*len(prop_f4excited)+jitter_f4exc, prop_f4excited, s=8, c='none', ec='grey')
ax.scatter([8]*len(prop_f4inhibited)+jitter_f4inh, prop_f4inhibited, s=8, c='none', ec='grey')
ax.scatter([10]*len(prop_f5excited)+jitter_f5exc, prop_f5excited, s=8, c='none', ec='grey')
ax.scatter([11]*len(prop_f5inhibited)+jitter_f5inh, prop_f5inhibited, s=8, c='none', ec='grey')

ax.set(ylim=(0,.13), xlim=(0, 12),
       xticks=[1,2,4,5,7,8,10,11], 
       xticklabels=['2f\nexc.','2f\ninh.','3f\nexc.','3f\ninh.','4f\nexc.','4f\ninh.','5f\nexc.','5f\ninh.'])

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
plt.show()