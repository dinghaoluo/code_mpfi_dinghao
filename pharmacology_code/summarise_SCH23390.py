# -*- coding: utf-8 -*-
"""
Created on Wed 13 Nov 16:53:32 2024

summarise pharmacological experiments with SCH23390

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
from scipy.stats import sem, ttest_rel

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

x_speed = np.arange(1801)/10
x_lick = np.arange(222)


#%% recording list
sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathSCH = rec_list.pathSCH
sessSCH = rec_list.sessSCH


#%% main
mean_speeds_pre = [] 
mean_speeds_drug = []
mean_speeds_post = []
mean_licks_pre = []
mean_licks_drug = []
mean_licks_post = []
for i, pathname in enumerate(pathSCH):    
    sessname = pathname[-13:]
    print(sessname)
    
    sesslist = sessSCH[i]
        
    for sess in sesslist:
        recname = sessname+f'-0{sess}'
        run_dist_file = sio.loadmat(pathname+'\\'+recname+'\\'+f'{recname}_runSpeedDist_Run0_msess1.mat')
        speed_over_dist = run_dist_file['speedOverDist'][0][0][0]  # trial x distance 
        if sess==1:
            mean_speeds_pre.append(np.mean(speed_over_dist, axis=0))
        if sess in [2,3]:
            mean_speeds_drug.append(np.mean(speed_over_dist, axis=0))
        if sess==4:
            mean_speeds_post.append(np.mean(speed_over_dist, axis=0))
        
        lick_dist_file = sio.loadmat(pathname+'\\'+recname+'\\'+f'{recname}_lickDist_msess1.mat')
        lick_over_dist = lick_dist_file['lickOverDist'][0][0][0]
        if sess==1:
            mean_licks_pre.append(np.mean(lick_over_dist, axis=0))
        if sess in [2,3]:
            mean_licks_drug.append(np.mean(lick_over_dist, axis=0))
        if sess==4:
            mean_licks_post.append(np.mean(lick_over_dist, axis=0))
        

#%% speed plot  
fig, ax = plt.subplots(figsize=(2.3,1.7))

ms_pre = np.mean(mean_speeds_pre, axis=0)/10
ms_drug = np.mean(mean_speeds_drug, axis=0)/10
ms_post = np.mean(mean_speeds_post, axis=0)/10
ss_pre = sem(mean_speeds_pre, axis=0)/10
ss_drug = sem(mean_speeds_drug, axis=0)/10
ss_post = sem(mean_speeds_post, axis=0)/10

lp, = ax.plot(x_speed, ms_pre, color='grey')
ax.fill_between(x_speed, ms_pre+ss_pre,
                         ms_pre-ss_pre,
                color='grey', edgecolor='none', alpha=.15)
ld, = ax.plot(x_speed, ms_drug, color='#004D80')
ax.fill_between(x_speed, ms_drug+ss_drug,
                         ms_drug-ss_drug,
                color='#004D80', edgecolor='none', alpha=.15)

plt.legend([lp, ld], ['baseline', 'exp.'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(0,180), xlabel='distance (cm)',
       ylabel='velocity (cm/s)')

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\SCH23390\speed_profile{}'.format(ext),
                dpi=300, bbox_inches='tight')
    

#%% lick plot  
fig, ax = plt.subplots(figsize=(2.3,1.7))

ml_pre = np.mean(mean_licks_pre, axis=0)/10
ml_drug = np.mean(mean_licks_drug, axis=0)/10
ml_post = np.mean(mean_licks_post, axis=0)/10
sl_pre = sem(mean_licks_pre, axis=0)/10
sl_drug = sem(mean_licks_drug, axis=0)/10
sl_post = sem(mean_licks_post, axis=0)/10

lp, = ax.plot(x_lick, ml_pre, color='grey')
ax.fill_between(x_lick, ml_pre+sl_pre,
                         ml_pre-sl_pre,
                color='grey', edgecolor='none', alpha=.15)
ld, = ax.plot(x_lick, ml_drug, color='#004D80')
ax.fill_between(x_lick, ml_drug+sl_drug,
                         ml_drug-sl_drug,
                color='#004D80', edgecolor='none', alpha=.15)

plt.legend([lp, ld], ['baseline', 'exp.'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(30,219), xlabel='distance (cm)',
       ylabel='hist. licks', ylim=(0,0.9), yticks=(0,0.5))

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\SCH23390\lick_profile{}'.format(ext),
                dpi=300, bbox_inches='tight')
    
    
#%% bar plot for mean speed with connected scatter + p-value
fig, ax = plt.subplots(figsize=(2.3,1.7))

# means over distance
avg_speed_pre = [np.mean(trace)/10 for trace in mean_speeds_pre]
avg_speed_drug = [np.mean(mean_speeds_drug[i*2:i*2+2])/10 for i in np.arange(len(avg_speed_pre))]

# means and sems
mean_ms_pre = np.mean(avg_speed_pre)
mean_ms_drug = np.mean(avg_speed_drug)
sem_ms_pre = sem(avg_speed_pre)
sem_ms_drug = sem(avg_speed_drug)

# paired t-test
tstat, pval = ttest_rel(avg_speed_pre, avg_speed_drug)
print(f'paired t-test p = {pval:.4f}')

# plot bars
positions = [1, 2]
bar_width = 0.5
ax.bar(positions[0], mean_ms_pre, yerr=sem_ms_pre, width=bar_width,
       color='grey', alpha=0.7, capsize=3)
ax.bar(positions[1], mean_ms_drug, yerr=sem_ms_drug, width=bar_width,
       color='#004D80', alpha=0.7, capsize=3)

# plot lines connecting individual animals
for b, d in zip(avg_speed_pre, avg_speed_drug):
    ax.plot(positions, [b, d], color='black', alpha=0.4, linewidth=0.8, marker='o', markersize=3)

# add p-value text
ax.text(1.5, max(mean_ms_pre, mean_ms_drug) * 1.05,
        f'p = {pval:.3f}', ha='center', va='bottom', fontsize=8)

# tidy up axes
ax.set(xticks=positions, xticklabels=['baseline', 'SCH23390'],
       ylabel='mean speed (cm/s)')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set_xlim(.5, 2.5)

# save
for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\SCH23390\mean_speed{}'.format(ext),
                dpi=300, bbox_inches='tight')