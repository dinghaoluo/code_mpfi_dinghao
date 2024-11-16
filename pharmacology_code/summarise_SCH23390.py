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
from scipy.stats import sem

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
       ylabel='hist. licks')

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\SCH23390\lick_profile{}'.format(ext),
                dpi=300, bbox_inches='tight')