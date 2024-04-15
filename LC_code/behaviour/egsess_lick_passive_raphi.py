# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:04:32 2023

plot lick curves (beh example)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial' 
import scipy.io as sio 
from scipy.stats import sem


#%% MAIN
rec_used = 'Raphi A009-20190128-01'
print('recording used: {}'.format(rec_used))

recdata_path = 'Z:\Raphael_tests\mice_expdata\ANM009\A009-20190128\A009-20190128-01\A009-20190128-01_DataStructure_mazeSection1_TrialType1_lickDist_msess1_Run1.mat'

recdata = sio.loadmat(recdata_path)
all_licks = recdata['lickOverDist'][0][0][0]

# access the depths of lick data (how is this so deep jesus)
sess_licks_mean = np.mean(all_licks, axis=0)[:220]
sess_licks_sem = sem(all_licks, axis=0)[:220]


#%% plotting 
fig, ax = plt.subplots(figsize=(3,2))
ax.set(title='avg. lick profile',
       xlabel='distance (cm)', ylabel='lick histogram',
       xlim=(30, 220), ylim=(0, 1.2))
for p in ['right', 'top']:
    ax.spines[p].set_visible(False)

xaxis = np.arange(len(sess_licks_mean))
ax.plot(xaxis, sess_licks_mean, 'k')
ax.fill_between(xaxis,
                sess_licks_mean-sess_licks_sem,
                sess_licks_mean+sess_licks_sem,
                color='grey', alpha=.25)
rew_ln, = ax.plot([180, 180], [0, 10], color='limegreen', alpha=.45)
ax.legend([rew_ln], ['reward'], loc='upper left', frameon=False, fontsize=7)

fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egsess_lick_passive.png',
            dpi=300,
            bbox_inches='tight')