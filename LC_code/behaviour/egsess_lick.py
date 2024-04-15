# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:04:32 2023

plot lick curves (beh example)

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio 

# for Illustrator
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'


#%% MAIN
rec_used = 'A014-20211201'
print('recording used: {}'.format(rec_used))

recdata_path = 'Z:/Dinghao/Behav/DataAnalysis/ANMD'+rec_used[1:4]+'/'+rec_used+'/'+rec_used+'_compSess.mat'

recdata = sio.loadmat(recdata_path)
all_licks = recdata['sessDataLick']

sess_used = 4
print('session used: {}'.format(sess_used))

# access the depths of lick data (how is this so deep jesus)
sess_licks_mean = all_licks['meanRun'][0][0][0,sess_used].reshape(-1)[:220]
sess_licks_sem = all_licks['SEMRun'][0][0][0,sess_used].reshape(-1)[:220]


#%% plotting 
fig, ax = plt.subplots(figsize=(3,2))
ax.set(title='avg lick profile',
       xlabel='distance (cm)', ylabel='lick rate (Hz)',
       xlim=(30, 220), ylim=(0, 5.5))
for p in ['right', 'top']:
    ax.spines[p].set_visible(False)

xaxis = np.arange(220)
ax.plot(xaxis, sess_licks_mean, 'k')
ax.vlines(180, 0, 10, color='limegreen', alpha=.5)
ax.fill_between(xaxis,
                sess_licks_mean-sess_licks_sem,
                sess_licks_mean+sess_licks_sem,
                color='grey', alpha=0.25,
                edgecolor='none')

fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egsess_lick.png',
            dpi=300,
            bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egsess_lick.pdf',
            dpi=300,
            bbox_inches='tight')