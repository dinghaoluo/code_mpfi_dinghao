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
import mat73
import h5py


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
fig, ax = plt.subplots(figsize=(2.2,1.8))
ax.set(title='lick profile',
       xlabel='distance (cm)', ylabel='lick rate (Hz)',
       xlim=(30, 220), ylim=(0, 5.5))
for p in ['right', 'top']:
    ax.spines[p].set_visible(False)

xaxis = np.arange(220)
ax.plot(xaxis, sess_licks_mean, 'k')
ax.fill_between(xaxis,
                sess_licks_mean-sess_licks_sem,
                sess_licks_mean+sess_licks_sem,
                color='grey', edgecolor='none', alpha=.25)

fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egsess_lick.png',
            dpi=300,
            bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egsess_lick.pdf',
            bbox_inches='tight')