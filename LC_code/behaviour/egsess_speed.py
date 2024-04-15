# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:31:35 2023

plot speed curve (beh example)

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
all_speeds = recdata['sessDataSpeed']

sess_used = 4
print('session used: {}'.format(sess_used))

# access the depths of lick data (how is this so deep jesus)
sess_speeds_mean = all_speeds['meanRun'][0][0][0,sess_used].reshape(-1)[:1800]
sess_speeds_sem = all_speeds['SEMRun'][0][0][0,sess_used].reshape(-1)[:1800]


#%% plotting 
fig, ax = plt.subplots(figsize=(3,2))
ax.set(title='avg speed profile',
       xlabel='distance (cm)', ylabel='speed (cm/s)',
       xlim=(0, 180),
       yticks=[0, 50])
for p in ['right', 'top']:
    ax.spines[p].set_visible(False)

xaxis = np.arange(1800)/10
ax.plot(xaxis, sess_speeds_mean, 'k')
ax.fill_between(xaxis,
                sess_speeds_mean-sess_speeds_sem,
                sess_speeds_mean+sess_speeds_sem,
                color='grey', alpha=.25,
                edgecolor='none')

fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egsess_speed.png',
            dpi=300,
            bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\egsess_speed.pdf',
            dpi=300,
            bbox_inches='tight')