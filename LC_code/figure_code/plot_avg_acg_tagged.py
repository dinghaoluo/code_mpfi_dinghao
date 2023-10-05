# -*- coding: utf-8 -*-
"""
Created on Tue 29 Aug 18:32:08 2023

plot averaged tagged Dbh+ cell ACGs

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
import sys

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% load data 
acgs = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_acg_baseline.npy',
               allow_pickle=True).item()

tag_list = list(np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
                allow_pickle=True).item().keys())
tot_tag = len(tag_list)


#%% main 
x = np.arange(-10, 10, 1)
sigma = 2
gaussian = [1 / (sigma*np.sqrt(2*np.pi)) * 
              np.exp(-t**2/(2*sigma**2)) for t in x]

take_length_half = 200
all_tagged_acg = np.zeros((tot_tag, take_length_half*2))

for i, clu in enumerate(tag_list):
    all_tagged_acg[i,:] = np.array(normalise(np.convolve(
        acgs[clu][10000-take_length_half:10000+take_length_half], gaussian, 
        mode='same')))
avg_tagged_acg = np.mean(all_tagged_acg, axis=0)
sem_tagged_acg = np.std(all_tagged_acg, axis=0)
    
    
#%% plot
fig, ax = plt.subplots(figsize=(4,3))
for p in ['right', 'top']:
    ax.spines[p].set_visible(False)

xaxis = np.arange(-200,200)

for i in range(tot_tag):
    ax.plot(xaxis, all_tagged_acg[i,:], c='grey', alpha=.05, linewidth=1)

ax.plot(xaxis, avg_tagged_acg, c='royalblue', linewidth=3)
# ax.fill_between(xaxis, 
#                 avg_tagged_acg+sem_tagged_acg, avg_tagged_acg-sem_tagged_acg,
#                 color='royalblue', alpha=.3)
ax.fill_between(xaxis, 
                0, avg_tagged_acg,
                color='royalblue', alpha=.3)

ax.set(xlim=(-195, 195), ylim=(0,1),
       xlabel='lag (ms)', ylabel='norm. correlation',
       xticks=[-150,-50,50,150], yticks=[0, 0.5, 1])
fig.suptitle('avg. autocorrelogram, tagged $\it{Dbh}$+')


#%% save
fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_avg_autocorrelogram.png',
            dpi=500,
            bbox_inches='tight',
            transparent=False)