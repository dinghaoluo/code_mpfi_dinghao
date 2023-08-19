# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:25:56 2023

calculate trough-to-AHP and asymmetry

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
    
tag_list = list(np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
                allow_pickle=True).item().keys())


#%% MAIN
waveforms = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_waveforms.npy', 
              allow_pickle=True).item()

asym_tagged = np.load('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_asym.npy',
                      allow_pickle=True).item()
                      # waveform asymmetry (B-A)/(B+A)
ttahp_tagged = np.load('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_ttahp.npy',
                       allow_pickle=True).item()
                       # trough to AHP
sr_tagged = np.load('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_sr.npy',
                    allow_pickle=True).item()  
                    # spike rate


#%% plotting with narrow v broad 
threshold = 1250  # divide at how many μs ttahp?
narrow = []; narrow_asym = []; narrow_sr = []
broad = []; broad_asym = []; broad_sr = []

for i, e in enumerate(list(ttahp_tagged.values())):
    if e < threshold:
        narrow.append(e)
        narrow_asym.append(list(asym_tagged.values())[i])
        narrow_sr.append(list(sr_tagged.values())[i])
    else:
        broad.append(e)
        broad_asym.append(list(asym_tagged.values())[i])
        broad_sr.append(list(sr_tagged.values())[i])

fig, axs = plt.subplots(2,2, figsize=(8, 8))

jitter_broad = np.random.randint(-20, 20, len(broad))
as_broad = axs[0,0].scatter(broad+jitter_broad, broad_asym,
                            s=4, color='coral',
                            marker='v')
jitter_narrow = np.random.randint(-20, 20, len(narrow))
as_narrow = axs[0,0].scatter(narrow+jitter_narrow, narrow_asym, 
                             s=4, color='darkcyan',
                             marker='o')
axs[0,0].set(ylabel='waveform asymmetry', xlabel='trough-to-AHP (μs)')
axs[0,0].legend([as_broad, as_narrow], ['broad', 'narrow'])
for spine in ['top', 'right']:
    axs[0,0].spines[spine].set_visible(False)
    
sr_broad = axs[1,0].scatter(broad+jitter_broad, broad_sr, 
                            s=4, color='coral',
                            marker='v')
sr_narrow = axs[1,0].scatter(narrow+jitter_narrow, narrow_sr, 
                             s=4, color='darkcyan',
                             marker='o')
axs[1,0].set(ylabel='spike rate (Hz)', xlabel='trough-to-AHP (μs)')
axs[1,0].legend([sr_broad, sr_narrow], ['broad', 'narrow'])
for spine in ['top', 'right']:
    axs[1,0].spines[spine].set_visible(False)


#%% box plot 
axs[0,1].set(ylabel='waveform asymmetry')
for p in ['top', 'right', 'bottom']:
    axs[0,1].spines[p].set_visible(False)
axs[0,1].set_xticks([], minor=False)

bp = axs[0,1].boxplot([broad_asym, narrow_asym],
                    positions=[.5, 1],
                    patch_artist=True,
                    notch='True')

colors = ['coral', 'darkcyan']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

bp['fliers'][0].set(marker ='v',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][1].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)

for median in bp['medians']:
    median.set(color='darkred',
                linewidth=1)
    
axs[1,1].set(ylabel='spike rate (Hz)')
for p in ['top', 'right', 'bottom']:
    axs[1,1].spines[p].set_visible(False)
axs[1,1].set_xticks([], minor=False)

bp = axs[1,1].boxplot([broad_sr, narrow_sr],
                    positions=[.5, 1],
                    patch_artist=True,
                    notch='True')

colors = ['coral', 'darkcyan']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

bp['fliers'][0].set(marker ='v',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][1].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)

for median in bp['medians']:
    median.set(color='darkred',
                linewidth=1)
    
fig.tight_layout()


#%% save 
fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\tagged_waveform_analysis.png',
            dpi=300,
            bbox_inches='tight',
            transparent=False)