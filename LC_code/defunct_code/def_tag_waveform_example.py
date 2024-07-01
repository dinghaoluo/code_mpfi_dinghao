# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:25:09 2023

SUMMARISE ALL TAGGED CELL WAVEFORMS

@author: Dinghao Luo
"""


#%% imports
import numpy as np
import matplotlib.pyplot as plt
import sys

# if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
#     sys.path.append('Z:\Dinghao\code_dinghao\common')
# from common import normalise


#%% global
conv_microvolt = (1/(2**16)) * 20 * (1/1000) * 1000 * 1000


#%% MAIN
all_tagged = {}

# addresses of all prepped files (avg and tagged waveforms and sem's)
if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')
rootfolder = 'Z:\Dinghao\code_dinghao\LC_tagged_by_sess'

pathname = 'Z:\Dinghao\MiceExp\ANMD056r\A056r-20230417\A056r-20230417-04'
sessname = pathname[-17:]
dict_stem = rootfolder+'\\'+sessname
    
spk_dict = np.load(dict_stem+'_avg_spk.npy', allow_pickle=True).item()
tag_spk_dict = np.load(dict_stem+'_tagged_spk.npy', allow_pickle=True).item()

spk = spk_dict['2']; tag_spk = tag_spk_dict['2']


#%% spont
fig, ax = plt.subplots(figsize=(4,4))

for p in ['top','right']:
    ax.spines[p].set_visible(False)
# ax.set_yticks([])

time_ax = np.arange(32)/20

ax.plot(time_ax, spk, 'k', linewidth=5)
ax.set(ylabel='voltage (μV)', xlabel='time (ms)')

plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged\egwaveform.'
fig.savefig(out_directory+'pdf', bbox_inches='tight')
fig.savefig(out_directory+'png', dpi=500, bbox_inches='tight')


#%% tagged 
fig, ax = plt.subplots(figsize=(4,4))
for p in ['top','right']:
    ax.spines[p].set_visible(False)
# ax.set_yticks([])

ax.plot(time_ax, tag_spk, 'royalblue', linewidth=5)
ax.set(ylabel='voltage (μV)', xlabel='time (ms)')

plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged\egwaveform_tagged.'
fig.savefig(out_directory+'pdf', bbox_inches='tight')
fig.savefig(out_directory+'png', dpi=500, bbox_inches='tight')
