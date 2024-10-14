# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:09:16 2023

first lick sensitive and sensitive types 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
# from scipy.stats import sem
import pandas as pd
import sys

sys.path.append(r'Z:/Dinghao/code_mpfi_dinghao/utils')
from common import normalise_to_all

# plotting parameters
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load dataframe  
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')
warped_dict = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_warped.npy', allow_pickle=True).item()


#%% function
def normalise_(data):  # data needs to be a 1-d vector/list
    norm_data = (data - min(data))/(max(data) - min(data))
    return norm_data


#%% obtain structure 
clu_list = list(cell_prop.index)

tag_list = []; put_list = []
tag_rop_list = []; put_rop_list = []
tag_rop_sensitive = []; put_rop_sensitive = []
tag_rop_sensitive_exc = []; put_rop_sensitive_exc = []
tag_rop_sensitive_inh = []; put_rop_sensitive_inh = []
tag_sensitive = []; put_sensitive = []
tag_sensitive_exc = []; put_sensitive_exc = []
tag_sensitive_inh = []; put_sensitive_inh = []

for clu in cell_prop.index:
    tg = cell_prop['tagged'][clu]
    pt = cell_prop['putative'][clu]
    rop = cell_prop['peakness'][clu]
    sensitive = cell_prop['lick_sensitive'][clu]
    stype = cell_prop['lick_sensitive_type'][clu]
    
    if tg:
        tag_list.append(clu)
        if rop:
            tag_rop_list.append(clu)
            if sensitive:
                tag_rop_sensitive.append(clu)
                if stype=='excitation':
                    tag_rop_sensitive_exc.append(clu)
                else:
                    tag_rop_sensitive_inh.append(clu)
        else:
            if sensitive:
                tag_sensitive.append(clu)
                if stype=='excitation':
                    tag_sensitive_exc.append(clu)
                else:
                    tag_sensitive_inh.append(clu)
    if pt:
        put_list.append(clu)
        if rop:
            put_rop_list.append(clu)
            if sensitive:
                put_rop_sensitive.append(clu)
                if stype=='excitation':
                    put_rop_sensitive_exc.append(clu)
                else:
                    put_rop_sensitive_inh.append(clu)
        else: 
            if sensitive:
                put_sensitive.append(clu)
                if stype=='excitation':
                    put_sensitive_exc.append(clu)
                else:
                    put_sensitive_inh.append(clu)
                    

#%% data wrangling
excitation = []
for clu in tag_rop_sensitive_exc:
    excitation.append(warped_dict[clu])
for clu in put_rop_sensitive_exc:
    excitation.append(warped_dict[clu])
mean_excitation = np.mean(excitation, axis=0)[0]*1250
# sem_excitation = sem(excitation, axis=0)[0]*1250

inhibition = []
for clu in tag_rop_sensitive_inh:
    inhibition.append(warped_dict[clu])
for clu in put_rop_sensitive_inh:
    inhibition.append(warped_dict[clu])
mean_inhibition = np.mean(inhibition, axis=0)[0]*1250
# sem_inhibition = sem(inhibition, axis=0)[0]*1250

norm_excitation = normalise_to_all(mean_excitation, np.hstack((mean_excitation, mean_inhibition)))
norm_inhibition = normalise_to_all(mean_inhibition, np.hstack((mean_excitation, mean_inhibition)))


#%% plotting 
cell_types = ('tagged\nDbh+', 'putative\nDbh+', 
              'tagged\nDbh+\nRO-peaking', 'putative\nDbh+\nRO-peaking')
colours = ['g', 'r', 'yellow', 'blue', 'orange']

exc_tag_rop = len(tag_rop_sensitive_exc)/len(tag_rop_list)
exc_put_rop = len(put_rop_sensitive_exc)/len(put_rop_list)
inh_tag_rop = len(tag_rop_sensitive_inh)/len(tag_rop_list)
inh_put_rop = len(put_rop_sensitive_inh)/len(put_rop_list)

exc_tag = len(tag_sensitive_exc)/len(tag_list)
exc_put = len(put_sensitive_exc)/len(put_list)
inh_tag = len(tag_sensitive_inh)/len(tag_list)
inh_put = len(put_sensitive_inh)/len(put_list)

excitation = [exc_tag, exc_put, exc_tag_rop, exc_put_rop]
inhibition = [inh_tag, inh_put, inh_tag_rop, inh_put_rop]

tagged_dist = [exc_tag_rop, inh_tag_rop, 1-exc_tag_rop-inh_tag_rop]

fig, ax = plt.subplots()

# ax.bar(cell_types, excitation, 0.5, color=colours)
    
# ax.legend(loc='upper right', frameon=False)

ax.pie(tagged_dist, colors=['g', 'r', 'grey'])
    
plt.show()


#%% plotting 
fig, ax = plt.subplots(figsize=(2.2,1.5))

le, = ax.plot(range(625), mean_excitation[:625], 'darkred')
# ax.fill_between(range(625), mean_excitation[:625]+sem_excitation[:625],
#                             mean_excitation[:625]-sem_excitation[:625],
#                             color='darkred', edgecolor='none', alpha=.1)
ax.plot(range(630, 1870), mean_excitation[630:1870], 'darkred')
# ax.fill_between(range(630, 1870), mean_excitation[630:1870]+sem_excitation[630:1870],
#                             mean_excitation[630:1870]-sem_excitation[630:1870],
#                             color='darkred', edgecolor='none', alpha=.1)
ax.plot(range(1875, 2500), mean_excitation[1875:2500], 'darkred')
# ax.fill_between(range(1875, 2500), mean_excitation[1875:2500]+sem_excitation[1875:2500],
#                             mean_excitation[1875:2500]-sem_excitation[1875:2500],
#                             color='darkred', edgecolor='none', alpha=.1)

li, = ax.plot(range(625), mean_inhibition[:625], 'forestgreen')
ax.plot(range(630, 1870), mean_inhibition[630:1870], 'forestgreen')
ax.plot(range(1875, 2500), mean_inhibition[1875:2500], 'forestgreen')

ax.axvspan(625,630, color='grey', alpha=.25)
ax.axvspan(1870,1875, color='grey', alpha=.25)

ax.set(xticks=[628, 1873], xticklabels=['run-onset', '1st-lick'],
       ylabel='spike rate (Hz)')

ax.legend([le, li], ['lick-exc.', 'lick-inh.'], frameon=False)

for s in ['top', 'right']: ax.spines[s].set_visible(False)

for ext in ['png', 'pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\lick_sensitive_profiles.{}'.format(ext),
                dpi=200,
                bbox_inches='tight')