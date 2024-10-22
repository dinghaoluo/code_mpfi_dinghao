# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:09:16 2023

first lick sensitive and sensitive types 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import sem
import pandas as pd
import sys
import os

sys.path.append(r'Z:/Dinghao/code_mpfi_dinghao/utils')
from common import normalise_to_all

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from logger_module import log_run

# plotting parameters
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% time 
from time import time 
start = time()


#%% load dataframe  
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')
warped_dict = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_warped.npy', allow_pickle=True).item()


#%% function
def normalise(data):  # data needs to be a 1-d vector/list
    norm_data = (data - min(data))/(max(data) - min(data))
    return norm_data


#%% obtain structure 
clu_list = list(cell_prop.index)

tag_list = []; put_list = []
tag_rop_list = []; put_rop_list = []
tag_rop_sensitive = []; put_rop_sensitive = []
tag_rop_sensitive_ON = []; put_rop_sensitive_ON = []
tag_rop_sensitive_OFF = []; put_rop_sensitive_OFF = []
tag_sensitive = []; put_sensitive = []
tag_sensitive_ON = []; put_sensitive_ON = []
tag_sensitive_OFF = []; put_sensitive_OFF = []

for clu in cell_prop.index:
    tg = cell_prop['tagged'][clu]
    pt = cell_prop['putative'][clu]
    rop = cell_prop['peakness'][clu]
    sensitive = cell_prop['lick_sensitive_shuf'][clu]
    stype = cell_prop['lick_sensitive_shuf_type'][clu]
    
    if tg:
        tag_list.append(clu)
        if rop:
            tag_rop_list.append(clu)
            if sensitive:
                tag_rop_sensitive.append(clu)
                if stype=='ON':
                    tag_rop_sensitive_ON.append(clu)
                else:
                    tag_rop_sensitive_OFF.append(clu)
        else:
            if sensitive:
                tag_sensitive.append(clu)
                if stype=='ON':
                    tag_sensitive_ON.append(clu)
                else:
                    tag_sensitive_OFF.append(clu)
    if pt:
        put_list.append(clu)
        if rop:
            put_rop_list.append(clu)
            if sensitive:
                put_rop_sensitive.append(clu)
                if stype=='ON':
                    put_rop_sensitive_ON.append(clu)
                else:
                    put_rop_sensitive_OFF.append(clu)
        else: 
            if sensitive:
                put_sensitive.append(clu)
                if stype=='ON':
                    put_sensitive_ON.append(clu)
                else:
                    put_sensitive_OFF.append(clu)
                    

#%% data wrangling
ON = []
for clu in tag_rop_sensitive_ON:
    ON.append(warped_dict[clu])
for clu in put_rop_sensitive_ON:
    ON.append(warped_dict[clu])
mean_ON = np.mean(ON, axis=0)[0]*1250
sem_ON = sem(ON, axis=0)[0]*1250

OFF = []
for clu in tag_rop_sensitive_OFF:
    OFF.append(warped_dict[clu])
for clu in put_rop_sensitive_OFF:
    OFF.append(warped_dict[clu])
mean_OFF = np.mean(OFF, axis=0)[0]*1250
sem_OFF = sem(OFF, axis=0)[0]*1250

norm_ON = normalise_to_all(mean_ON, np.hstack((mean_ON, mean_OFF)))
norm_OFF = normalise_to_all(mean_OFF, np.hstack((mean_ON, mean_OFF)))


#%% plotting 
import matplotlib.pyplot as plt

cell_types = ('tagged\nDbh+', 'putative\nDbh+\nRO-peaking', 
              'tagged\nDbh+\nRO-peaking', 'tagged\nDbh+', 
              'putative\nDbh+\nRO-peaking', 'putative\nDbh+\nRO-peaking')
colours = ['darkred', 'forestgreen', 'grey']

ON_tag_rop = len(tag_rop_sensitive_ON)/len(tag_rop_list)
ON_put_rop = len(put_rop_sensitive_ON)/len(put_rop_list)
ON_rop = (len(tag_rop_sensitive_ON)+len(put_rop_sensitive_ON))/(len(tag_rop_list)+len(put_rop_list))
OFF_tag_rop = len(tag_rop_sensitive_OFF)/len(tag_rop_list)
OFF_put_rop = len(put_rop_sensitive_OFF)/len(put_rop_list)
OFF_rop = (len(tag_rop_sensitive_OFF)+len(put_rop_sensitive_OFF))/(len(tag_rop_list)+len(put_rop_list))

ON_tag = len(tag_sensitive_ON)/len(tag_list)
ON_put = len(put_sensitive_ON)/len(put_list)
# ON_all = (len(tag_sensitive_ON)+len(put_sensitive_ON))/(len(tag_list)+len(put_list))
OFF_tag = len(tag_sensitive_OFF)/len(tag_list)
OFF_put = len(put_sensitive_OFF)/len(put_list)
# OFF_all = (len(tag_sensitive_OFF)+len(put_sensitive_OFF))/(len(tag_list)+len(put_list))

rop_pie = [ON_rop, OFF_rop, 1-ON_rop-OFF_rop]
# all_pie = [ON_all, OFF_all, 1-ON_all-OFF_all]
labelled_pie = {rop_pie[0]: 'lick-ON',
                rop_pie[1]: 'lick-OFF',
                rop_pie[2]: 'non-sensitive'}

fig, ax = plt.subplots(figsize=(2,2))

ax.pie(rop_pie, labels=[labelled_pie[tag] for tag in rop_pie], 
       colors=colours, autopct='%1.1f%%', textprops={'fontsize': 8})
    
plt.show()

for ext in ['png', 'pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\lick_sensitive_profiles_shuf_piechart.{}'.format(ext),
                dpi=200,
                bbox_inches='tight')


#%% plotting 
fig, ax = plt.subplots(figsize=(2.2,1.5))

le, = ax.plot(range(625), mean_ON[:625], 'darkred')
# ax.fill_between(range(625), mean_ON[:625]+sem_ON[:625],
#                             mean_ON[:625]-sem_ON[:625],
#                             color='darkred', edgecolor='none', alpha=.1)
ax.plot(range(630, 1870), mean_ON[630:1870], 'darkred')
# ax.fill_between(range(630, 1870), mean_ON[630:1870]+sem_ON[630:1870],
#                             mean_ON[630:1870]-sem_ON[630:1870],
#                             color='darkred', edgecolor='none', alpha=.1)
ax.plot(range(1875, 2500), mean_ON[1875:2500], 'darkred')
# ax.fill_between(range(1875, 2500), mean_ON[1875:2500]+sem_ON[1875:2500],
#                             mean_ON[1875:2500]-sem_ON[1875:2500],
#                             color='darkred', edgecolor='none', alpha=.1)

li, = ax.plot(range(625), mean_OFF[:625], 'forestgreen')
ax.plot(range(630, 1870), mean_OFF[630:1870], 'forestgreen')
ax.plot(range(1875, 2500), mean_OFF[1875:2500], 'forestgreen')

ax.axvspan(625,630, color='grey', alpha=.25)
ax.axvspan(1870,1875, color='grey', alpha=.25)

ax.set(xticks=[628, 1873], xticklabels=['run-onset', '1st-lick'],
       ylabel='spike rate (Hz)')

ax.legend([le, li], ['lick-ON', 'lick-OFF'], frameon=False)

for s in ['top', 'right']: ax.spines[s].set_visible(False)

for ext in ['png', 'pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\lick_sensitive_profiles_shuf.{}'.format(ext),
                dpi=200,
                bbox_inches='tight')
    
    
#%% logging 
runtime = '{} s'.format(time()-start)
params= {'run time': runtime}
log_run(os.path.basename(__file__), params)