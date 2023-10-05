# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:09:16 2023

calculate the proporton of first lick sensitive and sensitive types 

@author: Dinghao Luo
"""


#%% imports 
# import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
# import sys 
import pandas as pd


#%% load dataframe  
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


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

fig, ax = plt.subplots()

ax.bar(cell_types, excitation, 0.5, bottom=bottom, color=colours)
    
ax.legend(loc='upper right', frameon=False)
    
plt.show()