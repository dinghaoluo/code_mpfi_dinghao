# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:47:21 2023

Plot bar graph showing proportion of RO-peaking cells

@author: Dinghao Luo
"""


#%% imports 
import matplotlib.pyplot as plt 
import pandas as pd


#%% load dataframe
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
tag_list = []
for cell in cell_prop.index:
    tg = cell_prop['tagged'][cell]  # tagged
    
    if tg:
        tag_list.append(cell)
    
tot_tagged = len(tag_list)
peak_counter = 0
for key in tag_list:
    if cell_prop['union_peakness'][key]==True:
        peak_counter+=1

freq_peak = peak_counter/tot_tagged 
freq_others = 1-peak_counter/tot_tagged


#%% plot
print('plotting barplot for freq of occ...')

fig, ax = plt.subplots(figsize=(3, 6))
ax.set(title='tagged Dbh+ cells',
       ylabel='proportion of cells',
       ylim=(0, .7))
ax.spines[['right', 'top']].set_visible(False)

ax.bar(['RO-peaking', 'others'],
       [freq_peak, freq_others],
       color=['white']*3,
       edgecolor=['darkorange', 'grey'],
       linewidth=2)

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagged_ROpeaking_freqocc.png',
            dpi=300,
            bbox_inches='tight')