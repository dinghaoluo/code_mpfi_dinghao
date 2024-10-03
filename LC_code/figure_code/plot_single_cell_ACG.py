# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:36:21 2024

plot ACGs for single cells 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load cell properties 
cell_prop = pd.read_pickle(r'Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')
tag_list = [clu for clu in cell_prop.index if cell_prop['tagged'][clu]]
put_list = [clu for clu in cell_prop.index if cell_prop['putative'][clu]]


#%% load ACGs
acgs = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_acg_baseline.npy',
               allow_pickle=True).item()


#%% main 
xaxis = np.arange(-200, 200, 1)
for cluname, acg in acgs.items():
    suffix = ''
    if cluname in tag_list: suffix=' tgd'
    if cluname in put_list: suffix=' put'
    fig, ax = plt.subplots(figsize=(1,1))
    ax.plot(xaxis, acg[9800:10200], color='k')
    for s in ['left', 'top', 'right']: ax.spines[s].set_visible(False)
    ax.set(xlabel='lag (ms)', xticks=(-200, 0, 200),
           yticks=[], ylabel='', yticklabels='',
           title=cluname+suffix)
    for ext in ['png', 'pdf']:
        fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_ACG\{}.{}'.format(cluname+suffix, ext),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)