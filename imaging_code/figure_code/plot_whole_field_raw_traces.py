# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:30:59 2025

a stud script
plot traces for Ch1 and Ch2 for whole-field signals 

@author: Dinghao Luo
"""


#%% imports
import sys 
from scipy.stats import pearsonr
import numpy as np 
import matplotlib.pyplot as plt 

plot_first = 2000  # frames 


#%% recording lists 
sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE


#%% main 
all_r = []
all_p = []

for rec_path in pathGRABNE:
    whole_field_ch1 = np.squeeze(np.load(
        f'{rec_path}_grid_extract\grid_traces_dFF_496.npy',
        allow_pickle=True))
    whole_field_ch2 = np.squeeze(np.load(
        f'{rec_path}_grid_extract\grid_traces_dFF_496_ch2.npy',
        allow_pickle=True))
    
    recname = rec_path[-17:]
    print(recname)
    
    r, p = pearsonr(whole_field_ch1, whole_field_ch2)
    all_r.append(r); all_p.append(p)
    
    # plotting 
    fig, ax = plt.subplots(figsize=(5,2.4))
    
    ax.plot(whole_field_ch1[:plot_first], alpha=.8, linewidth=1)
    ax.plot(whole_field_ch2[:plot_first], alpha=.8, linewidth=1)
    
    ax.set(title=f'R = {r}\np = {p:.10e}')
    
    fig.tight_layout()
    
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\GRABNE\single_session_whole_field_raw_traces\{}_first{}.png'
        .format(recname, plot_first),
        dpi=300,
        bbox_inches='tight')
        
    plt.close(fig)
    
    
#%% pooled stats 
