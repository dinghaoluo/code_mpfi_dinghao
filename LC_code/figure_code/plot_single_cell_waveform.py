# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:22:01 2024

plot single cell waveform

@author: Dinghao Luo
"""


#%% imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd 

sys.path.extend([r'Z:\Dinghao\code_dinghao', r'Z:\Dinghao\code_dinghao\common'])
from common import normalise
import rec_list
pathLC = rec_list.pathLC

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% cell properties
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')
tag_list = []; put_list = []
for clu in cell_prop.index:
    tg = cell_prop['tagged'][clu]
    pt = cell_prop['putative'][clu]
    
    if tg:
        tag_list.append(clu)
    if pt:
        put_list.append(clu)


#%% MAIN
# dictionary containing all tagged cells:
#   keys include pathnames to recordings, a space, and cell number
#   values are 4x32 numpy arrays:
#       [0,:]: avg waveform
#       [1,:]: avg waveform sem
all_cells = {}

# addresses of all prepped files (avg and tagged waveforms and sem's)
if ('Z:\Dinghao\code_dinghao\LC_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_by_sess')
rootfolder = 'Z:\Dinghao\code_dinghao\LC_by_sess'

if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')
tagrootfolder = 'Z:\Dinghao\code_dinghao\LC_tagged_by_sess'

for pathname in pathLC:
    sessname = pathname[-17:]
    dict_stem = rootfolder+'\\'+sessname
    tagdict_stem = tagrootfolder+'\\'+sessname
    
    spk = np.load(dict_stem+'_avg_spk.npy', allow_pickle=True).item()
    sem = np.load(dict_stem+'_avg_sem.npy', allow_pickle=True).item()
    
    tag_spk = np.load(tagdict_stem+'_tagged_spk.npy', allow_pickle=True).item()
    
    all_clus = list(spk.keys())  # id's of all clu(s)
    
    avg_spk = {}  # stores cells' avg waveforms
    avg_sem = {}
    
    for clu in all_clus:
        avg_spk = spk[clu]
        avg_sem = sem[clu]
        
        packed = np.row_stack((avg_spk, 
                               avg_sem))
        
        clu_id_full = sessname+' clu'+clu
        all_cells[clu_id_full] = packed
    
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_all_waveforms.npy', 
        all_cells)


#%% plotting
tot_clu = len(all_cells)
time_ax = np.arange(32)

all_cells_list = list(all_cells.items())

for clu in all_cells_list:
    cluname = clu[0]
    
    spk = clu[1][0]
    spk_norm = normalise(spk)
    scaling_factor = spk_norm[0]/spk[0]
    sem = clu[1][1] * scaling_factor
    
    suffix = ''
    if cluname in tag_list: suffix=' tgd'
    if cluname in put_list: suffix=' put'
    
    fig, ax = plt.subplots(figsize=(1,1))
    ax.set(title=cluname+suffix)
    ax.plot(time_ax, spk_norm, color='k')
    ax.fill_between(time_ax, spk_norm+sem, spk_norm-sem, color='k', alpha=.25)
    ax.axis('off')

    plt.show()
    
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_waveform\{}.png'.format(cluname+suffix),
                dpi=200, bbox_inches='tight')
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\single_cell_waveform\{}.pdf'.format(cluname+suffix),
                bbox_inches='tight')
    
    plt.close(fig)