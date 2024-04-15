# -*- coding: utf-8 -*-
"""
Created on Mon 10 July 14:57:34 2023

SUMMARISE ALL CELL WAVEFORMS FROM HPC RECORDINGS

@author: Dinghao Luo
"""


#%% imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plc
import sys
import scipy.io as sio

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% MAIN
all_cells = {}

for pathname in pathHPC:
    sessname = pathname[-17:]
    
    # load NeuronQuality file
    nq_file = sio.loadmat(pathname+'\\'+pathname[-17:]+'.NeuronQuality.mat')
    
    # assume 6 shanks and load spikes into dict
    for shank in range(1, 7):
        avspk = nq_file['nqShank{}'.format(shank)]['AvSpk'][0][0]
        tot_clu = avspk.shape[0]
        
        for clu in range(tot_clu):
            clu_id = clu+2
            all_cells['{} shank{} clu{}'.format(sessname, shank, clu_id)] = avspk[clu, :]
    
np.save('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_waveforms.npy', 
        all_cells)


#%% plotting
tot_plots = len(all_cells)
col_plots = 8

row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
    
plc.rcParams['figure.figsize'] = (8*2, row_plots*2.5)

plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1)

time_ax = np.arange(320)/10

avg_spk_dict = {}

all_cells_list = list(all_cells.items())

for i in range(len(all_cells)):
    clu_name = all_cells_list[i][0]
    
    spk = all_cells_list[i][1]
    spk_norm = normalise(spk)
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
    ax.set_title(clu_name, fontsize = 8, color='grey')
    ax.plot(time_ax, spk_norm)
    ax.axis('off')

plt.subplots_adjust(hspace = 0.4)
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_waveforms_avg.png')