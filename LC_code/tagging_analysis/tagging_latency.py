# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:38:49 2023

calculate and plot mean latency after stim-onset for tagged spikes

**unit is converted to ms from original 20000Hz sampling rate**

@author: Dinghao Luo
"""


#%% imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import scipy.io as sio

all_tagged = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
                     allow_pickle=True).item()

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from param_to_array import param2array, get_clu


#%% MAIN
all_tagging_latency = {}

for cluname in all_tagged.keys():
    sessname = cluname[:17]
    datename = cluname[:14]
    animalname = cluname[1:5]
    fullname = 'Z:\Dinghao\MiceExp\ANMD'+animalname+'\\'+datename+'\\'+sessname
    
    print('\nprocessing {}'.format(cluname))
        
    # load .mat
    mat_BTDT = sio.loadmat(fullname+'\\'+sessname+'BTDT.mat')
    behEvents = mat_BTDT['behEventsTdt']
    spInfo = sio.loadmat(fullname+'\\'+sessname+'_DataStructure_mazeSection1_TrialType1_SpInfo_Run0')
    spike_rate = spInfo['spatialInfoSess'][0][0]['meanFR'][0][0][0]
    
    # global vars
    n_chan = 32  # need to change if using other probes
    n_spk_samp = 32  # arbitrary, equals to 1.6ms, default window in kilosort
    rd_samp_size = 60
    
    clu = param2array(fullname+'\\'+sessname+'.clu.1')  # load .clu
    res = param2array(fullname+'\\'+sessname+'.res.1')  # load .res
    
    clu = np.delete(clu, 0)  # delete 1st element (noise clusters)
    all_clus = np.delete(np.unique(clu), [0, 1])
    all_clus = np.array([int(x) for x in all_clus])
    all_clus = all_clus[all_clus>=2]
    tot_clus = len(all_clus)
    
    fspk = open(fullname+'\\'+sessname+'.spk.1', 'rb')  # load .spk into a byte bufferedreader

    # tagged
    stim_tp = np.zeros(60)  # hard-coded for LC stim protocol
    for i in behEvents['stimPulse'][0, 0][:, 4]-1:
        temp = (behEvents['stimPulse'][0, 0][int(i), 0] 
                + (behEvents['stimPulse'][0, 0][int(i), 1])/10000000)  # pulse time with highest precision
        temp_s = round(temp/20000, 4)  # f[sampling] = 20kHz
        stim_tp[int(i)] = temp  # time points of each stim 
    
    tagged_spk_index = np.zeros(60)
    tagged_spk_time = np.zeros(60)
    tagging_latency = np.zeros(60)
    
    nth_clu = int(cluname[21:])  # current clu number
    clu_n_id = np.transpose(get_clu(nth_clu, clu))
        
    for i in range(60):  # hard-coded
        t_0 = stim_tp[i]  # stim time point
        t_1 = stim_tp[i] + 300  # stim time point +15ms (Takeuchi et al.)
        spks_in_range = filter(lambda x: (int(res[x])>=t_0) and (int(res[x])<=t_1), clu_n_id)
        try:
            tagged_spk_index[i] = next(spks_in_range)  # 1st spike in range
        except StopIteration:
            pass
    
    for i in range(60):
        if tagged_spk_index[i]!=0:
            tagged_spk_time[i] = int(res[int(tagged_spk_index[i])])
    tagging_latency = tagged_spk_time - stim_tp
    tagging_latency = [x/20 for x in tagging_latency if x>=0]
    
    all_tagging_latency[cluname] = tagging_latency
    all_tagging_latency[cluname+' avg'] = np.mean(tagging_latency)
    
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_tagging_latency.npy', 
        all_tagging_latency)


#%% plotting
n_bins = np.linspace(0, 14, 20)
x_ticks = np.linspace(0, 15, 4)
y_ticks = np.linspace(0, 10, 3)
latencies = []
for clu in list(all_tagging_latency.items()):
    if 'avg' in clu[0]:
        latencies.append(clu[1])

fig, ax = plt.subplots(figsize=(2, 2), tight_layout=True)
ax.set(xlim=(0,15),
       xticks=x_ticks, yticks=y_ticks,
       title='average latencies',
       xlabel='latency (ms)',
       ylabel='# of cell')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

n, bins, patches = ax.hist(latencies, color='royalblue',
                           edgecolor='gray', linewidth=.5, bins=n_bins)

# fracs = ((n**(1 / 5)) / n.max())
# norm = colors.Normalize(fracs.min(), fracs.max())
 
# cind = 500
# for thisfrac, thispatch in zip(fracs, patches):
#     color = plt.cm.viridis(cind)
#     thispatch.set_facecolor(color)
#     cind-=30
    
# ax2 = ax.twinx()
# y_ticks_right = np.linspace(0, 0.26, 14)
# ax2.set(yticks=y_ticks_right,
#         ylim=(0,0.26))
# dens = sns.kdeplot(latencies, bw_adjust=0.2, color='orange')
    
out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
fig.savefig(out_directory + '\\'+'LC_tagging_latencies.png',
            dpi=300,
            bbox_inches='tight')
fig.savefig(out_directory + '\\'+'LC_tagging_latencies.pdf',
            bbox_inches='tight')