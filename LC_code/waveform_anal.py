# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:25:56 2023

calculate trough-to-AHP and asymmetry

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as plc 
import os 
import sys 
import scipy.io as sio
from scipy.stats import mannwhitneyu

if ('Z:\Dinghao\code_dinghao\LC_all' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_all')


#%% MAIN
wfs = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_waveforms.npy', 
              allow_pickle=True).item()

asym = {}  # waveform asymmetry (B-A)/(B+A)
asym_tagged = {}
ttahp = {}  # trough to AHP
ttahp_tagged = {}
sr = {}  # spike rate
sr_tagged = {}

for clu in list(wfs.items()):
    clu_name = clu[0]
    sessname = clu_name[:17]
    datename = clu_name[:14]
    animalname = clu_name[1:5]
    ind_clu = clu_name.index('clu')
    clu_num = int(clu_name[ind_clu+3:ind_clu+5])
    fullname = 'Z:\Dinghao\MiceExp\ANMD'+animalname+'\\'+datename+'\\'+sessname
    
    spInfo = sio.loadmat(fullname+'\\'+sessname+'_DataStructure_mazeSection1_TrialType1_SpInfo_Run0')
    spike_rate = spInfo['spatialInfoSess'][0][0]['meanFR'][0][0][0]
    
    curr_wf = clu[1][0]
    wf_min = np.argmin(curr_wf)  # find index of minimum point (trough)
    des = curr_wf[:wf_min]; asc = curr_wf[wf_min:]
    des_max = np.argmax(des); asc_max = np.argmax(asc)  # A and B
    a = des[des_max]; b = asc[asc_max]
    
    if 'tagged' in clu_name:
        # calculate trough-to-AHP
        ttahp_tagged[clu_name] = asc_max*50  # in usec    
        # calculate asymmetry
        asym_tagged[clu_name] = (b-a)/(b+a)
        # retrieve spike rate 
        sr_tagged[clu_name] = spike_rate[clu_num-2]
    else:
        # calculate trough-to-AHP
        ttahp[clu_name] = asc_max*50  # in usec    
        # calculate asymmetry
        asym[clu_name] = (b-a)/(b+a)
        # retrieve spike rate 
        sr[clu_name] = spike_rate[clu_num-2]
    
# np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\\'+'LC_all_ttahp.npy', 
        # ttahp)

#%% plotting with narrow v broad 
# div = 600  # separate at 600 μs ttahp
# narr = []; narr_asym = []
# brd = []; brd_asym = []

# for clu in list(ttahp.items()):
#     if clu[1] < 600:
#         narr.append(clu[1])
#         narr_asym.append(asym[clu[0]].item())
#     else:
#         brd.append(clu[1])
#         brd_asym.append(asym[clu[0]].item())

# fig, ax = plt.subplots()

# plt.figure(figsize=(14, 24))

# broad = ax.scatter(brd, brd_asym, s=4)
# narrow = ax.scatter(narr, narr_asym, s=4)
# ax.set(ylabel='waveform asymmetry', xlabel='trough-to-AHP (μs)')
# ax.legend([broad, narrow], ['broad', 'narrow'])
# for spine in ['top', 'right']:
#    ax.spines[spine].set_visible(False)

# plt.show()

# out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
# fig.savefig(out_directory+'\\'+'LC_waveform_asym_ttahp.png',
#             dpi=300,
#             bbox_inches='tight')


#%% plotting
fig, ax = plt.subplots(figsize=(5, 3))

jitter_tgd = np.random.randint(-20, 20, len(ttahp_tagged))
tgd = ax.scatter(list(ttahp_tagged.values())+jitter_tgd,
                 list(sr_tagged.values()), s=1.5, color='coral',
                 marker='v')
jitter_ntgd = np.random.randint(-20, 20, len(ttahp))
ntgd = ax.scatter(list(ttahp.values())+jitter_ntgd, 
                  list(sr.values()), s=1.5, color='darkcyan',
                  marker='o')
ax.set(ylabel='spike rate (Hz)', xlabel='trough-to-AHP (μs)')
ax.legend([tgd, ntgd], ['tagged', 'non-tagged'])
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all'
fig.savefig(out_directory+'\\'+'LC_ttahp_sr_comp.png',
            dpi=300,
            bbox_inches='tight')


#%% box plot 
fig, ax = plt.subplots(figsize=(2, 3))
plt.tick_params(labelbottom=False, bottom=False)
ax.set(ylabel='spike rate (Hz)')
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticks([], minor=False)

bp = ax.boxplot([list(sr_tagged.values()), list(sr.values())],
                positions=[.5, 1],
                patch_artist=True,
                notch='True')

colors = ['coral', 'darkcyan']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

bp['fliers'][0].set(marker ='v',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][1].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)

for median in bp['medians']:
    median.set(color='darkred',
               linewidth=1)
    

#%% Mann-Whitney U-test
res = mannwhitneyu(list(sr_tagged.values()), list(sr.values()), 
                   alternative='two-sided')
ax.set(title=res[1])

fig.savefig(out_directory+'\\'+'box.png',
            dpi=300,
            bbox_inches='tight')