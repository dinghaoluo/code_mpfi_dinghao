# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:25:56 2023

calculate trough-to-AHP and asymmetry

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
import sys 
import pandas as pd
import scipy.io as sio
from scipy.stats import mannwhitneyu, ttest_ind

if ('Z:\Dinghao\code_dinghao\LC_all' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_all')


#%% load data 
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')
tagged_keys = []
putative_keys = []
for cell in cell_prop.index:
    tg = cell_prop['tagged'][cell]  # if tagged 
    pt = cell_prop['putative'][cell]  # if putative 
    
    if tg:
        tagged_keys.append(cell)
    if pt:
        putative_keys.append(cell)


#%% MAIN
wfs = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_waveforms.npy', 
              allow_pickle=True).item()

asym = {}  # waveform asymmetry (B-A)/(B+A)
asym_tagged = {}
asym_putative = {}
sw = {}  # trough to AHP
sw_tagged = {}
sw_putative = {}
sr = {}  # spike rate
sr_tagged = {}
sr_putative = {}

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
    
    if clu_name in tagged_keys:
        # calculate trough-to-AHP
        sw_tagged[clu_name] = (asc_max+wf_min-des_max)*50  # in μsec    
        # calculate asymmetry
        asym_tagged[clu_name] = (b-a)/(b+a)
        # retrieve spike rate 
        sr_tagged[clu_name] = spike_rate[clu_num-2]
    elif clu_name in putative_keys:
        sw_putative[clu_name] = (asc_max+wf_min-des_max)*50
        asym_putative[clu_name] = (b-a)/(b+a)
        sr_putative[clu_name] = spike_rate[clu_num-2]
    else:
        sw[clu_name] = (asc_max+wf_min-des_max)*50
        asym[clu_name] = (b-a)/(b+a)
        sr[clu_name] = spike_rate[clu_num-2]
    
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_sw.npy', 
        sw_tagged)
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_asym.npy', 
        asym_tagged)
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_sr.npy', 
        sr_tagged)

np.save('Z:\Dinghao\code_dinghao\LC_all\LC_putative_sw.npy', 
        sw_putative)
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_putative_asym.npy', 
        asym_putative)
np.save('Z:\Dinghao\code_dinghao\LC_all\LC_putative_sr.npy', 
        sr_putative)


#%% plotting with narrow v broad 
# div = np.median(list(sw.values()))  # separate at 600 μs sw
# narr = []; narr_asym = []
# brd = []; brd_asym = []

# for clu in list(sw.items()):
#     if clu[1] < div:
#         narr.append(clu[1])
#         narr_asym.append(asym[clu[0]].item())
#     else:
#         brd.append(clu[1])
#         brd_asym.append(asym[clu[0]].item())

# fig, ax = plt.subplots(figsize=(4,3))

# broad = ax.scatter(brd, brd_asym, s=4)
# narrow = ax.scatter(narr, narr_asym, s=4)
# ax.set(ylabel='waveform asymmetry', xlabel='trough-to-AHP (ms)')
# ax.legend([broad, narrow], ['broad', 'narrow'])
# for spine in ['top', 'right']:
#     ax.spines[spine].set_visible(False)

# plt.show()

# out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
# fig.savefig(out_directory+'\\'+'LC_waveform_asym_sw.png',
#             dpi=300,
#             bbox_inches='tight')


#%% plotting
fig, ax = plt.subplots(figsize=(4, 2.8))

jitter_tgd = np.random.uniform(-20, 20, len(sw_tagged))
tgd = ax.scatter(list(sw_tagged.values())+jitter_tgd,
                 list(sr_tagged.values()), c='royalblue', ec='none',
                 s=4, lw=.5)
jitter_pt = np.random.uniform(-20, 20, len(sw_putative))
pt = ax.scatter(list(sw_putative.values())+jitter_pt,
                list(sr_putative.values()), c='orange', ec='none', 
                s=4, lw=.5)
jitter_ntgd = np.random.uniform(-20, 20, len(sw))
ntgd = ax.scatter(list(sw.values())+jitter_ntgd,
                  list(sr.values()),
                  s=4, lw=.5, c='grey', ec='none', alpha=.5)
ax.set(ylabel='spike rate (Hz)', xlabel='spike-width (μs)')
ax.legend([tgd, pt, ntgd], ['tagged', 'putative\nDbh+', 'putative\nDbh-'], 
          frameon=False, fontsize=7)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

out_directory = r'Z:\Dinghao\code_dinghao\LC_all'
fig.savefig(out_directory+'\\'+'LC_sw_sr_comp.png',
            dpi=500,
            bbox_inches='tight')


#%% box plot for sw
fig, ax = plt.subplots(figsize=(2.8, 3))
ax.set(ylabel='spike-width (μs)',
       xlim=(0,3.2))
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticklabels(['tagged', 'putative\nDbh+', 'putative\nDbh-'])

bp = ax.boxplot([list(sw_tagged.values()), list(sw_putative.values()), list(sw.values())],
                positions=[.5, 1.5, 2.5],
                patch_artist=True,
                notch='True')

jitter_tagged_swbox_x = np.random.uniform(-.1, .1, len(sw_tagged))
jitter_tagged_swbox_y = np.random.uniform(-20, 20, len(sw_tagged))
ax.scatter([.9]*len(sw_tagged)+jitter_tagged_swbox_x, 
           list(sw_tagged.values())+jitter_tagged_swbox_y, 
           s=4, c='royalblue', ec='none', lw=.5)

jitter_putative_swbox_x = np.random.uniform(-.1, .1, len(sw_putative))
jitter_putative_swbox_y = np.random.uniform(-20, 20, len(sw_putative))
ax.scatter([1.9]*len(sw_putative)+jitter_putative_swbox_x, 
           list(sw_putative.values())+jitter_putative_swbox_y, 
           s=4, c='orange', ec='none', lw=.5)

jitter_swbox_y = np.random.uniform(-.1, .1, len(sw))
jitter_swbox_x = np.random.uniform(-20, 20, len(sw))
ax.scatter([2.9]*len(sw)+jitter_swbox_y, 
           list(sw.values())+jitter_swbox_x, 
           s=4, fc='grey', ec='none', lw=.5, alpha=.5)

colors = ['royalblue', 'orange', 'grey']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

bp['fliers'][0].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][1].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][2].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)

for median in bp['medians']:
    median.set(color='darkred',
                linewidth=1)
    
p_tagged_pt_sw = ttest_ind(list(sw_tagged.values()), list(sw_putative.values()), 
                              alternative='two-sided')[1]
p_pt_nontagged_sw = ttest_ind(list(sw_putative.values()), list(sw.values()), 
                                 alternative='two-sided')[1]
p_tagged_nontagged_sw = ttest_ind(list(sw_tagged.values()), list(sw.values()), 
                                     alternative='two-sided')[1]
fig.suptitle('MWu p={}, {}, {}'.format(round(p_tagged_pt_sw,6),
                                       round(p_pt_nontagged_sw,6),
                                       round(p_tagged_nontagged_sw,6)))

fig.tight_layout()
plt.show()

fig.savefig(out_directory+'\\'+'sw_tagged_putative_nontagged_box.png',
            dpi=500,
            bbox_inches='tight')
    
plt.close(fig)

    
#%% box plot for rate
fig, ax = plt.subplots(figsize=(2.8, 3))
ax.set(ylabel='spike rate (Hz)',
       xlim=(0,3.2))
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticklabels(['tagged', 'putative\nDbh+', 'putative\nDbh-'])

bp = ax.boxplot([list(sr_tagged.values()), list(sr_putative.values()), list(sr.values())],
                positions=[.5, 1.5, 2.5],
                patch_artist=True,
                notch='True')

jitter_tagged_srbox_y = np.random.uniform(-.1, .1, len(sr_tagged))
jitter_tagged_srbox_x = np.random.uniform(-.1, .1, len(sr_tagged))
ax.scatter([.9]*len(sr_tagged)+jitter_tagged_srbox_y, 
           list(sr_tagged.values())+jitter_tagged_srbox_x, 
           s=4, c='royalblue', ec='none', lw=.5)

jitter_putative_srbox_y = np.random.uniform(-.1, .1, len(sr_putative))
jitter_putative_srbox_x = np.random.uniform(-.1, .1, len(sr_putative))
ax.scatter([1.9]*len(sr_putative)+jitter_putative_srbox_y, 
           list(sr_putative.values())+jitter_putative_srbox_x, 
           s=4, c='orange', ec='none', lw=.5)

jitter_srbox_y = np.random.uniform(-.1, .1, len(sr))
jitter_srbox_x = np.random.uniform(-.1, .1, len(sr))
ax.scatter([2.9]*len(sr.values())+jitter_srbox_y, 
           list(sr.values())+jitter_srbox_x, 
           s=4, c='grey', ec='none', lw=.5, alpha=.5)

colors = ['royalblue', 'orange', 'grey']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

bp['fliers'][0].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][1].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][2].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)

for median in bp['medians']:
    median.set(color='darkred',
                linewidth=1)

p_tagged_pt_sr = ttest_ind(list(sr_tagged.values()), list(sr_putative.values()), 
                            alternative='two-sided')[1]
p_pt_nontagged_sr = ttest_ind(list(sr_putative.values()), list(sr.values()), 
                              alternative='two-sided')[1]
p_tagged_nontagged_sr = ttest_ind(list(sr_tagged.values()), list(sr.values()), 
                                  alternative='two-sided')[1]
fig.suptitle('MWu p={}, {}, {}'.format(round(p_tagged_pt_sr,6),
                                       round(p_pt_nontagged_sr,6),
                                       round(p_tagged_nontagged_sr,6)))

fig.tight_layout()
plt.show()

fig.savefig(out_directory+'\\'+'spike_rate_tagged_putative_nontagged_box.png',
            dpi=300,
            bbox_inches='tight')

plt.close(fig)