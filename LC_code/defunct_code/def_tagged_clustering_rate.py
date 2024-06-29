# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:26:59 2023

calculate and save pre-burst, burst, and post-burst mean firing rates of all 
tagged LC cells 

@author: Dinghao Luo
"""


#%% imports
import numpy as np
import matplotlib.pyplot as plt 
import sys 

if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')
    
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
from single_unit import neu_peak_detection

    
#%% MAIN
all_tagged_train = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
                           allow_pickle=True).item()
pre_burst = {}; burst = {}; post_burst = {}; train = {}
b_a = {}  # burst divided by pre_burst
b_c = {}  # burst divided by post-burst 
clustered = {
    'burst': [],
    'burstv': [],
    'v': [],
    'other': []}

for clu in list(all_tagged_train.items()):
    cluname = clu[0]
    curr_spike_all = clu[1][0]
    curr_pre = []; curr_b = []; curr_post = []; curr = []
    
    for trial in curr_spike_all:
        curr.append(trial[:6250])
        curr_pre.append(trial[2500:3125])  # -1~-.5s, relative to ro
        curr_b.append(trial[3435:4065])  # -.25~.25s, empirical window
        curr_post.append(trial[5000:6250])  # 1~1.5s to account for inh window
        
    pre_burst[cluname] = np.mean(curr_pre, axis=0)*1250  # *1250 to show Hz
    burst[cluname] = np.mean(curr_b, axis=0)*1250
    post_burst[cluname] = np.mean(curr_post, axis=0)*1250
    train[cluname] = np.mean(curr, axis=0)*1250
    
    b_a[cluname] = np.mean(burst[cluname]) / np.mean(pre_burst[cluname])
    b_c[cluname] = np.mean(burst[cluname]) / np.mean(post_burst[cluname])


#%% plotting 
mba = np.mean(list(b_a.values()))
sba = np.std(list(b_a.values()))
mbc = np.mean(list(b_c.values()))
sbc = np.std(list(b_c.values()))

fig, ax = plt.subplots()

for clu in list(b_a.items()):
    name = clu[0]
    if name=='A045r-20221130-02 clu3':  # manually recategorised
        ax.scatter(np.log(clu[1]), np.log(b_c[name]), s=3, color='cornflowerblue')
        clustered['burstv'].append(name)
    elif neu_peak_detection(train[name])==True:
        if b_c[name]<mbc+sbc and b_c[name]>1.2:
            ax.scatter(np.log(clu[1]), np.log(b_c[name]), s=3, color='cornflowerblue')
            clustered['burstv'].append(name)
        if b_c[name]>mbc+sbc:
            ax.scatter(np.log(clu[1]), np.log(b_c[name]), s=3, color='tomato')
            clustered['burst'].append(name)
        if b_c[name]<1.2:
            ax.scatter(np.log(clu[1]), np.log(b_c[name]), s=3, color='grey')
            clustered['other'].append(name)
    else:
        ax.scatter(np.log(clu[1]), np.log(b_c[name]), s=3, color='mediumorchid')
        clustered['v'].append(name)

    # elif clu[1]>=mba+sba and b_c[name]<=mbc+sbc and b_c[name]>1:
    #     ax.scatter(clu[1], b_c[name], s=3, color='cornflowerblue')
    #     clustered['burstv'].append(name)
    # elif clu[1]>=mba+sba and b_c[name]<1:
    #     ax.scatter(clu[1], b_c[name], s=3, color='grey')
    #     clustered['other'].append(name)
    # elif clu[1]<mba+sba:
    #     ax.scatter(clu[1], b_c[name], s=3, color='mediumorchid')
    #     clustered['v'].append(name)
    # else:
    #     ax.scatter(clu[1], b_c[name], s=3, color='tomato')
    #     clustered['burst'].append(name)

ax.set(xlim=(-.75,2), ylim=(-1.5,2.5),
       xlabel='log(burst/pre)',
       ylabel='log(burst/post)')

plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_cluster.png',
            dpi=300,
            bbox_inches='tight')


#%% bar plot to show comparison in numbers 
fig, ax = plt.subplots()

ax.bar(['burst', 'burst-v', 'velocity', 'other'],
       [len(clustered['burst']), len(clustered['burstv']),
        len(clustered['v']), len(clustered['other'])],
       color=['tomato', 'cornflowerblue', 'mediumorchid', 'grey'],
       edgecolor='grey')

plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_cluster_hist.png',
            dpi=300,
            bbox_inches='tight')



#%% saving
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_clustered_bc.npy', 
        b_c)
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_clustered_ba.npy', 
        b_a)
np.save('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_clustered.npy', 
        clustered)
print('\nprocessed and saved to Z:\Dinghao\code_dinghao\LC_all_tagged\LC_clustered.npy')