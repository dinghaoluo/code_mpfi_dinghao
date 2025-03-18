# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:00:06 2023

PCA on waveforms/acgs of all neurones

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise


#%% load
waveforms = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
                    allow_pickle=True).item()

# acgs = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_acg.npy',
#                allow_pickle=True).item()
acgs = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_acg_baseline.npy',
               allow_pickle=True).item()

tag_list = list(np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
                allow_pickle=True).item().keys())


#%% waveforms
# wf_list = list(waveforms.values())
# wf_list = [wf[0,:] for wf in wf_list]
# wf_list_new = []
# for i in range(len(wf_list)):
#     imin = np.argmin(wf_list[i])
#     wf = wf_list[i][imin-10:imin+10]
#     if len(wf)==20:
#         wf_list_new.append(wf)
# wf_arr = np.array(wf_list_new)

# # data_scaler = StandardScaler()
# # data_scaler.fit(wf_arr)
# # wf_scaled = data_scaler.transform(wf_arr)  # standardise waveform data

# pca = PCA(n_components=2)
# pca.fit(wf_arr)
# wf_pca = pca.transform(wf_arr)

# wf_explained_variance = pca.explained_variance_ratio_


#%% acgs
acg_list = []
acg_keys = list(acgs.keys())
# for i in range(len(acg_keys)):
#     if acg_keys[i] in tag_list and acg_keys[i][:4]!='A056':
#         acg_list.append(acgs[acg_keys[i]])
for i in range(len(acg_keys)):
    acg_list.append(acgs[acg_keys[i]])
acg_list = [normalise(acg[9500:10500]) for acg in acg_list]
acg_arr = np.array(acg_list)

data_scaler = StandardScaler()
data_scaler.fit(acg_arr)
acg_scaled = data_scaler.transform(acg_arr)  # standardise waveform data

pca = PCA(n_components=2)
pca.fit(acg_scaled)
acg_pca = pca.transform(acg_scaled)

acg_explained_variance = pca.explained_variance_ratio_


#%% plot
fig, ax = plt.subplots(figsize=(2,3))

ax.hist(acg_pca[:,0], bins=30)
# ax.scatter(acg_pca[:,0], acg_pca[:,1], s=3)