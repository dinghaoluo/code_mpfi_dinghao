# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:26:49 2023

compiling cell properties into a dataframe

@author: Dinghao Luo
"""


#%% imports 
import os
import numpy as np 
import pandas as pd
import scipy.io as sio


#%% loads
# all rasters
rasters = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()
rasterkeys = list(rasters.keys())

# RO peak detection functions
from RO_peak_detection import RO_peak_detection, plot_RO_peak


#%% create dataframe if does not exist
overwrite = input('Write over previous dataframe?... (Y/N)\n')
if overwrite=='Y' or overwrite=='y' or os.path.isfile('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')==False:
    tempdict = {}
    sesslist = []
    clulist = []
    for cluname in rasterkeys:
        sesslist.append(cluname[:17])
        clulist.append(cluname)
    tempdict['session'] = sesslist
    
    df = pd.DataFrame(tempdict, index=clulist)
    
    df.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')

else:
    df = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% tagged column
tag_list = list(np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_waveforms.npy',
                allow_pickle=True).item().keys())
tagged = np.array([clu in tag_list for clu in rasterkeys])

df = df.assign(tagged=pd.Series(tagged).values)

df.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% firing rate column
spike_rate = []
for cluname in rasterkeys:
    sessname = 'Z:\Dinghao\MiceExp\ANMD{}\{}\{}\{}_DataStructure_mazeSection1_TrialType1_FR_Ctrl_Run0_mazeSess1'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    spike_rate_ctrl = sio.loadmat(sessname)['mFRStructSessCtrl']['mFR'][0][0][0]
    clunum = int(cluname.split('clu')[1])
    spike_rate.append(spike_rate_ctrl[clunum-2])
    
df = df.assign(spike_rate=pd.Series(spike_rate).values)

df.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% RO peak detection
peakness = []
for cluname in rasterkeys:
    print('peak detection: {}'.format(cluname))
    
    # find 1st stimulation index - we only want trials before stim
    sessname = 'Z:\Dinghao\MiceExp\ANMD{}\{}\{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    behPar = sio.loadmat(sessname)
    stimOn = behPar['behPar']['stimOn'][0][0][0]
    first_stim = next((i for i, j in enumerate(stimOn) if j), None)
    
    if type(first_stim)==int:
        [peak, avg_profile, sig_shuf] = RO_peak_detection(rasters[cluname], first_stim=first_stim)
    else:
        [peak, avg_profile, sig_shuf] = RO_peak_detection(rasters[cluname])
    peakness.append(peak)
    plot_RO_peak(cluname, avg_profile, sig_shuf)

df = df.assign(peakness=pd.Series(peakness).values)

df.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% cluster labels
# clusters = list(np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_clustered_hierarchical_centroid.npy',
#                         allow_pickle=True).item()['clustering result'].values())
# cluster_peakness = [c=='1' for c in clusters]

# df = df.assign(cluster_peakness=pd.Series(cluster_peakness).values)
# df = df.assign(clustering_result=pd.Series(clusters).values)

# df.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% unionise peakness
# union_peakness = []
# for ind in df.index:
#     p = df['peakness'][ind]
#     cp = df['cluster_peakness'][ind]
#     sr = df['spike_rate'][ind]
    
#     if p and cp and sr<10:
#         union_peakness.append(True)
#     else:
#         union_peakness.append(False)
        
# df = df.assign(union_peakness=pd.Series(union_peakness).values)

# df.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% min distances to nearest tagged cells after UMAP-embedding
# min_dist = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_UMAP_min_dist.npy',
#                    allow_pickle=True).item()
# min_dist_list = list(min_dist.values())
# df = df.assign(min_dist=pd.Series(min_dist_list).values)

# dist2mean = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_UMAP_dist2mean.npy',
#                     allow_pickle=True).item()
# dist2mean_list = list(dist2mean.values())
# df = df.assign(dist2mean=pd.Series(dist2mean_list).values)

# min_dist_list = np.array(min_dist_list)
# min_dist_std = np.std(min_dist_list[min_dist_list>0])  # only calculate based on non-tagged 
# # min_dist_mean = np.mean(min_dist_list[min_dist_list>0])  # same as above 

# dist2mean_list = np.array(dist2mean_list)
# dist2mean_threshold = np.percentile(dist2mean_list[dist2mean_list>0], 30)

# putative = [(d<dist2mean_threshold and d!=0) for d in dist2mean_list]
# for ind in df.index:
#     sr = df['spike_rate'][ind]
# for i, put in enumerate(putative):
#     if put and sr>10:  # not a putative if spike rate > 10 Hz
#         putative[i] = False

# df = df.assign(putative=pd.Series(putative).values)

# df.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% use kmeans result to cluster the cells into putative 
kmeans = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_UMAP_kmeans.npy',
                        allow_pickle=True)
putative = []
sr = list(df['spike_rate'])

for i, e in enumerate(kmeans):
    if e==1:
        putative.append(False)
    elif sr[i]>=10 or tagged[i]==True:
        putative.append(False)
    else:
        putative.append(True)

df = df.assign(putative=pd.Series(putative).values)

df.to_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')
