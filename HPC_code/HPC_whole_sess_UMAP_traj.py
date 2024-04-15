# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:26:17 2024

perform UMAP on the whole recording

@author: Dinghao Luo
"""


#%% imports 
# import umap.plot
import umap
import mat73
import scipy.io as sio
import sys
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from sklearn.preprocessing import StandardScaler 

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# import pre-processing functions 
if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise

# #%% load paths to recordings 
# if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
#     sys.path.append('Z:\Dinghao\code_dinghao')
# import rec_list
# pathHPC = rec_list.pathHPCLCopt


#%% loading
pathname = 'Z:\Dinghao\MiceExp\ANMD076r\A076r-20231214\A076r-20231214-01'
recname = pathname[-17:]

print(recname)
    
# load distance-converted spike data for the current session
spike = mat73.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_convSpikesTime9p6ms_Run0.mat'.format(pathname, recname))
spike_array = spike['filteredSpikeArray']

# # load if each neurones is an interneurone or a pyramidal cell 
# info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
# rec_info = info['rec'][0][0]
# intern_id = rec_info['isIntern'][0]
# pyr_id = [not(clu) for clu in intern_id]

# load place cell ID's 
classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
place_cells = classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]

# # locate baseline trials
# beh_info = info['beh'][0][0]
# behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
# stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
# baseline = np.arange(np.where(stimOn!=0)[0][0])


# #%% parameters
# tot_pc = len(place_cells)  # how many place cells are in this recording
# tot_dist = alignedRun[0].shape[1]  # tot distance bins (exclude beginning and end)
# tot_trial = 130  # tot baseline trials 


#%% UMAP 
reducer = umap.UMAP(metric='cosine',
                    output_metric='euclidean',
                    negative_sample_rate=5,
                    target_metric='categorical',
                    dens_lambda=2.0,
                    dens_frac=0.3,
                    dens_var_shift=0.1,
                    min_dist=0.1,
                    spread=1.0,
                    repulsion_strength=1.0,
                    learning_rate=1.0,
                    init='spectral',
                    n_neighbors=20,
                    random_state=100,
                    n_components=3)


#%% embedding
X = spike_array
X = np.transpose(X)

# X_scaled = StandardScaler().fit_transform(X)

X_embedding = reducer.fit_transform(X_scaled)

fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
plot = ax.scatter(xs=X_embedding[:, 0], ys=X_embedding[:, 1], zs=X_embedding[:, 2], s=3)
cbar = fig.colorbar(plot)