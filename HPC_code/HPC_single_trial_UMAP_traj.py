# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:26:17 2024

perform UMAP on a trial-by-trial basis, after fitting principal components to 
    averaged baseline activity

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
spike = mat73.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_convSpikesDistAligned_msess1_Run0.mat'.format(pathname, recname))
alignedRun = spike['filteredSpikeDistArrayRun']  # load alignedRun spike data

# load if each neurones is an interneurone or a pyramidal cell 
info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
rec_info = info['rec'][0][0]
intern_id = rec_info['isIntern'][0]
pyr_id = [not(clu) for clu in intern_id]

# load place cell ID's 
classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
place_cells = classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]

# locate baseline trials
beh_info = info['beh'][0][0]
behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
baseline = np.arange(np.where(stimOn!=0)[0][0])


#%% parameters
tot_pc = len(place_cells)  # how many place cells are in this recording
tot_dist = alignedRun[0].shape[1]  # tot distance bins (exclude beginning and end)
tot_trial = 130  # tot baseline trials 


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
                    n_components=3)

scaler = StandardScaler()


#%% single-trial embedding
X_curr = alignedRun[35][place_cells,:] 
X_curr = np.transpose(X_curr)

X_scaled = scaler.fit_transform(X_curr)

X_embedding = reducer.fit_transform(X_scaled)

d = np.arange(2001)  # dist for color mapping

fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
plot = ax.scatter(xs=X_embedding[:, 0], ys=X_embedding[:, 1], zs=X_embedding[:, 2], s=3, c=d, cmap=cm.jet)
cbar = fig.colorbar(plot)

for i in range(36,50):
    X_next = alignedRun[i][place_cells,:] 
    X_next = np.transpose(X_next)
    
    X_next_scaled = scaler.transform(X_next)
    
    X_next_embedding = reducer.fit_transform(X_next_scaled)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection ='3d')
    plot = ax.scatter(xs=X_next_embedding[:, 0], ys=X_next_embedding[:, 1], zs=X_next_embedding[:, 2], s=3, c=d, cmap=cm.jet)
    # cbar = fig.colorbar(plot)



#%% average PCA to find PC's
X_avg = np.zeros((tot_pc, tot_dist))  # to contain all pyramidal cells avg baseline activity
# X_avg_ordered = np.zeros((tot_pc, tot_dist))
for i, clu in enumerate(place_cells):
    temp_all = np.zeros((tot_trial, tot_dist))
    
    for trial in range(5, tot_trial):
        temp_all[trial, :] = alignedRun[trial][clu][300:-100]
    
    # X_avg[i, :] = normalise(np.mean(temp_all, axis=0))
    X_avg[i, :] = np.mean(temp_all, axis=0)
    
# # order stuff by argmax
# max_pt = {}  # argmax for all pcs
# for i in range(tot_pc):
#     max_pt[i] = np.argmax(X_avg[i,:])
# def helper(x):
#     return max_pt[x]
# ord_ind = sorted(np.arange(tot_pc), key=helper)
# for i, ind in enumerate(ord_ind): 
#     X_avg_ordered[i,:] = X_avg[ind,:]

# fig, ax = plt.subplots(figsize=(4,3))
# ax.imshow(X_avg_ordered, aspect='auto', interpolation='none', cmap='Greys', 
#           extent=[30, 190, 1, tot_pc])
# ax.set(xlabel='dist. (cm)', ylabel='place cell #')
        
X_avg = np.transpose(X_avg)
# X_avg[np.isnan(X_avg)] = 0
y = np.arange(tot_dist)
        
# PCA stuff
scaler = StandardScaler()
X_avg_norm = scaler.fit_transform(X_avg)

# split the dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_avg_norm, y, random_state=42, test_size=0.25, shuffle=True)

# Run PCA
pca = PCA(n_components=3)
X_avg_pca = pca.fit_transform(X_avg_norm)


#%% fitting every single trial to these axes
fig = plt.figure()
ax = plt.axes(projection ='3d')
for trial in range(20, 25):
    X_curr = alignedRun[trial][place_cells, 300:-100]
    X_curr = np.transpose(X_curr)
    
    X_curr_norm = scaler.transform(X_curr)
    X_curr_pca = pca.transform(X_curr_norm)

    # fig = plt.figure()
    ax.scatter(xs=X_curr_pca[:, 0], ys=X_curr_pca[:, 1], zs=X_curr_pca[:, 2], s=3, alpha=np.arange(tot_dist)/tot_dist/5)
    # ax.scatter(xs=X_curr_pca[:10, 0], ys=X_curr_pca[:10, 1], zs=X_curr_pca[:10, 2], s=3, alpha=.3)
    # ax.scatter(xs=X_curr_pca[-10:, 0], ys=X_curr_pca[-10:, 1], zs=X_curr_pca[-10:, 2], s=3, alpha=.9)

# mean 
ax.scatter(xs=np.mean(X_pca[:len(stim_trials), 0]),
           ys=np.mean(X_pca[:len(stim_trials), 0]),
           zs=np.mean(X_pca[:len(stim_trials), 0]),
           color='black')
ax.scatter(xs=np.mean(X_pca[len(stim_trials):, 0]),
           ys=np.mean(X_pca[len(stim_trials):, 0]),
           zs=np.mean(X_pca[len(stim_trials):, 0]),
           color='darkblue')

# legend
st = ax.scatter([],[], c='royalblue')
ct = ax.scatter([],[], c='grey')
ax.legend([st, ct], ['stim', 'baseline'], frameon=False)

ax.set(title='PCA, stim v stim-cont, {}'.format(recname))

fig.tight_layout()
        
# make folders if not exist
outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
if not os.path.exists(outdirroot):
    os.makedirs(outdirroot)
outdir = '{}\stim_stim_cont_pyr_only_PCA'.format(outdirroot)
        
fig.savefig(outdir,
            dpi=500,
            bbox_inches='tight')

stim_cent = np.array([np.mean(X_pca[:len(stim_trials), 0]),
                      np.mean(X_pca[:len(stim_trials), 0]),
                      np.mean(X_pca[:len(stim_trials), 0])])
cont_cent = np.array([np.mean(X_pca[len(stim_trials):, 0]),
                      np.mean(X_pca[len(stim_trials):, 0]),
                      np.mean(X_pca[len(stim_trials):, 0])])

squared_dist = np.sum((cont_cent-stim_cent)**2, axis=0)
dist = np.sqrt(squared_dist)


# shuffled
shuf_number = 500

dist_shuf = np.zeros(shuf_number)
for shuf in range(shuf_number):
    shuf_1 = np.random.randint(low=0, high=tot_trial, size=len(stim_trials))
    shuf_2 = np.random.randint(low=0, high=tot_trial, size=len(stim_trials))
    
    X_shuf = np.zeros((tot_clu, len(stim_trials)*2))
    for i in range(tot_clu):
        if pyr_id[i]==True:
            cluname = clu_list[i]
            temp_all = np.zeros(len(stim_trials)*2)

            for ind, trial in enumerate(shuf_1):
                temp_all[ind] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
            for ind, trial in enumerate(shuf_2):
                temp_all[len(stim_trials)+ind] = np.mean(trains[i][trial][comp_window[0]:comp_window[1]])*1250
            
            X_shuf[i, :] = temp_all
        
    X_shuf = np.transpose(X_shuf)
    X_shuf[np.isnan(X_shuf)] = 0
    y = np.arange(len(stim_trials)*2)
            
    # PCA stuff
    X_norm_shuf = scaler.fit_transform(X_shuf)

    # split the dataset into train and test datasets
    X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = train_test_split(X_norm_shuf, y, random_state=42, test_size=0.25, shuffle=True)

    # Run PCA
    pca = PCA(n_components=3)
    X_shuf_pca = pca.fit_transform(X_norm_shuf)
    
    shuf_1_cent = np.array([np.mean(X_shuf_pca[:len(stim_trials), 0]),
                            np.mean(X_shuf_pca[:len(stim_trials), 0]),
                            np.mean(X_shuf_pca[:len(stim_trials), 0])])
    shuf_2_cent = np.array([np.mean(X_shuf_pca[len(stim_trials):, 0]),
                            np.mean(X_shuf_pca[len(stim_trials):, 0]),
                            np.mean(X_shuf_pca[len(stim_trials):, 0])])
    
    squared_dist_shuf = np.sum((shuf_2_cent-shuf_1_cent)**2, axis=0)
    dist_shuf[shuf] = np.sqrt(squared_dist_shuf)
    

# histogram for distance comparison
fig, ax = plt.subplots(figsize=(4,3))

ax.hist(dist_shuf, bins=25, ec='dimgrey', color='grey', alpha=.85, density=True)
ax.vlines(dist, 0, 1.5, color='royalblue', linewidth=3)
ax.vlines(np.percentile(dist_shuf,99.9), 0, 1.5, color='dimgrey', linewidth=3)

ax.set(yticks=[], yticklabels=[],
       xlabel='distance btw. embedded means',
       ylim=(0,1.1))

fig.suptitle('stim v stim-cont, {}'.format(recname))

for s in ['top', 'left', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout() 

fig.savefig('{}_hist_dist'.format(outdir),
            dpi=500,
            bbox_inches='tight')