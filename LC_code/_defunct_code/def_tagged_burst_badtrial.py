# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:59:43 2023

define burst index as burst/post
correlate *trial by trial* burst index with bad trials

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import sem, ttest_rel
import sys

def burst_index(burst, post):
    """
    take burst & post means or time series and calculate burst/post as the 
    burst index
    """
    if type(burst)!=float:
        burst = np.mean(burst)
    if type(post)!=float:
        post = np.mean(post)
    if post==0:
        post = 1/1250
    return burst/post


#%% MAIN
all_tagged_train = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
                           allow_pickle=True).item()
burst_ind = {}
good_perc = []; bad_perc = []

for clu in list(all_tagged_train.items()):
    cluname = clu[0]
    curr_spike_all = clu[1][0]
    burst_ind[cluname] = []
    
    for trial in curr_spike_all:
        burst_ind[cluname].append(burst_index(trial[3435:4065], 
                                              trial[5000:6250]))
    curr_burst = burst_ind[cluname]
    
    # import bad beh ind
    root = 'Z:\Dinghao\MiceExp'
    fullpath = root+'\ANMD'+cluname[1:5]+'\\'+cluname[:14]+'\\'+cluname[:17]
    beh_par_file = sio.loadmat(fullpath+'\\'+cluname[:17]+
                               '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference
    ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                     # -1 to account for 0 being an empty trial
    ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
    # get bad burst id and good burst id
    temp_id = list(range(len(curr_burst)))
    
    # sort temp_id by curr_burst
    def myburst(e):
        return curr_burst[e]
    temp_id.sort(key=myburst)
    curr_burst.sort()
    
    # first index greater than 1.2
    border_id = next(x[0] for x in enumerate(curr_burst) if x[1]>1.2)
    
    bad_burst_id = temp_id[:border_id]
    good_burst_id = temp_id[border_id:]
    
    # how much of bad/good burst id's are bad trials
    bad_count = 0; good_count = 0
    for i in ind_bad_beh:
        if i in good_burst_id:
            good_count += 1
        if i in bad_burst_id:
            bad_count += 1
    bad_perc.append(bad_count / border_id)
    good_perc.append(good_count / (len(temp_id)-border_id))
    

#%% plotting 
print('plotting bad vs good trial percentages for bad and good bursts...')
t_res = ttest_rel(bad_perc, good_perc)
pval = t_res[1]

fig, ax = plt.subplots()

ax.plot([0,1], [0,1], color='grey')
ax.scatter(good_perc, bad_perc, s=5, color='grey', alpha=.5)
mean_gb = np.mean(good_perc); mean_bb = np.mean(bad_perc)
sem_gb = sem(good_perc); sem_bb = sem(bad_perc)
ax.scatter(mean_gb, mean_bb, s=15, color='royalblue', alpha=.9)
ax.plot([mean_gb, mean_gb], 
        [mean_bb+sem_bb, mean_bb-sem_bb], 
        color='cornflowerblue', alpha=.7)
ax.plot([mean_gb+sem_gb, mean_gb-sem_gb], 
        [mean_bb, mean_bb], 
        color='cornflowerblue', alpha=.7)

ax.set(title=pval,
       xlabel='good trials',
       ylabel='bad trials',
       xlim=(0,1), ylim=(0,1))

plt.show()