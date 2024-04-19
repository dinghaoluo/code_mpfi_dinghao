# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:50:42 2024

see if modulated cells are more likely place cells 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import sys 
import pandas as pd
from scipy.stats import sem, wilcoxon

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load paths and data 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt
df = pd.read_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.csv') 
pathHPCterm = rec_list.pathHPCLCtermopt
dfterm = pd.read_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_diff_profiles_pyr_only.csv')


#%% containers 
mc_overlap = []  # MC overlap with PC
nmc_overlap = []  # non-MC overlap with PC


#%% main loop 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
    place_cells = classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]
    place_cells = [i-2 for i in place_cells]  # correct for matlab 
    tot_pc = len(place_cells)
    
    if tot_pc!=0:
        
        mc = 0
        nmc = 0
            
        clu_list = np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_clu_list_{}.npy'.format(recname, recname))
        tot_clu = len(clu_list)
        
        # if each neurones is an interneurone or a pyramidal cell 
        info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
        rec_info = info['rec'][0][0]
        intern_id = rec_info['isIntern'][0]
        pyr_id = [not(clu) for clu in intern_id]
        tot_clu = len(pyr_id)
        
        # loop over all
        tot_mc = 0
        tot_nmc = 0
        for i in range(tot_clu):
            if pyr_id[i]==True:
                cluname = clu_list[i]
                reg = df.loc[df['Unnamed: 0'] == cluname]['regulated'].item()
                if reg!='none':
                    tot_mc+=1
                    if i in place_cells:
                        mc+=1
                else:
                    tot_nmc+=1
                    if i in place_cells:
                        nmc+=1
                        
        mc_overlap.append(mc/tot_mc)
        nmc_overlap.append(nmc/tot_nmc)


for pathname in pathHPCterm:
    recname = pathname[-17:]
    print(recname)
    
    classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
    place_cells = classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]
    place_cells = [i-2 for i in place_cells]  # correct for matlab 
    tot_pc = len(place_cells)
    
    if tot_pc!=0:
        
        mc = 0
        nmc = 0
            
        clu_list = np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_clu_list_{}.npy'.format(recname, recname))
        tot_clu = len(clu_list)
        
        # if each neurones is an interneurone or a pyramidal cell 
        info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
        rec_info = info['rec'][0][0]
        intern_id = rec_info['isIntern'][0]
        pyr_id = [not(clu) for clu in intern_id]
        tot_clu = len(pyr_id)
        
        # loop over all
        tot_mc = 1
        tot_nmc = 1
        for i in range(tot_clu):
            if pyr_id[i]==True:
                cluname = clu_list[i]
                reg = dfterm.loc[dfterm['Unnamed: 0'] == cluname]['regulated'].item()
                if reg!='none':
                    tot_mc+=1
                    if i in place_cells:
                        mc+=1
                else:
                    tot_nmc+=1
                    if i in place_cells:
                        nmc+=1
                        
        mc_overlap.append(mc/tot_mc)
        nmc_overlap.append(nmc/tot_nmc)


#%% as arrays 
mc_overlap = np.array(mc_overlap)
nmc_overlap = np.array(nmc_overlap)

mc_means = np.mean(mc_overlap)
nmc_means = np.mean(nmc_overlap)

mc_err = sem(mc_overlap)
nmc_err = sem(nmc_overlap)


#%% statistics
r = wilcoxon(mc_overlap, nmc_overlap) 


#%% plotting 
fig, ax = plt.subplots(figsize=(3,3))

# ax.bar([1,2], [mc_means, nmc_means], yerr=[mc_err, nmc_err])
# ax.scatter([1]*len(mc_overlap), mc_overlap, c='darkblue', s=3)
# ax.scatter([2]*len(nmc_overlap), nmc_overlap, c='grey', s=3)
# for i in range(len(mc_overlap)):
#     ax.plot([1,2], [mc_overlap[i],nmc_overlap[i]],
#             c='grey', alpha=.5)

ax.violinplot([mc_overlap, nmc_overlap])

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# ax.set(xlabel='log PN prop.', ylabel='log mod. prop.',
#        xlim=(-5.5, -.5), ylim=(-4, -.5),
#        xticks=[-5,-3,-1], yticks=[-1, -3],
#        title='stderr={}\npval={}'.format(stderr, pval))

# fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\pc_mc_proportion_comp.png',
#             dpi=300,
#             bbox_inches='tight')
