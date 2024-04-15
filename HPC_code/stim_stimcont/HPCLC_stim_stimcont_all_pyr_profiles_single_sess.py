# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:02:12 2024

plot profiles for all pyramidal cells in the HPC-LC stimulation sessions, 
    comparing stim-cont and stim trials
    ** single sessions 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import sys 
from scipy.stats import wilcoxon
# import scipy.stats as st

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
#     sys.path.append('Z:\Dinghao\code_dinghao\common')
# from common import normalise


#%% parameters
# run HPC-LC or HPC-LCterm
HPC_LC = 1


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
if HPC_LC:
    pathHPC = rec_list.pathHPCLCopt
elif not HPC_LC:
    pathHPC = rec_list.pathHPCLCtermopt


#%% lists to contain profiles
stim = []
stim_cont = []

stim_cont_rise_count = []
stim_cont_down_count = []
stim_rise_count = []
stim_down_count = []


#%% get profiles and place in lists
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    trains = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname), 
                          allow_pickle=True).item().values())
    tot_trial = len(trains[0])
    
    # determine if each cell is pyramidal or intern 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    tot_pyr = sum(pyr_id)
    
    # behaviour parameters
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2

    # rise and down counts for the current session
    curr_stim_cont_rise_count = 0 
    curr_stim_cont_down_count = 0 
    curr_stim_rise_count = 0 
    curr_stim_down_count = 0   

    # put each averaged profile into the separate lists
    for i, if_pyr in enumerate(pyr_id):
        if if_pyr:
            # take average 
            temp_cont = np.zeros((len(cont_trials), 6*1250))
            temp_stim = np.zeros((len(stim_trials), 6*1250))
            for ind, trial in enumerate(cont_trials):
                trial_length = len(trains[i][trial])-1250
                if trial_length<6*1250 and trial_length>0:
                    temp_cont[ind, :trial_length] = trains[i][trial][1250:1250+6*1250]  # use [-2, 4] to include -1.5 to -.5 seconds 
                elif trial_length>0:
                    temp_cont[ind, :] = trains[i][trial][1250:1250+6*1250]
            for ind, trial in enumerate(stim_trials):
                trial_length = len(trains[i][trial])-1250
                if trial_length<6*1250 and trial_length>0:
                    temp_stim[ind, :trial_length] = trains[i][trial][1250:1250+6*1250]
                elif trial_length>0:
                    temp_stim[ind, :] = trains[i][trial][1250:1250+6*1250]
            
            temp_cont_mean = np.nanmean(temp_cont, axis=0)
            temp_stim_mean = np.nanmean(temp_stim, axis=0)
            
            stim_cont.append(temp_cont_mean)  # we don't have to normalise here 
            stim.append(temp_stim_mean)
            
            curr_stim_cont_ratio = np.nanmean(temp_cont_mean[625:1875])/np.nanmean(temp_cont_mean[3125:4375])
            curr_stim_ratio = np.nanmean(temp_stim_mean[625:1875])/np.nanmean(temp_stim_mean[3125:4375])
            if curr_stim_cont_ratio < .9:
                curr_stim_cont_rise_count+=1
            elif curr_stim_cont_ratio > 1.1:
                curr_stim_cont_down_count+=1
            if curr_stim_ratio < .9:
                curr_stim_rise_count+=1
            elif curr_stim_ratio > 1.1:
                curr_stim_down_count+=1
            
    stim_cont_rise_count.append(curr_stim_cont_rise_count/tot_pyr)
    stim_cont_down_count.append(curr_stim_cont_down_count/tot_pyr)
    stim_rise_count.append(curr_stim_rise_count/tot_pyr)
    stim_down_count.append(curr_stim_down_count/tot_pyr)
    
tot_clu = len(stim_cont)


#%% plotting 
fig, ax = plt.subplots(figsize=(2,3))

# plot means 
ax.plot([.97, 1.03], [np.median(stim_cont_rise_count), np.median(stim_cont_rise_count)],
        color='grey', linewidth=4)
ax.plot([1.97, 2.03], [np.median(stim_rise_count), np.median(stim_rise_count)],
        color='royalblue', linewidth=4)
ax.plot([1.07, 1.93], [np.median(stim_cont_rise_count), np.median(stim_rise_count)],
        color='k', linewidth=1.5)

# # confidence interval 
# stim_cont_cf = st.t.interval(confidence=0.95, 
#                              df=len(stim_cont_rise_count)-1, 
#                              loc=np.mean(stim_cont_rise_count), 
#                              scale=st.sem(stim_cont_rise_count)) 
# stim_cf = st.t.interval(confidence=0.95, 
#                         df=len(stim_rise_count)-1, 
#                         loc=np.mean(stim_rise_count), 
#                         scale=st.sem(stim_rise_count))

# # plot confidence interval 
# ax.plot([0.5, 0.5], [stim_cont_cf[0], stim_cont_cf[1]], color='grey', linewidth=2, alpha=.3)
# ax.plot([2, 2], [stim_cf[0], stim_cf[1]], color='royalblue', linewidth=2, alpha=.3)

# plot scatter 
ax.scatter([1]*len(stim_cont_rise_count), 
           stim_cont_rise_count, 
           s=10, c='grey', ec='none', lw=.5, alpha=.1)
ax.scatter([2]*len(stim_rise_count), 
           stim_rise_count, 
           s=10, c='royalblue', ec='none', lw=.5, alpha=.1)
ax.plot([[1]*len(stim_cont_rise_count), [2]*len(stim_rise_count)], 
        [stim_cont_rise_count, stim_rise_count], 
        color='grey', alpha=.1, linewidth=1)

# boxplot
# bp = ax.boxplot([stim_cont_rise_count, stim_rise_count],
#                 positions=[.5, 2],
#                 patch_artist=True)

# colors = ['grey', 'royalblue']
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
# bp['fliers'][0].set(marker ='o',
#                 color ='#e7298a',
#                 markersize=2,
#                 alpha=0.5)
# bp['fliers'][1].set(marker ='o',
#                 color ='#e7298a',
#                 markersize=2,
#                 alpha=0.5)
# for median in bp['medians']:
#     median.set(color='darkblue',
#                linewidth=1)

ax.set(xticks=[1, 2], xticklabels=['cont', 'stim'],
       ylabel='% run-onset pyr.', 
       xlim=(.5, 2.5))

for s in ['top', 'bottom', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()

if HPC_LC:
    ax.set_title('HPC_LC\np={}'.format(wilcoxon(stim_cont_rise_count, stim_rise_count)[1]))
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_start_cells.png',
                dpi=500, bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_start_cells.pdf',
                bbox_inches='tight')
elif not HPC_LC:
    ax.set_title('HPC_LCterm\np={}'.format(wilcoxon(stim_cont_rise_count, stim_rise_count)[1]))
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_start_cells.png',
                dpi=500, bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_start_cells.pdf',
                bbox_inches='tight')