# -*- coding: utf-8 -*-
"""
Created on Wed 2 Aug 16:24:54 2023

plot all HPC LC rasters after running HPCLC_all_rasters.py to store things into
    separate files for each session 
    
dependencies:
    HPCLC_all_rasters.py

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import scipy.io as sio

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% MAIN 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    curr_rasters = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(pathname[-17:]),
                           allow_pickle=True).item()

    tot_plots = len(curr_rasters)  # how many clu's in this session
    col_plots = 10
    row_plots = tot_plots // col_plots
    if tot_plots % col_plots != 0:
        row_plots += 1
    plot_pos = np.arange(1, tot_plots+1)
    fig = plt.figure(1, figsize=[col_plots*4, row_plots*4])
    fig.suptitle(pathname[-17:])  # set sup title 
    
    # load stim trials
    behInfo = sio.loadmat('{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))['beh']
    stim_trial = np.squeeze(np.where(behInfo['pulseMethod'][0][0][0]!=0))-1  # -1 to match up with matlab indexing
    
    # # import bad beh trial id
    # root = 'Z:\Dinghao\MiceExp'
    # fullpath = root+'\ANMD'+curr_clu_name[1:5]+'\\'+curr_clu_name[:14]+'\\'+curr_clu_name[:17]
    # beh_par_file = sio.loadmat(fullpath+'\\'+curr_clu_name[:17]+
    #                            '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
    #                                # -1 to account for MATLAB Python difference
    # ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    #                                  # -1 to account for 0 being an empty trial
    # ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    # ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
    for i in range(tot_plots):
        curr_clu = list(curr_rasters.items())[i]
        curr_clu_name = curr_clu[0]
        print('plotting {}'.format(curr_clu_name))
        curr_raster = curr_clu[1]
        
        ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
        ax.set(xlim=(-3.0, 5.0), xlabel='time (s)',
                                 ylabel='trial #',
                                 title=curr_clu_name)
        ax.spines[['right','top']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(1)
            
        tot_trial = curr_raster.shape[0]  # how many trials
        for trial in range(tot_trial):
            curr_trial = np.where(curr_raster[trial]==1)[0]
            curr_trial = [(s-3750)/1250 for s in curr_trial]
            ax.scatter(curr_trial, [trial+1]*len(curr_trial),
                        color='grey', s=.35)
        ax.fill_between([-3, 5], [stim_trial[0], stim_trial[0]],
                                 [stim_trial[-1], stim_trial[-1]],
                                 color='royalblue', alpha=.15)
            
    plt.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters\{}.png'.format(pathname[-17:]),
                dpi=300,
                bbox_inches='tight')
    
    plt.close(fig)