# -*- coding: utf-8 -*-
"""
Created on Wed 23 Aug 17:18:12 2023

compare opto stim vs baseline lickdist

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats import ranksums
import sys

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathOpt = rec_list.pathLCopt

# 0-2-0
sess_list = [sess[-17:] for sess in pathOpt]

n_bst = 1000  # hyperparameter for bootstrapping
comp_method = 'baseline'
print('\n**BOOTSTRAP # = {}**'.format(n_bst))
print('**comparison method: {}**\n'.format(comp_method))


sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf
from common import mpl_formatting
mpl_formatting()


#%% MAIN 
all_licks_ctrl = []; all_licks_stim = []

for sessname in sess_list:
    print(sessname)
    
    infofilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    
    Info = sio.loadmat(infofilename)
    pulseMethod = Info['beh'][0][0]['pulseMethod'][0]
    
    # stim info
    tot_stims = len([t for t in pulseMethod if t!=0])
    stim_cond = pulseMethod[np.where(pulseMethod!=0)][0]  # check stim condition
    stim = [i for i, e in enumerate(pulseMethod) if e==stim_cond]
    
    # licks 
    lickfilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    alignRun = sio.loadmat(lickfilename)
    
    # ignore all 1st trials since it is before counting starts and is an empty cell
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    dist = alignRun['trialsRun']['xMM'][0][0][0][1:]  # distance at each sample
    
    first_licks = []
    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        # licks 
        lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
        if len(lk)!=0:  # append only if there is licks in this trial
            for i in range(len(lk)):
                ld = dist[trial][lk[0]-starts[trial]]/10                
                if ld > 30:  # filter out first licks before 30 (only starts counting at 30)
                    first_licks.append(dist[trial][lk[0]-starts[trial]]/10)
                    break
            if ld <= 30:
                first_licks.append(0)
        else:
            first_licks.append(0)
    
    # stim licks 
    licks_stim = [first_licks[i-1] for i in stim if first_licks[i-1]!=0 and first_licks[i+1]!=0]
    
    pval = []; 
    curr_licks_ctrl = []; curr_licks_stim = []

    for i in range(n_bst):
        # select same number of ctrl to match 
        ctrl_trials = np.where(pulseMethod==0)[0]
        if comp_method == 'baseline':
            selected_ctrl = ctrl_trials[np.random.randint(0, stim[0]-1, len(licks_stim))]
            licks_ctrl = []
            for t in selected_ctrl:
                if first_licks[t-1]!=0:
                    licks_ctrl.append(float(first_licks[t-1]))  # only compare trials with licks
                else:
                    licks_ctrl.append(float(first_licks[t]))
        elif comp_method == 'stim_cont': # stim_control 
            selected_ctrl = [i+2 for i in stim]
            licks_ctrl = [first_licks[i-1] for i in selected_ctrl if first_licks[i-1]!=0 and first_licks[i-3]!=0]
        
        curr_licks_ctrl.append(licks_ctrl)
        curr_licks_stim.append(licks_stim)
        
        pval.append(ranksums(licks_ctrl, licks_stim)[1])
        
    if stim_cond==2:
        all_licks_ctrl.append(np.median(curr_licks_ctrl))
        all_licks_stim.append(np.median(curr_licks_stim))
    
    
    data = [licks_ctrl, [l[0] for l in licks_stim]]
    
    fig, ax = plt.subplots(figsize=(3.3,2))
    
    # Remove top, right, and left spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    
    # Set title and axis limits
    ax.set(title=f'{sessname}, stim={stim_cond}', 
           xlim=(30, 225), ylim=(-0.5, 1.5), 
           ylabel='Condition', xlabel='dist. 1st lick (cm)')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['ctrl.', 'stim/'])
    
    # Plot box plots on separate tracks
    box_positions = [0, 1]
    boxplot = ax.boxplot(data, 
                         positions=box_positions, vert=False, widths=0.3,
                         patch_artist=True, 
                         boxprops=dict(facecolor='lightgrey', color='grey'),
                         medianprops=dict(color='black'))
    colors = ['grey', 'royalblue']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('k')
    
    ax.scatter(licks_ctrl, [.25]*len(licks_ctrl), color='grey', label='Baseline')
    ax.scatter(licks_stim, [.75]*len(licks_stim), color='royalblue', label='Stimulation')
    
    ax.plot([np.median(licks_ctrl), np.median(licks_stim)], [.25, .75], 
            color='grey', alpha=0.5, linestyle='--')
        
    for ext in ['.png', '.pdf']:
        if comp_method == 'baseline':
            fig.savefig(f'Z:/Dinghao/code_dinghao/LC_opto_ephys/opto_lickdist_0{stim_cond}0/{sessname}{ext}', 
                        dpi=300, bbox_inches='tight')
        elif comp_method == 'stim_cont':
            fig.savefig(f'Z:/Dinghao/code_dinghao/LC_opto_ephys/opto_lickdist_0{stim_cond}0_stim_cont/{sessname}{ext}', 
                        dpi=300, bbox_inches='tight')
    
    plt.show()
  
    
#%% summary statistics 
pf.plot_violin_with_scatter(all_licks_ctrl, all_licks_stim, 'grey', 'royalblue', 
                            paired=True,
                            xticklabels=['ctrl.', 'stim.'], ylabel='distance 1st-licks (cm)',
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_lickdist_020\summary', dpi=300)