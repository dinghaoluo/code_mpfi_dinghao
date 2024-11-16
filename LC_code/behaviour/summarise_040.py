# -*- coding: utf-8 -*-
"""
Created on Wed 13 Nov 16:53:32 2024

summarise pharmacological experiments with SCH23390

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
from scipy.stats import sem

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

from behaviour_processing import process_behavioural_data


#%% recording list
sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLCBehopt = rec_list.pathLCBehopt
sessLCBehopt = rec_list.sessLCBehopt
condLCBehopt = rec_list.condLCBehopt


#%% main
base_list = []
stim_list = []
for i, pathname in enumerate(pathLCBehopt[:2]):   
    sessname = pathname[-13:]
    curr_cond = condLCBehopt[i]
    curr_sess = sessLCBehopt[i]
    if str([0,4,0])[1:-1] not in str(curr_cond)[1:-1]:
        print(f'{sessname} does not have rew-stim.')
        continue
    file_idx = curr_sess[curr_cond.index(4)]  # rew
    text_file = r'{}\{}-0{}\{}-0{}T.txt'.format(pathname, 
                                                sessname, file_idx, 
                                                sessname, file_idx)
    curr_txt = process_behavioural_data(text_file)
    
    # Segment the session based on optogenetic protocol
    optogenetic_protocol = curr_txt['optogenetic_protocols']
    start_idx, end_idx = None, None
    
    # Find start of optogenetic session
    for trial_idx, protocol in enumerate(optogenetic_protocol):
        if protocol != '0':
            start_idx = trial_idx
            break
    
    # Find end of optogenetic session
    if start_idx is not None:
        for trial_idx in range(start_idx, len(optogenetic_protocol) - 2):
            if (optogenetic_protocol[trial_idx] == '0' and 
                optogenetic_protocol[trial_idx + 1] == '0' and 
                optogenetic_protocol[trial_idx + 2] == '0'):
                end_idx = trial_idx  # End before the 3 consecutive 0s
                break
    
    # Handle cases where optogenetic session runs until the end
    if start_idx is not None and end_idx is None:
        end_idx = len(optogenetic_protocol)
    
    # Handle edge case where no optogenetic session is found
    if start_idx is None:
        print(f"{sessname}: No optogenetic session found.")
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        optogenetic_trials = list(range(start_idx, end_idx))
    
    # Add to base and stim lists for further analysis
    base_list.append(baseline_trials)
    stim_list.append(optogenetic_trials)
    
    print(f"Session {sessname}:")
    print(f"  Baseline trials: {baseline_trials}")
    print(f"  Optogenetic trials: {optogenetic_trials}")
        

#%% speed plot  
fig, ax = plt.subplots(figsize=(2.3,1.7))

ms_pre = np.mean(mean_speeds_pre, axis=0)/10
ms_drug = np.mean(mean_speeds_drug, axis=0)/10
ms_post = np.mean(mean_speeds_post, axis=0)/10
ss_pre = sem(mean_speeds_pre, axis=0)/10
ss_drug = sem(mean_speeds_drug, axis=0)/10
ss_post = sem(mean_speeds_post, axis=0)/10

lp, = ax.plot(x_speed, ms_pre, color='grey')
ax.fill_between(x_speed, ms_pre+ss_pre,
                         ms_pre-ss_pre,
                color='grey', edgecolor='none', alpha=.15)
ld, = ax.plot(x_speed, ms_drug, color='#darkgreen')
ax.fill_between(x_speed, ms_drug+ss_drug,
                         ms_drug-ss_drug,
                color='darkgreen', edgecolor='none', alpha=.15)

plt.legend([lp, ld], ['baseline', 'exp.'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(0,180), xlabel='distance (cm)',
       ylabel='velocity (cm/s)')

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\propranolol\speed_profile{}'.format(ext),
                dpi=300, bbox_inches='tight')
    

#%% lick plot  
fig, ax = plt.subplots(figsize=(2.3,1.7))

ml_pre = np.mean(mean_licks_pre, axis=0)/10
ml_drug = np.mean(mean_licks_drug, axis=0)/10
ml_post = np.mean(mean_licks_post, axis=0)/10
sl_pre = sem(mean_licks_pre, axis=0)/10
sl_drug = sem(mean_licks_drug, axis=0)/10
sl_post = sem(mean_licks_post, axis=0)/10

lp, = ax.plot(x_lick, ml_pre, color='grey')
ax.fill_between(x_lick, ml_pre+sl_pre,
                        ml_pre-sl_pre,
                color='grey', edgecolor='none', alpha=.15)
ld, = ax.plot(x_lick, ml_drug, color='darkgreen')
ax.fill_between(x_lick, ml_drug+sl_drug,
                        ml_drug-sl_drug,
                color='darkgreen', edgecolor='none', alpha=.15)

plt.legend([lp, ld], ['baseline', 'exp.'], frameon=False)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(30,219), xlabel='distance (cm)',
       ylim=(0,0.4), ylabel='hist. licks', yticks=[0, 0.3])

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\pharmacology\propranolol\lick_profile{}'.format(ext),
                dpi=300, bbox_inches='tight')