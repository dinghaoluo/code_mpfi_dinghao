# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 17:18:06 2025

summarise optogenetics data from imaging animals

@author: Dinghao Luo
"""


#%% imports 
import sys

import numpy as np

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\behaviour_code\utils')
from behaviour_functions import process_behavioural_data

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf


#%% recording list
sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list

paths = rec_list.pathdLightLCOpto


#%% 020 
ctrl_lick_times_020 = []
stim_lick_times_020 = []
ctrl_lick_distances_020 = []
stim_lick_distances_020 = []

ctrl_mean_speeds_020 = []
stim_mean_speeds_020 = []
ctrl_perc_rew_020 = []
stim_perc_rew_020 = [] 


for i, path in enumerate(paths):
    recname = path.split('\\')[-1]
    print(recname)
    text_file = (
        rf'Z:\Dinghao\MiceExp\ANMD{recname[1:4]}'
        rf'\A{recname[1:4]}-{recname[6:]}T.txt'
        )
    curr_txt = process_behavioural_data(text_file)
    
    # segment the session based on optogenetic protocol
    optogenetic_protocol = [t[15] for t in curr_txt['trial_statements']]
    if '2' not in np.unique(optogenetic_protocol):
        continue
    
    start_idx = end_idx = None  # handle edge-cases
    for trial_idx, protocol in enumerate(optogenetic_protocol):  # find the start 
        if protocol != '0':
            start_idx = trial_idx
            break
    for trial_idx in range(start_idx, len(optogenetic_protocol) - 2):  # find the end
        if (optogenetic_protocol[trial_idx] == '0' and 
            optogenetic_protocol[trial_idx + 1] == '0' and 
            optogenetic_protocol[trial_idx + 2] == '0'):
            end_idx = trial_idx  # end before the 3 consecutive 0s
            break
    
    # handle cases where optogenetic session runs until the end
    if start_idx is not None and end_idx is None:
        end_idx = len(optogenetic_protocol)
    
    # handle edge case where no optogenetic session is found
    if start_idx is None:
        print(f'{recname}: no optogenetic stim. in this session')
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        
        opto_stim_trials = list(range(start_idx, end_idx, 3))
        opto_stim_trials = [i for i in opto_stim_trials if i < len(optogenetic_protocol)-1]
        
        opto_ctrl_trials = [i+2 for i in opto_stim_trials]
        opto_ctrl_trials = [i for i in opto_ctrl_trials if i < len(optogenetic_protocol)-1]
    
    baseline_dict = {key: [value[i] for i in baseline_trials if len(value)>1]
                     for key, value in curr_txt.items() if key != 'pulse_times'}
    opto_stim_dict = {key: [value[i] for i in opto_stim_trials if len(value)>1] 
                      for key, value in curr_txt.items() if key != 'pulse_times'}
    opto_ctrl_dict = {key: [value[i] for i in opto_ctrl_trials if len(value)>1] 
                      for key, value in curr_txt.items() if key != 'pulse_times'}
    
    # extract datapoints 
    ctrl_lick_times = []
    stim_lick_times = []
    ctrl_lick_distances = []
    stim_lick_distances = []
    for trial, licks in enumerate(baseline_dict['lick_times']):
        start_time = np.squeeze(baseline_dict['run_onsets'][trial])
        ctrl_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(opto_stim_dict['lick_times']):
        start_time = np.squeeze(opto_stim_dict['run_onsets'][trial])
        stim_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(baseline_dict['lick_distances_aligned']):
        ctrl_lick_distances.append(
            [l for l in licks if l > 30]
            )
    for trial, licks in enumerate(opto_stim_dict['lick_distances_aligned']):
        stim_lick_distances.append(
            [l for l in licks if l > 30]
            )
    
    ctrl_first_lick_time = np.median([licks[0] for licks in ctrl_lick_times if licks])
    stim_first_lick_time = np.median([licks[0] for licks in stim_lick_times if licks])
    ctrl_first_lick_distance = np.median([licks[0] for licks in ctrl_lick_distances if licks])
    stim_first_lick_distance = np.median([licks[0] for licks in stim_lick_distances if licks])
    
    if 2 < ctrl_first_lick_time < 10 and 2 < stim_first_lick_time < 10:
        ctrl_lick_times_020.append(ctrl_first_lick_time)
        stim_lick_times_020.append(stim_first_lick_time)
    
    if ctrl_first_lick_distance > 30 and stim_first_lick_distance > 30:
        ctrl_lick_distances_020.append(ctrl_first_lick_distance)
        stim_lick_distances_020.append(stim_first_lick_distance)
        
    # mean speed     
    speed_times_aligned = curr_txt['speed_times_aligned']
    ctrl_mean_speeds = []
    for trial in baseline_trials:
        speed_times = speed_times_aligned[trial]
        if speed_times:
            speeds = [s[1] for s in speed_times]
            ctrl_mean_speeds.append(np.mean(speeds))
    ctrl_mean_speeds_020.append(np.mean(ctrl_mean_speeds))
    
    stim_mean_speeds = []
    for trial in opto_stim_trials:
        speed_times = speed_times_aligned[trial]
        if speed_times:
            speeds = [s[1] for s in speed_times]
            stim_mean_speeds.append(np.mean(speeds))
    stim_mean_speeds_020.append(np.mean(stim_mean_speeds))
    
    # percent rewarded 
    reward_times = curr_txt['reward_times']
    ctrl_rewarded = [not np.isnan(reward_times[trial]) for trial in baseline_trials[1:]]
    ctrl_reward_perc = sum(ctrl_rewarded)/len(baseline_trials[1:])
    stim_rewarded = [not np.isnan(reward_times[trial]) for trial in opto_stim_trials]
    stim_reward_perc = sum(stim_rewarded)/len(opto_stim_trials)
    
    ctrl_perc_rew_020.append(ctrl_reward_perc)
    stim_perc_rew_020.append(stim_reward_perc)

pf.plot_violin_with_scatter(
    ctrl_lick_times_020, stim_lick_times_020, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='first-lick time (s)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\dLight\020_lick_times',
    dpi=300
    )

pf.plot_violin_with_scatter(
    ctrl_lick_distances_020, stim_lick_distances_020, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='first-lick distance (cm)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\dLight\020_lick_distances',
    dpi=300
    )

pf.plot_violin_with_scatter(
    ctrl_mean_speeds_020, stim_mean_speeds_020, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='mean speed (cm/s)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\dLight\020_mean_speed',
    dpi=300
    )

rmv = []
for i in range(len(ctrl_perc_rew_020)):
    if ctrl_perc_rew_020[i] < 0.5 or stim_perc_rew_020[i] < 0.5:
        rmv.append(i)
ctrl_perc_rew_020 = [trial for i, trial in enumerate(ctrl_perc_rew_020) if i not in rmv]
stim_perc_rew_020 = [trial for i, trial in enumerate(stim_perc_rew_020) if i not in rmv]

pf.plot_violin_with_scatter(
    ctrl_perc_rew_020, stim_perc_rew_020, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='reward perc.',
    ylim=(.5, 1.02),
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\dLight\020_perc_rew',
    dpi=300
    )