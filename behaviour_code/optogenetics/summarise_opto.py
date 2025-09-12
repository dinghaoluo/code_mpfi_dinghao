# -*- coding: utf-8 -*-
"""
Created on Wed 13 Nov 16:53:32 2024
Modified on Thur 6 Mar 2025

summarise optogenetic experiments

@author: Dinghao Luo
"""


#%% imports 
import numpy as np

from common import mpl_formatting
mpl_formatting()

from behaviour_functions import process_behavioural_data

import plotting_functions as pf


#%% recording list
import rec_list
pathLCBehopt = rec_list.pathLCBehopt
sessLCBehopt = rec_list.sessLCBehopt
condLCBehopt = rec_list.condLCBehopt

pathLCopt = rec_list.pathLCopt
pathHPCopt = rec_list.pathHPCLCopt
pathHPCLCtermopt = rec_list.pathHPCLCtermopt_beh


#%% 020 
ctrl_lick_times_020 = []
stim_lick_times_020 = []
ctrl_lick_distances_020 = []
stim_lick_distances_020 = []

ctrl_mean_speeds_020 = []
stim_mean_speeds_020 = []
ctrl_perc_rew_020 = []
stim_perc_rew_020 = [] 

for i, pathname in enumerate(pathLCBehopt):
    sessname = pathname[-13:]
    print(sessname)
    curr_cond = condLCBehopt[i]
    curr_sess = sessLCBehopt[i]
    if str([0,2,0])[1:-1] not in str(curr_cond)[1:-1]: 
        continue
    file_idx = curr_sess[curr_cond.index(2)]
    text_file = rf'{pathname}\{sessname}-0{file_idx}\{sessname}-0{file_idx}T.txt'
    curr_txt = process_behavioural_data(text_file)
    
    # segment the session based on optogenetic protocol
    optogenetic_protocol = [t[15] for t in curr_txt['trial_statements']]
    
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
        print(f'{sessname}: no optogenetic stim. in this session')
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        opto_stim_trials = list(range(start_idx, end_idx, 3))
        opto_ctrl_trials = [i+2 for i in opto_stim_trials]
    
    baseline_dict = {key: [value[i] for i in baseline_trials if len(value)>1]
                     for key, value in curr_txt.items()}
    opto_stim_dict = {key: [value[i] for i in opto_stim_trials if len(value)>1] 
                      for key, value in curr_txt.items()}
    opto_ctrl_dict = {key: [value[i] for i in opto_ctrl_trials if len(value)>1] 
                      for key, value in curr_txt.items()}
    
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
    
    # get first licks
    ctrl_first_lick_time = np.median([licks[0] for licks in ctrl_lick_times if licks])
    stim_first_lick_time = np.median([licks[0] for licks in stim_lick_times if licks])
    ctrl_first_lick_distance = np.median([licks[0] for licks in ctrl_lick_distances if licks])
    stim_first_lick_distance = np.median([licks[0] for licks in stim_lick_distances if licks])
    
    if 2 < ctrl_first_lick_time < 10 and 2 < stim_first_lick_time < 10:
        ctrl_lick_times_020.append(ctrl_first_lick_time)
        stim_lick_times_020.append(stim_first_lick_time)
    
    if ctrl_first_lick_distance > 100 and stim_first_lick_distance > 100:
        ctrl_lick_distances_020.append(ctrl_first_lick_distance)
        stim_lick_distances_020.append(stim_first_lick_distance)
        
    # get combined lists
    ctrl_lick_dist_time = [t for trial in ctrl_lick_times for t in trial if 2<t<4]
    stim_lick_dist_time = [t for trial in stim_lick_times for t in trial if 2<t<4]
    ctrl_lick_dist_distance = [t for trial in ctrl_lick_distances for t in trial if 100<t<220]
    stim_lick_dist_distance = [t for trial in stim_lick_distances for t in trial if 100<t<220]
    
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


for i, pathname in enumerate(pathLCopt):
    sessname = pathname[-17:]
    print(sessname)
    text_file = rf'{pathname}\{sessname}T.txt'
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
        print(f'{sessname}: no optogenetic stim. in this session')
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        opto_stim_trials = list(range(start_idx, end_idx, 3))
        opto_ctrl_trials = [i+2 for i in opto_stim_trials]
    
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
    
    if ctrl_first_lick_distance > 100 and stim_first_lick_distance > 100:
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
            
        
for i, pathname in enumerate(pathHPCopt):
    sessname = pathname[-17:]
    print(sessname)
    text_file = rf'{pathname}\{sessname}T.txt'
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
        print(f'{sessname}: no optogenetic stim. in this session')
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        opto_stim_trials = list(range(start_idx, end_idx, 3))
        opto_ctrl_trials = [i+2 for i in opto_stim_trials]
    
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
    
    if ctrl_first_lick_distance > 100 and stim_first_lick_distance > 100:
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
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\020_lick_times',
    dpi=300
    )

pf.plot_violin_with_scatter(
    ctrl_lick_distances_020, stim_lick_distances_020, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='first-lick distance (cm)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\020_lick_distances',
    dpi=300
    )

pf.plot_violin_with_scatter(
    ctrl_mean_speeds_020, stim_mean_speeds_020, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='mean speed (cm/s)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\020_mean_speed',
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
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\020_perc_rew',
    dpi=300
    )


#%% 030 
ctrl_lick_times_030 = []
stim_lick_times_030 = []
ctrl_lick_distances_030 = []
stim_lick_distances_030 = []

ctrl_mean_speeds_030 = []
stim_mean_speeds_030 = []
ctrl_perc_rew_030 = []
stim_perc_rew_030 = [] 

for i, pathname in enumerate(pathLCBehopt):
    sessname = pathname[-13:]
    print(sessname)
    curr_cond = condLCBehopt[i]
    curr_sess = sessLCBehopt[i]
    if str([0,3,0])[1:-1] not in str(curr_cond)[1:-1]: 
        continue
    file_idx = curr_sess[curr_cond.index(3)]
    text_file = rf'{pathname}\{sessname}-0{file_idx}\{sessname}-0{file_idx}T.txt'
    curr_txt = process_behavioural_data(text_file)
    
    # segment the session based on optogenetic protocol
    optogenetic_protocol = [t[15] for t in curr_txt['trial_statements']]
    
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
        print(f'{sessname}: no optogenetic stim. in this session')
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        opto_stim_trials = list(range(start_idx, end_idx, 3))
        opto_ctrl_trials = [i+2 for i in opto_stim_trials]
    
    baseline_dict = {key: [value[i] for i in baseline_trials if len(value)>1]
                     for key, value in curr_txt.items()}
    opto_stim_dict = {key: [value[i] for i in opto_stim_trials if len(value)>1] 
                      for key, value in curr_txt.items()}
    opto_ctrl_dict = {key: [value[i] for i in opto_ctrl_trials if len(value)>1] 
                      for key, value in curr_txt.items()}
    
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
    
    # get first licks
    ctrl_first_lick_time = np.median([licks[0] for licks in ctrl_lick_times if licks])
    stim_first_lick_time = np.median([licks[0] for licks in stim_lick_times if licks])
    ctrl_first_lick_distance = np.median([licks[0] for licks in ctrl_lick_distances if licks])
    stim_first_lick_distance = np.median([licks[0] for licks in stim_lick_distances if licks])
    
    if 2 < ctrl_first_lick_time < 10 and 2 < stim_first_lick_time < 10:
        ctrl_lick_times_030.append(ctrl_first_lick_time)
        stim_lick_times_030.append(stim_first_lick_time)
    
    if ctrl_first_lick_distance > 100 and stim_first_lick_distance > 100:
        ctrl_lick_distances_030.append(ctrl_first_lick_distance)
        stim_lick_distances_030.append(stim_first_lick_distance)
        
    # get combined lists
    ctrl_lick_dist_time = [t for trial in ctrl_lick_times for t in trial if 2<t<4]
    stim_lick_dist_time = [t for trial in stim_lick_times for t in trial if 2<t<4]
    ctrl_lick_dist_distance = [t for trial in ctrl_lick_distances for t in trial if 100<t<220]
    stim_lick_dist_distance = [t for trial in stim_lick_distances for t in trial if 100<t<220]
    
    # mean speed     
    speed_times_aligned = curr_txt['speed_times_aligned']
    ctrl_mean_speeds = []
    for trial in baseline_trials:
        speed_times = speed_times_aligned[trial]
        if speed_times:
            speeds = [s[1] for s in speed_times]
            ctrl_mean_speeds.append(np.mean(speeds))
    ctrl_mean_speeds_030.append(np.mean(ctrl_mean_speeds))
    
    stim_mean_speeds = []
    for trial in opto_stim_trials:
        speed_times = speed_times_aligned[trial]
        if speed_times:
            speeds = [s[1] for s in speed_times]
            stim_mean_speeds.append(np.mean(speeds))
    stim_mean_speeds_030.append(np.mean(stim_mean_speeds))
    
    # percent rewarded 
    reward_times = curr_txt['reward_times']
    ctrl_rewarded = [not np.isnan(reward_times[trial]) for trial in baseline_trials[1:]]
    ctrl_reward_perc = sum(ctrl_rewarded)/len(baseline_trials[1:])
    stim_rewarded = [not np.isnan(reward_times[trial]) for trial in opto_stim_trials]
    stim_reward_perc = sum(stim_rewarded)/len(opto_stim_trials)
    
    ctrl_perc_rew_030.append(ctrl_reward_perc)
    stim_perc_rew_030.append(stim_reward_perc)


for i, pathname in enumerate(pathLCopt):
    sessname = pathname[-17:]
    print(sessname)
    text_file = rf'{pathname}\{sessname}T.txt'
    curr_txt = process_behavioural_data(text_file)
    
    # segment the session based on optogenetic protocol
    optogenetic_protocol = [t[15] for t in curr_txt['trial_statements']]
    if '3' not in np.unique(optogenetic_protocol):
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
        print(f'{sessname}: no optogenetic stim. in this session')
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        opto_stim_trials = list(range(start_idx, end_idx, 3))
        opto_ctrl_trials = [i+2 for i in opto_stim_trials]
    
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
        ctrl_lick_times_030.append(ctrl_first_lick_time)
        stim_lick_times_030.append(stim_first_lick_time)
    
    if ctrl_first_lick_distance > 100 and stim_first_lick_distance > 100:
        ctrl_lick_distances_030.append(ctrl_first_lick_distance)
        stim_lick_distances_030.append(stim_first_lick_distance)
        
    # mean speed     
    speed_times_aligned = curr_txt['speed_times_aligned']
    ctrl_mean_speeds = []
    for trial in baseline_trials:
        speed_times = speed_times_aligned[trial]
        if speed_times:
            speeds = [s[1] for s in speed_times]
            ctrl_mean_speeds.append(np.mean(speeds))
    ctrl_mean_speeds_030.append(np.mean(ctrl_mean_speeds))
    
    stim_mean_speeds = []
    for trial in opto_stim_trials:
        speed_times = speed_times_aligned[trial]
        if speed_times:
            speeds = [s[1] for s in speed_times]
            stim_mean_speeds.append(np.mean(speeds))
    stim_mean_speeds_030.append(np.mean(stim_mean_speeds))
    
    # percent rewarded 
    reward_times = curr_txt['reward_times']
    ctrl_rewarded = [not np.isnan(reward_times[trial]) for trial in baseline_trials[1:]]
    ctrl_reward_perc = sum(ctrl_rewarded)/len(baseline_trials[1:])
    stim_rewarded = [not np.isnan(reward_times[trial]) for trial in opto_stim_trials]
    stim_reward_perc = sum(stim_rewarded)/len(opto_stim_trials)
    
    ctrl_perc_rew_030.append(ctrl_reward_perc)
    stim_perc_rew_030.append(stim_reward_perc)
            
        
for i, pathname in enumerate(pathHPCopt):
    sessname = pathname[-17:]
    print(sessname)
    text_file = rf'{pathname}\{sessname}T.txt'
    curr_txt = process_behavioural_data(text_file)
    
    # segment the session based on optogenetic protocol
    optogenetic_protocol = [t[15] for t in curr_txt['trial_statements']]
    if '3' not in np.unique(optogenetic_protocol):
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
        print(f'{sessname}: no optogenetic stim. in this session')
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        opto_stim_trials = list(range(start_idx, end_idx, 3))
        opto_ctrl_trials = [i+2 for i in opto_stim_trials]
    
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
        ctrl_lick_times_030.append(ctrl_first_lick_time)
        stim_lick_times_030.append(stim_first_lick_time)
    
    if ctrl_first_lick_distance > 100 and stim_first_lick_distance > 100:
        ctrl_lick_distances_030.append(ctrl_first_lick_distance)
        stim_lick_distances_030.append(stim_first_lick_distance)
        
    # mean speed     
    speed_times_aligned = curr_txt['speed_times_aligned']
    ctrl_mean_speeds = []
    for trial in baseline_trials:
        speed_times = speed_times_aligned[trial]
        if speed_times:
            speeds = [s[1] for s in speed_times]
            ctrl_mean_speeds.append(np.mean(speeds))
    ctrl_mean_speeds_030.append(np.mean(ctrl_mean_speeds))
    
    stim_mean_speeds = []
    for trial in opto_stim_trials:
        speed_times = speed_times_aligned[trial]
        if speed_times:
            speeds = [s[1] for s in speed_times]
            stim_mean_speeds.append(np.mean(speeds))
    stim_mean_speeds_030.append(np.mean(stim_mean_speeds))
    
    # percent rewarded 
    reward_times = curr_txt['reward_times']
    ctrl_rewarded = [not np.isnan(reward_times[trial]) for trial in baseline_trials[1:]]
    ctrl_reward_perc = sum(ctrl_rewarded)/len(baseline_trials[1:])
    stim_rewarded = [not np.isnan(reward_times[trial]) for trial in opto_stim_trials]
    stim_reward_perc = sum(stim_rewarded)/len(opto_stim_trials)
    
    ctrl_perc_rew_030.append(ctrl_reward_perc)
    stim_perc_rew_030.append(stim_reward_perc)

pf.plot_violin_with_scatter(
    ctrl_lick_times_030, stim_lick_times_030, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='first-lick time (s)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\030_lick_times',
    dpi=300
    )

pf.plot_violin_with_scatter(
    ctrl_lick_distances_030, stim_lick_distances_030, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='first-lick distance (cm)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\030_lick_distances',
    dpi=300
    )

pf.plot_violin_with_scatter(
    ctrl_mean_speeds_030, stim_mean_speeds_030, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='mean speed (cm/s)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\030_mean_speed',
    dpi=300
    )

rmv = []
for i in range(len(ctrl_perc_rew_030)):
    if ctrl_perc_rew_030[i] < 0.5 or stim_perc_rew_030[i] < 0.5:
        rmv.append(i)
ctrl_perc_rew_030 = [trial for i, trial in enumerate(ctrl_perc_rew_030) if i not in rmv]
stim_perc_rew_030 = [trial for i, trial in enumerate(stim_perc_rew_030) if i not in rmv]

pf.plot_violin_with_scatter(
    ctrl_perc_rew_030, stim_perc_rew_030, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='reward perc.',
    ylim=(.5, 1.03),
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\030_perc_rew',
    dpi=300
    )



#%% 040
all_opto_ctrl_num_licks_aft = [] 
all_opto_stim_num_licks_aft = []
for i, pathname in enumerate(pathLCBehopt):   
    sessname = pathname[-13:]
    curr_cond = condLCBehopt[i]
    curr_sess = sessLCBehopt[i]
    if str([0,4])[1:-1] not in str(curr_cond)[1:-1]:  # I have to commemmorate this as quite a clever trick
        print(curr_cond)
        print(f'{sessname} does not have rew-stim.')
        continue
    file_idx = curr_sess[curr_cond.index(4)]  # rew
    text_file = r'{}\{}-0{}\{}-0{}T.txt'.format(pathname, 
                                                sessname, file_idx, 
                                                sessname, file_idx)
    curr_txt = process_behavioural_data(text_file)
    
    # segment the session based on optogenetic protocol
    optogenetic_protocol = [t[15] for t in curr_txt['trial_statements']]
    
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
        print(f'{sessname}: no optogenetic stim. in this session')
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        opto_stim_trials = list(range(start_idx, end_idx, 3))
        opto_ctrl_trials = [i+2 for i in opto_stim_trials]
    
    baseline_dict = {key: [value[i] for i in baseline_trials if len(value)>1]
                     for key, value in curr_txt.items()}
    opto_stim_dict = {key: [value[i] for i in opto_stim_trials if len(value)>1] 
                      for key, value in curr_txt.items()}
    opto_ctrl_dict = {key: [value[i] for i in opto_ctrl_trials if len(value)>1] 
                      for key, value in curr_txt.items()}
    
    
    #### lick specific stuff, for commmittee meeting
    opto_stim_num_licks_aft = []
    for i, licks in enumerate(opto_stim_dict['lick_times']):
        pump_time = opto_stim_dict['reward_times'][i]
        opto_stim_num_licks_aft.append(
            len([l[0] for l in licks if l[0] > np.squeeze(pump_time)])
            )
    opto_ctrl_num_licks_aft = []
    for i, licks in enumerate(opto_ctrl_dict['lick_times']):
        pump_time = opto_ctrl_dict['reward_times'][i]
        opto_ctrl_num_licks_aft.append(
            len([l[0] for l in licks if l[0] > np.squeeze(pump_time)])
            )
    
    all_opto_ctrl_num_licks_aft.append(np.mean(opto_ctrl_num_licks_aft))
    all_opto_stim_num_licks_aft.append(np.mean(opto_stim_num_licks_aft))
    
pf.plot_violin_with_scatter(
    all_opto_ctrl_num_licks_aft, all_opto_stim_num_licks_aft, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='num. licks aft. reward',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\040_lick_aft_rew',
    dpi=300)


#%% 020 terminal 
ctrl_lick_times_020_term = []
stim_lick_times_020_term = []
    
for i, pathname in enumerate(pathHPCLCtermopt):
    sessname = pathname[-17:]
    text_file = rf'{pathname}\{sessname}T.txt'
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
        print(f'{sessname}: no optogenetic stim. in this session')
        baseline_trials = list(range(len(optogenetic_protocol)))
        optogenetic_trials = []
    else:
        baseline_trials = list(range(0, start_idx))
        opto_stim_trials = list(range(start_idx, end_idx, 3))
        opto_ctrl_trials = [i+2 for i in opto_stim_trials]
    
    baseline_dict = {key: [value[i] for i in baseline_trials if len(value)>1]
                     for key, value in curr_txt.items()}
    opto_stim_dict = {key: [value[i] for i in opto_stim_trials if len(value)>1] 
                      for key, value in curr_txt.items()}
    opto_ctrl_dict = {key: [value[i] for i in opto_ctrl_trials if len(value)>1] 
                      for key, value in curr_txt.items()}
    
    # extract datapoints 
    ctrl_lick_times = []
    stim_lick_times = []
    for trial, licks in enumerate(opto_ctrl_dict['lick_times']):
        start_time = np.squeeze(opto_ctrl_dict['run_onsets'][trial])
        ctrl_lick_times.append(
            [(l-start_time)/1000 for l in licks if l > start_time + 1000]
            )
    for trial, licks in enumerate(opto_stim_dict['lick_times']):
        start_time = np.squeeze(opto_stim_dict['run_onsets'][trial])
        stim_lick_times.append(
            [(l-start_time)/1000 for l in licks if l > start_time + 1000]
            )
    
    ctrl_first_lick_time = np.median([licks[0] for licks in ctrl_lick_times if licks])
    stim_first_lick_time = np.median([licks[0] for licks in stim_lick_times if licks])
    
    if ctrl_first_lick_time < 10 and stim_first_lick_time < 10:
        ctrl_lick_times_020_term.append(ctrl_first_lick_time)
        stim_lick_times_020_term.append(stim_first_lick_time)
        
pf.plot_violin_with_scatter(
    ctrl_lick_times_020_term, stim_lick_times_020_term, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='first-lick time (s)',
    save=False,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\020_lick_times',
    dpi=300)