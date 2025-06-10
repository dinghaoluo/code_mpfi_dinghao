# -*- coding: utf-8 -*-
"""
Created on Wed 13 Nov 16:53:32 2024
Modified on Thur 6 Mar 2025

summarise optogenetic experiments

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import sys

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

for i, pathname in enumerate(pathLCBehopt):   
    sessname = pathname[-13:]
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
    for trial, licks in enumerate(opto_ctrl_dict['lick_times']):
        start_time = np.squeeze(opto_ctrl_dict['run_onsets'][trial])
        ctrl_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(opto_stim_dict['lick_times']):
        start_time = np.squeeze(opto_stim_dict['run_onsets'][trial])
        stim_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(opto_ctrl_dict['lick_distances_aligned']):
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
    
    if ctrl_first_lick_time < 10 and stim_first_lick_time < 10:
        ctrl_lick_times_020.append(ctrl_first_lick_time)
        stim_lick_times_020.append(stim_first_lick_time)
    
    if ctrl_first_lick_distance > 30 and stim_first_lick_distance > 30:
        ctrl_lick_distances_020.append(ctrl_first_lick_distance)
        stim_lick_distances_020.append(stim_first_lick_distance)


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
    for trial, licks in enumerate(opto_ctrl_dict['lick_times']):
        start_time = np.squeeze(opto_ctrl_dict['run_onsets'][trial])
        ctrl_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(opto_stim_dict['lick_times']):
        start_time = np.squeeze(opto_stim_dict['run_onsets'][trial])
        stim_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(opto_ctrl_dict['lick_distances_aligned']):
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
    
    if ctrl_first_lick_time < 10 and stim_first_lick_time < 10:
        ctrl_lick_times_020.append(ctrl_first_lick_time)
        stim_lick_times_020.append(stim_first_lick_time)
    
    if ctrl_first_lick_distance > 100 and stim_first_lick_distance > 100:
        ctrl_lick_distances_020.append(ctrl_first_lick_distance)
        stim_lick_distances_020.append(stim_first_lick_distance)
            
        
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
    for trial, licks in enumerate(opto_ctrl_dict['lick_times']):
        start_time = np.squeeze(opto_ctrl_dict['run_onsets'][trial])
        ctrl_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(opto_stim_dict['lick_times']):
        start_time = np.squeeze(opto_stim_dict['run_onsets'][trial])
        stim_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(opto_ctrl_dict['lick_distances_aligned']):
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
    
    if ctrl_first_lick_time < 10 and stim_first_lick_time < 10:
        ctrl_lick_times_020.append(ctrl_first_lick_time)
        stim_lick_times_020.append(stim_first_lick_time)
    
    if ctrl_first_lick_distance > 100 and stim_first_lick_distance > 100:
        ctrl_lick_distances_020.append(ctrl_first_lick_distance)
        stim_lick_distances_020.append(stim_first_lick_distance)
    
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


#%% 030 
ctrl_lick_times_030 = []
stim_lick_times_030 = []
ctrl_lick_distances_030 = []
stim_lick_distances_030 = []

for i, pathname in enumerate(pathLCBehopt):   
    sessname = pathname[-13:]
    print(sessname)
    curr_cond = condLCBehopt[i]
    curr_sess = sessLCBehopt[i]
    if str([0,3,0])[1:-1] not in str(curr_cond)[1:-1]:  # I have to commemmorate this as quite a clever trick
        continue
    file_idx = curr_sess[curr_cond.index(3)]  # midtrial
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
    for trial, licks in enumerate(opto_ctrl_dict['lick_times']):
        start_time = np.squeeze(opto_ctrl_dict['run_onsets'][trial])
        ctrl_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(opto_stim_dict['lick_times']):
        start_time = np.squeeze(opto_stim_dict['run_onsets'][trial])
        stim_lick_times.append(
            [(l[0]-start_time)/1000 for l in licks if l[0] > start_time + 1000]
            )
    for trial, licks in enumerate(opto_ctrl_dict['lick_distances_aligned']):
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
    
    if ctrl_first_lick_time < 10 and stim_first_lick_time < 10:
        ctrl_lick_times_030.append(ctrl_first_lick_time)
        stim_lick_times_030.append(stim_first_lick_time)
    
    if ctrl_first_lick_distance > 100 and stim_first_lick_distance > 100:
        ctrl_lick_distances_030.append(ctrl_first_lick_distance)
        stim_lick_distances_030.append(stim_first_lick_distance)

pf.plot_violin_with_scatter(
    ctrl_lick_times_030, stim_lick_times_030, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='first-lick time (s)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\030_lick_times',
    dpi=300)

pf.plot_violin_with_scatter(
    ctrl_lick_distances_030, stim_lick_distances_030, 
    'grey', 'royalblue',
    xticklabels=['ctrl.', 'stim.'],
    ylabel='first-lick distance (s)',
    save=True,
    savepath=r'Z:\Dinghao\code_dinghao\behaviour\LC_opto\030_lick_distances',
    dpi=300)


#%% 040
all_opto_ctrl_num_licks_aft = [] 
all_opto_stim_num_licks_aft = []
for i, pathname in enumerate(pathLCBehopt):   
    sessname = pathname[-13:]
    curr_cond = condLCBehopt[i]
    curr_sess = sessLCBehopt[i]
    if str([0,4,0])[1:-1] not in str(curr_cond)[1:-1]:  # I have to commemmorate this as quite a clever trick
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
            len([l for l in licks if l > np.squeeze(pump_time)])
            )
    opto_ctrl_num_licks_aft = []
    for i, licks in enumerate(opto_ctrl_dict['lick_times']):
        pump_time = opto_ctrl_dict['reward_times'][i]
        opto_ctrl_num_licks_aft.append(
            len([l for l in licks if l > np.squeeze(pump_time)])
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