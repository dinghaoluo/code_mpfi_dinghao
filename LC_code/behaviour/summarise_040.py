# -*- coding: utf-8 -*-
"""
Created on Wed 13 Nov 16:53:32 2024

summarise pharmacological experiments with SCH23390

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import sys

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()

from behaviour_processing import process_behavioural_data

if (r'Z:\Dinghao\code_mpfi_dinghao\iutils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf


#%% recording list
sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLCBehopt = rec_list.pathLCBehopt
sessLCBehopt = rec_list.sessLCBehopt
condLCBehopt = rec_list.condLCBehopt


#%% main
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
    optogenetic_protocol = curr_txt['optogenetic_protocols']
    
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
    
    baseline_dict = {key: [value[i] for i in baseline_trials] for key, value in curr_txt.items()}
    opto_stim_dict = {key: [value[i] for i in opto_stim_trials] for key, value in curr_txt.items()}
    opto_ctrl_dict = {key: [value[i] for i in opto_ctrl_trials] for key, value in curr_txt.items()}
    
    
    #### lick specific stuff, for commmittee meeting
    opto_stim_num_licks_aft = []
    for i, licks in enumerate(opto_stim_dict['lick_times']):
        pump_time = opto_stim_dict['reward_times'][i]
        opto_stim_num_licks_aft.append(len([l for l in licks if l > pump_time]))
    opto_ctrl_num_licks_aft = []
    for i, licks in enumerate(opto_ctrl_dict['lick_times']):
        pump_time = opto_ctrl_dict['reward_times'][i]
        opto_ctrl_num_licks_aft.append(len([l for l in licks if l > pump_time]))
    
    all_opto_ctrl_num_licks_aft.append(np.mean(opto_ctrl_num_licks_aft))
    all_opto_stim_num_licks_aft.append(np.mean(opto_stim_num_licks_aft))
    
    

#%% plotting 
pf.plot_violin_with_scatter(all_opto_ctrl_num_licks_aft, all_opto_stim_num_licks_aft, 
                            'grey', 'royalblue',
                            xticklabels=['ctrl.', 'stim.'],
                            save=True,
                            savepath=r'Z:\Dinghao\code_dinghao\LC_opto_ephys\040_lick_aft_rew',
                            dpi=300)