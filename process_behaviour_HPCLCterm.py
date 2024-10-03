# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:35:35 2024

process and save behaviour files as dataframes
modified for HPCLCterm

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os
import pandas as pd

# import pre-processing functions 
sys.path.extend([r'Z:\Dinghao\code_mpfi_dinghao\utils', r'Z:\Dinghao\code_dinghao'])
import txt_processing_functions as tpf
import behaviour_functions as bf
import rec_list
pathHPC = rec_list.pathHPCLCtermopt


#%% container
sess = {'run_onsets': [],
        'pumps': [],
        'cues': [],
        'speed_times': [],
        'lick_times': [],
        'lick_distances': [],
        'lick_selectivity': [],
        'pulse_times': [],
        'pulse_descriptions': [],
        'trial_statements': [],
        'no_full_stop':[]}

df = pd.DataFrame(sess)


#%% main 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname+'\n')
    
    txt_path = r'Z:\Dinghao\MiceExp\ANMD{}\{}\{}\{}T.txt'.format(recname[1:5], recname[:-3], recname, recname)

    txt = tpf.process_txt(txt_path)
    
    ## timestamps
    pump_times, speed_times, movie_times, lick_times, pulse_times, pulse_descriptions, trial_statements = [
        txt[field] for field in 
        ['pump_times', 'speed_times', 'movie_times', 'lick_times', 'pulse_times', 'pulse_descriptions', 'trial_statements']
    ]
    
    tot_trial = len(speed_times)
    
    ## correct overflow
    pump_times = tpf.correct_overflow(pump_times, 'pump')
    speed_times = tpf.correct_overflow(speed_times, 'speed')
    movie_times = tpf.correct_overflow(movie_times, 'movie')
    lick_times = tpf.correct_overflow(lick_times, 'lick')
    pulse_times = tpf.correct_overflow(pulse_times, 'pulse')
    trial_statements = tpf.correct_overflow(trial_statements, 'trial_statement')
    first_time = speed_times[0][0][0]; last_time = speed_times[-1][-1][0]
    
    ## find run-onsets 
    speeds = []
    times = []
    for t in speed_times: # concatenate speeds of all trials in a session
        speeds.extend([p[1] for p in t])
        times.extend([p[0] for p in t])
    all_uni_time = np.linspace(first_time, last_time, int(last_time-first_time))
    all_uni_speed = np.interp(all_uni_time, times, speeds)
    all_uni_time = list(all_uni_time)
    
    run_onsets = []  # find the time point where speed continiously > 10 cm/s (after cue) first
    no_full_stop = []  # whether there is any point at which speed is lower then 10 cm/s
    last_rew = []  # last reward timepoint
    for trial in range(tot_trial):
        if trial==0:
            last_rew.append(first_time)  # initiate as the beginning of trial 0
        else:
            last_rew.append(pump_times[trial-1]+250)  # +250 ms to exclude situations where the reward does not immediately stop the animal
        curr_ITI_start = tpf.find_nearest(last_rew[trial], all_uni_time)
        curr_trial_end = tpf.find_nearest(pump_times[trial], all_uni_time)
        curr_time = all_uni_time[curr_ITI_start:curr_trial_end]
        curr_speed = all_uni_speed[curr_ITI_start:curr_trial_end]
        run_onsets.append(tpf.get_onset(curr_speed, curr_time))
        no_full_stop.append((curr_speed>10).all())
        
        # plotting for testing
        fig, ax = plt.subplots(figsize=(2,1))
        ax.plot([t/1000 for t in curr_time], curr_speed)
        ax.vlines(run_onsets[trial]/1000, 0, 50, 'orange')
        
        filename = r'Z:\Dinghao\code_dinghao\behaviour\single_trial_run_onset_detection\{}\t{}.png'.format(recname, trial)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=80, bbox_inches='tight')
        
        plt.close(fig)
                             
    lick_locs = bf.get_lick_locs(speed_times, lick_times)
    lick_selectivity = bf.lick_index(lick_locs)
        
    df.loc[recname] = np.asarray([run_onsets, 
                                  pump_times,
                                  movie_times,
                                  speed_times,
                                  lick_times,
                                  lick_locs,
                                  lick_selectivity,
                                  pulse_times,
                                  pulse_descriptions,
                                  trial_statements,
                                  no_full_stop,  # Jingyu, 9/27/2024
                                      ],
                                  dtype='object')


#%% save dataframe 
df.to_csv(r'Z:\Dinghao\code_dinghao\behaviour\all_HPCLCterm_sessions.csv')
df.to_pickle(r'Z:\Dinghao\code_dinghao\behaviour\all_HPCLCterm_sessions.pkl')