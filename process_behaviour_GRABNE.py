# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:35:35 2024

process and save behaviour files as dataframes

@author: Dinghao Luo
@modifiers: Dinghao Luo, Jingyu Cao
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os
import pandas as pd

# import pre-processing functions 
if (r'Z:\Dinghao\code_mpfi_dinghao\utils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import txt_processing_functions as tpf
import behaviour_functions as bf


#%% recording list
if r'Z:\Dinghao\code_dinghao' not in sys.path:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE


#%% container
sess = {'run_onsets': [],
        'run_onset_frames': [],
        'pumps': [],
        'pump_frames': [],
        'cues': [],
        'cue_frames': [],
        'speed_times': [],
        'lick_times': [],
        'lick_distances': [],
        'lick_selectivity': [],
        'pulse_times': [],
        'pulse_descriptions': [],
        'trial_statements': [],
        'frame_times': [],
        'no_full_stop':[]}  # Jingyu, 9/27/2024

df = pd.DataFrame(sess)


#%% main 
for pathname in pathGRABNE:
    recname = pathname[-17:]
    print(recname+'\n')
    
    txt_path = r'Z:\Dinghao\MiceExp\ANMD{}\{}{}T.txt'.format(recname[1:4], recname[:4], recname[5:])

    txt = tpf.process_txt(txt_path)
    
    ## timestamps
    pump_times = txt['pump_times']  # pumps 
    speed_times = txt['speed_times']  # speeds 
    movie_times = txt['movie_times']  # cues 
    lick_times = txt['lick_times']  # licks 
    pulse_times = txt['pulse_times']  # pulses
    pulse_descriptions = txt['pulse_descriptions']
    trial_statements = txt['trial_statements']
    frame_times = txt['frame_times']  # frames 
    
    tot_trial = len(speed_times)
    tot_frame = len(frame_times)
    
    ## correct overflow
    pump_times = tpf.correct_overflow(pump_times, 'pump')
    speed_times = tpf.correct_overflow(speed_times, 'speed')
    movie_times = tpf.correct_overflow(movie_times, 'movie')
    lick_times = tpf.correct_overflow(lick_times, 'lick')
    pulse_times = tpf.correct_overflow(pulse_times, 'pulse')
    trial_statements = tpf.correct_overflow(trial_statements, 'trial_statement')
    frame_times = tpf.correct_overflow(frame_times, 'frame')
    first_frame = frame_times[0]; last_frame = frame_times[-1]
    first_time = speed_times[0][0][0]; last_time = speed_times[-1][-1][0]
    
    ## **fill in dropped $FM signals
    # since the 2P system does not always successfully transmit frame signals to
    # the behavioural recording system every time it acquires a frame, one needs to
    # manually interpolate the frame signals in between 2 frame signals that are 
    # further apart than 50 ms
    for i in range(len(frame_times)-1):
        if frame_times[i+1]-frame_times[i]>50:
            interp_fm = (frame_times[i+1]+frame_times[i])/2
            frame_times.insert(i+1, interp_fm)
    
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
            last_rew.append(pump_times[trial-1]+100)  # +100 ms to exclude situations where the reward does not immediately stop the animal
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
                         
      
    ## determine frames for each variable     
    run_onset_frames = []
    for trial in run_onsets:
        if trial<first_frame or trial>last_frame:
            run_onset_frames.append(-1)
        elif trial!=-1:  # if there is a clear run-onset in this trial
            rf = tpf.find_nearest(trial, frame_times)
            if rf!=0:
                run_onset_frames.append(rf)  # find the nearest frame
            else:
                run_onset_frames.append(-1)
        else:
            run_onset_frames.append(-1)
    pump_frames = [] 
    for trial in pump_times:
        if trial==0: #Unrewarded trials, Jingyu, 20240730
            pump_frames.append(-1)
        elif trial<first_frame or trial>last_frame:
            pump_frames.append(-1)
        else:
            pf = tpf.find_nearest(trial, frame_times)
            if pf!=0:
                pump_frames.append(pf)
            # else:
            #     pump_frames.append(-1)
    cue_frames = []
    for trial in movie_times:
        if trial[0][0]<first_frame or trial[0][0]>last_frame:
            cue_frames.append(-1)
        else:
            cf = tpf.find_nearest(trial[0][0], frame_times)
            if cf!=0:
                cue_frames.append(cf)
            else:
                cue_frames.append(-1)
        
    lick_locs = bf.get_lick_locs(speed_times, lick_times)
    lick_selectivity = bf.lick_index(lick_locs)
        
    df.loc[recname] = np.asarray([run_onsets, 
                                  run_onset_frames,
                                  pump_times,
                                  pump_frames,
                                  movie_times,
                                  cue_frames,
                                  speed_times,
                                  lick_times,
                                  lick_locs,
                                  lick_selectivity,
                                  pulse_times,
                                  pulse_descriptions,
                                  trial_statements,
                                  frame_times,
                                  no_full_stop,  # Jingyu, 9/27/2024
                                      ],
                                  dtype='object')


#%% save dataframe 
df.to_csv(r'Z:\Dinghao\code_dinghao\behaviour\all_GRABNE_sessions.csv')
df.to_pickle(r'Z:\Dinghao\code_dinghao\behaviour\all_GRABNE_sessions.pkl')
