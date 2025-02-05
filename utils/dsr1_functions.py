# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:25:06 2025

using local DeepSeek-R1 implementation to help with a bunch of things

@author: Dinghao Luo
"""


#%% imports 
import ollama 
import re 


#%% main chat function
def chat(message, use_abliterated=True):
    if use_abliterated:
        model = 'huihui_ai/deepseek-r1-abliterated:8b-llama-distill'
    else:
        model = 'deepseek_r1:8b'
    
    response = ollama.chat(
        model = model,
        messages = [
            {'role': 'user',
             'content': message}
        ]
        )
    
    # extract the content
    content = response['message']['content']
    
    # Parse <think> tags for thought process and remaining text for output
    thought_process = '\n'.join(re.findall(r'<think>(.*?)</think>', content, re.DOTALL))
    output = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    return thought_process, output


#%% functions 
def generate_docstring(code):
    message = f'''
    Please generate a docstring for the code below:
    \n{code}
    \nStrictly follow these guidelines: use LOWER case letters for the start of a sentence and use British spellings
    \nOne example:
        processes behavioural data from a txt file, aligning speed, lick, and reward events 
        to both time and distance bases, while extracting metrics such as run onsets, 
        lick selectivity, trial quality, and full stop status.
        
        parameters:
        ----------
        txtfile : str
            path to the txt file containing behavioural data.
        max_distance : int, optional
            maximum distance for aligning data across trials (default is 220 cm).
        distance_resolution : int, optional
            distance step size in cm for interpolated distance base (default is 1 cm).
        run_onset_initial : float, optional
            initial speed threshold for detecting run onset (default is 3.0 cm/s).
        run_onset_sustained : float, optional
            sustained speed threshold for detecting run onset (default is 10.0 cm/s).
        run_onset_duration : int, optional
            duration (in ms) for which the sustained speed threshold must be held 
            to confirm run onset (default is 300 ms).
        
        returns:
        -------
        dict
            a dictionary containing aligned behavioural data across time and distance bases, including:
            - 'speed_times': list of lists, each containing [timestamp, speed] pairs for each trial.
            - 'speed_distances': list of arrays, each containing speeds aligned to a common distance base.
            - 'lick_times': list of lists, each containing lick event timestamps for each trial.
            - 'lick_distances': list of arrays, each containing lick events aligned to the distance base.
            - 'lick_maps': list of arrays, each binary array representing lick events mapped onto the distance base.
            - 'start_cue_times': list of lists, each containing timestamps of start cues.
            - 'reward_times': list of lists, each containing timestamps of reward deliveries.
            - 'reward_distance': list of arrays, each containing reward events aligned to the distance base.
            - 'run_onsets': list of timestamps for detected run-onset events in each trial.
            - 'lick_selectivities': list of float values, one for each trial, representing lick selectivity indices.
            - 'trial_statements': list of lists containing trial-specific metadata and protocols.
            - 'full_stops': list of booleans indicating whether the animal fully stopped (speed < 10 cm/s) 
              between the previous trial's reward and the current trial's reward.
            - 'bad_trials': list of booleans indicating whether each trial was classified as "bad" based on:
                1. Presence of licks between 30 and 90 cm.
                2. Speeds below 10 cm/s for a cumulative duration >5 seconds between run-onset and reward.
                3. Lack of a detected run-onset or delivered reward.
            - 'frame_times': list of timestamps for each movie frame, corrected for overflow.
    '''
    
    return chat(message)[1]



generate_docstring('''line = file.readline().rstrip('\n').split(',')
if len(line) == 1: # read an empty line
    line = file.readline().rstrip('\n').split(',')
return line''')