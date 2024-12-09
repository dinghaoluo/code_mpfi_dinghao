# -*- coding: utf-8 -*-
"""
Created on Fri 6 Dec 17:45:35 2024

plot continuous trial traces 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os 
import sys 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import replace_outlier

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% parameters 
save_every = 5  # save every n trials


#%% load data 
exp_name = 'GRABNE'
df = pd.read_pickle(r'Z:/Dinghao/code_dinghao/behaviour/all_{}_sessions.pkl'.format(
    exp_name))


#%% main 
os.makedirs(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{}'.format(exp_name), 
            exist_ok=True)

for recname, row in df.iterrows():
    print(recname)
    os.makedirs(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{}\{}'.format(exp_name, recname), 
                exist_ok=True)
    
    speeds_rec = row['speed_times']
    tot_trial = len(speeds_rec)
    
    fig, ax = plt.subplots(figsize=(4,.75))
    speed_max = 0
    all_speeds = []  # combine speeds across trials to circumvent the 'missing connection' problem
    holdout_licks = []; holdout_rews = []
    for trial in range(tot_trial):
        if np.mod(trial, save_every)==0 and trial!=0:  # if 5 trials have been held, plot, save and close the current fig and create another fig 
            all_speeds = np.vstack(all_speeds)
            times = all_speeds[:,0]/1000
            speeds = replace_outlier(all_speeds[:,1])
            speed_max = max(speed_max, max(speeds))
            ax.plot(times, speeds, linewidth=1, color='k')
        
            for lick in holdout_licks:
                ax.vlines(x=lick, ymin=speed_max+speed_max*.01, ymax=speed_max+speed_max*.1, 
                          linewidth=.5, color='orchid')
            for rew in holdout_rews:
                ax.vlines(x=rew, ymin=speed_max+speed_max*.01, ymax=speed_max+speed_max*.1, 
                          linewidth=.5, color='darkgreen')
            
            for s in ['top','right']: ax.spines[s].set_visible(False)
            ax.set(xlabel='time (s)',
                   ylabel='speed (cm/s)', ylim=(0, speed_max+speed_max*.11),
                   title='{}_{}'.format(trial-(save_every-1), trial))
            for ext in ['.png', '.pdf']:
                fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{}\{}\{}{}'.format(
                    exp_name, recname, f'{trial-(save_every-1)}_{trial}', ext
                    ), 
                    dpi=300,
                    bbox_inches='tight')
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(4,.75))
            speed_max = 0
            all_speeds = []
            holdout_licks = []; holdout_rews = []
                
        # combine speeds
        curr_speeds = np.asarray(speeds_rec[trial])
        all_speeds.append(curr_speeds)
        
        curr_start = row['run_onsets'][trial]/1000
        ax.axvline(x=curr_start, linewidth=1, color='red', alpha=.8)
        
        if not not(row['lick_times'][trial]):  # if the current trial has any lick 
            curr_licks = np.asarray(row['lick_times'][trial])/1000
            holdout_licks = np.concatenate((holdout_licks, curr_licks))
        
        if not not(row['reward_times'][trial]):  # if the current trial has a reward 
            curr_rew = row['reward_times'][trial][0]/1000
            holdout_rews.append(curr_rew)