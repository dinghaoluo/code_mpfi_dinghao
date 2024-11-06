# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:45:35 2024

plot trial-profiles for LC animals 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os 

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load data 
df = pd.read_pickle(r'Z:/Dinghao/code_dinghao/behaviour/all_LC_sessions.pkl')


#%% main 
for recname, row in df.iterrows():
    print(recname)
    
    os.makedirs(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{}'.format(recname), exist_ok=True)
    
    speeds = row['speed_times']
    tot_trial = len(speeds)
    
    fig, ax = plt.subplots(figsize=(5,.75))
    speed_max = 0
    holdout_licks = []; holdout_rews = []
    for trial in range(tot_trial):
        if np.mod(trial, 10)==0 and trial!=0:    # if 5 trials have been plotted, save and close the current fig and create another fig 
            for lick in holdout_licks:
                ax.vlines(x=lick, ymin=speed_max+speed_max*.01, ymax=speed_max+speed_max*.1, linewidth=.5, color='orchid')
            for rew in holdout_rews:
                ax.vlines(x=rew, ymin=speed_max+speed_max*.01, ymax=speed_max+speed_max*.1, linewidth=.5, color='darkgreen')
            for s in ['top','right']: ax.spines[s].set_visible(False)
            ax.set(xlabel='time (s)',
                   ylabel='speed (cm/s)', ylim=(0, speed_max+speed_max*.11),
                   title='{}_{}'.format(trial-9, trial))
            fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{}\{}.png'.format(recname, '{}_{}'.format(trial-9, trial)),
                        dpi=200, bbox_inches='tight')
            fig.savefig(r'Z:\Dinghao\code_dinghao\behaviour\trial_profiles\{}\{}.pdf'.format(recname, '{}_{}'.format(trial-9, trial)),
                        bbox_inches='tight')
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(5,.75))
            speed_max = 0
            holdout_licks = []; holdout_rews = []
        
        curr_speeds = np.asarray(speeds[trial])
        speed_max = max([speed_max, max(curr_speeds[:,1])])
        ax.plot(curr_speeds[:,0]/1000, curr_speeds[:,1], linewidth=1, color='k')
        
        curr_start = row['run_onsets'][trial]/1000
        ax.axvline(x=curr_start, linewidth=1, color='red', alpha=.8)
        
        if not not(row['lick_times'][trial]):  # if the current trial has any lick 
            curr_licks = np.asarray(row['lick_times'][trial])[:,0]/1000
            holdout_licks = np.concatenate((holdout_licks, curr_licks))
        
        if not not(row['pumps'][trial]):  # if the current trial has a reward 
            curr_rew = row['pumps'][trial]/1000
            holdout_rews.append(curr_rew)