# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:28:55 2025

MODIFIED from stim_ctrl_all_pyr_decay_time.py to work on pooled ON and OFF cells
analyse the decay time constants of pyramidal cells in stim. vs ctrl. trials 

@author: Dinghao Luo
"""


#%% imports 
from pathlib import Path

import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.stats import sem, linregress

from plotting_functions import plot_violin_with_scatter
from common import mpl_formatting
mpl_formatting()

import rec_list
pathHPCLCopt = rec_list.pathHPCLCopt
pathHPCLCtermopt = rec_list.pathHPCLCtermopt


#%% parameters 
SAMP_FREQ = 1250  # Hz
TIME = np.arange(-SAMP_FREQ*3, SAMP_FREQ*7)/SAMP_FREQ  # 10 seconds 

MAX_TIME = 10  # collect (for each trial) a maximum of 10 s of spiking-profile
MAX_SAMPLES = SAMP_FREQ * MAX_TIME

XAXIS = np.arange(-1*1250, 4*1250) / 1250


#%% path stems 
all_sess_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/all_sessions')
all_beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments')
ctrl_stim_stem = Path('Z:/Dinghao/code_dinghao/HPC_ephys/run_onset_response/ctrl_stim_pooled')


#%% load data 
print('loading dataframes...')

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
    )
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 


#%% main 
def main(paths, exp='HPCLC'):
    mean_prof_ctrl_ON = []
    mean_prof_stim_ON = []
    mean_prof_ctrl_OFF = []
    mean_prof_stim_OFF = []
    
    # for pooling first and then averaging across single neurones 
    mean_prof_ctrl_ON_single_cell = []
    mean_prof_stim_ON_single_cell = []
    mean_prof_ctrl_OFF_single_cell = []
    mean_prof_stim_OFF_single_cell = []
    
    all_ctrl_stim_lick_time_delta = []
    all_ctrl_stim_lick_distance_delta = []
    
    all_amp_ON_delta_sum = []
    all_amp_ON_delta_mean = []
    
    all_amp_OFF_delta_sum = []
    all_amp_OFF_delta_mean = []
    
    ## ACCUMULATE DATA ##
    for path in paths:
        recname = Path(path).name
        print(f'\n{recname}')
        
        train_path = all_sess_stem / recname / f'{recname}_all_trains.npy'
        trains = np.load(train_path, allow_pickle=True).item()
        
        if (all_beh_stem / 'HPCLC' / f'{recname}.pkl').exists():
            with open(all_beh_stem / 'HPCLC' / f'{recname}.pkl', 'rb') as f:
                beh = pickle.load(f)
        else:
            with open(all_beh_stem / 'HPCLCterm' / f'{recname}.pkl', 'rb') as f:
                beh = pickle.load(f)
        
        stim_conds = [t[15] for t in beh['trial_statements']][1:]
        stim_idx = [trial for trial, cond in enumerate(stim_conds)
                    if cond!='0']
        ctrl_idx = [trial+2 for trial in stim_idx]
        
        first_lick_times = [t[0][0] - s for t, s
                            in zip(beh['lick_times'], beh['run_onsets'])
                            if t]
        ctrl_stim_lick_time_delta = np.mean(
            [t for i, t in enumerate(first_lick_times) if i in stim_idx]
            ) - np.mean(
                [t for i, t in enumerate(first_lick_times) if i in ctrl_idx]
                )
        all_ctrl_stim_lick_time_delta.append(ctrl_stim_lick_time_delta)
                
        first_lick_distances = [t[0]
                                if type(t)!=float and len(t)>0
                                else np.nan
                                for t 
                                in beh['lick_distances_aligned']
                                ][1:]
        ctrl_stim_lick_distance_delta = np.mean(
            [t for i, t in enumerate(first_lick_distances) if i in stim_idx]
            ) - np.mean(
                [t for i, t in enumerate(first_lick_distances) if i in ctrl_idx]
                )
        all_ctrl_stim_lick_distance_delta.append(ctrl_stim_lick_distance_delta)
        
        curr_df_pyr = df_pyr[df_pyr['recname']==recname]
    
        ## ACCUMULATE OVER CLUS ##
        curr_prof_ctrl_ON = []
        curr_prof_stim_ON = []
        curr_prof_ctrl_OFF = []
        curr_prof_stim_OFF = []
        
        for idx, session in curr_df_pyr.iterrows():
            cluname = idx    
            
            ## CTRL OR STIM ON ##
            if session['class_ctrl']=='run-onset ON' or session['class_stim']=='run-onset ON':
                
                mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
                mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
                
                curr_prof_ctrl_ON.append(mean_prof_ctrl)
                curr_prof_stim_ON.append(mean_prof_stim)
                
                mean_prof_ctrl_ON_single_cell.append(mean_prof_ctrl)
                mean_prof_stim_ON_single_cell.append(mean_prof_stim)
            
            ## CTRL OR STIM OFF ##
            if session['class_ctrl']=='run-onset OFF' or session['class_stim']=='run-onset OFF':
                
                mean_prof_ctrl = np.mean(trains[cluname][ctrl_idx], axis=0)
                mean_prof_stim = np.mean(trains[cluname][stim_idx], axis=0)
                
                curr_prof_ctrl_OFF.append(mean_prof_ctrl)
                curr_prof_stim_OFF.append(mean_prof_stim)
                
                mean_prof_ctrl_OFF_single_cell.append(mean_prof_ctrl)
                mean_prof_stim_OFF_single_cell.append(mean_prof_stim)
        
        
        ## APPEND TO LISTS ##
        if len(curr_prof_ctrl_ON) > 0:
            sess_mean_ctrl_ON = np.mean(curr_prof_ctrl_ON, axis=0)
            mean_prof_ctrl_ON.append(sess_mean_ctrl_ON)
        if len(curr_prof_stim_ON) > 0:
            sess_mean_stim_ON = np.mean(curr_prof_stim_ON, axis=0)
            mean_prof_stim_ON.append(sess_mean_stim_ON)
            
        if len(curr_prof_ctrl_OFF) > 0:
            sess_mean_ctrl_OFF = np.mean(curr_prof_ctrl_OFF, axis=0)
            mean_prof_ctrl_OFF.append(sess_mean_ctrl_OFF)
        if len(curr_prof_stim_OFF) > 0:
            sess_mean_stim_OFF = np.mean(curr_prof_stim_OFF, axis=0)
            mean_prof_stim_OFF.append(sess_mean_stim_OFF)
            
        if len(curr_prof_ctrl_ON) > 0 and len(curr_prof_stim_ON) > 0:
            amp_ON_delta_sum = np.sum(sess_mean_stim_ON[3750+625:3750+1825]) - np.mean(sess_mean_ctrl_ON[3750+625:3750+1825])
            all_amp_ON_delta_sum.append(amp_ON_delta_sum)
            amp_ON_delta_mean = np.mean(sess_mean_stim_ON[3750+625:3750+1825]) - np.mean(sess_mean_ctrl_ON[3750+625:3750+1825])
            all_amp_ON_delta_mean.append(amp_ON_delta_mean)
        
        if len(curr_prof_ctrl_OFF) > 0 and len(curr_prof_stim_OFF) > 0:
            amp_OFF_delta_sum = np.sum(sess_mean_stim_OFF[3750+625:3750+1825]) - np.mean(sess_mean_ctrl_OFF[3750+625:3750+1825])
            all_amp_OFF_delta_sum.append(amp_OFF_delta_sum)
            amp_OFF_delta_mean = np.mean(sess_mean_stim_OFF[3750+625:3750+1825]) - np.mean(sess_mean_ctrl_OFF[3750+625:3750+1825])
            all_amp_OFF_delta_mean.append(amp_OFF_delta_mean)
    ## SINGLE-SESSION PROCESSING ENDS ##
    
     
    ## EXTRACTION ##
    mean_ctrl_ON = np.mean(mean_prof_ctrl_ON, axis=0)[2500:2500+5*1250]
    sem_ctrl_ON = sem(mean_prof_ctrl_ON, axis=0)[2500:2500+5*1250]
    
    mean_stim_ON = np.mean(mean_prof_stim_ON, axis=0)[2500:2500+5*1250]
    sem_stim_ON = sem(mean_prof_stim_ON, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_OFF = np.mean(mean_prof_ctrl_OFF, axis=0)[2500:2500+5*1250]
    sem_ctrl_OFF = sem(mean_prof_ctrl_OFF, axis=0)[2500:2500+5*1250]
    
    mean_stim_OFF = np.mean(mean_prof_stim_OFF, axis=0)[2500:2500+5*1250]
    sem_stim_OFF = sem(mean_prof_stim_OFF, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_ON_single_cell = np.mean(mean_prof_ctrl_ON_single_cell, axis=0)[2500:2500+5*1250]
    sem_ctrl_ON_single_cell = sem(mean_prof_ctrl_ON_single_cell, axis=0)[2500:2500+5*1250]
    
    mean_stim_ON_single_cell = np.mean(mean_prof_stim_ON_single_cell, axis=0)[2500:2500+5*1250]
    sem_stim_ON_single_cell = sem(mean_prof_stim_ON_single_cell, axis=0)[2500:2500+5*1250]
    
    mean_ctrl_OFF_single_cell = np.mean(mean_prof_ctrl_OFF_single_cell, axis=0)[2500:2500+5*1250]
    sem_ctrl_OFF_single_cell = sem(mean_prof_ctrl_OFF_single_cell, axis=0)[2500:2500+5*1250]
    
    mean_stim_OFF_single_cell = np.mean(mean_prof_stim_OFF_single_cell, axis=0)[2500:2500+5*1250]
    sem_stim_OFF_single_cell = sem(mean_prof_stim_OFF_single_cell, axis=0)[2500:2500+5*1250]
    ## EXTRACTION ENDS ##
    
    
    ## PLOTTING ALL TRACES ##    
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_ON, label='mean_ctrl_ON', color='lightcoral')
    ax.fill_between(XAXIS, mean_ctrl_ON + sem_ctrl_ON, mean_ctrl_ON - sem_ctrl_ON,
                    color='lightcoral', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_ON, label='mean_stim_ON', color='firebrick')
    ax.fill_between(XAXIS, mean_stim_ON + sem_stim_ON, mean_stim_ON - sem_stim_ON,
                    color='firebrick', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_ctrl_OFF, label='mean_ctrl_OFF', color='violet')
    ax.fill_between(XAXIS, mean_ctrl_OFF + sem_ctrl_OFF, mean_ctrl_OFF - sem_ctrl_OFF,
                    color='violet', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_OFF, label='mean_stim_OFF', color='purple')
    ax.fill_between(XAXIS, mean_stim_OFF + sem_stim_OFF, mean_stim_OFF - sem_stim_OFF,
                    color='purple', edgecolor='none', alpha=.15)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrUp (<2/3) and PyrDown (>3/2)', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(ctrl_stim_stem / f'{exp}_ctrl_stim_all_curves{ext}', dpi=300, bbox_inches='tight')
    ## ALL TRACES PLOTTED ##
    
    
    ## PLOTTING ALL TRACES (NEURONE-AVERAGED) ##    
    fig, ax = plt.subplots(figsize=(2.6,2))
    
    ax.plot(XAXIS, mean_ctrl_ON_single_cell, label='mean_ctrl_ON', color='lightcoral')
    ax.fill_between(XAXIS, mean_ctrl_ON_single_cell + sem_ctrl_ON_single_cell, mean_ctrl_ON_single_cell - sem_ctrl_ON_single_cell,
                    color='lightcoral', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_ON_single_cell, label='mean_stim_ON', color='firebrick')
    ax.fill_between(XAXIS, mean_stim_ON_single_cell + sem_stim_ON_single_cell, mean_stim_ON_single_cell - sem_stim_ON_single_cell,
                    color='firebrick', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_ctrl_OFF_single_cell, label='mean_ctrl_OFF', color='violet')
    ax.fill_between(XAXIS, mean_ctrl_OFF_single_cell + sem_ctrl_OFF_single_cell, mean_ctrl_OFF_single_cell - sem_ctrl_OFF_single_cell,
                    color='violet', edgecolor='none', alpha=.15)
    
    ax.plot(XAXIS, mean_stim_OFF_single_cell, label='mean_stim_OFF', color='purple')
    ax.fill_between(XAXIS, mean_stim_OFF_single_cell + sem_stim_OFF_single_cell, mean_stim_OFF_single_cell - sem_stim_OFF_single_cell,
                    color='purple', edgecolor='none', alpha=.15)
    
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.set(xlabel='Time from run-onset (s)', xticks=[0, 2, 4],
           ylabel='Firing rate (Hz)', yticks=[1, 2, 3, 4], ylim=(1, 4.1))
    ax.set_title(f'{exp}\nPyrUp (<2/3) and PyrDown (>3/2) (neu.-ave.)', fontsize=10)
    ax.legend(fontsize=4, frameon=False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(ctrl_stim_stem / f'{exp}_ctrl_stim_all_curves_neu_ave{ext}', dpi=300, bbox_inches='tight')
    ## ALL TRACES PLOTTED ##
    
    
    # amplitude statistics
    amp_ctrl_ON = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_ctrl_ON
        ], 
        axis=1)
    amp_stim_ON = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_stim_ON
        ], 
        axis=1)
    
    plot_violin_with_scatter(amp_ctrl_ON, amp_stim_ON, 
                             'firebrick', 'royalblue',
                             xticklabels=['ctrl.', 'stim.'],
                             paired=False,
                             showscatter=True,
                             showmedians=True,
                             ylabel='Firing rate (Hz)',
                             dpi=300,
                             save=True,
                             savepath=ctrl_stim_stem / f'{exp}_ctrl_stim_ON_amp')
    
    amp_ctrl_only_OFF = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_ctrl_OFF
        ], 
        axis=1)
    amp_stim_only_OFF = np.mean([
        prof[3750+625:3750+1875] for prof
        in mean_prof_stim_OFF
        ], 
        axis=1)
    
    plot_violin_with_scatter(amp_ctrl_only_OFF, amp_stim_only_OFF, 
                             'purple', 'royalblue',
                             xticklabels=['ctrl.', 'stim.'],
                             paired=False,
                             showscatter=True,
                             showmedians=True,
                             ylabel='Firing rate (Hz)',
                             dpi=300,
                             save=True,
                             savepath=ctrl_stim_stem / f'{exp}_ctrl_stim_OFF_amp')
    
    
    ## DELTA DIST V AMP LINREGRESS ##
    all_amp_ON_delta_mean_filt = [v for i, v in enumerate(all_amp_ON_delta_mean)
                                  if not np.isnan(all_ctrl_stim_lick_distance_delta[i])]
    all_ctrl_stim_lick_distance_delta_filt = [v for i, v in enumerate(all_ctrl_stim_lick_distance_delta)
                                              if not np.isnan(all_ctrl_stim_lick_distance_delta[i])]
    
    # regression
    slope, intercept, r, p, _ = linregress(all_amp_ON_delta_mean_filt, all_ctrl_stim_lick_distance_delta_filt)
    
    # plot
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    
    # scatter
    ax.scatter(all_amp_ON_delta_mean_filt, all_ctrl_stim_lick_distance_delta_filt,
               color='teal', edgecolor='white', s=30, alpha=0.8)
    
    # regression line
    x_vals = np.linspace(min(all_amp_ON_delta_mean_filt), max(all_amp_ON_delta_mean_filt), 100)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='k', lw=1)
    
    # r and p text
    ax.text(0.05, 0.95, f'$R = {r:.2f}$\n$p = {p:.3g}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)
    
    # labels and style
    ax.set_xlabel('delta ON (Hz)', fontsize=10)
    ax.set_ylabel('delta lick dist. (stim − ctrl)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(ctrl_stim_stem / f'{exp}_delta_amp_mean_dist{ext}',
                    dpi=300,
                    bbox_inches='tight')

if __name__ == '__main__':
    main(pathHPCLCopt, 'HPCLC')
    main(pathHPCLCtermopt, 'HPCLCterm')