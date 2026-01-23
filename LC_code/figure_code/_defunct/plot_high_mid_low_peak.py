# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:43:21 2025

High and low rate LC peak examples

@author: Dinghao Luo
"""


#%% imports
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from scipy.stats import sem
import matplotlib.pyplot as plt

from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathLC


#%% paths and parameters
all_sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
LC_beh_stem   = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')
plot_stem     = Path('Z:/Dinghao/code_dinghao/LC_ephys/single_cell_high_low_peak')

SAMP_FREQ     = 1250
SAMP_FREQ_BEH = 1000
RUN_ONSET_IDX = 3 * SAMP_FREQ
BURST_WINDOW  = (-.5, .5)  # for amplitude 

XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

colours = [
    '#4169e1',
    '#4a86e8',
    '#8ab4f8'
]


#%% load cell properties
cell_prop = pd.read_pickle('Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% iterate through recordings 
for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

    # load behaviour
    with open(LC_beh_stem / f'{recname}.pkl', 'rb') as f:
        beh = pickle.load(f)
    
    run_onsets = beh['run_onsets'][1:]
    trials_sts = beh['trial_statements'][1:]

    # find opto trials
    opto_idx = [i for i, t in enumerate(trials_sts) if t[15] != '0']
    
    # load all trains
    all_trains_path = all_sess_stem / recname / f'{recname}_all_trains.npy'
    all_trains = np.load(all_trains_path, allow_pickle=True).item()

    # eligible cells
    curr_cell_prop = cell_prop[cell_prop['sessname'] == recname]
    eligible_cells = [
        cluname for cluname, row in curr_cell_prop.iterrows()
        if row['identity'] != 'other' and row['run_onset_peak']
    ]
    if not eligible_cells:
        print('No eligible cells. Skipping.')
        continue

    # collect per-trial data
    valid_trials = [t for t, ro in enumerate(run_onsets[:-1])
                    if t not in opto_idx and t-1 not in opto_idx and not np.isnan(ro)]
    
    for cluname in eligible_cells:
        train = all_trains[cluname]
        
        # get peak rates
        rates = [np.mean(
                    trial[RUN_ONSET_IDX + int(BURST_WINDOW[0] * SAMP_FREQ) :
                          RUN_ONSET_IDX + int(BURST_WINDOW[1] * SAMP_FREQ)], axis=0
                        )
                 for trial in train]
        thres = np.percentile(rates, 100/2)
        where_high = np.where(rates > thres)[0]
        where_low  = np.where(rates < thres)[0]
        
        high_profs = train[where_high, RUN_ONSET_IDX - SAMP_FREQ : RUN_ONSET_IDX + 4 * SAMP_FREQ]
        low_profs  = train[where_low, RUN_ONSET_IDX - SAMP_FREQ : RUN_ONSET_IDX + 4 * SAMP_FREQ]
        
        high_mean = np.mean(high_profs, axis=0)
        high_sem  = sem(high_profs, axis=0)
        low_mean  = np.mean(low_profs, axis=0)
        low_sem   = sem(low_profs, axis=0)
        
        # single-cell plot 
        fig, ax = plt.subplots(figsize=(3,2.4))
        
        ax.plot(XAXIS, high_mean, colours[0])
        ax.fill_between(XAXIS, high_mean+high_sem,
                               high_mean-high_sem,
                        color=colours[0], edgecolor='none', alpha=.35)
        ax.plot(XAXIS, low_mean, colours[2])
        ax.fill_between(XAXIS, low_mean+low_sem,
                               low_mean-low_sem,
                        color=colours[1], edgecolor='none', alpha=.35)
        
        for ext in ['.pdf', '.png']:
            fig.savefig(
                plot_stem / f'{cluname}{ext}',
                dpi=300, bbox_inches='tight'
                )