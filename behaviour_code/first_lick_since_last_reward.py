# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 12:03:43 2025
Modified on Friday to get reward-aligned firing profiles 
Duplicated and modified on Thursday 8 Jan 2026 to 
    correlate w first lick time (n+1 trial)

Correlation between time since last reward and first-lick time 

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
from scipy.stats import sem, linregress, ttest_1samp, wilcoxon
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from common import mpl_formatting, smooth_convolve
import GLM_functions as gf
mpl_formatting()

import rec_list
paths = rec_list.pathLC


#%% paths and parameters
all_sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
LC_beh_stem   = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')
all_beh_stem  = Path('Z:/Dinghao/code_dinghao/behaviour')

SAMP_FREQ_BEH = 1000

PERMS = 1000  # permutate for 1000 times (per session) for signif test 

# colours (still used for low/high etc.)
greens = ['#b9e4c9',  # light
          '#4fb66d',  # mid
          '#145a32']  # dark

# saving switch 
save = False


#%% iterate recordings
# grand lists
all_t_since     = []
all_first_licks = []

# counters
animals = set()
n_sess = 0

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    n_sess -= -1
    
    anmname = recname.split('-')[0]
    animals.add(anmname)

    # load behaviour
    with open(LC_beh_stem / f'{recname}.pkl', 'rb') as f:
        beh = pickle.load(f)
    
    lick_times    = beh['lick_times_aligned'][1:]
    reward_times  = beh['reward_times'][1:]
    run_onsets    = beh['run_onsets'][1:]
    trials_sts    = beh['trial_statements'][1:]

    # find opto trials
    opto_idx = [i for i, t in enumerate(trials_sts) if t[15] != '0']

    # collect per-trial data
    valid_trials = [t for t, ro in enumerate(run_onsets[:-1])
                    if t not in opto_idx and t-1 not in opto_idx and not np.isnan(ro)]
    
    for ti in valid_trials:
        onset_time = run_onsets[ti] / SAMP_FREQ_BEH

        # time since last reward
        curr_t_since = gf.time_since_last_reward(
            reward_times, onset_time, ti
            )
        
        if np.isnan(curr_t_since) or curr_t_since < 0 or curr_t_since > 8:  # filter out invalid (<0) and extreme (>5) values
            continue

        # this trial's first-lick time
        curr_licks = lick_times[ti]
        if type(curr_licks) == list:
            if len(curr_licks)<=1:
                continue
        else:
            continue
        curr_first_lick = lick_times[ti][0] / SAMP_FREQ_BEH
        
        if curr_first_lick > 8 or curr_first_lick < 0.5:
            continue
        
        # collect data 
        all_t_since.append(curr_t_since)
        all_first_licks.append(curr_first_lick)
        
print(f'n = {len(animals)}')
print(f'n_sess = {n_sess}')    

        
#%% plotting 
fig, ax = plt.subplots(figsize=(4, 4))

x = np.asarray(all_t_since, dtype=float)
y = np.asarray(all_first_licks, dtype=float)

# drop any stray nans (shouldn't be there, but safe)
m = np.isfinite(x) & np.isfinite(y)
x = x[m]
y = y[m]

ax.scatter(x, y, s=20, alpha=0.7, edgecolor='none')

# regression
res = linregress(x, y)
xx = np.linspace(x.min(), x.max(), 200)
yy = res.intercept + res.slope * xx
ax.plot(xx, yy, linewidth=2)

# stats box
stats_txt = (
    f'n = {len(x)}\n'
    f'y = {res.slope:.3f}x + {res.intercept:.3f}\n'
    f'r = {res.rvalue:.3f}\n'
    f'p = {res.pvalue:.3g}'
)

ax.text(
    0.02, 0.98, stats_txt,
    transform=ax.transAxes,
    va='top', ha='left',
    fontsize=12,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, linewidth=0.5)
)

ax.set_xlabel('time since last reward (s)')
ax.set_ylabel('first lick time (s)')
ax.set_title('time since last reward vs first lick time')

plt.tight_layout()
plt.show()