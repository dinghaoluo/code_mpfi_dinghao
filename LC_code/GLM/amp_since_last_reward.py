# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 12:03:43 2025
Modified on Friday to get reward-aligned firing profiles 

Compare LC run-onset amplitudes between trials with low and high 'time since 
    last reward' according to GLM results 

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

from common import mpl_formatting
import GLM_functions as gf
mpl_formatting()

import rec_list
paths = rec_list.pathLC


#%% paths
all_sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
LC_beh_stem   = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')
GLM_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/GLM')

SAMP_FREQ     = 1250
SAMP_FREQ_BEH = 1000
RUN_ONSET_IDX = 3 * SAMP_FREQ
BURST_WINDOW  = (-.5, .5)  # for amplitude 

PERMS = 1000  # permutate for 1000 times (per session) for signif test 


#%% load cell properties
cell_prop = pd.read_pickle('Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% iterate recordings
low_prof_list   = []
high_prof_list  = []

regress_r      = []
regress_shuf_r = []

# aligned to rewards 
prof_rew_list_1 = []
prof_rew_list_2 = []
prof_rew_list_3 = []
prof_rew_list_4 = []
prof_rew_list_5 = []

# per cell 
all_cell_rate_dict = {}
all_cell_time_dict = {}

for path in paths:
    recname = Path(path).name
    print(recname)

    # load behaviour
    with open(LC_beh_stem / f'{recname}.pkl', 'rb') as f:
        beh = pickle.load(f)
    
    reward_times  = beh['reward_times'][1:]
    run_onsets    = beh['run_onsets'][1:]
    trials_sts    = beh['trial_statements'][1:]

    # find opto trials
    opto_idx = [i for i, t in enumerate(trials_sts) if t[15] != '0']

    # new--get reward aligned to full spike maps
    rec_stem = Path(f'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}') / recname[:14] / recname
    aligned_rew_path = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRew_msess1.mat'
    aligned_rew = sio.loadmat(aligned_rew_path)['trialsRew'][0][0]
    rew_spike = aligned_rew['startLfpInd'][0][1:]
    
    # aligned_run_path = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
    # aligned_run = sio.loadmat(aligned_run_path)['trialsRun'][0][0]
    # run_spike = aligned_run['startLfpInd'][0][1:]

    # get spike maps 
    sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions') / recname
    spike_maps = np.load(sess_stem / f'{recname}_smoothed_spike_map.npy', allow_pickle=True)

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
        print(f'No eligible cells for {recname}. Skipping.')
        continue

    # collect per-trial data
    valid_trials = [t for t, ro in enumerate(run_onsets[:-1])
                    if t not in opto_idx and t-1 not in opto_idx and not np.isnan(ro)]
    
    trial_times, trial_rates = [], []
    cell_rate_dict = {clu: [] for clu in eligible_cells}  # store per-cell FRs across valid trials
    cell_time_dict = {clu: [] for clu in eligible_cells}  # store matching t_since values
    for ti in valid_trials:
        onset_time = run_onsets[ti] / SAMP_FREQ_BEH

        # time since last reward
        t_since = gf.time_since_last_reward(reward_times, onset_time, ti)
        if np.isnan(t_since) or t_since < 0 or t_since > 5:  # filter out invalid (<0) and extreme (>5) values
            continue

        # mean FR across all eligibles 
        curr_rates = []
        for cluname in eligible_cells:
            clu_idx = int(cluname.split('clu')[-1]) - 2  # for retrieving spike map 
            
            # run-aligned 
            train = all_trains[cluname][ti]
            curr_rate = np.mean(train[RUN_ONSET_IDX + int(BURST_WINDOW[0]*SAMP_FREQ):
                                      RUN_ONSET_IDX + int(BURST_WINDOW[1]*SAMP_FREQ)])
            curr_rates.append(curr_rate)
            
            # store per-cell rates and corresponding t_since
            cell_rate_dict[cluname].append(curr_rate)
            cell_time_dict[cluname].append(t_since)
            
            # last-rew-aligned
            train_rew = spike_maps[clu_idx][rew_spike[ti] - 3*SAMP_FREQ : rew_spike[ti] + 7*SAMP_FREQ]
            
            if t_since < 1.5:
                low_prof_list.append(train)
            else:
                high_prof_list.append(train)
                
            if t_since < 1:
                prof_rew_list_1.append(train_rew)
            elif t_since < 2:
                prof_rew_list_2.append(train_rew)
            elif t_since < 3:
                prof_rew_list_3.append(train_rew)
            elif t_since < 4:
                prof_rew_list_4.append(train_rew)
            else:
                prof_rew_list_5.append(train_rew)
        
        trial_times.append(t_since)
        trial_rates.append(np.mean(curr_rates))
        
    # store in big dictionaries 
    all_cell_rate_dict.update(cell_rate_dict)
    all_cell_time_dict.update(cell_time_dict)
        
    # compute regression 
    slope, intercept, r, p, _ = linregress(trial_times, trial_rates)
    regress_r.append(r)
    xfit = np.linspace(min(trial_times), max(trial_times), 2)
    yfit = intercept + slope * xfit
    
    # permutations
    curr_r_shuf = []
    for perm in range(PERMS):
        trial_times_shuf = np.random.permutation(trial_times)
        trial_rates_shuf = np.random.permutation(trial_rates)
        slope_shuf, intercept_shuf, r_shuf, p_shuf, _ = linregress(trial_times_shuf, trial_rates_shuf)
        curr_r_shuf.append(r_shuf)
        yfit_shuf = intercept_shuf + slope_shuf * xfit
    regress_shuf_r.append(np.mean(curr_r_shuf))

    # plot one scatter per session
    fig, axes = plt.subplots(1, 2, figsize=(3.2, 2.2), sharex=True, sharey=True)

    axes[0].scatter(trial_times, trial_rates, s=10, color='forestgreen', ec='none', alpha=0.7)
    axes[0].plot(xfit, yfit, color='black', lw=1)
    axes[0].text(0.05, 0.95, f'r = {r:.2f}\np = {p:.3f}',
                 transform=axes[0].transAxes, ha='left', va='top',
                 fontsize=7, color='black')
    axes[0].set(xlabel='Time since rew. (s)',
                ylabel='Run-onset FR (Hz)')
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].scatter(trial_times, trial_rates_shuf, s=10, color='gray', ec='none', alpha=0.6)
    axes[1].plot(xfit, yfit_shuf, color='black', lw=1)
    axes[1].text(0.05, 0.95, f'r = {r_shuf:.2f}\np = {p_shuf:.3f}',
                 transform=axes[1].transAxes, ha='left', va='top',
                 fontsize=7, color='black')
    axes[1].set(xlabel='Time since rew. (s)')
    axes[1].spines[['top', 'right']].set_visible(False)
    
    fig.suptitle(recname)
    plt.tight_layout()
    plt.show()
    
    for ext in ['.pdf', '.png']:
        fig.savefig(
            GLM_stem / 'rew_to_run_single_session' /  f'{recname}{ext}',
            dpi=300,
            bbox_inches='tight'
            )
        
low_prof_list   = np.array(low_prof_list)
high_prof_list  = np.array(high_prof_list)


#%% population summary
mean_low_prof  = np.mean(low_prof_list, axis=0)
sem_low_prof   = sem(low_prof_list, axis=0)
mean_high_prof = np.mean(high_prof_list, axis=0)
sem_high_prof  = sem(high_prof_list, axis=0)

mean_prof_rew_1  = np.mean(prof_rew_list_1, axis=0)
sem_prof_rew_1   = sem(prof_rew_list_1, axis=0)
mean_prof_rew_2  = np.mean(prof_rew_list_2, axis=0)
sem_prof_rew_2   = sem(prof_rew_list_2, axis=0)
mean_prof_rew_3  = np.mean(prof_rew_list_3, axis=0)
sem_prof_rew_3   = sem(prof_rew_list_3, axis=0)
mean_prof_rew_4  = np.mean(prof_rew_list_4, axis=0)
sem_prof_rew_4   = sem(prof_rew_list_4, axis=0)
mean_prof_rew_5  = np.mean(prof_rew_list_5, axis=0)
sem_prof_rew_5   = sem(prof_rew_list_5, axis=0)

mean_prof_rew_0_3 = np.mean(prof_rew_list_1+prof_rew_list_2+prof_rew_list_3, axis=0)
sem_prof_rew_0_3 = sem(prof_rew_list_1+prof_rew_list_2+prof_rew_list_3, axis=0)
mean_prof_rew_3_5 = np.mean(prof_rew_list_4+prof_rew_list_5, axis=0)
sem_prof_rew_3_5 = sem(prof_rew_list_4+prof_rew_list_5, axis=0)


#%% plotting 
XAXIS = np.arange(len(mean_low_prof)) / SAMP_FREQ - 3

fig, ax = plt.subplots(figsize=(2.4, 2.0))

ax.plot(XAXIS, mean_low_prof, color='forestgreen', alpha=.5, label='Shorter time')
ax.fill_between(XAXIS, mean_low_prof-sem_low_prof,
                       mean_low_prof+sem_low_prof,
                color='forestgreen', alpha=.15)
ax.plot(XAXIS, mean_high_prof, color='forestgreen', label='Longer time')
ax.fill_between(XAXIS, mean_high_prof-sem_high_prof,
                       mean_high_prof+sem_high_prof,
                color='forestgreen', alpha=.3)

ax.set(xlabel='Time from run onset (s)', xlim=(-1, 4),
       ylabel='Firing rate (Hz)', ylim=(1.6,5))
ax.legend(frameon=False)

ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

for ext in ['.pdf', '.png']:
    fig.savefig(
        GLM_stem / f'rew_to_run_profiles{ext}',
        dpi=300,
        bbox_inches='tight'
        )


#%% r comparison 
tval, p_t = ttest_1samp(regress_r, 0)
wstat, p_w = wilcoxon(regress_r)

fig, ax = plt.subplots(figsize=(1.6, 2.2))

# violin
parts = ax.violinplot(regress_r, positions=[1],
                      showmeans=False, showmedians=True, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('forestgreen')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)
parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

# scatter individual r's
ax.scatter(np.ones(len(regress_r)), regress_r,
           color='forestgreen', ec='none', s=10, alpha=0.5, zorder=3)

# baseline at 0
ax.axhline(0, color='gray', lw=1, ls='--')

# mean ± sem text just above the violin
mean_r, sem_r = np.nanmean(regress_r), sem(regress_r)
ymax = np.max(regress_r)
ax.text(1, ymax + 0.05*(ymax - np.min(regress_r)),
        f'{mean_r:.2f} ± {sem_r:.2f}',
        ha='center', va='bottom', fontsize=7, color='forestgreen')

# stats text neatly *below* the violin
ax.text(1, np.min(regress_r) - 0.10*(ymax - np.min(regress_r)),
        f't(1-samp)={tval:.2f}, p={p_t:.2e}\n'
        f'Wilcoxon={wstat:.2f}, p={p_w:.2e}',
        ha='center', va='top', fontsize=6.5, color='black')

# formatting
ax.set(xlim=(0.5, 1.5),
       xticks=[1],
       xticklabels=['Real r'],
       ylabel='Correlation (r)',
       title='across-sess. r')
ax.spines[['top', 'right', 'bottom']].set_visible(False)

plt.tight_layout()
plt.show()

for ext in ['.pdf', '.png']:
    fig.savefig(
        GLM_stem / f'rew_to_run_r_violinplot{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    
    
#%% rates
low_rates_cells  = []
high_rates_cells = []

for cluname in all_cell_rate_dict:
    rates = np.array(all_cell_rate_dict[cluname])
    times = np.array(all_cell_time_dict[cluname])

    curr_low  = []
    curr_high = []
    for idx, t in enumerate(times):
        if t < 1.5:
            curr_low.append(rates[idx])
        else:
            curr_high.append(rates[idx])

    mean_low  = np.mean(curr_low)
    mean_high = np.mean(curr_high)

    low_rates_cells.append(mean_low)
    high_rates_cells.append(mean_high)

from plotting_functions import plot_violin_with_scatter
plot_violin_with_scatter(low_rates_cells, high_rates_cells, 
                         'forestgreen', 'darkgreen',
                         paired=True,
                         xticklabels=['Low', 'High'],
                         ylabel='Firing rate (Hz)',
                         save=True,
                         savepath=GLM_stem / f'rew_to_run_FR_violinplot{ext}',
                         print_statistics=True)

    
#%% reward train 
XAXIS = np.arange(len(mean_prof_rew_1)) / SAMP_FREQ - 3

greens = ['#b9e4c9',
          '#7fd199',
          '#4fb66d',
          '#2d914d',
          '#145a32']

fig, ax = plt.subplots(figsize=(2.4, 2.0))

ax.plot(XAXIS, mean_prof_rew_1, color=greens[0], label='0-1 s')
ax.fill_between(XAXIS, mean_prof_rew_1-sem_prof_rew_1,
                       mean_prof_rew_1+sem_prof_rew_1,
                color=greens[0], edgecolor='none', alpha=.3)
ax.plot(XAXIS, mean_prof_rew_2, color=greens[1], label='1-2 s')
ax.fill_between(XAXIS, mean_prof_rew_2-sem_prof_rew_2,
                       mean_prof_rew_2+sem_prof_rew_2,
                color=greens[1], edgecolor='none', alpha=.3)
ax.plot(XAXIS, mean_prof_rew_3, color=greens[2], label='2-3 s')
ax.fill_between(XAXIS, mean_prof_rew_3-sem_prof_rew_3,
                       mean_prof_rew_3+sem_prof_rew_3,
                color=greens[2], edgecolor='none', alpha=.3)
ax.plot(XAXIS, mean_prof_rew_4, color=greens[3], label='3-4 s')
ax.fill_between(XAXIS, mean_prof_rew_4-sem_prof_rew_4,
                       mean_prof_rew_4+sem_prof_rew_4,
                color=greens[3], edgecolor='none', alpha=.3)
ax.plot(XAXIS, mean_prof_rew_5, color=greens[4], label='4-5 s')
ax.fill_between(XAXIS, mean_prof_rew_5-sem_prof_rew_5,
                       mean_prof_rew_5+sem_prof_rew_5,
                color=greens[4], edgecolor='none', alpha=.3)

ax.set(xlabel='Time from run onset (s)',
       ylabel='Firing rate (Hz)')
ax.legend(frameon=False)

ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

for ext in ['.pdf', '.png']:
    fig.savefig(
        GLM_stem / f'rew_to_run_rew_aligned_profiles{ext}',
        dpi=300,
        bbox_inches='tight'
        )


#%% 
fig, ax = plt.subplots(figsize=(3,3))
ax.plot(mean_prof_rew_0_3)
ax.plot(mean_prof_rew_3_5)