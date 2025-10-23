# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 12:03:43 2025
Modified on Fri Oct 17 to use baseline rate as the predictor 

Compare LC run-onset amplitudes between trials with low and high baseline rates

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path
import warnings 

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

regress_r = []
regress_shuf_r = []

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
    first_opto = opto_idx[0] if opto_idx else len(trials_sts) - 1
    
    # load alignment
    rec_stem = Path(f'Z:/Dinghao/MiceExp/ANMD{recname[1:5]}') / recname[:14] / recname
    aligned_run_path = rec_stem / f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
    aligned_run = sio.loadmat(aligned_run_path)['trialsRun'][0][0]
    run_onsets_spike = aligned_run['startLfpInd'][0]  # integer indices into spike_maps sampling
    
    # load spike maps 
    sess_stem = all_sess_stem / recname
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

    # single-cell regression 
    curr_cell_prop = cell_prop[cell_prop['sessname'] == recname]
    for cluname, row in curr_cell_prop.iterrows():
        if row['identity'] == 'other' or not row['run_onset_peak']:
            continue

        clu_idx = int(cluname.split('clu')[-1]) - 2

        # loop across all baseline trials now 
        trial_baselines, trial_rates = [], []
        for ti, onset in enumerate(run_onsets[:first_opto]):
            if ti in opto_idx or ti - 1 in opto_idx or np.isnan(onset):
                continue

            # base rate 
            onset_idx = int(run_onsets_spike[ti])
            base_start = onset_idx - int(SAMP_FREQ * 10)
            base_end   = onset_idx - int(SAMP_FREQ * 2.5)
            if base_start < 0 or base_end > spike_maps.shape[1]:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                base_rate = np.nanmean(spike_maps[clu_idx, base_start:base_end])
                
            if np.isfinite(base_rate):
                trial_baselines.append(base_rate)

        # midpoint for splitting trials 
        baseline_midpoint = np.nanmedian(trial_baselines)
        
        # get the rates 
        for ti, onset in enumerate(run_onsets[:first_opto]):
            if ti in opto_idx or ti - 1 in opto_idx or np.isnan(onset):
                continue
        
            onset_idx = int(run_onsets_spike[ti])
            base_start = onset_idx - int(SAMP_FREQ * 10)
            base_end   = onset_idx - int(SAMP_FREQ * 2.5)
            if base_start < 0 or base_end > spike_maps.shape[1]:
                continue
        
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                base_rate = np.nanmean(spike_maps[clu_idx, base_start:base_end])
        
            train = all_trains[cluname][ti]
            rate = gf.run_onset_amplitude(train, SAMP_FREQ, RUN_ONSET_IDX)
        
            if np.isfinite(base_rate) and np.isfinite(rate):
                trial_rates.append(rate)
            if base_rate > baseline_midpoint:
                high_prof_list.append(train)
            else:
                low_prof_list.append(train)

        # regression
        slope, intercept, r, p, _ = linregress(trial_baselines, trial_rates)
        regress_r.append(r)

        xfit = np.linspace(min(trial_baselines), max(trial_baselines), 2)
        yfit = intercept + slope * xfit

        curr_r_shuf = []
        for perm in range(PERMS):
            x_shuf = np.random.permutation(trial_baselines)
            y_shuf = np.random.permutation(trial_rates)
            slope_shuf, intercept_shuf, r_shuf, p_shuf, _ = linregress(x_shuf, y_shuf)
            curr_r_shuf.append(r_shuf)
        regress_shuf_r.append(np.mean(curr_r_shuf))

        # plot
        fig, axes = plt.subplots(1, 2, figsize=(3.2, 2.2), sharex=True, sharey=True)

        axes[0].scatter(trial_baselines, trial_rates, s=10, color='coral', ec='none', alpha=0.7)
        axes[0].plot(xfit, yfit, color='black', lw=1)
        axes[0].text(0.05, 0.95, f'r = {r:.2f}\np = {p:.3f}',
                     transform=axes[0].transAxes, ha='left', va='top',
                     fontsize=7, color='black')
        axes[0].set(xlabel='Baseline FR (Hz)',
                    ylabel='Run-onset amplitude (Hz)')
        axes[0].spines[['top', 'right']].set_visible(False)

        axes[1].scatter(x_shuf, y_shuf, s=10, color='gray', ec='none', alpha=0.6)
        axes[1].text(0.05, 0.95, f'mean r_shuf = {np.mean(curr_r_shuf):.2f}',
                     transform=axes[1].transAxes, ha='left', va='top',
                     fontsize=7, color='black')
        axes[1].set(xlabel='Baseline FR (Hz)')
        axes[1].spines[['top', 'right']].set_visible(False)

        fig.suptitle(f'{recname} {cluname}', fontsize=8)
        plt.tight_layout()
        for ext in ['.pdf', '.png']:
            fig.savefig(
                GLM_stem / 'baseline_single_session_single_cell' / f'{cluname}{ext}',
                dpi=300,
                bbox_inches='tight'
            )
        plt.close(fig)
        
low_prof_list   = np.array(low_prof_list)
high_prof_list  = np.array(high_prof_list)


#%% population summary
mean_low_prof  = np.mean(low_prof_list, axis=0)
sem_low_prof   = sem(low_prof_list, axis=0)
mean_high_prof = np.mean(high_prof_list, axis=0)
sem_high_prof  = sem(high_prof_list, axis=0)


#%% plotting 
XAXIS = np.arange(len(mean_low_prof)) / SAMP_FREQ - 3

fig, ax = plt.subplots(figsize=(2.4, 2.0))

ax.plot(XAXIS, mean_low_prof, color='coral', alpha=.5, label='Lower baseline')
ax.fill_between(XAXIS, mean_low_prof-sem_low_prof,
                       mean_low_prof+sem_low_prof,
                color='coral', alpha=.15)
ax.plot(XAXIS, mean_high_prof, color='coral', label='Higher baseline')
ax.fill_between(XAXIS, mean_high_prof-sem_high_prof,
                       mean_high_prof+sem_high_prof,
                color='coral', alpha=.3)

ax.set(xlabel='Time from run onset (s)', xlim=(-1, 4),
       ylabel='Firing rate (Hz)', ylim=(1.5,5.5))
ax.legend(frameon=False)

ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

for ext in ['.pdf', '.png']:
    fig.savefig(
        GLM_stem / f'baseline_profiles{ext}',
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
    pc.set_facecolor('coral')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)
parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

# scatter individual r's
ax.scatter(np.ones(len(regress_r)), regress_r,
           color='coral', ec='none', s=10, alpha=0.5, zorder=3)

# baseline at 0
ax.axhline(0, color='gray', lw=1, ls='--')

# mean ± sem text just above the violin
mean_r, sem_r = np.nanmean(regress_r), sem(regress_r)
ymax = np.max(regress_r)
ax.text(1, ymax + 0.05*(ymax - np.min(regress_r)),
        f'{mean_r:.2f} ± {sem_r:.2f}',
        ha='center', va='bottom', fontsize=7, color='coral')

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
        GLM_stem / f'baseline_r_violinplot{ext}',
        dpi=300,
        bbox_inches='tight'
        )