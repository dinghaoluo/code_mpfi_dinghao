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
from matplotlib.cm import ScalarMappable

from common import mpl_formatting, smooth_convolve
import GLM_functions as gf
mpl_formatting()

import rec_list
paths = rec_list.pathLC


#%% paths and parameters
all_sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
LC_beh_stem   = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')
GLM_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/GLM')

SAMP_FREQ     = 1250
SAMP_FREQ_BEH = 1000
RUN_ONSET_IDX = 3 * SAMP_FREQ
BURST_WINDOW  = (-.5, .5)  # for amplitude 

PERMS = 1000  # permutate for 1000 times (per session) for signif test 

# colours (still used for low/high etc.)
greens = ['#b9e4c9',  # light
          '#4fb66d',  # mid
          '#145a32']  # dark

# fine bins for time-since-last-reward traces
BIN_START = 1.0
BIN_END   = 2.1
BIN_WIDTH = 0.1
N_BINS    = int((BIN_END - BIN_START) / BIN_WIDTH)
bin_edges = np.arange(BIN_START, BIN_END + 1e-6, BIN_WIDTH)

# saving switch 
save = True


#%% load cell properties
cell_prop = pd.read_pickle('Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% iterate recordings
# grand list 
all_t_since = []

# containers 
low_prof_list      = []
high_prof_list     = []
low_prof_rew_list  = []
high_prof_rew_list = []

regress_r      = []
regress_shuf_r = []

# reward-aligned traces in 0.1-s bins between 0.5 and 1.5
all_prof_rew_list_bins = [[] for _ in range(N_BINS)]

# per cell 
all_cell_rate_dict = {}
all_cell_time_dict = {}
cell_bin_traces = {}  # for quantification of baselines and ramp rates 

for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')

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
        print('No eligible cells. Skipping.')
        continue
    
    # initiate bins 
    for cluname in eligible_cells:
        if cluname not in cell_bin_traces:
            cell_bin_traces[cluname] = [[] for _ in range(N_BINS)]


    # per-session containers for reward-aligned traces
    prof_rew_list_bins = [[] for _ in range(N_BINS)]
    trial_counts = [0] * N_BINS  # counts per 0.1-s bin (in trials)

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
        all_t_since.append(t_since)
        
        if np.isnan(t_since) or t_since < 0 or t_since > 8:  # filter out invalid (<0) and extreme (>5) values
            continue

        # bin index for 0.1-s reward-aligned traces (0.5–1.5 s only)
        bin_idx = None
        if BIN_START <= t_since < BIN_END:
            bin_idx = int((t_since - BIN_START) // BIN_WIDTH)
            if bin_idx < 0:
                bin_idx = 0
            if bin_idx > N_BINS - 1:
                bin_idx = N_BINS - 1
            trial_counts[bin_idx] += 1

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
            
            # last-rew-aligned (long window, low vs high split)
            train_rew_full = spike_maps[clu_idx][
                rew_spike[ti] - 3 * SAMP_FREQ : rew_spike[ti] + 7 * SAMP_FREQ
                ]
            if t_since < 1.5:
                low_prof_list.append(train)
                low_prof_rew_list.append(train_rew_full)
            else:
                high_prof_list.append(train)
                high_prof_rew_list.append(train_rew_full)
            
            # shorter reward-aligned trace, stopping at t_since
            train_rew_short = spike_maps[clu_idx][
                rew_spike[ti] - 1 * SAMP_FREQ : rew_spike[ti] + int(t_since * SAMP_FREQ)
                ]
            if bin_idx is not None:
                prof_rew_list_bins[bin_idx].append(train_rew_short)
                cell_bin_traces[cluname][bin_idx].append(train_rew_short)

        
        trial_times.append(t_since)
        trial_rates.append(np.mean(curr_rates))

    # require at least some trials overall in 0.5–1.5 s
    if sum(trial_counts) < 10:
        print('Not enough trials. Skipping reward-binned plot for this session.')
    else:
        # plot one mean trace plot per session with all bins that have enough trials
        fig, ax = plt.subplots(figsize=(3, 2))

        for bi in range(N_BINS):
            if trial_counts[bi] < 3:
                continue
            bin_trials = prof_rew_list_bins[bi]
            if len(bin_trials) == 0:
                continue

            bin_min = min(len(trial) for trial in bin_trials)
            xaxis_bin = np.arange(bin_min) / SAMP_FREQ - 1
            prof_rew_array_bin = np.array([trial[:bin_min] for trial in bin_trials])
            mean_prof_rew_bin = np.mean(prof_rew_array_bin, axis=0)

            # colour gradient across bins
            colour = plt.cm.Greens(0.3 + 0.6 * bi / (N_BINS - 1))
            ax.plot(xaxis_bin, mean_prof_rew_bin, color=colour)

        fig.suptitle(recname)
        plt.tight_layout()
        
        if save:
            for ext in ['.pdf', '.png']:
                fig.savefig(
                    GLM_stem / 'rew_to_run_single_session_split' /  f'{recname}{ext}',
                    dpi=300,
                    bbox_inches='tight'
                    )
        plt.close()

        # accumulate into global lists only if we didn't skip
        for bi in range(N_BINS):
            all_prof_rew_list_bins[bi].extend(prof_rew_list_bins[bi])
    
    # store in big dictionaries 
    all_cell_rate_dict.update(cell_rate_dict)
    all_cell_time_dict.update(cell_time_dict)
        
    # compute regression (run-onset FR vs time-since-last-reward)
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
    
    if save:
        for ext in ['.pdf', '.png']:
            fig.savefig(
                GLM_stem / 'rew_to_run_single_session' /  f'{recname}{ext}',
                dpi=300,
                bbox_inches='tight'
                )
    
    plt.close()
        
    # plot one scatter per session (lim restricted)
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
    
    for x in range(2):
        axes[x].set(xlim=(.5,1.5))
    
    fig.suptitle(recname)
    plt.tight_layout()
    
    if save:
        for ext in ['.pdf', '.png']:
            fig.savefig(
                GLM_stem / 'rew_to_run_single_session' /  f'{recname}_lim{ext}',
                dpi=300,
                bbox_inches='tight'
                )
    
    plt.close()
        
low_prof_list   = np.array(low_prof_list)
high_prof_list  = np.array(high_prof_list)


#%% per-cell ramp quantification 
BASELINE_OFFSET = 0.25
SMOOTH_WINDOW   = 0.05
MIN_TRIALS_PER_BIN = 3

cell_ramp_rates = {clu: np.full(N_BINS, np.nan) for clu in cell_bin_traces}
cell_baselines  = {clu: np.full(N_BINS, np.nan) for clu in cell_bin_traces}
cell_ends       = {clu: np.full(N_BINS, np.nan) for clu in cell_bin_traces}   # NEW

for cluname, bin_lists in cell_bin_traces.items():
    for bi in range(N_BINS):
        trials = bin_lists[bi]
        if len(trials) < MIN_TRIALS_PER_BIN:
            continue

        # align traces in this cell+bin
        bin_min = min(len(tr) for tr in trials)
        arr = np.stack([tr[:bin_min] for tr in trials])
        mean_prof = np.mean(arr, axis=0)
        xaxis = np.arange(bin_min) / SAMP_FREQ - 1

        # define times
        t_end  = xaxis[-1]
        t_base = t_end - BASELINE_OFFSET
        if t_base <= xaxis[0]:
            continue

        # baseline index
        mask_base = (xaxis >= (t_base - SMOOTH_WINDOW)) & (xaxis <= (t_base + SMOOTH_WINDOW))
        idx_base = np.where(mask_base)[0]
        if idx_base.size == 0:
            idx_base = np.array([np.argmin(np.abs(xaxis - t_base))])

        # end index (actual end)
        mask_end = (xaxis >= (t_end - SMOOTH_WINDOW)) & (xaxis <= (t_end + SMOOTH_WINDOW))
        idx_end = np.where(mask_end)[0]
        if idx_end.size == 0:
            idx_end = np.array([np.argmin(np.abs(xaxis - t_end))])

        # amplitudes
        baseline = np.nanmean(mean_prof[idx_base])
        end      = np.nanmean(mean_prof[idx_end])

        # slope across the last 0.5 seconds
        slope    = (end - baseline) / BASELINE_OFFSET

        # store
        cell_baselines[cluname][bi]  = baseline
        cell_ramp_rates[cluname][bi] = slope
        cell_ends[cluname][bi]       = end
        

#%% plot all t_since 
# fig, ax = plt.subplots(figsize=(3,3))
# ax.hist(all_t_since, bins=1000)
# ax.set(xlim=(0, 20))
# plt.show()

# fig, ax = plt.subplots(figsize=(3,3))
# ax.hist(all_t_since, bins=1000)
# ax.set(xlim=(0, 10))
# plt.show()

# fig, ax = plt.subplots(figsize=(3,3))
# ax.hist(all_t_since, bins=1500)
# ax.set(xlim=(0, 5))
# plt.show()

# fig, ax = plt.subplots(figsize=(3,3))
# ax.hist(all_t_since, bins=1500)
# ax.set(xlim=(0, 3))
# plt.show()


#%% population summary (run-onset low vs high)
mean_low_prof  = np.mean(low_prof_list, axis=0)
sem_low_prof   = sem(low_prof_list, axis=0)
mean_high_prof = np.mean(high_prof_list, axis=0)
sem_high_prof  = sem(high_prof_list, axis=0)

xaxis_bins = []
mean_prof_rew_bins = []
sem_prof_rew_bins  = []

for bi in range(N_BINS):
    bin_trials = all_prof_rew_list_bins[bi]

    if len(bin_trials) == 0:
        xaxis_bins.append(None)
        mean_prof_rew_bins.append(None)
        sem_prof_rew_bins.append(None)
        continue

    bin_end = BIN_START + bi * BIN_WIDTH
    duration = bin_end + 1.0
    n_samples = int(duration * SAMP_FREQ)

    arr = []
    for tr in bin_trials:
        if len(tr) >= n_samples:
            arr.append(tr[:n_samples])
        else:
            arr.append(np.pad(tr, (0, n_samples - len(tr)), mode='edge'))
    arr = np.vstack(arr)

    mean_prof = np.mean(arr, axis=0)
    sem_prof  = sem(arr, axis=0)

    xaxis = np.arange(n_samples) / SAMP_FREQ - 1

    xaxis_bins.append(xaxis)
    mean_prof_rew_bins.append(mean_prof)
    sem_prof_rew_bins.append(sem_prof)


#%% plotting run-onset low vs high
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

if save:
    for ext in ['.pdf', '.png']:
        fig.savefig(
            GLM_stem / f'rew_to_run_profiles{ext}',
            dpi=300,
            bbox_inches='tight'
            )


#%% r comparison 
tval, p_t = ttest_1samp(regress_r, 0)
wstat, p_w = wilcoxon(regress_r)

# compute shuffle 95% interval across sessions
shuf_mean = np.nanmean(regress_shuf_r)
shuf_std  = np.nanstd(regress_shuf_r)
ci_low  = shuf_mean - 1.96 * shuf_std
ci_high = shuf_mean + 1.96 * shuf_std

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


# shuffle + CI
ax.axhline(shuf_mean, color='gray', lw=1.2, ls='--')
ax.fill_between([0.5, 1.5], [ci_low, ci_low], [ci_high, ci_high],
                color='gray', alpha=0.20, edgecolor='none')

# real mean ± sem
mean_r, sem_r = np.nanmean(regress_r), sem(regress_r)
ymax = np.max(regress_r)
ax.text(1, ymax + 0.05*(ymax - np.min(regress_r)),
        f'{mean_r:.2f} ± {sem_r:.2f}',
        ha='center', va='bottom', fontsize=7, color='forestgreen')

# significance
ax.text(1, np.min(regress_r) - 0.10*(ymax - np.min(regress_r)),
        f't(1-samp)={tval:.2f}, p={p_t:.2e}\n'
        f'Wilcoxon={wstat:.2f}, p={p_w:.2e}',
        ha='center', va='top', fontsize=6.5, color='black')

# formatting
ax.set(xlim=(0.5, 1.5),
       xticks=[1],
       xticklabels=['Real r'],
       ylabel='Correlation (r)',
       title='Across-sess. r')
ax.spines[['top', 'right', 'bottom']].set_visible(False)

plt.tight_layout()
plt.show()

if save:
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
                         savepath=GLM_stem / 'rew_to_run_FR_violinplot',
                         print_statistics=True)

    
#%% reward-aligned train
fig, ax = plt.subplots(figsize=(3, 2.4))

for bi in range(N_BINS):
    if xaxis_bins[bi] is None:
        continue

    colour = plt.cm.Greens(0.3 + 0.6 * bi / (N_BINS - 1))
    label = f'{BIN_START + bi*BIN_WIDTH:.1f}-{BIN_START + (bi+1)*BIN_WIDTH:.1f} s'

    # plot the curve
    ax.plot(xaxis_bins[bi], mean_prof_rew_bins[bi], color=colour, label=label, linewidth=1.2, alpha=1.0)
    
    x_end = xaxis_bins[bi][-1]
    ax.axvline(x_end, color=colour, linestyle='--', linewidth=.8, alpha=.6)

ax.set(xlabel='Time from last reward (s)',
       ylabel='Firing rate (Hz)')

ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=6, ncol=2)

# colorbar
norm = plt.Normalize(vmin=BIN_START, vmax=BIN_START + (N_BINS-1)*BIN_WIDTH)
cmap = plt.cm.Greens
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=.35, aspect=10)
cbar.set_label('Time bin (s)', fontsize=8)
cbar.set_ticks([1, 2])

plt.tight_layout()
plt.show()

if save:
    for ext in ['.pdf', '.png']:
        fig.savefig(
            GLM_stem / f'rew_to_run_rew_aligned_profiles{ext}',
            dpi=300,
            bbox_inches='tight'
            )
        

#%% plot fitted 
fig, ax = plt.subplots(figsize=(3, 2.4))

for bi in range(N_BINS):
    if xaxis_bins[bi] is None:
        continue

    colour = plt.cm.Greens(0.3 + 0.6 * bi / (N_BINS - 1))
    label = f'{BIN_START + bi*BIN_WIDTH:.1f}-{BIN_START + (bi+1)*BIN_WIDTH:.1f} s'

    # plot the curve
    ax.plot(xaxis_bins[bi], mean_prof_rew_bins[bi], color=colour, label=label, linewidth=1.2, alpha=.75)
    
    x_end = xaxis_bins[bi][-1]
    ax.axvline(x_end, color=colour, linestyle='--', linewidth=.8, alpha=.6)

## ---- fit curves ---- ## 
curves_padded = np.zeros((len(mean_prof_rew_bins), len(mean_prof_rew_bins[-1])), dtype=float)
curves_padded[curves_padded==0] = np.nan
for i, prof in enumerate(mean_prof_rew_bins):
    curves_padded[i, :len(prof)] = prof
smooth_axis = xaxis_bins[-1]
smooth_prof = smooth_convolve(np.nanmean(curves_padded, axis=0), 50)
ax.plot(smooth_axis, smooth_prof, color='k', linewidth=2)
## ---- fit curves end ---- ##

ax.set(xlabel='Time from last reward (s)',
       ylabel='Firing rate (Hz)')

ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=6, ncol=2)

# colorbar
norm = plt.Normalize(vmin=BIN_START, vmax=BIN_START + (N_BINS-1)*BIN_WIDTH)
cmap = plt.cm.Greens
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=.35, aspect=10)
cbar.set_label('Time bin (s)', fontsize=8)
cbar.set_ticks([1, 2])

plt.tight_layout()
plt.show()

if save:
    for ext in ['.pdf', '.png']:
        fig.savefig(
            GLM_stem / f'rew_to_run_rew_aligned_profiles_fitted{ext}',
            dpi=300,
            bbox_inches='tight'
            )