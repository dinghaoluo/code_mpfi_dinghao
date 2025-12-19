# -*- coding: utf-8 -*- #
"""
Created on 5 Dec 2025

analyse pupil size changes aligned to run onset

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import ttest_1samp, wilcoxon, sem 

from common import mpl_formatting, normalise, smooth_convolve
mpl_formatting()

import behaviour_functions as bf


#%% paths and parameters 
pupil_stem  = Path('Z:/Dinghao/pupil_tracking')
output_stem = Path('Z:/Dinghao/code_dinghao/behaviour/pupil_tracking')

SAMP_FREQ     = 30  # fps 
SAMP_FREQ_BEH = 1000  # for Arduino

BEF = 1  # s 
AFT = 14

N_SHUF = 500


#%% define fullpaths to loop over
paths = [
    r'Z:\Dinghao\MiceExp\ANMD057\A057-20230510-03',
    r'Z:\Dinghao\MiceExp\ANMD057\A057-20230511-03',
    r'Z:\Dinghao\MiceExp\ANMD057\A057-20230516-03',
    r'Z:\Dinghao\MiceExp\ANMD057\A057-20230519-03',
    
    r'Z:\Dinghao\MiceExp\ANMD059\A059-20230424-01',
    r'Z:\Dinghao\MiceExp\ANMD059\A059-20230424-02',
    r'Z:\Dinghao\MiceExp\ANMD059\A059-20230425-02',
    r'Z:\Dinghao\MiceExp\ANMD059\A059-20230503-02',
    r'Z:\Dinghao\MiceExp\ANMD059\A059-20230509-02',
    r'Z:\Dinghao\MiceExp\ANMD059\A059-20230512-02'
]


#%% main loop
avg_start_traces   = []

for path in paths:
    recname = Path(path).name
    animal  = f'ANMD{recname[1:4]}'
    day     = recname[:-3]
    print(f'\n{recname}...')

    face_path  = pupil_stem / animal / day / f'{recname}_proc.npy'
    ctime_path = pupil_stem / animal / day / f'{recname}_tsdict.npy'
    txt_path   = Path(f'{path}T.txt')

    if not face_path.exists() or not ctime_path.exists() or not txt_path.exists():
        print('Missing data files; skipped')
        continue

    face  = np.load(face_path, allow_pickle=True).item()
    ctime = np.load(ctime_path, allow_pickle=True).item()['ctime']
    ctime = [t*1000 for t in ctime]
    ctime = [t-ctime[0] for t in ctime]

    height = face['Ly']
    width = face['Lx']
    
    try:
        pupil_area = face['pupil'][0]['area']
    except KeyError:
        continue
            
    beh = bf.process_behavioural_data(txt_path)
    
    # get stim
    trial_statements = beh['trial_statements']
    opto_cds         = [t[15] for t in trial_statements]
    try:
        first_stim = [trial for trial, cond in enumerate(opto_cds) if cond != '0'][0]
    except IndexError:
        first_stim = None
    
    t_PE = beh['reward_times']
    t_ST = beh['run_onsets']

    logfile = open(txt_path, 'r')
    def get_next_line(file):
        line = file.readline().rstrip('\n').split(",")
        if len(line) == 1:
            line = file.readline().rstrip('\n').split(",")
        return line

    line = ['$']
    t_camsync, t_camsync_by_trial, camsync_trial = [], [], []
    trial_statements = []

    while line[0].find('$') == 0:
        if line[0] == '$SY':
            t_camsync.append(float(line[1]))
            camsync_trial.append(float(line[1]))
        if line[0] == '$NT':
            t_camsync_by_trial.append(camsync_trial)
            camsync_trial = []
        line = get_next_line(logfile)

    t_PE = t_PE[5:first_stim] if first_stim is not None else t_PE[5:]
    t_ST = t_ST[5:first_stim] if first_stim is not None else t_ST[5:]

    tot_camsync = len(t_camsync)
    for sync in range(1, tot_camsync):
        dt_curr = t_camsync[sync] - t_camsync[sync - 1]
        if dt_curr < 25 or dt_curr > 75:
            if dt_curr > 0:
                raise Exception(f'{recname}: sync #{sync} invalid dt')
            else:
                overflow_sync = sync
                overflow_t = t_camsync[sync]
                overflow_t_bef = t_camsync[sync - 1]

    tot_frame = len(ctime)
    if tot_frame != pupil_area.shape[0]:
        raise Exception(f'{recname}: ctime length mismatch')

    ctime = np.array([t + t_camsync[0] for t in ctime])

    timebef = BEF * SAMP_FREQ_BEH
    timeaft = AFT * SAMP_FREQ_BEH
    
    frames_by_start, pupil_by_start = [], []
    for start in t_ST:
        window = [start - timebef, start + timeaft]
        frame_curr = [f for f in range(tot_frame) if window[0] < ctime[f] < window[1]]
        if frame_curr:
            frames_by_start.append(frame_curr)
            pupil_by_start.append(smooth_convolve(pupil_area[frame_curr], sigma=SAMP_FREQ/20))  # smoothing 
            
    if not pupil_by_start:
        print('Start alignment failed; skipped')
        continue
    min_len_start = min(len(p) for p in pupil_by_start)
    
    # organise data 
    session_traces = [p[:min_len_start] for p in pupil_by_start]
    
    # plotting
    fig, ax = plt.subplots(figsize=(2.9,2.3))
    ax.set(xlabel='time (s)', xlim=(-1,4), ylabel='pupil size')
    ax.set_xticks([0,2,4])

    avg_start = np.nanmean(session_traces, axis=0)
    sem_start = np.nanstd(session_traces, axis=0) / np.sqrt(len(pupil_by_start))
    x = np.arange(min_len_start) / SAMP_FREQ - BEF  # - 1 seccond
    ax.plot(x, avg_start, 'k', lw=2)
    ax.fill_between(x, 
                    avg_start+sem_start, 
                    avg_start-sem_start, 
                    color='k', 
                    edgecolor='none',
                    alpha=.25)
    ax.set(title=recname)
    for p in ['right', 'top']: 
        ax.spines[p].set_visible(False)
    
    fig.tight_layout()
    plt.show()
    for ext in ['.png', '.pdf']:
        fig.savefig(
            output_stem / 'single_sessions' / f'{recname}_run_aligned{ext}',
            dpi=500, bbox_inches='tight'
            )
    
    avg_start_traces.append(normalise(avg_start))
    
    
#%% quantify per-session pupil modulation
session_real_deltas = []
session_shuf_means  = []
session_shuf_stds   = []
session_shuf_bands  = []   # (low95, high95)

for trace in avg_start_traces:
    
    n = len(trace)
    t = np.arange(n) / SAMP_FREQ - BEF

    # analysis windows
    baseline_mask = (t >= -0.5) & (t <= 0)
    response_mask = (t >= 1.0) & (t <= 2.0)

    if baseline_mask.sum() == 0 or response_mask.sum() == 0:
        print('Warning: window out of bounds, skipping one session.')
        continue

    # real
    base = np.nanmean(trace[baseline_mask])
    resp = np.nanmean(trace[response_mask])
    real_delta = resp - base
    session_real_deltas.append(real_delta)

    # shuffle
    shuf_deltas = []
    for _ in range(N_SHUF):
        shift = np.random.randint(n)
        shuf_trace = np.roll(trace, shift)

        base_s = np.nanmean(shuf_trace[baseline_mask])
        resp_s = np.nanmean(shuf_trace[response_mask])

        shuf_deltas.append(resp_s - base_s)

    shuf_deltas = np.array(shuf_deltas)
    session_shuf_means.append(np.nanmean(shuf_deltas))
    session_shuf_stds.append(np.nanstd(shuf_deltas))

    low95  = np.percentile(shuf_deltas, 2.5)
    high95 = np.percentile(shuf_deltas, 97.5)
    session_shuf_bands.append((low95, high95))
    

#%% mean 
minlen_all = min(len(trace) for trace in avg_start_traces)
trimmed = np.array([
    (trace[:minlen_all] - np.nanmin(trace[:minlen_all])) /
    (np.nanmax(trace[:minlen_all]) - np.nanmin(trace[:minlen_all]))
    for trace in avg_start_traces
])

grand_avg = np.nanmean(trimmed, axis=0)
grand_sem = np.nanstd(trimmed, axis=0) / np.sqrt(trimmed.shape[0])
x_axis = np.arange(minlen_all) / 30 - 1000 / 600  # same as before

fig, ax = plt.subplots(figsize=(2.8, 2.3))
ax.plot(x_axis, grand_avg, 'k', lw=2)
ax.fill_between(x_axis, grand_avg+grand_sem, grand_avg-grand_sem,
                color='k', edgecolor='none', alpha=0.15)
ax.set(xlabel='Time from run onset (s)',
       ylabel='Norm. pupil size',
       title='Run-onset-aligned pupil trace',
       xlim=(-1, 4), xticks=[0,2,4])
for p in ['right', 'top']:
    ax.spines[p].set_visible(False)
fig.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        output_stem / f'mean_run_aligned_pupil{ext}',
        dpi=500, bbox_inches='tight'
        )
    
    
#%% violin plot for deltas 
tval, p_t = ttest_1samp(session_real_deltas, 0)
wstat, p_w = wilcoxon(session_real_deltas)

fig, ax = plt.subplots(figsize=(1.6, 2.2))

# violin
parts = ax.violinplot(session_real_deltas, positions=[1],
                      showmeans=False, showmedians=True, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('k')
    pc.set_edgecolor('none')
    pc.set_alpha(0.35)
parts['cmedians'].set_color('k')
parts['cmedians'].set_linewidth(1.2)

# scatter individual r's
ax.scatter(np.ones(len(session_real_deltas)), session_real_deltas,
           color='k', ec='none', s=10, alpha=0.5, zorder=3)

# shuffle + CI
shuf_mean = np.nanmean(session_shuf_means)
shuf_std  = np.nanmean(session_shuf_means)

lower_95 = shuf_mean - 1.96 * shuf_std
upper_95 = shuf_mean + 1.96 * shuf_std

ax.axhline(shuf_mean, color='gray', lw=1, ls='--')

ax.fill_between(
    [0, 2],
    lower_95, upper_95,
    color='gray', alpha=0.2, edgecolor='none', zorder=0
)

mean_r, sem_r = np.nanmean(session_real_deltas), sem(session_real_deltas)
ymax = np.max(session_real_deltas)
ax.text(1, ymax + 0.05*(ymax - np.min(session_real_deltas)),
        f'{mean_r:.2f} ± {sem_r:.2f}',
        ha='center', va='bottom', fontsize=7, color='k')

ax.text(1, np.min(session_real_deltas) - 0.10*(ymax - np.min(session_real_deltas)),
        f't(1-samp)={tval:.2f}, p={p_t:.2e}\n'
        f'Wilcoxon={wstat:.2f}, p={p_w:.2e}',
        ha='center', va='top', fontsize=6.5, color='k')

# formatting
ax.set(xlim=(0.5, 1.5), xticks=[],
       ylabel='Norm. pupil size change',
       title='Pupil size delta')
ax.spines[['top', 'right', 'bottom']].set_visible(False)

plt.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        output_stem / f'mean_run_aligned_deltas_violin{ext}',
        dpi=500, bbox_inches='tight'
        )