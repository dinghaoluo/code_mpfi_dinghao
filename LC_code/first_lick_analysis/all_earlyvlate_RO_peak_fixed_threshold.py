# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:37:16 2023
Modified on Tue 22 Apr 2025:
    - reworking the script to calculate multiple other factors other than 
        peak amplitude 

loop over all cells for early v late trials

@author: Dinghao Luo
"""


#%% imports
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import sys 
import scipy.io as sio
from scipy.stats import sem, ranksums

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
from plotting_functions import plot_violin_with_scatter
mpl_formatting()

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC


#%% load data 
print('loading data...')
cell_prop = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )


#%% get keys for different categories of cells 
clu_keys = list(cell_prop.index)

tagged_keys = []; putative_keys = []
tagged_RO_keys = []; putative_RO_keys = []
RO_keys = []  # pooled run-onset bursting cells 
for clu in cell_prop.itertuples():
    if clu.identity == 'tagged':
        tagged_keys.append(clu.Index)
        if clu.run_onset_peak:
            tagged_RO_keys.append(clu.Index)
            RO_keys.append(clu.Index)
    if clu.identity == 'putative':
        putative_keys.append(clu.Index)
        if clu.run_onset_peak:
            putative_RO_keys.append(clu.Index)
            RO_keys.append(clu.Index)
    

#%% parameters for processing
SAMP_FREQ = 1250 
RUN_ONSET_BIN = 3750
BEF = 1  # s, how much time before run-onset to get
AFT = 4  # same as above 
WINDOW_HALF_SIZE = .5

RO_WINDOW = [
    int(RUN_ONSET_BIN - WINDOW_HALF_SIZE * SAMP_FREQ), 
    int(RUN_ONSET_BIN + WINDOW_HALF_SIZE * SAMP_FREQ)
    ]  # window for spike summation, half a sec around run onsets


#%% support functions
def get_peak_std(
        trains, 
        trials,
        RO_WINDOW,
        RUN_ONSET_BIN=3750,
        SAMP_FREQ=1250
        ):
    times = []
    for trial in trials:
        curr_train = trains[trial][RO_WINDOW[0] : RO_WINDOW[1]]
        if curr_train.size == 0:
            continue
        peak_idx = np.argmax(curr_train)
        times.append((peak_idx - RUN_ONSET_BIN) / SAMP_FREQ)
    if len(times)>1:
        return np.std(times)
    else:
        return np.nan

def get_profiles_and_spike_rates(
        trains, 
        trials,
        RO_WINDOW,
        RUN_ONSET_BIN=3750, 
        SAMP_FREQ=1250, 
        BEF=1, 
        AFT=4, 
        ):
    """
    extract peri-run-onset spike profiles and spike rates for a list of trials

    parameters:
        - trains: np.ndarray, shape (n_trials, n_timepoints), spike train array
        - trials: list of ints, trial indices to include
        - RO_WINDOW: list of two ints, index range for spike rate calculation around run-onset
        - RUN_ONSET_BIN: int, index of run-onset timepoint
        - SAMP_FREQ: int, sampling frequency (Hz)
        - BEF: float, seconds before run-onset to include in extracted profile
        - AFT: float, seconds after run-onset to include in extracted profile

    returns:
        - profiles: list of np.ndarrays, peri-run-onset spike profiles (from RUN_ONSET_BIN - BEF*s to +AFT*s)
        - spike_rates: list of floats, mean spike rate within RO_WINDOW
    """
    profiles = []
    spike_rates = []
    
    for trial in trials:
        curr_train = trains[trial]
        profiles.append(curr_train[
            RUN_ONSET_BIN - BEF * SAMP_FREQ : RUN_ONSET_BIN + AFT * SAMP_FREQ
            ])
        spike_rates.append(np.mean(
            curr_train[RO_WINDOW[0] : RO_WINDOW[1]]
            ))
    
    return profiles, spike_rates


#%% main
## container lists 
# profile containers 
early_profiles = []
earlymid_profiles = [] 
midlate_profiles = []
late_profiles = []

# spike rate containers 
early_spike_rates = []
earlymid_spike_rates = []
midlate_spike_rates = []
late_spike_rates = [] 

# new containers for peak timings
early_peak_std = []
earlymid_peak_std = []
midlate_peak_std = []
late_peak_std = []

recname = ''  # for keeping track of current/next session states to load new files 

for cluname in RO_keys:
    temp_recname = cluname.split(' ')[0]
    if temp_recname != recname:  # load new files if starting a new session 
        recname = temp_recname
        print(f'\n{recname}')
        
        alignRun = sio.loadmat(
            rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:-3]}'
            rf'\{recname}\{recname}_DataStructure_mazeSection1_'
            r'TrialType1_alignRun_msess1.mat'
            )
        licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
        starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
        tot_trial = licks.shape[0]
        
        behPar = sio.loadmat(
            rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:-3]}'
            rf'\{recname}\{recname}_DataStructure_mazeSection1_'
            r'TrialType1_behPar_msess1.mat'
            )
        bad_idx = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                             # -1 to account for 0 being an empty trial
        good_idx = np.arange(behPar['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
        good_idx = [t for t in good_idx if t not in bad_idx]
        
        # get first-lick time
        first_licks = []
        for trial in range(tot_trial):
            lk = [l for l in licks[trial] 
                  if l-starts[trial] > .5*SAMP_FREQ]  # only if the animal does not lick in the first half a second (carry-over licks)
            
            if len(lk)==0:  # no licks in the current trial
                first_licks.append(np.nan)
            else:  # if there are licks, append relative time of first lick
                first_licks.extend(lk[0]-starts[trial])
        
        # convert first licks to seconds
        first_licks_sec = np.array(first_licks) / SAMP_FREQ
        
        # initialise trial groups
        early_trials = []
        earlymid_trials = []
        midlate_trials = []
        late_trials = []
        
        for trial, t in enumerate(first_licks_sec):
            if trial in bad_idx:  # break if bad trial 
                continue
            
            if .5 < t < 2.5:
                early_trials.append(trial)
            elif .5 < t < 2.8:
                earlymid_trials.append(trial)
            elif t > 3.1 and t <= 3.4:
                midlate_trials.append(trial)
            elif t > 3.4:
                late_trials.append(trial)
        
        print(
            f'found {len(early_trials)} early trials, '
            f'{len(earlymid_trials)} earlymid trials, '
            f'{len(midlate_trials)} midlate trials and '
            f'{len(late_trials)} ultra-midlate trials'
            )
        
        all_trains = np.load(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}'
            rf'\{recname}_all_trains.npy',
            allow_pickle=True
            ).item()
    
    print(cluname)
    
    trains = all_trains[cluname]
    
    if len(early_trials) > 10:
        temp_profiles, temp_spike_rates = get_profiles_and_spike_rates(
            trains, early_trials, RO_WINDOW
            )
        early_profiles.extend(temp_profiles)
        early_spike_rates.extend(temp_spike_rates)
        early_peak_std.append(
            get_peak_std(trains, early_trials, RO_WINDOW)
            )
    
    if len(earlymid_trials) > 10:
        temp_profiles, temp_spike_rates = get_profiles_and_spike_rates(
            trains, earlymid_trials, RO_WINDOW
            )
        earlymid_profiles.extend(temp_profiles)
        earlymid_spike_rates.extend(temp_spike_rates)
        earlymid_peak_std.append(
            get_peak_std(trains, earlymid_trials, RO_WINDOW)
            )
    
    if len(midlate_trials) > 10:
        temp_profiles, temp_spike_rates = get_profiles_and_spike_rates(
            trains, midlate_trials, RO_WINDOW
            )
        midlate_profiles.extend(temp_profiles)
        midlate_spike_rates.extend(temp_spike_rates)
        midlate_peak_std.append(
            get_peak_std(trains, midlate_trials, RO_WINDOW)
            )
    
    if len(late_trials) > 10:
        temp_profiles, temp_spike_rates = get_profiles_and_spike_rates(
            trains, late_trials, RO_WINDOW
            )
        late_profiles.extend(temp_profiles)
        late_spike_rates.extend(temp_spike_rates)
        late_peak_std.append(
            get_peak_std(trains, late_trials, RO_WINDOW)
            )


#%% plotting 
XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

early_mean = np.mean(early_profiles, axis=0)
early_sem = sem(early_profiles, axis=0)

earlymid_mean = np.mean(earlymid_profiles, axis=0)
earlymid_sem = sem(earlymid_profiles, axis=0)

midlate_mean = np.mean(midlate_profiles, axis=0)
midlate_sem = sem(midlate_profiles, axis=0)

late_mean = np.mean(late_profiles, axis=0)
late_sem = sem(late_profiles, axis=0)

early_c     = (0.55, 0.65, 0.95)  # soft light royal blue
earlymid_c  = (0.35, 0.50, 0.85)  # medium royal blue
midlate_c   = (0.20, 0.35, 0.65)  # darker royal blue
late_c      = (0.10, 0.25, 0.40)  # deep navy-toned blue


fig, ax = plt.subplots(figsize=(2.2, 2.5))

ax.plot(XAXIS, early_mean, c='grey', label='early')
ax.fill_between(XAXIS, early_mean+early_sem,
                       early_mean-early_sem,
                       color='grey', edgecolor='none', alpha=.25)

ax.plot(XAXIS, earlymid_mean, color='royalblue', label='late')
ax.fill_between(XAXIS, earlymid_mean+earlymid_sem,
                       earlymid_mean-earlymid_sem,
                       color='royalblue', edgecolor='none', alpha=.25)

ax.plot(XAXIS, midlate_mean, c=midlate_c, label='midlate')
ax.fill_between(XAXIS, midlate_mean+midlate_sem,
                       midlate_mean-midlate_sem,
                       color=midlate_c, edgecolor='none', alpha=.25)

ax.plot(XAXIS, late_mean, color=late_c, label='late')
ax.fill_between(XAXIS, late_mean+late_sem,
                       late_mean-late_sem,
                       color=late_c, edgecolor='none', alpha=.25)

plt.legend(fontsize=5, frameon=False)

ax.set(xlabel='time from run-onset (s)', xlim=(-1,4),
       ylabel='spike rate (Hz)')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
comparison_pairs = [
    ('early', early_spike_rates, 'earlymid', earlymid_spike_rates),
    ('earlymid', earlymid_spike_rates, 'midlate', midlate_spike_rates),
    ('midlate', midlate_spike_rates, 'late', late_spike_rates),
    ('early', early_spike_rates, 'midlate', midlate_spike_rates),
    ('early', early_spike_rates, 'late', late_spike_rates),
    ('earlymid', earlymid_spike_rates, 'late', late_spike_rates)
]

title_lines = []
for name1, a, name2, b in comparison_pairs:
    a = [x for x in a if not np.isnan(x)]
    b = [x for x in b if not np.isnan(x)]
    stat, p = ranksums(a, b)
    if p < 0.0001:
        p_str = 'p<0.0001'
    else:
        p_str = f'p={p:.3g}'
    title_lines.append(f'{name1} vs {name2}: {p_str}')

ax.set_title('\n' + '\n'.join(title_lines), fontsize=6)
    
fig.tight_layout()
plt.show()

# for ext in ['.png', '.pdf']:
#     fig.savefig(
#         r'Z:\Dinghao\code_dinghao\LC_ephys\first_lick_analysis'
#         rf'\all_run_onset_mean_profiles{ext}',
#         dpi=300,
#         bbox_inches='tight'
#         )


#%% std comparison 
# clean NaNs
early_std = [x for x in early_peak_std if not np.isnan(x)]
late_std = [x for x in late_peak_std if not np.isnan(x)]

# plot 
plot_violin_with_scatter(
    early_std, late_std,
    early_c, late_c,
    paired=False,
    showscatter=True,
    xticklabels=['early', 'late'],
    ylabel='peak timing std.',
    title='run-onset peak time var.',
    dpi=300,
    save=True,
    savepath=(
        r'Z:\Dinghao\code_dinghao\LC_ephys'
        r'\first_lick_analysis\early_late_peak_timing_std'
        )
    )