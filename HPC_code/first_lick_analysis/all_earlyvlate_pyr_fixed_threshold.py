# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:56:30 2025

replicate what we have done on LC run onset-peaking cells with HPC recordings

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
pathHPCLCopt = rec_list.pathHPCLCopt
pathHPCLCtermopt = rec_list.pathHPCLCtermopt


#%% parameters 
SAMP_FREQ = 1250 
RUN_ONSET_BIN = 3750
BEF = 1  # s, how much time before run-onset to get
AFT = 4  # same as above 


#%% helper 
def get_profiles(
        trains, 
        trials,
        RUN_ONSET_BIN=RUN_ONSET_BIN, 
        SAMP_FREQ=SAMP_FREQ, 
        BEF=BEF, 
        AFT=AFT, 
        ):
    """
    extract peri-run-onset spike profiles for a list of trials

    parameters:
        - trains: np.ndarray, shape (n_trials, n_timepoints), spike train array
        - trials: list of ints, trial indices to include
        - RUN_ONSET_BIN: int, index of run-onset timepoint
        - SAMP_FREQ: int, sampling frequency (Hz)
        - BEF: float, seconds before run-onset to include in extracted profile
        - AFT: float, seconds after run-onset to include in extracted profile

    returns:
        - profiles: list of np.ndarrays, peri-run-onset spike profiles (from RUN_ONSET_BIN - BEF*s to +AFT*s)
    """
    profiles = []
    
    for trial in trials:
        curr_train = trains[trial]
        profiles.append(curr_train[
            RUN_ONSET_BIN - BEF * SAMP_FREQ : RUN_ONSET_BIN + AFT * SAMP_FREQ
            ])
    
    return profiles


#%% load data 
print('loading dataframes...')

cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
    )
df_pyr = cell_profiles[cell_profiles['cell_identity']=='pyr']  # pyramidal only 

pyrON = df_pyr[df_pyr['class']=='run-onset ON']


#%% main
## container lists 
# profile containers 
early_profiles = []
mid_profiles = [] 
late_profiles = []
verylate_profiles = []

recname = ''  # for keeping track of current/next session states to load new files 

for cluname in pyrON.index:
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
        for trial in good_idx:
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
        mid_trials = []
        late_trials = []
        verylate_trials = []
        
        for trial, t in enumerate(first_licks_sec):
            if trial in bad_idx:  # break if bad trial 
                continue
            
            if 1.5 < t < 2.5:
                early_trials.append(trial)
            if 2.5 < t < 3.5:
                mid_trials.append(trial)
            if 3.5 < t < 4.5:
                late_trials.append(trial)
            if t > 4.5:
                verylate_trials.append(trial)
        
        print(
            f'found {len(early_trials)} early trials, '
            f'{len(mid_trials)} mid trials, '
            f'{len(late_trials)} late trials and '
            f'{len(verylate_trials)} ultra-late trials'
            )
        
        all_trains = np.load(
            rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}'
            rf'\{recname}_all_trains.npy',
            allow_pickle=True
            ).item()
    
    print(cluname)
    
    trains = all_trains[cluname]

    if (len(early_trials) > 15 and len(mid_trials) > 15):
        temp_profiles = get_profiles(
            trains, early_trials
        )
        early_profiles.extend(temp_profiles)
    
        temp_profiles = get_profiles(
            trains, mid_trials
        )
        mid_profiles.extend(temp_profiles)
        
        temp_profiles = get_profiles(
            trains, late_trials
        )
        late_profiles.extend(temp_profiles)
    
        temp_profiles = get_profiles(
            trains, verylate_trials
        )
        verylate_profiles.extend(temp_profiles)


#%% plotting 
XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

early_mean = np.mean(early_profiles, axis=0)
early_sem = sem(early_profiles, axis=0)

mid_mean = np.mean(mid_profiles, axis=0)
mid_sem = sem(mid_profiles, axis=0)

late_mean = np.mean(late_profiles, axis=0)
late_sem = sem(late_profiles, axis=0)

verylate_mean = np.mean(verylate_profiles, axis=0)
verylate_sem = sem(verylate_profiles, axis=0)

early_c     = (0.55, 0.65, 0.95)  # soft light royal blue
mid_c       = (0.35, 0.50, 0.85)  # medium royal blue
late_c      = (0.20, 0.35, 0.65)  # darker royal blue
verylate_c  = (0.10, 0.25, 0.40)  # deep navy-toned blue


fig, ax = plt.subplots(figsize=(2.2, 2.5))

ax.plot(XAXIS, early_mean, c='grey', label='early')
ax.fill_between(XAXIS, early_mean+early_sem,
                       early_mean-early_sem,
                       color='grey', edgecolor='none', alpha=.25)

ax.plot(XAXIS, mid_mean, color='royalblue', label='verylate')
ax.fill_between(XAXIS, mid_mean+mid_sem,
                       mid_mean-mid_sem,
                       color='royalblue', edgecolor='none', alpha=.25)

ax.plot(XAXIS, late_mean, c=late_c, label='late')
ax.fill_between(XAXIS, late_mean+late_sem,
                       late_mean-late_sem,
                       color=late_c, edgecolor='none', alpha=.25)

ax.plot(XAXIS, verylate_mean, color=verylate_c, label='verylate')
ax.fill_between(XAXIS, verylate_mean+verylate_sem,
                       verylate_mean-verylate_sem,
                       color=verylate_c, edgecolor='none', alpha=.25)

plt.legend(fontsize=5, frameon=False)

ax.set(xlabel='time from run-onset (s)', xlim=(-1,4),
       ylabel='spike rate (Hz)')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
fig.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis'
        rf'\all_run_onset_mean_profiles{ext}',
        dpi=300,
        bbox_inches='tight'
        )