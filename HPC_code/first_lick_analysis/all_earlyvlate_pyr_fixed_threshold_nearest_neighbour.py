# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:56:30 2025

replicate what we have done on LC run onset-peaking cells with HPC recordings

@author: Dinghao Luo
"""


#%% imports
import os
import sys 

import numpy as np 
import pandas as pd
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt 
from scipy.stats import sem
from sklearn.neighbors import NearestNeighbors

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
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
late_profiles = [] 

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
        late_trials = []
        
        for trial, t in enumerate(first_licks_sec):
            if trial in bad_idx:  # break if bad trial 
                continue
            
            if t < 2.5:
                early_trials.append(trial)
            if 2.5 < t < 3.5:
                late_trials.append(trial)
        
        print(
            f'found {len(early_trials)} early trials, '
            f'{len(late_trials)} late trials'
            )
        
        all_trains = np.load(
            rf'Z:\Dinghao\code_dinghao\HPC_ephys\all_sessions\{recname}'
            rf'\{recname}_all_trains.npy',
            allow_pickle=True
            ).item()
        
        try:
            beh_path = os.path.join(
                r'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLC',
                f'{recname}.pkl'
            )
            with open(beh_path, 'rb') as f:
                beh = pickle.load(f)
        except FileNotFoundError:
            beh_path = os.path.join(
                r'Z:\Dinghao\code_dinghao\behaviour\all_experiments\HPCLCterm',
                f'{recname}.pkl'
            )
            with open(beh_path, 'rb') as f:
                beh = pickle.load(f)
        
        speed_times = beh['speed_times_aligned'][1:]
        
        # speed matching
        matched_early = []
        matched_late = []
        
        if len(early_trials) > 15 and len(late_trials) > 15:
            early_speeds = []
            early_valid = []
            for t in early_trials:
                try:
                    speed = [pt[1] for pt in speed_times[t]]
                    if len(speed) >= 4000:
                        vec = speed[:4000]
                    else:
                        vec = speed + [0]*(4000 - len(speed))
                    early_speeds.append(vec)
                    early_valid.append(t)
                except:
                    continue
        
            late_speeds = []
            late_valid = []
            for t in late_trials:
                try:
                    speed = [pt[1] for pt in speed_times[t]]
                    if len(speed) >= 4000:
                        vec = speed[:4000]
                    else:
                        vec = speed + [0]*(4000 - len(speed))
                    late_speeds.append(vec)
                    late_valid.append(t)
                except:
                    continue
        
            early_arr = np.array(early_speeds)
            late_arr = np.array(late_speeds)
        
            # NN: early → late
            nbrs_el = NearestNeighbors(n_neighbors=1).fit(late_arr)
            _, idx_el = nbrs_el.kneighbors(early_arr)
        
            # NN: late → early
            nbrs_le = NearestNeighbors(n_neighbors=1).fit(early_arr)
            _, idx_le = nbrs_le.kneighbors(late_arr)
        
            for ei, li in enumerate(idx_el[:, 0]):
                if idx_le[li][0] == ei:
                    matched_early.append(early_valid[ei])
                    matched_late.append(late_valid[li])
        
            print(f'found {len(matched_early)} matched trial pairs with full speed profile match')
        
            # plot matched traces
            fig, ax = plt.subplots(figsize=(2.8, 1.6))
            for i in range(min(10, len(matched_early))):
                t_early = matched_early[i]
                t_late = matched_late[i]
                sp_e = [pt[1] for pt in speed_times[t_early]]
                sp_l = [pt[1] for pt in speed_times[t_late]]
                ax.plot(sp_e[:4000], color='grey', alpha=0.5)
                ax.plot(sp_l[:4000], color=(0.2, 0.35, 0.65), alpha=0.5)
            ax.set(xlabel='time (ms)', ylabel='speed (cm/s)', title=f'{recname} matched traces')
            fig.tight_layout()
        
            vis_path = os.path.join(
                r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\single_session_speed_matching',
                f'{recname}_matched_speed_traces.png'
            )
            fig.savefig(vis_path, dpi=200)
            plt.close(fig)
            
        else:  # if not enough trials
            continue
    
    
    # main
    print(cluname)
    
    trains = all_trains[cluname]

    if len(matched_early) >= 10:
        early_profiles.extend(get_profiles(trains, matched_early))
        late_profiles.extend(get_profiles(trains, matched_late))



#%% plotting 
XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

early_mean = np.mean(early_profiles, axis=0)
early_sem = sem(early_profiles, axis=0)

late_mean = np.mean(late_profiles, axis=0)
late_sem = sem(late_profiles, axis=0)

early_c     = (0.55, 0.65, 0.95)  # soft light royal blue
late_c      = (0.20, 0.35, 0.65)  # darker royal blue


fig, ax = plt.subplots(figsize=(2.4, 2.1))

ax.plot(XAXIS, early_mean, c='grey', label='<2.5')
ax.fill_between(XAXIS, early_mean+early_sem,
                       early_mean-early_sem,
                       color='grey', edgecolor='none', alpha=.25)

ax.plot(XAXIS, late_mean, c=late_c, label='2.5~3.5')
ax.fill_between(XAXIS, late_mean+late_sem,
                       late_mean-late_sem,
                       color=late_c, edgecolor='none', alpha=.25)

plt.legend(fontsize=5, frameon=False)

ax.set(xlabel='time from run-onset (s)', xlim=(-1,4),
       ylabel='spike rate (Hz)')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
fig.tight_layout()
plt.show()

# for ext in ['.png', '.pdf']:
#     fig.savefig(
#         r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis'
#         rf'\all_run_onset_mean_profiles{ext}',
#         dpi=300,
#         bbox_inches='tight'
#         )