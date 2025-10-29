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
from scipy.stats import sem, chi2  # added chi2

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
BEF = 1  # seconds before run-onset
AFT = 4  # seconds after run-onset

# mahala params
N_BINS = 7
BIN_SIZE = 500          # 0–3500 ms
CALIPER_ALPHA = 0.95    # 95% chi-square cutoff
CALIPER = np.sqrt(chi2.ppf(CALIPER_ALPHA, df=N_BINS))
RIDGE = 1e-6            # ridge added to covariance for stability


#%% helper 
def compute_bin_speeds_7(trial_indices, n_bins=N_BINS, bin_size=BIN_SIZE):
    means = []
    valid = []
    total_len = n_bins * bin_size  # 3500
    for t in trial_indices:
        try:
            sp = [pt[1] for pt in speed_times[t]]
            if len(sp) < total_len:
                continue
            s = np.asarray(sp[:total_len], dtype=float)               # truncate to 0–3500 ms
            m = s.reshape(n_bins, bin_size).mean(axis=1)              # per-bin means (shape: 7,)
            means.append(m)
            valid.append(t)
        except:
            continue
    if len(means) == 0:
        return np.empty((0, n_bins)), []
    return np.vstack(means), valid    

def get_profiles(trains, trials,
                 RUN_ONSET_BIN=RUN_ONSET_BIN,
                 SAMP_FREQ=SAMP_FREQ,
                 BEF=BEF, AFT=AFT):
    profiles = []
    for trial in trials:
        curr_train = trains[trial]
        profiles.append(curr_train[
            RUN_ONSET_BIN - BEF*SAMP_FREQ : RUN_ONSET_BIN + AFT*SAMP_FREQ
        ])
    return profiles

def _session_mean_speed(trial_list, speed_times, n=3500):
    arrs = []
    for t in trial_list:
        sp = [pt[1] for pt in speed_times[t]]
        if len(sp) >= n:
            arrs.append(np.asarray(sp[:n], dtype=float))
    if len(arrs) == 0:
        return None
    return np.nanmean(np.vstack(arrs), axis=0)

def _mahalanobis_filter(X, mu, Sigma_inv, caliper):
    # returns boolean mask for rows of X with d_M <= caliper
    diffs = X - mu
    # (diffs @ Sigma_inv) * diffs, row-wise
    left = diffs @ Sigma_inv
    d2 = np.einsum('ij,ij->i', left, diffs)
    return (d2 <= caliper**2)

def _fit_mu_cov_inv(X, ridge=RIDGE):
    # estimate mean and inverse covariance with a small ridge for stability
    mu = X.mean(axis=0)
    diffs = X - mu
    # sample covariance (rows=obs)
    Sigma = (diffs.T @ diffs) / max(1, (X.shape[0] - 1))
    # ridge
    Sigma = Sigma + ridge * np.eye(Sigma.shape[0])
    # inverse (pinv is safer in case of near-singularity)
    Sigma_inv = np.linalg.pinv(Sigma)
    return mu, Sigma_inv


#%% load data 
print('loading dataframes...')
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
)
df_pyr = cell_profiles[cell_profiles['cell_identity'] == 'pyr']
pyrON = df_pyr[df_pyr['class'] == 'run-onset ON']


#%% main
early_profiles = []
late_profiles = []

# speed after speed matching for each session 
sess_early_speed_means = []
sess_late_speed_means = []

recname = ''

for cluname in pyrON.index:
    temp_recname = cluname.split(' ')[0]
    if temp_recname != recname:
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
        good_idx = [t for t in range(tot_trial) if t not in bad_idx]

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

        early_trials = []
        late_trials = []
        for trial, t in enumerate(first_licks_sec):
            if trial in bad_idx or np.isnan(t): continue
            if t < 2.5:
                early_trials.append(trial)
            elif 2.5 < t < 3.5:
                late_trials.append(trial)

        print(f'found {len(early_trials)} early trials, {len(late_trials)} late trials')

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
        
        matched_early = []
        matched_late = []
        
        if len(early_trials) < 10 or len(late_trials) < 10:
            continue

        # 7-bin speed extraction (0–3500 ms)
        E_bins, e_valid = compute_bin_speeds_7(early_trials)  # shape: (nE, 7)
        L_bins, l_valid = compute_bin_speeds_7(late_trials)   # shape: (nL, 7)

        # mutual mahalanobis caliper matching
        if len(E_bins) and len(L_bins):
            # fit early distribution
            e_mu, e_Sinv = _fit_mu_cov_inv(E_bins, ridge=RIDGE)
            # late inside early ellipsoid
            l_mask_in_e = _mahalanobis_filter(L_bins, e_mu, e_Sinv, CALIPER)

            # fit late distribution
            l_mu, l_Sinv = _fit_mu_cov_inv(L_bins, ridge=RIDGE)
            # early inside late ellipsoid
            e_mask_in_l = _mahalanobis_filter(E_bins, l_mu, l_Sinv, CALIPER)

            matched_late  = [l_valid[i] for i in np.where(l_mask_in_e)[0]]
            matched_early = [e_valid[i] for i in np.where(e_mask_in_l)[0]]
        else:
            matched_early, matched_late = [], []
        
        print(f'{len(matched_early)} early and {len(matched_late)} late trials passed mutual Mahalanobis filtering (α={CALIPER_ALPHA:.2f}, cutoff≈{CALIPER:.2f})')

        # collect session means for later plotting 
        e_mean_sp = _session_mean_speed(matched_early, speed_times, n=3500)
        l_mean_sp = _session_mean_speed(matched_late, speed_times, n=3500)
        
        if e_mean_sp is not None and l_mean_sp is not None:
            sess_early_speed_means.append(e_mean_sp)
            sess_late_speed_means.append(l_mean_sp)
        
        # plot trace examples
        fig, ax = plt.subplots(figsize=(2.8, 1.6))
        for i in range(min(10, len(matched_early))):
            sp = [pt[1] for pt in speed_times[matched_early[i]]]
            ax.plot(sp[:3500], color='grey', alpha=0.5)
        for i in range(min(10, len(matched_late))):
            sp = [pt[1] for pt in speed_times[matched_late[i]]]
            ax.plot(sp[:3500], color=(0.2, 0.35, 0.65), alpha=0.5)
        ax.set(xlabel='time (ms)', ylabel='speed (cm/s)', title=f'{recname} mahalanobis-filtered')
        fig.tight_layout()
        vis_path = os.path.join(
            r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\single_session_speed_matching',
            f'{recname}_mahalanobis_speed_traces.png'
        )
        fig.savefig(vis_path, dpi=200)
        plt.close(fig)


    # main 
    trains = all_trains[cluname]
    if len(matched_early) >= 10 and len(matched_late) >= 10:
        early_profiles.extend(get_profiles(trains, matched_early))
        late_profiles.extend(get_profiles(trains, matched_late))


#%% average speed curves post-speed matching 
if len(sess_early_speed_means) and len(sess_late_speed_means):
    E = np.vstack(sess_early_speed_means)  # shape: n_sessions x 3500
    L = np.vstack(sess_late_speed_means)

    # mean ± sem across sessions
    E_mean = np.mean(E, axis=0)
    E_sem  = sem(E, axis=0)
    L_mean = np.mean(L, axis=0)
    L_sem  = sem(L, axis=0)

    XAXIS_MS = np.arange(3500)  # milliseconds

    fig, ax = plt.subplots(figsize=(2.4, 2.1))
    ax.plot(XAXIS_MS, E_mean, c='grey', label='<2.5 s')
    ax.fill_between(XAXIS_MS, E_mean + E_sem, E_mean - E_sem, color='grey', edgecolor='none', alpha=.25)

    late_c = (0.20, 0.35, 0.65)
    ax.plot(XAXIS_MS, L_mean, c=late_c, label='2.5–3.5 s')
    ax.fill_between(XAXIS_MS, L_mean + L_sem, L_mean - L_sem, color=late_c, edgecolor='none', alpha=.25)

    ax.set(xlabel='time (ms)', ylabel='speed (cm/s)', title='Across-session speed (matched)')
    ax.legend(fontsize=5, frameon=False)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.show()
else:
    print('no sessions with valid matched early/late speed means to summarise.')


#%% mean spiking curves for early v late 
XAXIS = np.arange(5 * SAMP_FREQ) / SAMP_FREQ - 1

early_mean = np.mean(early_profiles, axis=0)
early_sem = sem(early_profiles, axis=0)

late_mean = np.mean(late_profiles, axis=0)
late_sem = sem(late_profiles, axis=0)

early_c = (0.55, 0.65, 0.95)
late_c  = (0.20, 0.35, 0.65)

fig, ax = plt.subplots(figsize=(2.3, 1.9))
ax.plot(XAXIS, early_mean, c='grey', label='<2.5')
ax.fill_between(XAXIS, early_mean+early_sem, early_mean-early_sem,
                color='grey', edgecolor='none', alpha=.25)
ax.plot(XAXIS, late_mean, c=late_c, label='2.5~3.5')
ax.fill_between(XAXIS, late_mean+late_sem, late_mean-late_sem,
                color=late_c, edgecolor='none', alpha=.25)

ax.legend(fontsize=5, frameon=False)
ax.set(xlabel='time from run-onset (s)', xlim=(-1, 4),
       ylabel='spike rate (Hz)')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout()
plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(rf'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis\all_run_onset_mean_profiles{ext}',
                dpi=300,
                bbox_inches='tight')
