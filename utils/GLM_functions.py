# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:14:41 2025

helper functions for GLM building 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np
import statsmodels.api as sm


#%% utils: basis functions and design matrices
def lick_rate_last5s(lick_times, onset_time, window=5.0):
    """
    compute lick rate (licks/sec) in the last 5 s before run onset.

    parameters:
    - lick_times: 1d array of lick times (s)
    - onset_time: run onset time (s)
    - window: lookback window (s)

    returns:
    - rate: float (licks/sec)
    """
    mask = (lick_times >= onset_time - window) & (lick_times < onset_time)
    n_licks = np.sum(mask)
    return n_licks / window if window > 0 else np.nan

def first_lick_to_reward_last_trial(lick_times_trials, reward_times, ti):
    if ti == 0:
        return np.nan
    try:
        last_first_lick = lick_times_trials[ti-1][0]
    except IndexError:  # if no licks 
        return np.nan
    last_rew = reward_times[ti-1]
    if np.isnan(last_rew): return np.nan
    return (last_rew - last_first_lick) / 1000.0  # convert ms → s

def time_since_last_reward(reward_times, onset_time, trial_index):
    last_reward_time = reward_times[trial_index - 1]
    if np.isnan(last_reward_time) or np.isnan(onset_time):
        return np.nan
    else:
        last_reward_time /= 1000.0
    return (onset_time - last_reward_time)

def stop_duration_before_onset(timestamps_s, speeds_cm_s, reward_times, run_onsets, trial_idx, speed_thresh=10):
    """
    computes stop duration before run onset.

    parameters:
    - timestamps_s: 1d array of behavioural timestamps (s)
    - speeds_cm_s: 1d array of running speed (cm/s)
    - reward_times: list of reward times per trial (s)
    - run_onsets: list or array of run-onset times (s)
    - trial_idx: index of current trial
    - speed_thresh: speed threshold for defining stop (cm/s), default 10

    returns:
    - stop_dur: duration (s) between first dip below threshold after previous reward
                and the run onset of current trial. np.nan if not measurable.
    """
    try:
        # time of last reward
        if trial_idx == 0 or np.isnan(reward_times[trial_idx - 1]):
            return np.nan
        last_rew_t = reward_times[trial_idx - 1] / 1000.0  # last reward time of previous trial
        onset_t = run_onsets[trial_idx] / 1000.0

        # mask for post-reward to current onset
        mask = (timestamps_s > last_rew_t) & (timestamps_s < onset_t)
        if not np.any(mask):
            return np.nan

        post_rew_times = timestamps_s[mask]
        post_rew_speeds = speeds_cm_s[mask]

        # find first below-threshold time
        below_idx = np.where(post_rew_speeds < speed_thresh)[0]
        if len(below_idx) == 0:
            return np.nan

        first_below_t = post_rew_times[below_idx[0]]
        stop_dur = onset_t - first_below_t
        return stop_dur if stop_dur > 0 else np.nan

    except Exception as e:
        print(e)
        return np.nan

def mean_speed_prev_trial(timestamps_s, speeds_cm_s, run_onsets_s, ti):
    if ti == 0: 
        return np.nan
    # trial boundaries defined by successive run onsets; last trial ends at this onset
    t_start = run_onsets_s[ti-1]
    t_end   = run_onsets_s[ti] if ti < len(run_onsets_s) else timestamps_s[-1]
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        return np.nan
    mask = (timestamps_s >= t_start) & (timestamps_s < t_end)
    if not np.any(mask): 
        return np.nan
    return float(np.nanmean(speeds_cm_s[mask]))


def prev_run_amp(amplitudes, ti):
    """
    amplitude of the previous trial, if any.

    parameters:
    - amplitudes: list/array of trial amplitudes up to current trial
    - ti: current trial index

    returns:
    - float, np.nan if no previous trial
    """
    if ti == 0:
        return np.nan
    return amplitudes[ti - 1]


def preonset_rate(train, samp_freq=1250, onset_idx=3750, window=(2.5, 1.5)):
    """
    mean firing rate in [onset - window[0], onset - window[1]] (s).
    """
    lo = int(onset_idx - window[0]*samp_freq)
    hi = int(onset_idx - window[1]*samp_freq)
    return float(np.nanmean(train[lo:hi]))


#%% target (run onset rates)
def run_onset_amplitude(spk_rate: np.ndarray, sr: float, onset_idx: int) -> float:
    """
    sum of spike rates in [-0.5, +0.5] s window around run-onset.

    parameters:
    - spk_rate: spike rate vector (hz)
    - sr: sampling rate (hz)
    - onset_idx: sample index of run-onset

    returns:
    - amp: summed spike rate in window (float)
    """
    half_win = int(0.5 * sr)
    lo = onset_idx - half_win
    hi = onset_idx + half_win
    return float(np.nanmean(spk_rate[lo:hi]))


#%% fit GLM
def fit_glm_log_gaussian(X: np.ndarray, y: np.ndarray, eps: float = 1e-6):
    """
    fit a gaussian glm to log(y+eps).

    parameters:
    - X: design matrix (n_trials, n_features)
    - y: target vector (n_trials,)
    - eps: small constant to avoid log(0)

    returns:
    - result: fitted statsmodels glm result
    """
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or inf after clipping")

    if np.nanstd(y) < 1e-8:
        return None  # or skip fit
    
    y_log = np.log(y.astype(float) + eps)
    Xc = sm.add_constant(X, has_constant='add')
    fam = sm.families.Gaussian()
    model = sm.GLM(y_log, Xc, family=fam)
    return model.fit()