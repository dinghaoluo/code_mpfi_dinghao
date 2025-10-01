# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:14:41 2025

helper functions for GLM building 

@author: Dinghao Luo
"""

#%% imports 
import numpy as np
from scipy.ndimage import gaussian_filter1d
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

def time_since_last_reward(reward_times, onset_time):
    """
    compute time since the most recent reward before run-onset.
    returns np.nan if no previous reward.
    """
    past_rewards = [rt for rt in reward_times if rt < onset_time]
    if len(past_rewards) == 0:
        return np.nan
    return onset_time - past_rewards[-1]


def stop_fraction_before_onset(timestamps_s, speeds, onset_time, window=3.0, thresh=5.0):
    mask = (timestamps_s >= onset_time - window) & (timestamps_s < onset_time)
    if not np.any(mask):
        return np.nan
    return np.mean(speeds[mask] < thresh)

def mean_speed_last_trial(timestamps_s, speeds, run_onsets_s, ti):
    if ti == 0:
        return np.nan
    start, end = run_onsets_s[ti-1], run_onsets_s[ti]
    mask = (timestamps_s >= start) & (timestamps_s < end)
    if not np.any(mask):
        return np.nan
    return np.nanmean(speeds[mask])


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


def baseline_rate(spk_rate, sr, onset_idx, window=(2.5, 1.5)):
    """
    mean firing rate in [onset - window[0], onset - window[1]] (s).
    """
    lo = int(onset_idx - window[0]*sr)
    hi = int(onset_idx - window[1]*sr)
    return float(np.nanmean(spk_rate[lo:hi]))


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