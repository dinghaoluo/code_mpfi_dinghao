# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:12:36 2025

a Python implementation of the decay-time extraction scripts in Heldman et al.
    2025:
        - applicable to both run-onset-ON and -OFF cells

@author: Dinghao Luo
"""

#%% imports 
import numpy as np 
import pandas as pd 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks


#%% parameters
SAMP_FREQ = 1250  # Hz 
GOODNESS_THRESHOLD = 0.7  # gof


#%% functions 
def compute_tau(time: np.array,
                profile: np.array, 
                peak_idx: int,
                cell_type='run-onset ON'):
    """
    fit an exponential decay to the post-peak portion of a spike rate profile and compute the decay time constant (tau).
    
    parameters:
    - time: 1d array of time values corresponding to the profile
    - profile: 1d array of spike rates (same length as time)
    - peak_idx: index of the peak or trough used as the decay starting point
    - cell_type: string indicating cell response type; default is 'run-onset ON'
    
    returns:
    - tau: estimated decay time constant (in seconds), or None if fitting fails
    - fit_params: dictionary containing fit parameters ('a', 'b'), r², and adjusted r²; or None if fitting fails
    """
    time = np.array(time)
    profile = np.array(profile)

    # subtract baseline, 18 Sept 2025
    baseline = np.min(profile)
    profile = profile - baseline

    x_data = time[peak_idx:] - time[peak_idx]
    y_data = profile[peak_idx:]

    if len(y_data) < 2: 
        print('profile length improper for fitting')
        return None, None

    try:
        popt, _ = curve_fit(exp_decay, x_data, y_data)
        a_fit, b_fit = popt
        tau = -1 / b_fit

        # predicted values
        y_pred = exp_decay(x_data, *popt)

        # r2
        ss_res = np.sum((y_data - y_pred)**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # adjusted r2
        n = len(y_data)
        p = 2
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    except RuntimeError:
        print('curve fitting failed')
        return None, None

    return tau, {
        'a': a_fit,
        'b': b_fit,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'baseline': baseline
    }

def detect_min_max(
        profile: np.array,
        cell_type='run-onset ON',
        run_onset_bin=1250,
        SAMP_FREQ=1250):
    if cell_type == 'run-onset ON':
        return np.argmax(profile[run_onset_bin:run_onset_bin+SAMP_FREQ*3])
    elif cell_type == 'run-onset OFF':
        return np.argmin(profile[run_onset_bin:run_onset_bin+SAMP_FREQ*3])
    else:
        raise ValueError("cell_type must be 'run-onset ON' or 'run-onset OFF'")

def detect_peak(profile, cell_type='run-onset ON',
                run_onset_bin=1250, SAMP_FREQ=1250,
                num_shuffles=500, p_sig=99):
    """
    detects the first significant peak (or trough) in the firing rate profile.

    parameters:
    - profile: np.array, firing rate profile over time
    - cell_type: str, 'run-onset ON' or 'run-onset OFF'
    - run_onset_bin: int, index of run-onset (default: 1250 for 1s)
    - SAMP_FREQ: int, sampling frequency in Hz
    - num_shuffles: int, number of permutations for significance testing
    - p_sig: float, percentile threshold for significance (default: 99)

    returns:
    - peak_idx: int, index of the detected peak or trough in the full profile
    """
    # search window: from run-onset to 3 seconds after
    search_start = run_onset_bin
    search_end = run_onset_bin + SAMP_FREQ * 3
    segment = profile[search_start:search_end]

    if cell_type == 'run-onset ON':
        # detect peaks
        peaks, _ = find_peaks(segment)
        if len(peaks) == 0:
            return np.argmax(segment) + search_start  # fallback

        # permutation test
        shuffled_peaks = np.zeros((num_shuffles, len(peaks)))
        for i in range(num_shuffles):
            shuffled = np.random.permutation(segment)
            shuffled_peaks[i, :] = shuffled[peaks]

        thresholds = np.percentile(shuffled_peaks, p_sig, axis=0)
        sig_peak_indices = np.where(segment[peaks] > thresholds)[0]

        if len(sig_peak_indices) > 0:
            return peaks[sig_peak_indices[0]] + search_start
        else:
            return np.argmax(segment) + search_start  # fallback

    elif cell_type == 'run-onset OFF':
        # detect troughs by inverting the signal
        troughs, _ = find_peaks(-segment)
        if len(troughs) == 0:
            return np.argmin(segment) + search_start  # fallback

        shuffled_troughs = np.zeros((num_shuffles, len(troughs)))
        for i in range(num_shuffles):
            shuffled = np.random.permutation(segment)
            shuffled_troughs[i, :] = shuffled[troughs]

        thresholds = np.percentile(shuffled_troughs, 100 - p_sig, axis=0)
        sig_trough_indices = np.where(segment[troughs] < thresholds)[0]

        if len(sig_trough_indices) > 0:
            return troughs[sig_trough_indices[0]] + search_start
        else:
            return np.argmin(segment) + search_start  # fallback

    else:
        raise ValueError("cell_type must be 'run-onset ON' or 'run-onset OFF'")

def exp_decay(x, a, b):
    return a * np.exp(b * x)

def plot_fit(time: np.array, 
             profile: np.array, 
             peak_idx: int, 
             fit_params: dict, 
             cluname: str,
             cluclass: str,
             filename=None):
    x_data = time[peak_idx:] - time[peak_idx]
    y_data = profile[peak_idx:]

    if fit_params is None:
        print('no valid fit--skipping plot')
        return

    a_fit, b_fit = fit_params['a'], fit_params['b']
    fit_curve = exp_decay(x_data, a_fit, b_fit)

    fig, ax = plt.subplots(figsize=(3,2))
    data_scatter = ax.scatter(x_data, y_data, s=.5, alpha=0.5)
    fit_line, = ax.plot(x_data, fit_curve, color='red')
    ax.set(xlabel='time from peak (s)',
           ylabel='spike rate (Hz)',
           title=f'exp. fit {cluname}')
    ax.legend([data_scatter, fit_line],
              ['data', f'fit (τ = {(-1/b_fit):.2f})'],
              frameon=False)
    
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\single_cell_decay_time'
        rf'\{cluname} {cluclass}.png',
        dpi=300,
        bbox_inches='tight'
        )
    
def plot_fit_compare(time: np.array, 
                     profile1: np.array, 
                     peak_idx1: int, 
                     fit_params1: dict,
                     profile2: np.array,
                     peak_idx2: int,
                     fit_params2: dict,
                     cluname: str,
                     cluclass: str,
                     filename=None,
                     SAVE=True):
    x_data1 = time[peak_idx1:] - time[peak_idx1]
    y_data1 = profile1[peak_idx1:]

    if fit_params1 is None:
        print('no valid fit--skipping plot')
        return

    a_fit1, b_fit1 = fit_params1['a'], fit_params1['b']
    fit_curve1 = exp_decay(x_data1, a_fit1, b_fit1)
    
    x_data2 = time[peak_idx2:] - time[peak_idx2]
    y_data2 = profile2[peak_idx2:]

    if fit_params2 is None:
        print('no valid fit--skipping plot')
        return

    a_fit1, b_fit1 = fit_params1['a'], fit_params1['b']
    fit_curve1 = exp_decay(x_data1, a_fit1, b_fit1)
    
    a_fit2, b_fit2 = fit_params2['a'], fit_params2['b']
    fit_curve2 = exp_decay(x_data2, a_fit2, b_fit2)

    fig, ax = plt.subplots(figsize=(3,2))
    data1_ln, = ax.plot(x_data1, y_data1, c='firebrick', lw=1, alpha=.5)
    fit1_line, = ax.plot(x_data1, fit_curve1, c='red', lw=1, alpha=.5)
    data2_ln, = ax.plot(x_data2, y_data2, c='firebrick', lw=1)
    fit2_line, = ax.plot(x_data2, fit_curve2, c='red', lw=1)
    ax.set(xlabel='time from peak (s)',
           ylabel='spike rate (Hz)',
           title=f'exp. fit {cluname}')
    ax.legend([data1_ln, fit1_line, data2_ln, fit2_line],
              ['data1', f'fit (τ = {(-1/b_fit1):.2f})', 'data2', f'fit (τ = {(-1/b_fit2):.2f})'],
              frameon=False)
    
    if SAVE:
        fig.savefig(
            r'Z:\Dinghao\code_dinghao\HPC_ephys\first_lick_analysis'
            r'\single_cell_decay_constant_early_v_late'
            rf'\{cluname} {cluclass}.png',
            dpi=300,
            bbox_inches='tight'
            )
        

#%% main 
def main():
    cell_profiles = pd.read_pickle(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl'
        )
    df_classified = cell_profiles[cell_profiles['class']!='run-onset unresponsive']
    
    tau_values = []
    fit_results = []
    
    for cluname, clu in df_classified.iterrows():
        print(f'\n{cluname}: {clu["class"]}')
        
        mean_prof = clu['prof_mean'][SAMP_FREQ*(3-1):SAMP_FREQ*(3+4)]
    
        # get peak or trough 
        peak_idx = detect_min_max(mean_prof, clu['class'])
    
        # compute decay constant
        time = np.arange(-SAMP_FREQ, SAMP_FREQ*4)/SAMP_FREQ  # 5 seconds 
        tau, fit_params = compute_tau(time, mean_prof, peak_idx, clu['class'])
    
        tau_values.append(tau)
        fit_results.append(fit_params)
    
        # optional display
        plot_fit(time, mean_prof, peak_idx, fit_params, cluname, clu['class'])


if __name__ == '__main__':
    main()