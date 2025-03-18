# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:12:36 2025

a Python implementation of the decay-time extraction scripts in Heldman et al.
    2025:
        - applicable to both run-onset-ON and -OFF cells

@author: LuoD
"""

#%% imports 
import numpy as np 
import pandas as pd 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 


#%% parameters
SAMP_FREQ = 1250  # Hz 


#%% functions 
def compute_tau(time: np.array,
                profile: np.array, 
                peak_idx: int,  # note that this is the raw index, not time
                cell_type='run-onset ON'):
    time = np.array(time)
    profile = np.array(profile)

    # extract time and firing rate from peak/trough onwards
    x_data = time[peak_idx:] - time[peak_idx]  # shift time so peak starts at x=0
    y_data = profile[peak_idx:]
    
    # check if there is enough data to fit to 
    if len(y_data) < 2: 
        print('profile length improper for fitting')
        return None, None

    # initial parameter guess: peak firing rate, assumed decay rate, for better convergence
    p0 = [y_data[0], -1] if cell_type == 'run-onset ON' else [y_data[0], 1]
    
    try:
        popt, _ = curve_fit(exp_decay, x_data, y_data, p0=p0)
        a_fit, b_fit = popt
        tau = -1 / b_fit  # decay time constant
    except RuntimeError:
        print('curve fitting failed')
        return None, None

    return tau, {'a': a_fit, 'b': b_fit}

def detect_min_max(
        profile: np.array,
        cell_type='run-onset ON'):
    if cell_type == 'run-onset ON':
        return np.argmax(profile)
    elif cell_type == 'run-onset OFF':
        return np.argmin(profile)
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
                     filename=None):
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