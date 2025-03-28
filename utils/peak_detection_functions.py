# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:39:04 2023
Modified on Wed 18 Dec 2024 15:26:12:
    - added GPU acceleration support using CuPy
    - tidies up the peak detection function

peak detection (originally written for LC run-onset responses) for any recording

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
from itertools import groupby


#%% functions
def neu_shuffle(trains,
                around=6, 
                bootstrap=1000, 
                samp_freq=1250, 
                GPU_AVAILABLE=False,
                VERBOSE=True):
    """
    shuffles spike trains and calculates mean and significance thresholds

    parameters:
    - trains (numpy.ndarray): spike train data (trials x timepoints)
    - around (int): time window in seconds, default 6
    - bootstrap (int): number of shuffles, default 1000
    - samp_freq (int): sampling frequency in Hz, default 1250
    - GPU_AVAILABLE (bool): if true, uses GPU for calculations, default false
    - VERBOSE (bool): if true, shows a progress bar, default true

    returns:
    - tuple: (mean shuffled spike profile, percentile thresholds)
    """
    if VERBOSE:
        from tqdm import tqdm
        iterator = tqdm(range(bootstrap), desc=f'peak detection ({"GPU" if GPU_AVAILABLE else "CPU"})')
    else:
        iterator = range(bootstrap)

    if GPU_AVAILABLE: 
        import cupy as xp
    else: 
        xp = np

    trains = xp.asarray(trains)
    tot_trials = trains.shape[0]
    shuf_mean_arr = xp.zeros([bootstrap, samp_freq * around])
    
    indices = xp.arange(samp_freq * around)
    
    for shuf in iterator:
        rand_shift = xp.random.randint(1, samp_freq * around, tot_trials)
        shifted_indices = (indices[None, :] - rand_shift[:, None]) % (samp_freq * around)
        shuf_arr = trains[xp.arange(tot_trials)[:, None], shifted_indices]
        shuf_mean_arr[shuf, :] = xp.mean(shuf_arr, axis=0)
    
    shuf_mean = xp.mean(shuf_mean_arr, axis=0)
    shuf_sig = xp.percentile(shuf_mean_arr, [99.9, 99, 95, 50, 5, 1, .1], axis=0)
    
    if GPU_AVAILABLE:
        shuf_mean = shuf_mean.get()
        shuf_sig = [sig.get() for sig in shuf_sig]
    
    return shuf_mean, shuf_sig

def peak_detection(trains, 
                   first_stim=None,
                   around=6,
                   peak_width=1,
                   min_peak=.2,
                   samp_freq=1250,
                   centre_bin=3750,
                   bootstrap=1000,
                   no_boundary=False,
                   GPU_AVAILABLE=False,
                   VERBOSE=True):
    """
    detects spiking peaks around run-onset and evaluates significance

    parameters:
    - trains: spike train data (trials x time bins)
    - first_stim: index of the first stimulation trial, default None
    - around: time window in seconds, default 6
    - peak_width: expected width of peaks in seconds, default 1
    - min_peak: minimum peak length in seconds, default 0.2
    - samp_freq: sampling frequency in Hz, default 1250
    - centre_bin: centre bin index, default 3750
    - bootstrap: number of times to shuffle, default 1000
    - no_boundary: if True, allows peaks that touch the edges of the window
    - GPU_AVAILABLE: if True, uses GPU for calculations, default false

    returns:
    - tuple:
        1. bool: whether a significant peak was detected
        2. numpy.ndarray: mean spike profile around the run-onset
        3. numpy.ndarray: significance thresholds for peaks
    """
    if first_stim is None:
        baseline_trains = trains
    else:
        baseline_trains = trains[:first_stim]
    baseline_trains = [t[:samp_freq * around] 
                       if len(t) > samp_freq * around 
                       else np.pad(t, (0, samp_freq * around - len(t)), mode='reflect')
                       for t in baseline_trains]
    
    shuf_mean, shuf_sigs = neu_shuffle(
        baseline_trains, 
        around=around,
        samp_freq=samp_freq,
        bootstrap=bootstrap,
        GPU_AVAILABLE=GPU_AVAILABLE,
        VERBOSE=VERBOSE
        )

    peak_window = [int(centre_bin - samp_freq * (peak_width / 2)),
                   int(centre_bin + samp_freq * (peak_width / 2))]
    
    shuf_sig_99_around = shuf_sigs[1][peak_window[0]:peak_window[1]] * samp_freq
    mean_train_around = np.mean(baseline_trains, axis=0)[peak_window[0]:peak_window[1]] * samp_freq

    diffs_mean_shuf = mean_train_around - shuf_sig_99_around
    idx_diffs = [diff > 0 for diff in diffs_mean_shuf]

    tot_groups = len(list(groupby(idx_diffs)))
    groups_0_1 = groupby(idx_diffs)
    
    max_truths = 0
    for group_count, (key, group) in enumerate(groups_0_1):
        if key:  # only look at True blocks
            consecutive_truths = sum(group)
            if no_boundary:
                if consecutive_truths > max_truths:
                    max_truths = consecutive_truths
            else:
                if group_count != 0 and group_count != tot_groups - 1:
                    if consecutive_truths > max_truths:
                        max_truths = consecutive_truths

    return max_truths > int(min_peak * samp_freq), mean_train_around, shuf_sig_99_around

def plot_peak_v_shuf(cluname,
                     mean_prof, 
                     shuf_prof,
                     peak,
                     savepath,
                     peak_width=1,
                     samp_freq=1250):
    """
    plots the mean spike profile against shuffled profile for a single cell

    parameters:
    - cluname (str): identifier for the cell
    - mean_prof (numpy.ndarray): mean spike profile around run-onset
    - shuf_prof (numpy.ndarray): shuffled spike profile around run-onset
    - peak (boolean): whether this cell has a peak
    - samp_freq (int): sampling frequency in Hz, default 1250

    saves the plot as a png and pdf in the specified directory
    """
    import sys 
    import matplotlib.pyplot as plt 
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
    from common import mpl_formatting
    mpl_formatting()
        
    fig, ax = plt.subplots(figsize=(2,1.6))

    ax.set(title=f'{cluname}\nRO_peak={peak}',
           xlim=(-peak_width/2, peak_width/2),
           xlabel='time from run-onset (s)',
           ylabel='spike rate (Hz)')
    
    xaxis = np.arange(-samp_freq*peak_width/2, samp_freq*peak_width/2) / samp_freq
    mean_line, = ax.plot(xaxis, mean_prof)
    shuf_line, = ax.plot(xaxis, shuf_prof, color='grey')
    
    ax.legend([mean_line, shuf_line],
              ['mean', 'shuf.'],
              frameon=False,
              fontsize=5)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    # for ext in ['.png', '.pdf']:
    #     fig.savefig(
    #         f'{savepath}{ext}',
    #         dpi=300,
    #         bbox_inches='tight'
    #         )
    fig.savefig(
        f'{savepath}.png',
        dpi=300,
        bbox_inches='tight'
        )
        
    plt.close(fig)