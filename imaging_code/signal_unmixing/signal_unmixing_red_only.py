# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:03:06 2025

attempts to regress the movement out of red, for either the whole field or a 
    single ROI

@author: Dinghao Luo
"""

#%% imports
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% core functions
def compute_motion_absdiff(R):
    T = R.shape[0]
    diffs = np.abs(R[1:] - R[:-1]).reshape(T-1, -1)
    M = np.empty(T, dtype=np.float32)
    M[1:] = diffs.mean(axis=1)
    M[0] = M[1]
    return M

def compute_motion_corr(R):
    T = R.shape[0]
    M = np.empty(T, dtype=np.float32)
    for t in range(1, T):
        f0 = R[t-1].ravel()
        f1 = R[t].ravel()
        corr = np.corrcoef(f1, f0)[0,1]
        M[t] = -corr
    M[0] = M[1]
    return M

def fit_motion_regression(R_ts, M, robust=True):
    X = np.vstack([M, np.ones_like(M)]).T
    if robust:
        model = HuberRegressor(fit_intercept=False)
        model.fit(X, R_ts)
        a, c = model.coef_
    else:
        a, c = np.linalg.lstsq(X, R_ts, rcond=None)[0]
    return a, c


#%% main function
def regress_motion_red(bin_path, ops_path=None, roi_idx=None):
    '''
    performs motion regression on a red channel movie, either on
    the whole field (if roi_idx=None) or on a single ROI's trace.

    parameters:
    - bin_path: path to data_chan2.bin
    - ops_path: path to ops.npy; if None, inferred from bin_path
    - roi_idx: index of ROI; if None, uses whole-field mean

    returns:
    - results: dict with keys
        - 'a_abs', 'c_abs', 'a_corr', 'c_corr'
        - 'R_ts', 'M_abs', 'M_corr'
        - 'Rcorr_abs_ts', 'Rcorr_corr_ts'
    '''
    # brute force recname
    if 'AC' in bin_path:
        div = bin_path.find('-202')  # the most likely occurrence 
        recdate = bin_path[div-5 : div+9]  # full recdate 
        recsess = bin_path[div+10 : div+12]  # sess
        recname = f'{recdate}-{recsess}'
    else:
        recname = 'unknown'
    
    print(f'processing {recname}')
    
    folder = os.path.dirname(bin_path)
    if ops_path is None:
        ops_path = os.path.join(folder, 'ops.npy')

    ops = np.load(ops_path, allow_pickle=True).item()
    T, H, W = ops['nframes'], ops['Ly'], ops['Lx']

    if roi_idx is None:
        # whole-field mode
        mov = np.memmap(bin_path, mode='r', dtype='int16', shape=(T, H, W)).astype(np.float32)
        R_ts = mov.mean(axis=(1,2))
        M_abs = compute_motion_absdiff(mov)
        M_corr = compute_motion_corr(mov)
    else:
        # single ROI mode
        # stat_path = os.path.join(folder, 'stat.npy')
        F_chan2_path = os.path.join(folder, 'F_chan2.npy')

        # stat = np.load(stat_path, allow_pickle=True)
        F_chan2 = np.load(F_chan2_path)

        R_ts = F_chan2[:, roi_idx]

        # compute simpler motion regressors directly on ROI trace
        M_abs = np.abs(R_ts[1:] - R_ts[:-1])
        M_abs = np.insert(M_abs, 0, M_abs[0])

        M_corr = np.empty_like(R_ts)
        for t in range(1, len(R_ts)):
            # for single scalar traces, correlation is ill-defined
            # so approximate with negative diff
            M_corr[t] = -(R_ts[t] - R_ts[t-1])
        M_corr[0] = M_corr[1]

    # fit regressions
    print('fitting regressions...')
    a_abs, c_abs = fit_motion_regression(R_ts, M_abs, robust=True)
    a_corr, c_corr = fit_motion_regression(R_ts, M_corr, robust=True)

    print(f'{"ROI " + str(roi_idx) if roi_idx is not None else "Whole field"}:')
    print(f'absolute diff regressor: a = {a_abs:.4f}, c = {c_abs:.4f}')
    print(f'correlation  regressor: a = {a_corr:.4f}, c = {c_corr:.4f}')

    # compute corrected traces
    Rcorr_abs_ts = R_ts - a_abs * M_abs + c_abs
    Rcorr_corr_ts = R_ts - a_corr * M_corr + c_corr

    # plotting 
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex='col')

    # abs diff plots
    axes[0,0].plot(R_ts, color='tab:blue')
    axes[0,0].set_ylabel('Original')
    axes[0,0].set_title('AbsDiff: Original Trace')

    axes[1,0].plot(a_abs*M_abs, color='tab:orange')
    axes[1,0].set_ylabel('Removed')
    axes[1,0].set_title('AbsDiff: Motion Component')

    axes[2,0].plot(Rcorr_abs_ts, color='tab:green')
    axes[2,0].set_ylabel('Corrected')
    axes[2,0].set_xlabel('Frame')
    axes[2,0].set_title('AbsDiff: Corrected Trace')

    # corr plots
    axes[0,1].plot(R_ts, color='tab:blue')
    axes[0,1].set_title('Corr: Original Trace')

    axes[1,1].plot(a_corr*M_corr, color='tab:orange')
    axes[1,1].set_title('Corr: Motion Component')

    axes[2,1].plot(Rcorr_corr_ts, color='tab:green')
    axes[2,1].set_xlabel('Frame')
    axes[2,1].set_title('Corr: Corrected Trace')

    for ax in axes.flat:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.suptitle(recname)

    plt.tight_layout()
    plt.show()
    
    for ext in ['.png', '.pdf']:
        fig.savefig(os.path.join(
            r'Z:\Dinghao\code_dinghao\tests\signal_unmixing',
            f'{recname}{ext}'
            ),
            dpi=300,
            bbox_inches='tight')
    
    return {
        'a_abs': a_abs, 'c_abs': c_abs,
        'a_corr': a_corr, 'c_corr': c_corr,
        'R_ts': R_ts,
        'M_abs': M_abs,
        'M_corr': M_corr,
        'Rcorr_abs_ts': Rcorr_abs_ts,
        'Rcorr_corr_ts': Rcorr_corr_ts
    }


#%% run
bin_path = r'Z:\Jingyu\2P_Recording\AC976\AC976-20250424\06\RegOnly\suite2p\plane0\data_chan2.bin'
roi_idx = None  # this corresponds to the roi index in stat.npy

results = regress_motion_red(
    bin_path, 
    roi_idx=roi_idx  # if this is left as None then we do whole-field correction
    )
