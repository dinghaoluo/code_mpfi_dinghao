# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:03:06 2025

attempts to regress the movement out of red 

@author: Dinghao Luo
"""

#%% imports
import numpy as np
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt


#%% functions
def compute_motion_absdiff(R):
    """
    compute frame-wise motion index via pixel-wise absolute difference in the RED channel.

    parameters:
    - R: array (T, H, W), observed red channel

    returns:
    - M: 1D array length T, motion regressor
    """
    T = R.shape[0]
    diffs = np.abs(R[1:] - R[:-1]).reshape(T-1, -1)
    M = np.empty(T, dtype=np.float32)
    M[1:] = diffs.mean(axis=1)
    M[0] = M[1]
    return M


def compute_motion_corr(R):
    """
    compute frame-wise motion index via negative Pearson correlation between successive frames.

    parameters:
    - R: array (T, H, W), observed red channel

    returns:
    - M: 1D array length T, motion regressor
    """
    T = R.shape[0]
    M = np.empty(T, dtype=np.float32)
    for t in range(1, T):
        f0 = R[t-1].ravel()
        f1 = R[t].ravel()
        corr = np.corrcoef(f1, f0)[0,1]
        M[t] = -corr
    M[0] = M[1]
    return M


def fit_red_motion(R_ts, M, robust=True):
    """
    fit R_ts = a*M + c via Huber (or OLS) regression.

    parameters:
    - R_ts: 1D array length T, mean red per frame
    - M:    1D array length T, motion regressor
    - robust: bool, if True use HuberRegressor

    returns:
    - a: float, motion coefficient
    - c: float, intercept (baseline red)
    """
    X = np.vstack([M, np.ones_like(M)]).T
    if robust:
        model = HuberRegressor(fit_intercept=False)
        model.fit(X, R_ts)
        a, c = model.coef_
    else:
        a, c = np.linalg.lstsq(X, R_ts, rcond=None)[0]
    return a, c


def correct_red_movie(R, M, a, add_baseline=False, c=0):
    """
    subtract motion-related component from the red movie.

    parameters:
    - R:           array (T, H, W), observed red channel
    - M:           1D array length T, motion regressor
    - a:           float, motion coefficient
    - add_baseline: bool, if True re-add the intercept c
    - c:           float, intercept term (baseline)
    """
    R_corr = R - a * M[:, None, None]
    if add_baseline:
        R_corr += c
    return R_corr


#%% main 
# ops = np.load(r'Z:\Jingyu\2P_Recording\AC967\AC967-20250213\04\rigid_reg\suite2p\plane0\ops.npy', allow_pickle=True).item()
ops = np.load(r'Z:\Jingyu\2P_Recording\AC976\AC976-20250424\06\RegOnly\suite2p\plane0\ops.npy', allow_pickle=True).item()
T, H, W = ops['nframes'], ops['Ly'], ops['Lx']

# load red channel movie
# mov = np.memmap(r'Z:\Jingyu\2P_Recording\AC967\AC967-20250213\04\rigid_reg\suite2p\plane0\data_chan2.bin',
#                 mode='r', dtype='int16', shape=(T, H, W)).astype(np.float32)
mov = np.memmap(r'Z:\Jingyu\2P_Recording\AC976\AC976-20250424\06\RegOnly\suite2p\plane0\data_chan2.bin',
                mode='r', dtype='int16', shape=(T, H, W)).astype(np.float32)

#%% processing 
# mean 
R_ts = mov.mean(axis=(1, 2))

# motion regressors
M_abs = compute_motion_absdiff(mov)
M_corr = compute_motion_corr(mov)

# fit motion models
a_abs, c_abs = fit_red_motion(R_ts, M_abs, robust=True)
a_corr, c_corr = fit_red_motion(R_ts, M_corr, robust=True)

print(f'Absolute diff regressor: a = {a_abs:.4f}, c = {c_abs:.4f}')
print(f'Correlation  regressor: a = {a_corr:.4f}, c = {c_corr:.4f}')

# corrected time‑courses (global means)
Rcorr_abs_ts = R_ts - a_abs * M_abs + c_abs  # keep baseline for display
Rcorr_corr_ts = R_ts - a_corr * M_corr + c_corr


#%% abs diff plotting 
# compute the three traces
removed_ts   =   a_abs * M_abs
corrected_ts = R_ts - a_abs * M_abs

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# original trace 
axes[0].plot(R_ts, color='tab:blue')
axes[0].set_ylabel('Original\nMean red')
axes[0].set_title('Original Mean Red Trace')

# removed component
axes[1].plot(removed_ts, color='tab:orange')
axes[1].set_ylabel('Removed\n(a·M)')
axes[1].set_title('Motion Component Removed')

# corrected trace 
axes[2].plot(corrected_ts, color='tab:green')
axes[2].set_ylabel('Corrected\nMean red')
axes[2].set_xlabel('Frame')
axes[2].set_title('Corrected Mean Red Trace')

fig.suptitle('abs-diff-based correction')

plt.tight_layout()

plt.show()


#%% corr plotting 
# compute the three traces
removed_ts   =   a_corr * M_corr
corrected_ts = R_ts - a_corr * M_corr

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# original trace 
axes[0].plot(R_ts, color='tab:blue')
axes[0].set_ylabel('Original\nMean red')
axes[0].set_title('Original Mean Red Trace')

# removed component
axes[1].plot(removed_ts, color='tab:orange')
axes[1].set_ylabel('Removed\n(a·M)')
axes[1].set_title('Motion Component Removed')

# corrected trace 
axes[2].plot(corrected_ts, color='tab:green')
axes[2].set_ylabel('Corrected\nMean red')
axes[2].set_xlabel('Frame')
axes[2].set_title('Corrected Mean Red Trace')

fig.suptitle('correlation-based correction')

plt.tight_layout()

plt.show()